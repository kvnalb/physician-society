"""Batch LLM survey over cohort physicians."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Set, Tuple

import pandas as pd

from simulation.llm_client import call_llm, get_api_key, make_client
from simulation.persona_methods import PROMPT_VERSION, build_prompts_for_persona_variant
from simulation.questions_io import Question, load_questions

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "data" / "output" / "runs"
DEFAULT_COHORT = PROJECT_ROOT / "data" / "output" / "tirzepatide_simulation_cohort_100.tsv"
DEFAULT_MODEL_TOGETHER = "zai-org/GLM-5.1"
DEFAULT_MODEL_OPENAI = "gpt-4o-mini"

_cache_lock = threading.Lock()


def _synthetic_response_rows(
    cohort_df: pd.DataFrame,
    questions: List[Question],
    *,
    model: str,
    temperature: float,
) -> List[dict[str, Any]]:
    """Deterministic A/B rows for CI or environments without an API key (not real LLM output)."""
    rows: List[dict[str, Any]] = []
    for _, row in cohort_df.iterrows():
        npi = str(row.get("npi", ""))
        for q in questions:
            n_opt = len(q.options)
            if n_opt == 0:
                continue
            h = int(hashlib.md5(f"{npi}|{q.question_id}".encode()).hexdigest(), 16)
            ia = h % n_opt
            ib = (h + 1) % n_opt
            opt_a = q.options[ia].option_id
            opt_b = q.options[ib].option_id
            for letter, parsed in [("a", opt_a), ("b", opt_b)]:
                rows.append(
                    {
                        "npi": npi,
                        "question_id": q.question_id,
                        "method": f"method_{letter}",
                        "model": model,
                        "temperature": temperature,
                        "persona_variant": "offline_seed",
                        "run_id": None,
                        "prompt_version": PROMPT_VERSION,
                        "raw": "",
                        "parsed_option": parsed,
                        "reasoning": "Offline deterministic seed (no LLM call).",
                        "latency_ms": 0,
                        "error": None,
                        "cache_hit": False,
                        "offline_seed": True,
                    }
                )
    return rows


def run_offline_seed_demo(
    *,
    cohort_path: Path,
    output_dir: Path,
    limit_npis: Optional[int],
    write_demo_bundle: bool,
    run_id: Optional[str],
) -> int:
    questions = load_questions()
    if not cohort_path.is_file():
        print(f"Cohort not found: {cohort_path}")
        return 1
    df = pd.read_csv(cohort_path, sep="\t", low_memory=False)
    if limit_npis is not None:
        df = df.head(limit_npis)
    if df.empty:
        print("Cohort is empty.")
        return 1
    model = "offline_seed"
    temperature = 0.0
    rows_out = _synthetic_response_rows(df, questions, model=model, temperature=temperature)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "responses.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as fh:
        for r in rows_out:
            fh.write(json.dumps(r) + "\n")
    print(f"Wrote {jsonl_path.relative_to(PROJECT_ROOT)} (offline seed, n={len(rows_out)} rows).")
    _write_run_manifest(
        output_dir=output_dir,
        run_id=run_id,
        persona_variant="offline_seed",
        model=model,
        temperature=temperature,
        methods=["A", "B"],
        questions_spec="all",
        concurrency=1,
        limit_npis=limit_npis,
        n_npis=len(df),
        cohort_path=cohort_path,
        llm_provider="offline",
        base_url_set=False,
        offline=True,
    )
    if write_demo_bundle:
        _write_demo_bundle(
            rows_out=rows_out,
            cohort_df=df,
            questions=questions,
            methods=["A", "B"],
            model=model,
        )
        demo_summary_path = PROJECT_ROOT / "artifacts" / "demo" / "summary.json"
        data = json.loads(demo_summary_path.read_text(encoding="utf-8"))
        data["is_placeholder"] = False
        data["offline_seed"] = True
        demo_summary_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print("Updated artifacts/demo/summary.json (offline seed, not real LLM output).")
    return 0


def _cache_key(
    *,
    model: str,
    temperature: float,
    method: str,
    question_id: str,
    npi: str,
    persona_variant: str,
    prompt_version: str,
) -> str:
    h = hashlib.sha256(
        f"{prompt_version}|{persona_variant}|{model}|{temperature}|{method}|"
        f"{question_id}|{npi}".encode()
    ).hexdigest()[:32]
    return h


def _read_cache(cache_dir: Path, key: str, max_age_hours: float = 24.0) -> Optional[dict[str, Any]]:
    p = cache_dir / f"{key}.json"
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        ts = float(data.get("ts", 0))
        if (datetime.now().timestamp() - ts) / 3600.0 > max_age_hours:
            return None
        return data
    except (json.JSONDecodeError, OSError, TypeError):
        return None


def _write_run_manifest(
    *,
    output_dir: Path,
    run_id: Optional[str],
    persona_variant: str,
    model: str,
    temperature: float,
    methods: List[str],
    questions_spec: str,
    concurrency: int,
    limit_npis: Optional[int],
    n_npis: int,
    cohort_path: Path,
    llm_provider: str,
    base_url_set: bool,
    offline: bool,
) -> None:
    """Snapshot run configuration next to responses.jsonl for reproducibility."""
    try:
        cohort_rel = str(cohort_path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        cohort_rel = str(cohort_path)
    manifest = {
        "schema_version": 1,
        "written_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "offline": offline,
        "run_id": run_id,
        "persona_variant": persona_variant,
        "prompt_version": PROMPT_VERSION,
        "llm_provider": llm_provider,
        "model": model,
        "temperature": temperature,
        "methods": methods,
        "questions_spec": questions_spec,
        "concurrency": concurrency,
        "limit_npis": limit_npis,
        "n_npis_in_run": n_npis,
        "cohort_path": cohort_rel,
        "base_url_set": base_url_set,
    }
    p = output_dir / "run_manifest.json"
    p.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {p.relative_to(PROJECT_ROOT)}")


def _write_cache(cache_dir: Path, key: str, payload: dict[str, Any]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = cache_dir / f"{key}.json"
    out = {**payload, "ts": datetime.now().timestamp()}
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")


def _select_questions(all_q: List[Question], spec: str) -> List[Question]:
    spec = spec.strip().lower()
    if spec in ("all", "*"):
        return all_q
    ids = {x.strip() for x in spec.split(",") if x.strip()}
    return [q for q in all_q if q.question_id in ids]


def _execute_one_llm_task(
    *,
    task_idx: int,
    npi: str,
    method: str,
    q: Question,
    rowd: dict[str, Any],
    persona_variant: str,
    run_id: Optional[str],
    model: str,
    temperature: float,
    client: Any,
    cache_dir: Path,
) -> Tuple[int, dict[str, Any], bool, bool]:
    """Returns (idx, row_dict, cache_hit, had_error)."""
    ck = _cache_key(
        model=model,
        temperature=temperature,
        method=method,
        question_id=q.question_id,
        npi=npi,
        persona_variant=persona_variant,
        prompt_version=PROMPT_VERSION,
    )
    with _cache_lock:
        cached = _read_cache(cache_dir, ck)
    if cached and cached.get("parsed_option"):
        return (
            task_idx,
            {
                "npi": npi,
                "question_id": q.question_id,
                "method": f"method_{method.lower()}",
                "model": model,
                "temperature": temperature,
                "persona_variant": persona_variant,
                "run_id": run_id,
                "prompt_version": PROMPT_VERSION,
                "raw": cached.get("raw", ""),
                "parsed_option": cached.get("parsed_option"),
                "reasoning": cached.get("reasoning", ""),
                "latency_ms": int(cached.get("latency_ms", 0)),
                "error": None,
                "cache_hit": True,
            },
            True,
            False,
        )

    system, user = build_prompts_for_persona_variant(persona_variant, method, rowd, q)
    raw, parsed, reason, lat, err = call_llm(
        client,
        system=system,
        user=user,
        model=model,
        temperature=temperature,
        question=q,
    )
    had_error = bool(err)
    if parsed:
        payload = {
            "raw": raw,
            "parsed_option": parsed,
            "reasoning": reason,
            "latency_ms": lat,
        }
        with _cache_lock:
            _write_cache(cache_dir, ck, payload)

    return (
        task_idx,
        {
            "npi": npi,
            "question_id": q.question_id,
            "method": f"method_{method.lower()}",
            "model": model,
            "temperature": temperature,
            "persona_variant": persona_variant,
            "run_id": run_id,
            "prompt_version": PROMPT_VERSION,
            "raw": raw,
            "parsed_option": parsed,
            "reasoning": reason,
            "latency_ms": lat,
            "error": err,
            "cache_hit": False,
        },
        False,
        had_error,
    )


def run(
    *,
    cohort_path: Path,
    output_dir: Path,
    limit_npis: Optional[int],
    questions_spec: str,
    methods: List[str],
    model: str,
    temperature: float,
    llm_provider: str,
    base_url: Optional[str],
    api_key: Optional[str],
    save_demo_bundle: bool,
    persona_variant: str,
    concurrency: int,
    run_id: Optional[str],
) -> int:
    questions = load_questions()
    qs = _select_questions(questions, questions_spec)
    if not qs:
        print("No questions selected.")
        return 1

    if not cohort_path.is_file():
        print(f"Cohort not found: {cohort_path}")
        print("Run scripts/06_tirzepatide_simulation_cohort.py first or pass --cohort-path.")
        return 1

    df = pd.read_csv(cohort_path, sep="\t", low_memory=False, dtype={"npi": str})
    if limit_npis is not None:
        df = df.head(limit_npis)

    client = make_client(api_key=api_key, base_url=base_url)
    if client is None:
        print("OPENAI_API_KEY (or TOGETHER_API_KEY with --base-url) not set; skipping API calls.")
        print("Use --save-as-demo-bundle only after setting a key, or commit pre-built artifacts/demo/.")
        return 0

    cache_dir = output_dir / "cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "responses.jsonl"

    work: List[Tuple[int, str, str, Question, dict[str, Any]]] = []
    t = 0
    for _, row in df.iterrows():
        rowd = row.to_dict()
        npi = str(rowd.get("npi", ""))
        for method in methods:
            for q in qs:
                work.append((t, npi, method, q, rowd))
                t += 1

    results: List[Optional[dict[str, Any]]] = [None] * len(work)
    cache_hits = 0
    errors = 0

    def _submit(w: Tuple[int, str, str, Question, dict[str, Any]]) -> Tuple[int, dict[str, Any], bool, bool]:
        idx, npi, method, q, rowd = w
        return _execute_one_llm_task(
            task_idx=idx,
            npi=npi,
            method=method,
            q=q,
            rowd=rowd,
            persona_variant=persona_variant,
            run_id=run_id,
            model=model,
            temperature=temperature,
            client=client,
            cache_dir=cache_dir,
        )

    n_workers = max(1, min(concurrency, len(work)))
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(_submit, w) for w in work]
        for fut in as_completed(futures):
            idx, row_dict, hit, had_err = fut.result()
            results[idx] = row_dict
            if hit:
                cache_hits += 1
            if had_err:
                errors += 1

    rows_out = [r for r in results if r is not None]

    with open(jsonl_path, "w", encoding="utf-8") as fh:
        for r in rows_out:
            fh.write(json.dumps(r) + "\n")

    n_calls = len(rows_out)
    print(
        f"Wrote {jsonl_path.relative_to(PROJECT_ROOT)}: {n_calls} rows, "
        f"{cache_hits} cache hits, {errors} rows with LLM errors, "
        f"concurrency={n_workers}, persona_variant={persona_variant}, prompt_v={PROMPT_VERSION}."
    )

    _write_run_manifest(
        output_dir=output_dir,
        run_id=run_id,
        persona_variant=persona_variant,
        model=model,
        temperature=temperature,
        methods=methods,
        questions_spec=questions_spec,
        concurrency=n_workers,
        limit_npis=limit_npis,
        n_npis=len(df),
        cohort_path=cohort_path,
        llm_provider=llm_provider,
        base_url_set=bool(base_url),
        offline=False,
    )

    if save_demo_bundle:
        _write_demo_bundle(
            rows_out=rows_out,
            cohort_df=df,
            questions=qs,
            methods=methods,
            model=model,
        )

    return 0


def _write_demo_bundle(
    *,
    rows_out: List[dict[str, Any]],
    cohort_df: pd.DataFrame,
    questions: List[Question],
    methods: List[str],
    model: str,
) -> None:
    demo_dir = PROJECT_ROOT / "artifacts" / "demo"
    demo_dir.mkdir(parents=True, exist_ok=True)

    # method_comparison: per question, distribution for method_a and method_b
    by_q: Dict[str, Dict[str, Counter]] = defaultdict(lambda: {"method_a": Counter(), "method_b": Counter()})
    for r in rows_out:
        if not r.get("parsed_option"):
            continue
        qid = r["question_id"]
        m = r["method"]
        if m == "method_a":
            by_q[qid]["method_a"][r["parsed_option"]] += 1
        elif m == "method_b":
            by_q[qid]["method_b"][r["parsed_option"]] += 1

    method_comparison: Dict[str, Any] = {}
    for q in questions:
        d = by_q[q.question_id]
        method_comparison[q.question_id] = {
            "method_a_distribution": dict(d["method_a"]),
            "method_b_distribution": dict(d["method_b"]),
        }

    # adoption by archetype (empirical from cohort)
    actual_adoption: Dict[str, Any] = {}
    if "adoption_archetype" in cohort_df.columns and "has_tirzepatide_2023" in cohort_df.columns:
        g = cohort_df.groupby("adoption_archetype")["has_tirzepatide_2023"].agg(["mean", "count"])
        for idx, row in g.iterrows():
            actual_adoption[str(idx)] = {"rate": float(row["mean"]), "n": int(row["count"])}

    # sample_personas: merge a few NPIs with q1 answers
    npis = list(cohort_df["npi"].astype(str).head(10))
    sample_personas: List[Dict[str, Any]] = []
    q1 = questions[0].question_id if questions else None
    for npi in npis:
        sub = cohort_df[cohort_df["npi"].astype(str) == npi]
        if sub.empty:
            continue
        srow = sub.iloc[0]
        entry: Dict[str, Any] = {
            "npi": npi,
            "specialty": str(srow.get("specialty", "")),
            "city": str(srow.get("city", "")),
            "state": str(srow.get("state", "")),
            "adoption_archetype": str(srow.get("adoption_archetype", "")),
        }
        if q1:
            for mlabel, letter in [("q1_response_method_a", "a"), ("q1_response_method_b", "b")]:
                hit = next(
                    (
                        r
                        for r in rows_out
                        if r["npi"] == npi
                        and r["question_id"] == q1
                        and r["method"] == f"method_{letter}"
                    ),
                    None,
                )
                entry[mlabel] = hit.get("parsed_option") if hit else None
        sample_personas.append(entry)

    summary = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "cohort_note": "Regenerate after tirzepatide_simulation_cohort_100.tsv is updated.",
        "n_npis": int(len(cohort_df)),
        "n_questions": len(questions),
        "n_methods": len(methods),
        "model": model,
        "total_api_calls": len(rows_out),
        "total_cost_usd": None,
        "method_comparison": method_comparison,
        "adoption_by_archetype_actual": actual_adoption,
        "sample_personas": sample_personas,
    }
    (demo_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    sample_path = demo_dir / "sample_responses.jsonl"
    with open(sample_path, "w", encoding="utf-8") as fh:
        for line in rows_out[:10]:
            fh.write(json.dumps(line) + "\n")
    print(f"Wrote {demo_dir / 'summary.json'} and {sample_path.name}")


def main() -> None:
    p = argparse.ArgumentParser(description="Run LLM survey batch over physician cohort.")
    p.add_argument("--cohort-path", type=Path, default=DEFAULT_COHORT)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Explicit output directory (overrides --run-id). Default: data/output/runs/<run-id>.",
    )
    p.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Subdirectory under data/output/runs/ (default: latest). E.g. v0_naive, v2_ab_rich.",
    )
    p.add_argument("--limit-npis", type=int, default=None)
    p.add_argument("--questions", type=str, default="all", help="all or comma-separated question_ids")
    p.add_argument("--method", type=str, default="both", choices=["A", "B", "both"])
    p.add_argument(
        "--persona-variant",
        type=str,
        default="ab",
        choices=["naive", "b", "a", "ab", "a_numeric"],
        help="Prompting strategy: naive|b|a force single stream (method_a rows only); ab|a_numeric for A/B.",
    )
    p.add_argument("--concurrency", type=int, default=12, help="Parallel LLM calls (default 12).")
    p.add_argument(
        "--provider",
        type=str,
        default="together",
        choices=["together", "openai"],
        help="together = native Together SDK (default); openai = OpenAI client (optional --base-url).",
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"Chat model id. Default: {DEFAULT_MODEL_TOGETHER} with --provider together, "
        f"{DEFAULT_MODEL_OPENAI} with --provider openai.",
    )
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Only for --provider openai: OpenAI-compatible API base URL (e.g. https://api.together.xyz/v1).",
    )
    p.add_argument(
        "--save-as-demo-bundle",
        action="store_true",
        help="Write artifacts/demo/summary.json and sample_responses.jsonl",
    )
    p.add_argument(
        "--offline-seed-demo",
        action="store_true",
        help="No API: deterministic A/B responses from cohort + questions (for CI / missing keys)",
    )
    p.add_argument(
        "--write-demo-bundle",
        action="store_true",
        help="With --offline-seed-demo, also refresh artifacts/demo/ (default: do not overwrite demo).",
    )
    args = p.parse_args()

    if args.model is None:
        args.model = DEFAULT_MODEL_TOGETHER if args.provider == "together" else DEFAULT_MODEL_OPENAI

    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = RUNS_ROOT / (args.run_id or "latest")

    if args.offline_seed_demo:
        raise SystemExit(
            run_offline_seed_demo(
                cohort_path=args.cohort_path,
                output_dir=output_dir,
                limit_npis=args.limit_npis,
                write_demo_bundle=args.write_demo_bundle,
                run_id=args.run_id,
            )
        )

    pv = args.persona_variant.strip().lower()
    if pv in ("naive", "b", "a"):
        methods = ["A"]
        if args.method != "A" and args.method != "both":
            print("Note: naive/b/a variants emit a single stream labeled method_a; ignoring --method for list size.")
    elif args.method == "both":
        methods = ["A", "B"]
    else:
        methods = [args.method]

    if args.provider == "together":
        key = get_api_key("together")
    else:
        key = get_api_key("openai")
        if args.base_url and os.environ.get("TOGETHER_API_KEY"):
            key = os.environ.get("TOGETHER_API_KEY") or key

    raise SystemExit(
        run(
            cohort_path=args.cohort_path,
            output_dir=output_dir,
            limit_npis=args.limit_npis,
            questions_spec=args.questions,
            methods=methods,
            model=args.model,
            temperature=args.temperature,
            llm_provider=args.provider,
            base_url=args.base_url,
            api_key=key,
            save_demo_bundle=args.save_as_demo_bundle,
            persona_variant=pv,
            concurrency=args.concurrency,
            run_id=args.run_id,
        )
    )


if __name__ == "__main__":
    main()
