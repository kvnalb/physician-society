"""Batch LLM survey over cohort physicians."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Set, Tuple

import pandas as pd
from tqdm import tqdm

from simulation.env_bootstrap import load_local_dotenv
from simulation.llm_client import call_llm_survey_json, get_api_key, make_client
from simulation.persona_methods import PROMPT_VERSION, build_survey_prompts_for_persona_variant
from simulation.questions_io import Question, load_questions
from simulation.responses_schema import (
    RESPONSE_ROW_SCHEMA_VERSION,
    flatten_survey_rows,
    responses_filename_for_model,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "data" / "output" / "runs"
DEFAULT_COHORT = PROJECT_ROOT / "data" / "output" / "tirzepatide_simulation_cohort_100.tsv"
DEFAULT_MODEL_TOGETHER = "zai-org/GLM-5.1"
DEFAULT_MODEL_OPENAI = "gpt-4o-mini"

_cache_lock = threading.Lock()


def _rel_project(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _synthetic_survey_rows_v2(
    cohort_df: pd.DataFrame,
    questions: List[Question],
    *,
    temperature: float,
) -> List[dict[str, Any]]:
    """Deterministic v2 rows (one per NPI) for CI or environments without an API key."""
    rows: List[dict[str, Any]] = []
    for _, row in cohort_df.iterrows():
        npi = str(row.get("npi", ""))
        method_a: Dict[str, Any] = {}
        for q in questions:
            n_opt = len(q.options)
            if n_opt == 0:
                continue
            h = int(hashlib.md5(f"{npi}|{q.question_id}".encode()).hexdigest(), 16)
            ia = h % n_opt
            opt_a = q.options[ia].option_id
            reason = "Offline deterministic seed (no LLM call)."
            method_a[q.question_id] = {"option_id": opt_a, "reasoning": reason}
        rows.append(
            {
                "schema_version": RESPONSE_ROW_SCHEMA_VERSION,
                "npi": npi,
                "persona_variant": "offline_seed",
                "run_id": None,
                "prompt_version": PROMPT_VERSION,
                "temperature": temperature,
                "method_a": method_a,
                "raw_by_method": {"method_a": ""},
                "latency_ms_by_method": {"method_a": 0},
                "survey_error_by_method": {"method_a": None},
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
    rows_out = _synthetic_survey_rows_v2(df, questions, temperature=temperature)
    output_dir.mkdir(parents=True, exist_ok=True)
    responses_name = responses_filename_for_model(model)
    jsonl_path = output_dir / responses_name
    with open(jsonl_path, "w", encoding="utf-8") as fh:
        for r in rows_out:
            fh.write(json.dumps(r) + "\n")
    print(f"Wrote {_rel_project(jsonl_path)} (offline seed, n={len(rows_out)} NPI rows).")
    _write_run_manifest(
        output_dir=output_dir,
        run_id=run_id,
        persona_variant="offline_seed",
        model=model,
        temperature=temperature,
        methods=["A"],
        questions_spec="all",
        concurrency=1,
        limit_npis=limit_npis,
        n_npis=len(df),
        cohort_path=cohort_path,
        llm_provider="offline",
        base_url_set=False,
        offline=True,
        responses_filename=responses_name,
        response_row_schema_version=RESPONSE_ROW_SCHEMA_VERSION,
    )
    if write_demo_bundle:
        _write_demo_bundle(
            rows_out=rows_out,
            cohort_df=df,
            questions=questions,
            methods=["A"],
            model=model,
            n_llm_calls=0,
        )
        demo_summary_path = PROJECT_ROOT / "artifacts" / "demo" / "summary.json"
        data = json.loads(demo_summary_path.read_text(encoding="utf-8"))
        data["is_placeholder"] = False
        data["offline_seed"] = True
        demo_summary_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print("Updated artifacts/demo/summary.json (offline seed, not real LLM output).")
    return 0


def _cache_key_survey(
    *,
    model: str,
    temperature: float,
    method: str,
    npi: str,
    persona_variant: str,
    prompt_version: str,
    question_ids_sig: str,
) -> str:
    h = hashlib.sha256(
        f"{prompt_version}|{persona_variant}|{model}|{temperature}|{method}|"
        f"{question_ids_sig}|{npi}".encode()
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
    responses_filename: str,
    response_row_schema_version: int,
    shuffle_questions: bool = False,
    shuffle_seed: Optional[int] = None,
) -> None:
    """Snapshot run configuration next to the responses JSONL for reproducibility."""
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
        "responses_filename": responses_filename,
        "response_row_schema_version": response_row_schema_version,
        "shuffle_questions": shuffle_questions,
        "shuffle_seed": shuffle_seed,
    }
    p = output_dir / "run_manifest.json"
    p.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {_rel_project(p)}")


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


def _cache_has_full_survey(cached: dict[str, Any], qs: List[Question]) -> bool:
    ans = cached.get("answers")
    if not isinstance(ans, dict):
        return False
    return all(
        q.question_id in ans
        and isinstance(ans[q.question_id], dict)
        and ans[q.question_id].get("option_id")
        for q in qs
    )


def _execute_one_npi_method_survey(
    *,
    task_idx: int,
    npi: str,
    method: str,
    rowd: dict[str, Any],
    persona_variant: str,
    run_id: Optional[str],
    model: str,
    temperature: float,
    client: Any,
    cache_dir: Path,
    qs_ordered: List[Question],
) -> Tuple[int, str, str, dict[str, Any], bool, bool]:
    """
    One LLM call per (NPI, method) for the full question set.
    Returns (task_idx, npi, method_key, partial_row_fragment, cache_hit, had_error).
    partial_row_fragment has keys: method_block, raw, latency_ms, error, cache_hit
    where method_block is dict question_id -> {option_id, reasoning}.
    """
    method_key = f"method_{method.lower()}"
    q_sig = ",".join(q.question_id for q in qs_ordered)
    ck = _cache_key_survey(
        model=model,
        temperature=temperature,
        method=method,
        npi=npi,
        persona_variant=persona_variant,
        prompt_version=PROMPT_VERSION,
        question_ids_sig=q_sig,
    )
    with _cache_lock:
        cached = _read_cache(cache_dir, ck)
    if cached and _cache_has_full_survey(cached, qs_ordered):
        ans = cached.get("answers")
        assert isinstance(ans, dict)
        return (
            task_idx,
            npi,
            method_key,
            {
                "method_block": ans,
                "raw": str(cached.get("raw", "")),
                "latency_ms": int(cached.get("latency_ms", 0)),
                "error": None,
                "cache_hit": True,
            },
            True,
            False,
        )

    system, user = build_survey_prompts_for_persona_variant(persona_variant, method, rowd, qs_ordered)
    raw, parsed, lat, err = call_llm_survey_json(
        client,
        system=system,
        user=user,
        model=model,
        temperature=temperature,
        questions=qs_ordered,
    )
    had_error = len(parsed) == 0
    if not err and len(parsed) == len(qs_ordered):
        with _cache_lock:
            _write_cache(cache_dir, ck, {"answers": parsed, "raw": raw, "latency_ms": lat})

    return (
        task_idx,
        npi,
        method_key,
        {
            "method_block": parsed,
            "raw": raw,
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
    shuffle_questions: bool = False,
    shuffle_seed: int = 0,
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

    client = make_client(api_key=api_key, base_url=base_url, provider=llm_provider)
    if client is None:
        if llm_provider == "together":
            print("No API key for Together: set TOGETHER_API_KEY (or OPENAI_API_KEY if you use that env).")
        else:
            print("OPENAI_API_KEY not set (or TOGETHER_API_KEY when using --base-url with Together).")
        print("Use --save-as-demo-bundle only after setting a key, or commit pre-built artifacts/demo/.")
        return 0

    qs_survey = list(qs)
    if shuffle_questions:
        random.Random(int(shuffle_seed)).shuffle(qs_survey)

    cache_dir = output_dir / "cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    responses_name = responses_filename_for_model(model)
    jsonl_path = output_dir / responses_name

    work: List[Tuple[int, str, str, dict[str, Any]]] = []
    t = 0
    for _, row in df.iterrows():
        rowd = row.to_dict()
        npi = str(rowd.get("npi", ""))
        for method in methods:
            work.append((t, npi, method, rowd))
            t += 1

    results: List[Optional[Tuple[str, str, dict[str, Any]]]] = [None] * len(work)
    cache_hits = 0
    errors = 0

    def _submit(w: Tuple[int, str, str, dict[str, Any]]) -> Tuple[int, str, str, dict[str, Any], bool, bool]:
        idx, npi, method, rowd = w
        return _execute_one_npi_method_survey(
            task_idx=idx,
            npi=npi,
            method=method,
            rowd=rowd,
            persona_variant=persona_variant,
            run_id=run_id,
            model=model,
            temperature=temperature,
            client=client,
            cache_dir=cache_dir,
            qs_ordered=qs_survey,
        )

    n_workers = max(1, min(concurrency, len(work)))
    n_tasks = len(work)
    desc = f"LLM survey ({n_tasks} calls, workers={n_workers})"
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(_submit, w) for w in work]
        for fut in tqdm(
            as_completed(futures),
            total=n_tasks,
            desc=desc,
            unit="call",
            file=sys.stdout,
            mininterval=0.5,
        ):
            idx, npi, method_key, frag, hit, had_err = fut.result()
            results[idx] = (npi, method_key, frag)
            if hit:
                cache_hits += 1
            if had_err:
                errors += 1

    by_npi: Dict[str, dict[str, Any]] = {}
    for item in results:
        if item is None:
            continue
        npi, method_key, frag = item
        row = by_npi.setdefault(
            npi,
            {
                "schema_version": RESPONSE_ROW_SCHEMA_VERSION,
                "npi": npi,
                "persona_variant": persona_variant,
                "run_id": run_id,
                "prompt_version": PROMPT_VERSION,
                "temperature": temperature,
                "raw_by_method": {},
                "latency_ms_by_method": {},
                "survey_error_by_method": {},
                "cache_hits_by_method": {},
            },
        )
        row[method_key] = frag.get("method_block") or {}
        row["raw_by_method"][method_key] = frag.get("raw", "")
        row["latency_ms_by_method"][method_key] = int(frag.get("latency_ms", 0))
        row["survey_error_by_method"][method_key] = frag.get("error")
        row["cache_hits_by_method"][method_key] = bool(frag.get("cache_hit"))

    npi_order = [str(x) for x in df["npi"].tolist()]
    rows_out = [by_npi[n] for n in npi_order if n in by_npi]

    with open(jsonl_path, "w", encoding="utf-8") as fh:
        for r in rows_out:
            fh.write(json.dumps(r) + "\n")

    n_llm_calls = len(work)
    print(
        f"Wrote {_rel_project(jsonl_path)}: {len(rows_out)} NPI rows ({n_llm_calls} LLM calls), "
        f"{cache_hits} cache hits, {errors} calls with LLM errors, "
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
        responses_filename=responses_name,
        response_row_schema_version=RESPONSE_ROW_SCHEMA_VERSION,
        shuffle_questions=shuffle_questions,
        shuffle_seed=shuffle_seed if shuffle_questions else None,
    )

    if save_demo_bundle:
        _write_demo_bundle(
            rows_out=rows_out,
            cohort_df=df,
            questions=qs,
            methods=methods,
            model=model,
            n_llm_calls=n_llm_calls,
        )

    return 0


def _write_demo_bundle(
    *,
    rows_out: List[dict[str, Any]],
    cohort_df: pd.DataFrame,
    questions: List[Question],
    methods: List[str],
    model: str,
    n_llm_calls: int,
) -> None:
    demo_dir = PROJECT_ROOT / "artifacts" / "demo"
    demo_dir.mkdir(parents=True, exist_ok=True)

    flat = flatten_survey_rows(rows_out)
    by_q: Dict[str, Counter] = defaultdict(Counter)
    for r in flat:
        if not r.get("parsed_option"):
            continue
        qid = r["question_id"]
        if r["method"] == "method_a":
            by_q[qid][r["parsed_option"]] += 1

    simulated_distributions: Dict[str, Any] = {}
    for q in questions:
        simulated_distributions[q.question_id] = {"distribution": dict(by_q[q.question_id])}

    # adoption by archetype (empirical from cohort)
    actual_adoption: Dict[str, Any] = {}
    if "adoption_archetype" in cohort_df.columns and "has_tirzepatide_2023" in cohort_df.columns:
        g = cohort_df.groupby("adoption_archetype")["has_tirzepatide_2023"].agg(["mean", "count"])
        for idx, row in g.iterrows():
            actual_adoption[str(idx)] = {"rate": float(row["mean"]), "n": int(row["count"])}

    # sample_personas: merge a few NPIs with q1 answers
    npis = list(cohort_df["npi"].astype(str).head(10))
    sample_personas: List[Dict[str, Any]] = []
    q_first = questions[0].question_id if questions else None
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
        if q_first:
            hit = next(
                (
                    r
                    for r in flat
                    if r["npi"] == npi and r["question_id"] == q_first and r["method"] == "method_a"
                ),
                None,
            )
            entry["first_question_response_method_a"] = hit.get("parsed_option") if hit else None
        sample_personas.append(entry)

    summary = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "cohort_note": "Regenerate after tirzepatide_simulation_cohort_100.tsv is updated.",
        "n_npis": int(len(cohort_df)),
        "n_questions": len(questions),
        "n_llm_calls": n_llm_calls,
        "model": model,
        "total_api_calls": n_llm_calls,
        "total_cost_usd": None,
        "simulated_distributions": simulated_distributions,
        "adoption_by_archetype_actual": actual_adoption,
        "sample_personas": sample_personas,
    }
    (demo_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    sample_path = demo_dir / "sample_responses.jsonl"
    with open(sample_path, "w", encoding="utf-8") as fh:
        for line in flat[:40]:
            fh.write(json.dumps(line) + "\n")
    print(f"Wrote {demo_dir / 'summary.json'} and {sample_path.name}")


def main() -> None:
    load_local_dotenv(override=False)
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
        help="Subdirectory under data/output/runs/ (default: latest). E.g. v0_naive, smoke_test.",
    )
    p.add_argument("--limit-npis", type=int, default=None)
    p.add_argument("--questions", type=str, default="all", help="all or comma-separated question_ids")
    p.add_argument(
        "--persona-variant",
        type=str,
        default="production",
        choices=["production", "naive", "a"],
        help="production = 2022-only rich persona (default); naive = specialty/geo only; a = rich with 2022 tirzepatide line.",
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
        help="No API: deterministic single-stream responses from cohort + questions (for CI / missing keys)",
    )
    p.add_argument(
        "--write-demo-bundle",
        action="store_true",
        help="With --offline-seed-demo, also refresh artifacts/demo/ (default: do not overwrite demo).",
    )
    p.add_argument(
        "--shuffle-questions",
        action="store_true",
        help="Shuffle question block order in the joint survey prompt (same seed for all NPIs in run).",
    )
    p.add_argument(
        "--shuffle-seed",
        type=int,
        default=0,
        help="RNG seed when --shuffle-questions is set (default 0).",
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
    methods = ["A"]

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
            shuffle_questions=bool(args.shuffle_questions),
            shuffle_seed=int(args.shuffle_seed),
        )
    )


if __name__ == "__main__":
    main()
