"""Batch LLM survey over cohort physicians."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Set

import pandas as pd

from simulation.llm_client import call_llm, get_api_key, make_client
from simulation.persona_methods import build_prompts
from simulation.questions_io import Question, load_questions

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COHORT = PROJECT_ROOT / "data" / "output" / "tirzepatide_simulation_cohort_100.tsv"


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
) -> str:
    h = hashlib.sha256(
        f"{model}|{temperature}|{method}|{question_id}|{npi}".encode()
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


def run(
    *,
    cohort_path: Path,
    output_dir: Path,
    limit_npis: Optional[int],
    questions_spec: str,
    methods: List[str],
    model: str,
    temperature: float,
    base_url: Optional[str],
    api_key: Optional[str],
    save_demo_bundle: bool,
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

    df = pd.read_csv(cohort_path, sep="\t", low_memory=False)
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

    rows_out: List[dict[str, Any]] = []
    cache_hits = 0
    errors = 0

    for _, row in df.iterrows():
        rowd = row.to_dict()
        npi = str(rowd.get("npi", ""))
        for method in methods:
            for q in qs:
                ck = _cache_key(
                    model=model,
                    temperature=temperature,
                    method=method,
                    question_id=q.question_id,
                    npi=npi,
                )
                cached = _read_cache(cache_dir, ck)
                if cached and cached.get("parsed_option"):
                    cache_hits += 1
                    rows_out.append(
                        {
                            "npi": npi,
                            "question_id": q.question_id,
                            "method": f"method_{method.lower()}",
                            "model": model,
                            "temperature": temperature,
                            "raw": cached.get("raw", ""),
                            "parsed_option": cached.get("parsed_option"),
                            "reasoning": cached.get("reasoning", ""),
                            "latency_ms": int(cached.get("latency_ms", 0)),
                            "error": None,
                            "cache_hit": True,
                        }
                    )
                    continue

                system, user = build_prompts(method, rowd, q)
                raw, parsed, reason, lat, err = call_llm(
                    client,
                    system=system,
                    user=user,
                    model=model,
                    temperature=temperature,
                    question=q,
                )
                if err:
                    errors += 1
                if parsed:
                    _write_cache(
                        cache_dir,
                        ck,
                        {
                            "raw": raw,
                            "parsed_option": parsed,
                            "reasoning": reason,
                            "latency_ms": lat,
                        },
                    )
                rows_out.append(
                    {
                        "npi": npi,
                        "question_id": q.question_id,
                        "method": f"method_{method.lower()}",
                        "model": model,
                        "temperature": temperature,
                        "raw": raw,
                        "parsed_option": parsed,
                        "reasoning": reason,
                        "latency_ms": lat,
                        "error": err,
                        "cache_hit": False,
                    }
                )

    with open(jsonl_path, "w", encoding="utf-8") as fh:
        for r in rows_out:
            fh.write(json.dumps(r) + "\n")

    n_calls = len(rows_out)
    print(
        f"Wrote {jsonl_path.relative_to(PROJECT_ROOT)}: {n_calls} rows, "
        f"{cache_hits} cache hits, {errors} errors."
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
    p.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "output" / "runs" / "latest")
    p.add_argument("--limit-npis", type=int, default=None)
    p.add_argument("--questions", type=str, default="all", help="all or comma-separated question_ids")
    p.add_argument("--method", type=str, default="both", choices=["A", "B", "both"])
    p.add_argument("--model", type=str, default="gpt-4o-mini")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--base-url", type=str, default=None, help="Together or other OpenAI-compatible base URL")
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
    args = p.parse_args()

    if args.offline_seed_demo:
        raise SystemExit(
            run_offline_seed_demo(
                cohort_path=args.cohort_path,
                output_dir=args.output_dir,
                limit_npis=args.limit_npis,
            )
        )

    methods = ["A", "B"] if args.method == "both" else [args.method]
    key = get_api_key("openai")
    if args.base_url and os.environ.get("TOGETHER_API_KEY"):
        key = os.environ.get("TOGETHER_API_KEY") or key

    raise SystemExit(
        run(
            cohort_path=args.cohort_path,
            output_dir=args.output_dir,
            limit_npis=args.limit_npis,
            questions_spec=args.questions,
            methods=methods,
            model=args.model,
            temperature=args.temperature,
            base_url=args.base_url,
            api_key=key,
            save_demo_bundle=args.save_as_demo_bundle,
        )
    )


if __name__ == "__main__":
    main()
