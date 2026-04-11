"""Load one NPI's cohort row + survey row(s) from a run's responses JSONL (v2: one row per NPI)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from simulation.responses_schema import resolve_responses_jsonl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "data" / "output" / "runs"
DEFAULT_COHORT = PROJECT_ROOT / "data" / "output" / "tirzepatide_simulation_cohort_100.tsv"


def load_persona_bundle(
    *,
    npi: str,
    cohort_path: Path,
    responses_path: Path,
) -> Dict[str, Any]:
    npi = str(npi).strip()
    cohort = pd.read_csv(cohort_path, sep="\t", low_memory=False, dtype={"npi": str})
    row = cohort[cohort["npi"].astype(str) == npi]
    if row.empty:
        raise SystemExit(f"NPI {npi} not found in cohort {cohort_path}")
    profile = row.iloc[0].to_dict()

    answers: List[dict[str, Any]] = []
    with open(responses_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if str(r.get("npi")) != npi:
                continue
            answers.append(r)

    answers.sort(key=lambda x: (x.get("question_id", ""), x.get("method", "")))
    return {
        "npi": npi,
        "cohort_profile": profile,
        "n_response_rows": len(answers),
        "responses": answers,
        "responses_file": str(responses_path),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Merge cohort TSV + run JSONL for one NPI.")
    p.add_argument("--npi", type=str, required=True)
    p.add_argument("--cohort-path", type=Path, default=DEFAULT_COHORT)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--responses-file", type=Path, help="Path to responses.jsonl")
    g.add_argument(
        "--run-id",
        type=str,
        help=f"Use {RUNS_ROOT}/<run-id>/responses.jsonl",
    )
    args = p.parse_args()

    if args.responses_file is not None:
        rpath = args.responses_file
    else:
        rpath = RUNS_ROOT / args.run_id / "responses.jsonl"
    resolved = resolve_responses_jsonl(rpath)
    rpath = resolved if resolved is not None else rpath
    if not rpath.is_file():
        raise SystemExit(f"Missing responses file: {rpath}")

    bundle = load_persona_bundle(
        npi=args.npi,
        cohort_path=args.cohort_path,
        responses_path=rpath,
    )
    print(json.dumps(bundle, indent=2, default=str))


if __name__ == "__main__":
    main()
