"""Rebuild ``artifacts/demo/`` from an existing responses JSONL (no LLM API calls)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from eval.metrics import compute_metrics_bundle
from simulation.questions_io import Question, load_questions
from simulation.responses_schema import resolve_responses_jsonl
from simulation.run_batch import _write_demo_bundle

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESPONSES = PROJECT_ROOT / "data" / "output" / "runs" / "latest" / "responses.jsonl"
DEFAULT_COHORT = PROJECT_ROOT / "data" / "output" / "tirzepatide_simulation_cohort_100.tsv"
DEFAULT_METRICS_OUT = PROJECT_ROOT / "artifacts" / "demo" / "metrics.json"
QUESTIONS_DEFAULT = PROJECT_ROOT / "simulation" / "questions.yaml"


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _cohort_for_rows(cohort_path: Path, rows_out: list[dict[str, Any]]) -> pd.DataFrame:
    """Restrict cohort to NPIs present in ``rows_out`` (same scope as the run)."""
    npi_order = [str(r["npi"]) for r in rows_out if r.get("npi") is not None]
    npi_set = set(npi_order)
    if not npi_set:
        return pd.DataFrame()
    df = pd.read_csv(cohort_path, sep="\t", low_memory=False, dtype={"npi": str})
    df["npi"] = df["npi"].astype(str).str.strip()
    sub = df[df["npi"].isin(npi_set)].copy()
    order = {n: i for i, n in enumerate(npi_order)}
    sub["_ord"] = sub["npi"].map(lambda x: order.get(str(x), 10**9))
    sub = sub.sort_values("_ord").drop(columns=["_ord"])
    return sub


def _read_run_manifest(run_dir: Path) -> dict[str, Any]:
    mf = run_dir / "run_manifest.json"
    if not mf.is_file():
        return {}
    try:
        return json.loads(mf.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def refresh_demo_from_responses(
    *,
    responses_file: Path,
    cohort_path: Path,
    questions: list[Question],
    metrics_output: Path,
    questions_yaml: Path | None,
) -> None:
    """
    Rewrite ``artifacts/demo/summary.json``, ``sample_responses.jsonl``, and metrics JSON
    using the given responses file and current eval / flatten logic.
    """
    resolved = resolve_responses_jsonl(responses_file) or responses_file
    if not resolved.is_file():
        raise FileNotFoundError(f"Responses file not found: {resolved}")

    rows_out = _load_jsonl_rows(resolved)
    if not rows_out:
        raise ValueError(f"No JSON lines in {resolved}")

    if not cohort_path.is_file():
        raise FileNotFoundError(f"Cohort TSV not found: {cohort_path}")

    cohort_df = _cohort_for_rows(cohort_path, rows_out)
    manifest = _read_run_manifest(resolved.parent)
    model = str(manifest.get("model") or "").strip() or "unknown_model"
    n_llm = int(manifest.get("n_npis_in_run") or len(rows_out))
    if manifest.get("offline") is True:
        n_llm = 0

    _write_demo_bundle(
        rows_out=rows_out,
        cohort_df=cohort_df,
        questions=questions,
        methods=["method_a"],
        model=model,
        n_llm_calls=n_llm,
    )

    summary_path = PROJECT_ROOT / "artifacts" / "demo" / "summary.json"
    summary_obj = json.loads(summary_path.read_text(encoding="utf-8"))
    if rows_out and all(bool(r.get("offline_seed")) for r in rows_out):
        summary_obj["offline_seed"] = True
        summary_obj["is_placeholder"] = False
    else:
        summary_obj.pop("offline_seed", None)
    summary_path.write_text(json.dumps(summary_obj, indent=2), encoding="utf-8")

    cohort_rel: str | None
    try:
        cohort_rel = str(cohort_path.resolve().relative_to(PROJECT_ROOT)) if cohort_path.is_file() else None
    except ValueError:
        cohort_rel = str(cohort_path) if cohort_path.is_file() else None

    bundle = compute_metrics_bundle(
        resolved,
        questions_yaml=questions_yaml if questions_yaml and questions_yaml.is_file() else None,
        cohort_path=cohort_path if cohort_path.is_file() else None,
    )
    if manifest:
        bundle["run_manifest"] = manifest
    q_yaml_meta: str | None
    if questions_yaml and questions_yaml.is_file():
        try:
            q_yaml_meta = str(questions_yaml.resolve().relative_to(PROJECT_ROOT))
        except ValueError:
            q_yaml_meta = str(questions_yaml)
    else:
        q_yaml_meta = None

    try:
        responses_rel = str(resolved.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        responses_rel = str(resolved.resolve())

    bundle["eval_meta"] = {
        "responses_file": responses_rel,
        "cohort_path": cohort_rel,
        "questions_yaml": q_yaml_meta,
        "refreshed_via": "simulation.refresh_demo_from_responses",
    }

    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_text = json.dumps(bundle, indent=2) + "\n"
    metrics_output.write_text(metrics_text, encoding="utf-8")
    print(f"Wrote {metrics_output.relative_to(PROJECT_ROOT)}")

    sidecar = resolved.parent / "metrics.json"
    if sidecar.resolve() != metrics_output.resolve():
        sidecar.write_text(metrics_text, encoding="utf-8")
        print(f"Wrote {sidecar.relative_to(PROJECT_ROOT)}")


def main() -> int:
    p = argparse.ArgumentParser(
        description="Rebuild artifacts/demo/ (summary, sample JSONL, metrics) from existing responses JSONL — no API."
    )
    p.add_argument("--responses-file", type=Path, default=DEFAULT_RESPONSES)
    p.add_argument("--cohort-path", type=Path, default=DEFAULT_COHORT)
    p.add_argument("--questions-yaml", type=Path, default=QUESTIONS_DEFAULT)
    p.add_argument("--metrics-output", type=Path, default=DEFAULT_METRICS_OUT)
    args = p.parse_args()

    questions = load_questions(args.questions_yaml)
    refresh_demo_from_responses(
        responses_file=args.responses_file,
        cohort_path=args.cohort_path,
        questions=questions,
        metrics_output=args.metrics_output,
        questions_yaml=args.questions_yaml,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
