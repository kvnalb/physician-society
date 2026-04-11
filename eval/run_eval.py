"""CLI: responses.jsonl -> metrics.json"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from eval.metrics import compute_metrics_bundle

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESPONSES = PROJECT_ROOT / "data" / "output" / "runs" / "latest" / "responses.jsonl"
DEFAULT_OUT = PROJECT_ROOT / "artifacts" / "demo" / "metrics.json"


def _sidecar_metrics_path(responses_file: Path) -> Optional[Path]:
    """If responses live under data/output/runs/<id>/, metrics sit beside them."""
    try:
        rf = responses_file.resolve()
    except OSError:
        return None
    parts = rf.parts
    if rf.name != "responses.jsonl":
        return None
    if "runs" not in parts:
        return None
    if "output" not in parts or "data" not in parts:
        return None
    return rf.parent / "metrics.json"


def main() -> None:
    p = argparse.ArgumentParser(description="Compute eval metrics from responses JSONL.")
    p.add_argument("--responses-file", type=Path, default=DEFAULT_RESPONSES)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--questions-yaml",
        type=Path,
        default=None,
        help="Optional path to questions YAML (defaults to simulation/questions.yaml)",
    )
    p.add_argument(
        "--cohort-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "output" / "tirzepatide_simulation_cohort_100.tsv",
        help="Cohort TSV for behavioral pseudo-label alignment (skip if missing).",
    )
    p.add_argument(
        "--method-for-alignment",
        type=str,
        default="method_a",
        choices=["method_a", "method_b"],
        help="Which LLM method rows to score vs pseudo-labels.",
    )
    args = p.parse_args()

    if not args.responses_file.is_file():
        print(f"Missing responses file: {args.responses_file}")
        # Write minimal metrics so downstream tools do not crash
        args.output.parent.mkdir(parents=True, exist_ok=True)
        placeholder = {
            "error": "no_responses_file",
            "survey": {
                "method_agreement_kappa_mean": None,
                "per_question": {},
                "stability": "No responses.jsonl found; run simulation.run_batch with an API key.",
            },
        }
        args.output.write_text(json.dumps(placeholder, indent=2), encoding="utf-8")
        print(f"Wrote placeholder {args.output}")
        return

    cohort_p = args.cohort_path if args.cohort_path.is_file() else None
    bundle = compute_metrics_bundle(
        args.responses_file,
        questions_yaml=args.questions_yaml,
        cohort_path=cohort_p,
        method_for_alignment=args.method_for_alignment,
    )

    manifest_path = args.responses_file.parent / "run_manifest.json"
    if manifest_path.is_file():
        try:
            bundle["run_manifest"] = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            bundle["run_manifest_error"] = "could_not_read_run_manifest"

    try:
        cohort_rel = (
            str(args.cohort_path.resolve().relative_to(PROJECT_ROOT))
            if args.cohort_path.is_file()
            else None
        )
    except ValueError:
        cohort_rel = str(args.cohort_path) if args.cohort_path.is_file() else None

    bundle["eval_meta"] = {
        "responses_file": str(args.responses_file.resolve()),
        "cohort_path": cohort_rel,
        "method_for_alignment": args.method_for_alignment,
        "questions_yaml": str(args.questions_yaml) if args.questions_yaml else None,
    }

    text = json.dumps(bundle, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(f"Wrote {args.output}")

    sidecar = _sidecar_metrics_path(args.responses_file)
    if sidecar and sidecar.resolve() != args.output.resolve():
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(text, encoding="utf-8")
        print(f"Wrote sidecar {sidecar.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
