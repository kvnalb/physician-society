"""CLI: responses.jsonl -> metrics.json"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.metrics import compute_metrics_bundle

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESPONSES = PROJECT_ROOT / "data" / "output" / "runs" / "latest" / "responses.jsonl"
DEFAULT_OUT = PROJECT_ROOT / "artifacts" / "demo" / "metrics.json"


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

    bundle = compute_metrics_bundle(args.responses_file, questions_yaml=args.questions_yaml)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
