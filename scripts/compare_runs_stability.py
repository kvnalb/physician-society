#!/usr/bin/env python3
"""Compare two survey response JSONL files (flattened v2) for retest / stability analysis."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eval.instrument_health import load_raw_response_rows
from simulation.responses_schema import flatten_survey_rows


def _load_flat(path: Path) -> List[dict[str, Any]]:
    return flatten_survey_rows(load_raw_response_rows(path))


def _index(rows: List[dict[str, Any]]) -> Dict[Tuple[str, str, str], str]:
    out: Dict[Tuple[str, str, str], str] = {}
    for r in rows:
        key = (str(r["npi"]), str(r["question_id"]), str(r["method"]))
        opt = r.get("parsed_option")
        if opt:
            out[key] = str(opt)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Item agreement between two response JSONL runs (same NPIs).")
    p.add_argument("--run-a", type=Path, required=True, help="First responses__*.jsonl or legacy flat file")
    p.add_argument("--run-b", type=Path, required=True, help="Second responses file (e.g. duplicate run)")
    args = p.parse_args()

    a = _index(_load_flat(args.run_a))
    b = _index(_load_flat(args.run_b))
    keys = sorted(set(a) & set(b))
    if not keys:
        print("No overlapping (npi, question_id, method) keys with parsed_option in both files.")
        raise SystemExit(1)
    agree = sum(1 for k in keys if a[k] == b[k])
    total = len(keys)
    by_method: Dict[str, List[bool]] = defaultdict(list)
    for k in keys:
        by_method[k[2]].append(a[k] == b[k])
    print(json.dumps({"n_comparable_cells": total, "item_agreement_rate": agree / total, "by_method": {m: sum(v) / len(v) for m, v in by_method.items()}}, indent=2))


if __name__ == "__main__":
    main()
