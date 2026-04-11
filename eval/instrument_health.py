"""Run-level health: coverage, errors, latency from responses JSONL (v2-aware)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from simulation.questions_io import Question, load_questions
from simulation.responses_schema import flatten_survey_rows, is_v2_survey_row


def load_raw_response_rows(path: Path) -> List[dict[str, Any]]:
    rows: List[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def compute_instrument_health(
    responses_jsonl: Path,
    *,
    questions_yaml: Path | None = None,
) -> dict[str, Any]:
    """
    Summarize parse/coverage and LLM-call health from raw + flattened rows.

    - v2: survey_error_by_method, latency_ms_by_method, missing per-question cells.
    - Legacy flat rows: error / missing parsed_option counts only.
    """
    raw = load_raw_response_rows(responses_jsonl)
    questions = load_questions(questions_yaml)
    qids = [q.question_id for q in questions]
    flat = flatten_survey_rows(raw)

    n_raw = len(raw)
    n_v2 = sum(1 for r in raw if is_v2_survey_row(r))
    n_legacy = n_raw - n_v2

    latencies: List[int] = []
    survey_errors = 0
    for r in raw:
        if not is_v2_survey_row(r):
            continue
        lat = r.get("latency_ms_by_method") or {}
        if isinstance(lat, dict):
            for v in lat.values():
                try:
                    if v is not None:
                        latencies.append(int(v))
                except (TypeError, ValueError):
                    pass
        errm = r.get("survey_error_by_method") or {}
        if isinstance(errm, dict):
            survey_errors += sum(1 for v in errm.values() if v)

    # Count every expected (NPI × question) cell for method_a: empty or failed joint
    # surveys must count as missing, not skipped (previously ``if not block: continue`` hid them).
    missing_cells = 0
    for r in raw:
        if not is_v2_survey_row(r):
            continue
        for mk in ("method_a",):
            block = r.get(mk)
            if not isinstance(block, dict):
                block = {}
            for qid in qids:
                cell = block.get(qid)
                if not isinstance(cell, dict) or not cell.get("option_id"):
                    missing_cells += 1

    flat_with_error = sum(1 for r in flat if r.get("error"))
    # ``missing_cells`` is the authoritative v2 expected-vs-answered gap (including empty ``method_a``).
    # Legacy flat rows (non-v2 source) may lack ``method_a``; count those separately to avoid double-counting
    # v2 rows that already appear in ``missing_cells``.
    legacy_flat_missing = sum(
        1 for r in flat if r.get("method") != "method_a" and not r.get("parsed_option")
    )
    flat_cells_missing_option = missing_cells + legacy_flat_missing

    v2_expected_answer_cells = n_v2 * len(qids)
    missing_answer_cell_rate: float | None
    if v2_expected_answer_cells > 0:
        missing_answer_cell_rate = missing_cells / float(v2_expected_answer_cells)
    else:
        missing_answer_cell_rate = None

    return {
        "schema_notes": "v2 rows carry latency_ms_by_method and survey_error_by_method; "
        "legacy rows are flat only.",
        "n_jsonl_rows": n_raw,
        "n_v2_rows": n_v2,
        "n_legacy_rows": n_legacy,
        "n_flat_cells": len(flat),
        "flat_cells_with_error": flat_with_error,
        "flat_cells_missing_option": flat_cells_missing_option,
        "v2_missing_question_cells": missing_cells,
        "v2_expected_answer_cells": v2_expected_answer_cells,
        "missing_answer_cell_rate": missing_answer_cell_rate,
        "v2_survey_level_errors": survey_errors,
        "latency_ms": {
            "n_calls_with_latency": len(latencies),
            "mean": sum(latencies) / len(latencies) if latencies else None,
            "p50": sorted(latencies)[len(latencies) // 2] if latencies else None,
            "max": max(latencies) if latencies else None,
        },
        "claims_map_file": "simulation/question_claims_map.yaml",
    }
