"""Survey response JSONL schema (v2: one row per NPI, model in filename)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

RESPONSE_ROW_SCHEMA_VERSION = 2


def responses_filename_for_model(model: str) -> str:
    """Stable filename segment from model id (no path separators)."""
    slug = re.sub(r"[^0-9A-Za-z._-]+", "_", model.strip().replace("/", "_"))
    slug = slug.strip("_") or "unknown_model"
    return f"responses__{slug}.jsonl"


def is_v2_survey_row(row: Mapping[str, Any]) -> bool:  # noqa: UP006
    if row.get("schema_version") == RESPONSE_ROW_SCHEMA_VERSION:
        return True
    if row.get("question_id") is not None:
        return False
    return isinstance(row.get("method_a"), dict)


def flatten_survey_rows(rows: List[dict[str, Any]]) -> List[dict[str, Any]]:
    """
    Expand v2 rows (one per NPI, nested by method) into legacy-shaped rows
    for metrics (npi, question_id, method, parsed_option, reasoning, error).
    """
    flat: List[dict[str, Any]] = []
    for r in rows:
        if not is_v2_survey_row(r):
            flat.append(r)
            continue
        npi = str(r["npi"])
        err_by = r.get("survey_error_by_method") or {}
        for mk in ("method_a",):
            block = r.get(mk)
            if not isinstance(block, dict):
                continue
            top_err = err_by.get(mk) if isinstance(err_by, dict) else None
            for qid, cell in block.items():
                if not isinstance(cell, dict):
                    continue
                flat.append(
                    {
                        "npi": npi,
                        "question_id": str(qid),
                        "method": mk,
                        "parsed_option": cell.get("option_id"),
                        "reasoning": cell.get("reasoning", ""),
                        "error": top_err or cell.get("error"),
                    }
                )
    return flat


def resolve_responses_jsonl(requested: Path) -> Optional[Path]:
    """
    Resolve the canonical responses file for a run directory.

    If ``requested`` is ``responses.jsonl`` **and** ``run_manifest.json`` names a different
    ``responses_filename`` that exists (typical v2: ``responses__<model>.jsonl``), return the
    manifest path so stale legacy flat ``responses.jsonl`` files do not shadow the real run.
    If ``responses.jsonl`` is missing, fall back to manifest → newest ``responses__*.jsonl``.
    """
    parent = requested.parent
    manifest_cand: Optional[Path] = None
    mf = parent / "run_manifest.json"
    if mf.is_file():
        try:
            data = json.loads(mf.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
        fn = data.get("responses_filename")
        if isinstance(fn, str) and fn.strip():
            p = parent / fn.strip()
            if p.is_file():
                manifest_cand = p

    if requested.is_file():
        if (
            requested.name == "responses.jsonl"
            and manifest_cand is not None
            and manifest_cand.resolve() != requested.resolve()
        ):
            return manifest_cand
        return requested

    if requested.name != "responses.jsonl":
        return None
    if manifest_cand is not None:
        return manifest_cand
    globs = sorted(parent.glob("responses__*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return globs[0] if globs else None
