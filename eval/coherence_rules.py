"""Deterministic cross-item consistency checks (persona coherence proxy, no cognitive claim)."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Set

from simulation.responses_schema import is_v2_survey_row

# Option ids must match simulation/questions.yaml
_Q3_NO_PRESCRIBE: Set[str] = {"q3_no_unlikely", "q3_no_not_yet", "q3_unsure"}
_Q3_YES = "q3_yes"
_Q4_NA = "q4_na"
_Q4_SPEED: Set[str] = {"q4_early", "q4_mainstream", "q4_late"}


def _get_answer(answers: Mapping[str, str], qid: str) -> Optional[str]:
    v = answers.get(qid)
    return str(v).strip() if v else None


def coherence_violations_for_npi(
    answers_by_qid: Mapping[str, str],
    *,
    method_label: str,
    npi: str,
) -> List[dict[str, Any]]:
    """
    Return a list of violation dicts for one method's completed survey (question_id -> option_id).
    """
    out: List[dict[str, Any]] = []
    q3 = _get_answer(answers_by_qid, "q3_tirzepatide_prescribed")
    q4 = _get_answer(answers_by_qid, "q4_tirzepatide_adoption_speed")

    if q4 and q4 in _Q4_SPEED and q3 != _Q3_YES:
        out.append(
            {
                "rule_id": "q4_speed_requires_q3_yes",
                "npi": npi,
                "method": method_label,
                "detail": f"q4={q4} but q3={q3} (speed options require prescribing yes)",
            }
        )

    if q3 and q3 in _Q3_NO_PRESCRIBE and q4 and q4 not in (_Q4_NA,):
        out.append(
            {
                "rule_id": "no_prescribe_implies_q4_na",
                "npi": npi,
                "method": method_label,
                "detail": f"q3={q3} but q4={q4} (expected q4_na when not prescribing)",
            }
        )

    return out


def _answers_from_method_block(
    block: Mapping[str, Any],
    question_ids: List[str],
) -> Dict[str, str]:
    ans: Dict[str, str] = {}
    for qid in question_ids:
        cell = block.get(qid)
        if isinstance(cell, dict):
            oid = cell.get("option_id")
            if oid:
                ans[qid] = str(oid)
    return ans


def compute_persona_coherence(
    raw_rows: List[dict[str, Any]],
    *,
    question_ids: List[str],
) -> dict[str, Any]:
    """
    Aggregate cross-item rule violations across NPIs (v2 rows only).
    """
    all_violations: List[dict[str, Any]] = []
    n_blocks = 0
    for r in raw_rows:
        if not is_v2_survey_row(r):
            continue
        npi = str(r.get("npi", ""))
        for mk in ("method_a", "method_b"):
            block = r.get(mk)
            if not isinstance(block, dict) or not block:
                continue
            answers = _answers_from_method_block(block, question_ids)
            if len(answers) < 2:
                continue
            n_blocks += 1
            all_violations.extend(
                coherence_violations_for_npi(answers, method_label=mk, npi=npi),
            )

    n_violations = len(all_violations)
    rate = n_violations / n_blocks if n_blocks else None
    return {
        "rules_version": "1",
        "note": "Heuristic cross-item checks; not a measure of real physician cognition.",
        "n_method_blocks_checked": n_blocks,
        "n_violations": n_violations,
        "violation_rate_per_method_block": rate,
        "violations_sample": all_violations[:50],
    }
