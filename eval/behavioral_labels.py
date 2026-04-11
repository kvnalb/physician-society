"""Map cohort rows to pseudo–ground-truth option_ids for forward survey items (hold-out Part D).

Rules use **post-2022** administrative fields for labels; persona text for production runs must
exclude those fields. RULES_VERSION must be bumped when binning logic changes.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

RULES_VERSION = "2-forward-holdout"


def _f(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or (isinstance(x, float) and str(x) == "nan"):
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def _i(x: Any, default: int = 0) -> int:
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return default


def _dir_up_flat_down(delta: float, *, eps: float) -> str:
    if delta > eps:
        return "up"
    if delta < -eps:
        return "down"
    return "flat"


def pseudo_label_for_question(row: Mapping[str, Any], question_id: str) -> Optional[str]:
    """Return option_id aligned to questions.yaml, or None if fields missing."""
    if question_id == "f_q1_tirzepatide_12m":
        if "has_tirzepatide_2023" not in row:
            return None
        if not _i(row.get("has_tirzepatide_2023")):
            return "f_q1_expect_unlikely"
        n_claims = _f(row.get("tirzepatide_claims_2023"))
        if n_claims >= 3.0:
            return "f_q1_expect_active"
        return "f_q1_expect_selective"

    if question_id == "f_q2_glp1_trajectory":
        d = _f(row.get("glp1_penetration_2023")) - _f(row.get("glp1_penetration_2022"))
        tag = _dir_up_flat_down(d, eps=0.03)
        return f"f_q2_{tag}"

    if question_id == "f_q3_branded_trajectory":
        d = _f(row.get("branded_share_2023")) - _f(row.get("branded_share_2022"))
        tag = _dir_up_flat_down(d, eps=0.02)
        return f"f_q3_{tag}"

    if question_id == "f_q4_diabetes_mix_trajectory":
        d = _f(row.get("diabetes_share_2023")) - _f(row.get("diabetes_share_2022"))
        tag = _dir_up_flat_down(d, eps=0.02)
        return f"f_q4_{tag}"

    if question_id == "f_q5_panel_scale_trajectory":
        c22 = _f(row.get("claims_2022"))
        c23 = _f(row.get("claims_2023"))
        if c22 <= 0:
            return None
        rel = (c23 - c22) / c22
        tag = _dir_up_flat_down(rel, eps=0.05)
        return f"f_q5_{tag}"

    if question_id == "f_q6_molecule_breadth_trajectory":
        d = float(_i(row.get("drug_diversity_2023"))) - float(_i(row.get("drug_diversity_2022")))
        if d >= 1.0:
            tag = "up"
        elif d <= -1.0:
            tag = "down"
        else:
            tag = "flat"
        return f"f_q6_{tag}"

    return None


def pseudo_labels_for_row(row: Mapping[str, Any]) -> Dict[str, Optional[str]]:
    qids = [
        "f_q1_tirzepatide_12m",
        "f_q2_glp1_trajectory",
        "f_q3_branded_trajectory",
        "f_q4_diabetes_mix_trajectory",
        "f_q5_panel_scale_trajectory",
        "f_q6_molecule_breadth_trajectory",
    ]
    return {qid: pseudo_label_for_question(row, qid) for qid in qids}
