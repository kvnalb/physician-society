"""Two persona prompt builders: rich (full cohort row) vs minimal (specialty/geo/archetype)."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

from simulation.questions_io import Question, format_question_block


def _pct(x: Any) -> str:
    try:
        v = float(x)
        return f"{100.0 * v:.1f}%"
    except (TypeError, ValueError):
        return "n/a"


def build_prompts_method_a(row: Mapping[str, Any], question: Question) -> Tuple[str, str]:
    """Rich persona: NPPES + 2022 Part D + OP + archetype/tier."""
    spec = str(row.get("specialty", "Unknown"))
    city = str(row.get("city", ""))
    state = str(row.get("state", ""))
    npi = str(row.get("npi", ""))
    system = (
        "You are a board-eligible or board-certified U.S. physician. "
        "Answer only from the perspective of the prescribing clinician described below. "
        "Use the quantitative practice profile as a prior; you may infer typical patterns "
        "but stay consistent with those numbers. Do not invent a personal name."
    )
    profile = [
        f"Practice profile (Medicare Part D–visible, 2022 baseline; NPI {npi}):",
        f"- Specialty: {spec}",
        f"- Location: {city}, {state}",
        f"- Organization / site line: {row.get('organization_name', '')}",
        f"- Credentials: {row.get('credentials', '')}",
        f"- Adoption archetype (derived from 2022 prescribing): {row.get('adoption_archetype', '')}",
        f"- Pharma engagement tier (Open Payments 2022): {row.get('pharma_engagement_tier', '')}",
        f"- Total Part D claims (2022): {row.get('claims_2022', '')}",
        f"- Part D beneficiaries (2022): {row.get('beneficiaries_2022', '')}",
        f"- Diabetes share of claims (2022): {_pct(row.get('diabetes_share_2022'))}",
        f"- GLP-1 share of diabetes claims (2022): {_pct(row.get('glp1_penetration_2022'))}",
        f"- Branded share of Part D rows (2022): {_pct(row.get('branded_share_2022'))}",
        f"- Distinct diabetes drugs (2022): {row.get('drug_diversity_2022', '')}",
        f"- Any tirzepatide claims in 2022 data: {row.get('has_tirzepatide_2022', '')}",
        f"- Total general payments (Open Payments 2022 USD): {row.get('total_payments_2022', '')}",
        f"- Novo Nordisk-attributed payments (2022 USD): {row.get('novo_nordisk_payments', '')}",
        f"- Eli Lilly–attributed payments (2022 USD): {row.get('eli_lilly_payments', '')}",
    ]
    user = "\n".join(profile) + "\n\n---\n\n" + format_question_block(question)
    return system, user


def build_prompts_method_b(row: Mapping[str, Any], question: Question) -> Tuple[str, str]:
    """Minimal persona: specialty, city, state, adoption archetype only."""
    spec = str(row.get("specialty", "Unknown"))
    city = str(row.get("city", ""))
    state = str(row.get("state", ""))
    arch = str(row.get("adoption_archetype", "Unclassified"))
    system = (
        "You are a U.S. physician. You have minimal context; infer typical practice patterns "
        "for your specialty and location. Do not invent a personal name."
    )
    user = (
        f"You are a {spec} physician practicing in {city}, {state}. "
        f"Your prescribing style is summarized as: {arch} (label only; no detailed metrics).\n\n"
        + format_question_block(question)
    )
    return system, user


def build_prompts(method: str, row: Mapping[str, Any], question: Question) -> Tuple[str, str]:
    m = method.upper()
    if m == "A":
        return build_prompts_method_a(row, question)
    if m == "B":
        return build_prompts_method_b(row, question)
    raise ValueError(f"Unknown method {method!r}; use 'A' or 'B'.")
