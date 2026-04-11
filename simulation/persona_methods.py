"""Persona prompt builders: naive, rich legacy (A), and production (2022-only, hold-out–safe)."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Tuple

from simulation.questions_io import Question, format_multi_question_json_survey, format_question_block

# Bump when any prompt string changes so disk cache invalidates across runs.
PROMPT_VERSION = "2026-04-11-production-forward-holdout"


def _pct(x: Any) -> str:
    try:
        v = float(x)
        return f"{100.0 * v:.1f}%"
    except (TypeError, ValueError):
        return "n/a"


def build_prompts_production_persona(row: Mapping[str, Any], question: Question) -> Tuple[str, str]:
    """
    Primary demo persona: rich **pre–outcome-holdout** context (through CY2022 Part D + OP 2022).

    Omits **any 2023 Part D outcomes** and omits in-year tirzepatide flags so forward tirzepatide items
    are not answered straight from a printed yes/no line. Archetype / tiers summarize **2022-era**
    patterns only.
    """
    spec = str(row.get("specialty", "Unknown"))
    city = str(row.get("city", ""))
    state = str(row.get("state", ""))
    npi = str(row.get("npi", ""))
    system = (
        "You are a board-eligible or board-certified U.S. physician. "
        "Answer only from the perspective of the prescribing clinician described below. "
        "The profile is built from **Medicare Part D and Open Payments through calendar-year 2022** "
        "plus registry attributes. It deliberately **excludes any 2023 Part D outcomes**—treat "
        "forward-looking survey items as **June 2022 judgments** about the next program year, not "
        "as knowledge of what already happened in 2023 claims. "
        "Stay consistent with the numeric 2022 practice profile; you may infer typical patterns "
        "but do not invent a personal name."
    )
    profile = [
        f"Practice profile (Medicare Part D–visible **2022 baseline only**; NPI {npi}):",
        f"- Specialty: {spec}",
        f"- Location: {city}, {state}",
        f"- Organization / site line: {row.get('organization_name', '')}",
        f"- Credentials: {row.get('credentials', '')}",
        f"- Adoption archetype (derived from **2022** prescribing patterns): {row.get('adoption_archetype', '')}",
        f"- Pharma engagement tier (Open Payments 2022): {row.get('pharma_engagement_tier', '')}",
        f"- Total Part D claims (2022): {row.get('claims_2022', '')}",
        f"- Part D beneficiaries (2022): {row.get('beneficiaries_2022', '')}",
        f"- Diabetes share of claims (2022): {_pct(row.get('diabetes_share_2022'))}",
        f"- GLP-1 share of diabetes claims (2022): {_pct(row.get('glp1_penetration_2022'))}",
        f"- Branded share of Part D rows (2022): {_pct(row.get('branded_share_2022'))}",
        f"- Distinct diabetes drugs (2022): {row.get('drug_diversity_2022', '')}",
        f"- Total general payments (Open Payments 2022 USD): {row.get('total_payments_2022', '')}",
        f"- Novo Nordisk-attributed payments (2022 USD): {row.get('novo_nordisk_payments', '')}",
        f"- Eli Lilly–attributed payments (2022 USD): {row.get('eli_lilly_payments', '')}",
    ]
    user = "\n".join(profile) + "\n\n---\n\n" + format_question_block(question)
    return system, user


def build_prompts_method_a(row: Mapping[str, Any], question: Question) -> Tuple[str, str]:
    """Rich persona: NPPES + 2022 Part D + OP + archetype/tier (includes 2022 tirzepatide flag)."""
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


def build_prompts(method: str, row: Mapping[str, Any], question: Question) -> Tuple[str, str]:
    """Legacy name: single stream labeled ``method_a`` in JSONL; ``method`` must be ``A``."""
    m = method.upper()
    if m == "A":
        return build_prompts_method_a(row, question)
    raise ValueError(f"Unknown method {method!r}; use 'A'.")


def build_prompts_naive(row: Mapping[str, Any], question: Question) -> Tuple[str, str]:
    """v0: specialty + city + state only (model prior)."""
    spec = str(row.get("specialty", "Unknown"))
    city = str(row.get("city", ""))
    state = str(row.get("state", ""))
    system = (
        "You are a U.S. physician. You have almost no practice-specific data. "
        "Answer as a typical clinician for your stated specialty and location. "
        "Do not invent a personal name."
    )
    user = (
        f"You are a {spec} physician practicing in {city}, {state}.\n\n"
        + format_question_block(question)
    )
    return system, user


def build_prompts_for_persona_variant(
    persona_variant: str,
    method: str,
    row: Mapping[str, Any],
    question: Question,
) -> Tuple[str, str]:
    """
    persona_variant:
      naive — only specialty/geo (single ``method_a`` stream in runs).
      production — default rich persona without 2023 outcomes / without 2022 tirzepatide line (hold-out eval).
      a — rich grounded (2022 tirzepatide line included).
    """
    pv = persona_variant.strip().lower()
    m = method.upper()

    if pv == "naive":
        return build_prompts_naive(row, question)
    if pv == "production":
        if m != "A":
            raise ValueError("persona_variant 'production' uses a single Method A stream only")
        return build_prompts_production_persona(row, question)
    if pv == "a":
        return build_prompts_method_a(row, question)
    raise ValueError(
        f"Unknown persona_variant {persona_variant!r}; use naive, production, or a"
    )


def build_survey_prompts_for_persona_variant(
    persona_variant: str,
    method: str,
    row: Mapping[str, Any],
    questions: List[Question],
) -> Tuple[str, str]:
    """
    Same persona framing as ``build_prompts_for_persona_variant``, but one user block
    listing all questions and requesting a single JSON object (see ``format_multi_question_json_survey``).
    """
    if not questions:
        raise ValueError("questions list is empty")
    system, first_user = build_prompts_for_persona_variant(persona_variant, method, row, questions[0])
    tail = format_question_block(questions[0])
    if tail not in first_user:
        raise ValueError("internal: could not strip single-question tail for survey prompt")
    prefix = first_user.replace(tail, "", 1).rstrip()
    survey_block = format_multi_question_json_survey(questions)
    user = prefix + "\n\n" + survey_block
    return system, user
