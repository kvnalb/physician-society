"""Persona prompt builders: naive, minimal (B), rich (A), A+numeric grounding, and AB (default)."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Tuple

from simulation.questions_io import Question, format_multi_question_json_survey, format_question_block

# Bump when any prompt string changes so disk cache invalidates across runs.
PROMPT_VERSION = "2026-04-11-survey-json"


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


def build_prompts_method_a_numeric(row: Mapping[str, Any], question: Question) -> Tuple[str, str]:
    """Rich persona plus explicit numeric facts the model must not contradict."""
    system, user = build_prompts_method_a(row, question)
    numerics = [
        "VERIFIED NUMERICS (Medicare Part D 2022; do not contradict these on share/volume questions):",
        f"- GLP-1 share of diabetes claims: {_pct(row.get('glp1_penetration_2022'))}",
        f"- Branded share of Part D rows: {_pct(row.get('branded_share_2022'))}",
        f"- Distinct diabetes drugs (count): {row.get('drug_diversity_2022', '')}",
        f"- Any tirzepatide claims in 2022 extract: {row.get('has_tirzepatide_2022', '')}",
        f"- Diabetes share of claims: {_pct(row.get('diabetes_share_2022'))}",
    ]
    user = "\n".join(numerics) + "\n\n---\n\n" + user
    return system, user


def build_prompts_for_persona_variant(
    persona_variant: str,
    method: str,
    row: Mapping[str, Any],
    question: Question,
) -> Tuple[str, str]:
    """
    persona_variant:
      naive — only specialty/geo (single-method runs use method_a label).
      B — minimal grounded (Method B style).
      A — rich grounded (Method A style) for all method letters (single-method runs).
      AB — standard A vs B comparison.
      A_numeric — Method A gets numeric grounding; Method B stays minimal.
    """
    pv = persona_variant.strip().lower()
    m = method.upper()

    if pv == "naive":
        return build_prompts_naive(row, question)
    if pv == "b":
        return build_prompts_method_b(row, question)
    if pv == "a":
        return build_prompts_method_a(row, question)
    if pv == "a_numeric":
        if m == "B":
            return build_prompts_method_b(row, question)
        return build_prompts_method_a_numeric(row, question)
    if pv == "ab":
        if m == "A":
            return build_prompts_method_a(row, question)
        if m == "B":
            return build_prompts_method_b(row, question)
        raise ValueError(f"AB variant requires method A or B, got {method!r}")
    raise ValueError(
        f"Unknown persona_variant {persona_variant!r}; "
        "use naive, B, A, AB, A_numeric"
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
