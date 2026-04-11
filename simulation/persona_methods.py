"""Persona prompt builders: naive, exec-style segment card (B), rich (A), A+numeric grounding, and AB (default)."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Tuple

from simulation.questions_io import Question, format_multi_question_json_survey, format_question_block

# Bump when any prompt string changes so disk cache invalidates across runs.
PROMPT_VERSION = "2026-04-10-method-b-exec-briefing"


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


def _method_b_pharma_paragraph(row: Mapping[str, Any]) -> str:
    tier = str(row.get("pharma_engagement_tier") or "").strip()
    mapping = {
        "Low_Engagement": (
            "**Low** aggregate Sunshine / Open Payments footprint versus peers—think **light touch**, "
            "not **never interacts with industry**."
        ),
        "Medium_Engagement": (
            "**Moderate** Sunshine footprint—some meals, consulting, or speaking shows up; stay realistic "
            "about access conversations but do not invent dollar amounts."
        ),
        "High_Engagement": (
            "**Heavier** Sunshine footprint than many peers—field teams would treat them as higher-touch; "
            "still do not invent specific transfers of value."
        ),
    }
    return mapping.get(
        tier,
        "Sunshine band is **unspecified on this card**—stay neutral; do not assume access favors any brand.",
    )


def _method_b_geo_paragraph(row: Mapping[str, Any]) -> str:
    gc = str(row.get("geo_cluster") or "").strip()
    if not gc:
        return ""
    parts = gc.split("_", 1)
    if len(parts) < 2:
        return (
            f"Commercial planning tags them in bucket **{gc.replace('_', ' ')}**—use typical access and peer "
            "dynamics implied for that slice."
        )
    st_abbrev, tail = parts[0], parts[1]
    city_words = tail.replace("_", " ")
    state_long = {"TX": "Texas", "CA": "California", "FL": "Florida", "NY": "New York"}.get(
        st_abbrev, st_abbrev
    )
    return (
        f"Geography bucket on the brand map: **{city_words}** ({state_long}). "
        "Use that as implied patient mix and access pressure—not a census tract."
    )


def build_prompts_method_b(row: Mapping[str, Any], question: Question) -> Tuple[str, str]:
    """
    Exec-style “segment card” persona: specialty + home practice city/state, plus light brand-team
    context (site line, archetype tag, Sunshine band, geo bucket)—deliberately **without** Part D tables
    (those stay in Method A).
    """
    spec = str(row.get("specialty", "Unknown"))
    city = str(row.get("city", ""))
    state = str(row.get("state", ""))
    org = str(row.get("organization_name", "")).strip()
    creds = str(row.get("credentials", "")).strip()
    arch_raw = str(row.get("adoption_archetype", "Unclassified"))
    arch_readable = arch_raw.replace("_", " ")
    system = (
        "You are role-playing **one** U.S. outpatient prescriber for a **fast workshop / ChatGPT-style drill**. "
        "A brand-side colleague pasted a **short segment card** below—**not** the full data-room workbook "
        "(no claim-by-claim tables). Stay internally consistent with the card. "
        "Do not invent a personal name, a specific patient, or numeric claims metrics that are not implied."
    )
    lines = [
        "### Segment card (the kind of sketch a pharma lead would paste into ChatGPT before a workshop)",
        "",
        f"You are a **{spec}** clinician whose **home practice market** is **{city}, {state}**—anchor payer norms, "
        "peer behavior, and “what feels typical here” to that community.",
    ]
    if creds:
        lines.append(f"- **Registry credentials:** {creds}")
    if org:
        lines.append(f"- **Practice / site line on file (often an address or legal entity text):** {org}")
    else:
        lines.append(
            "- **Practice / site:** Only the city/state above—assume a mainstream community or employed "
            "setting unless the questions force a choice."
        )
    lines.extend(
        [
            f"- **Adoption posture tag from the forecasting team:** “{arch_readable}” "
            f"(internal shorthand `{arch_raw}`—interpret it; do not quote the code back unless asked).",
            f"- **Sunshine / Open Payments band (relationship intensity, not a moral score):** "
            f"{_method_b_pharma_paragraph(row)}",
        ]
    )
    geo = _method_b_geo_paragraph(row)
    if geo:
        lines.append(f"- **Geo / priority-market context:** {geo}")
    lines.extend(
        [
            "",
            "Answer the survey as that clinician in **June 2022** (GLP-1 momentum; tirzepatide newly in market).",
            "",
            "---",
            "",
            format_question_block(question),
        ]
    )
    user = "\n".join(lines)
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
      B — exec-style segment card (Method B; no Part D tables).
      A — rich grounded (Method A style) for all method letters (single-method runs).
      AB — standard A vs B comparison.
      A_numeric — Method A gets numeric grounding; Method B stays the segment-card style.
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
