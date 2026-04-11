"""Tirzepatide adoption simulation — offline-first demo (Streamlit)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from simulation.env_bootstrap import load_local_dotenv
from simulation.questions_io import Question, load_questions

PROJECT_ROOT = Path(__file__).resolve().parent
QUESTIONS_YAML = PROJECT_ROOT / "simulation" / "questions.yaml"
SUMMARY_PATH = PROJECT_ROOT / "artifacts" / "demo" / "summary.json"
SAMPLE_JSONL = PROJECT_ROOT / "artifacts" / "demo" / "sample_responses.jsonl"
DEFAULT_METRICS_PATH = PROJECT_ROOT / "artifacts" / "demo" / "metrics.json"
RUNS_ROOT = PROJECT_ROOT / "data" / "output" / "runs"
ARTIFACT_RUNS_DIR = PROJECT_ROOT / "artifacts" / "runs"
COHORT_PATH = PROJECT_ROOT / "data" / "output" / "tirzepatide_simulation_cohort_100.tsv"
DEFAULT_TOGETHER_MODEL = "zai-org/GLM-5.1"


@lru_cache(maxsize=1)
def _cached_questions() -> tuple[Question, ...]:
    return tuple(load_questions(QUESTIONS_YAML))


def _question_by_id(qid: str) -> Question | None:
    for q in _cached_questions():
        if q.question_id == qid:
            return q
    return None


def _question_short_title(qid: str, max_len: int = 120) -> str:
    q = _question_by_id(qid)
    if not q:
        return qid
    t = " ".join(q.text.split())
    return t if len(t) <= max_len else t[: max_len - 1] + "…"


def _option_labels_for_question(qid: str) -> dict[str, str]:
    q = _question_by_id(qid)
    if not q:
        return {}
    return {o.option_id: o.label for o in q.options}


def _pretty_geo_cluster(raw: object) -> str:
    s = str(raw).strip()
    if not s:
        return ""
    parts = s.split("_", 1)
    if len(parts) < 2:
        return s.replace("_", " ")
    abbr, tail = parts[0], parts[1]
    city = tail.replace("_", " ")
    return f"{city}, {abbr}"


def _pretty_archetype(raw: object) -> str:
    return str(raw).strip().replace("_", " ") or "—"


def _load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _discover_metrics_files() -> list[tuple[str, Path]]:
    """(label, path) for eval metrics JSON; deduped by resolved path."""
    found: list[tuple[str, Path]] = []
    if DEFAULT_METRICS_PATH.is_file():
        found.append(("Demo bundle (default)", DEFAULT_METRICS_PATH))
    if RUNS_ROOT.is_dir():
        for d in sorted(RUNS_ROOT.iterdir()):
            if not d.is_dir():
                continue
            mp = d / "metrics.json"
            if mp.is_file():
                found.append((f"Saved run: {d.name}", mp))
    if ARTIFACT_RUNS_DIR.is_dir():
        for mp in sorted(ARTIFACT_RUNS_DIR.glob("*metrics*.json")):
            if mp.is_file():
                found.append((f"Archived metrics: {mp.name}", mp))
    seen: set[Path] = set()
    out: list[tuple[str, Path]] = []
    for label, p in found:
        try:
            r = p.resolve()
        except OSError:
            continue
        if r not in seen:
            seen.add(r)
            out.append((label, p))
    return out


def _read_cohort_tsv() -> pd.DataFrame | None:
    if not COHORT_PATH.is_file():
        return None
    df = pd.read_csv(COHORT_PATH, sep="\t", low_memory=False)
    return df if len(df) else None


def _cohort_adoption_by_archetype(df: pd.DataFrame | None) -> dict:
    if df is None or "adoption_archetype" not in df.columns or "has_tirzepatide_2023" not in df.columns:
        return {}
    g = df.groupby("adoption_archetype")["has_tirzepatide_2023"].agg(["mean", "count"])
    out = {}
    for idx, row in g.iterrows():
        out[str(idx)] = {"rate": float(row["mean"]), "n": int(row["count"])}
    return out


def _dist_counts_to_df(dist: dict[str, int] | None, labels: dict[str, str]) -> pd.DataFrame:
    if not dist:
        return pd.DataFrame(columns=["Answer choice", "Simulated count"])
    rows = [
        {"Answer choice": labels.get(k, k), "Simulated count": int(v)}
        for k, v in sorted(dist.items(), key=lambda x: -x[1])
    ]
    return pd.DataFrame(rows)


def _render_sample_description(cohort_df: pd.DataFrame | None) -> None:
    st.markdown(
        """
        **Who is in the sample**

        - **Unit of analysis:** Individual prescribers (NPPES type-1) with practice locations in **six priority metros**
          (Houston, Los Angeles, New York City, Miami, Dallas, San Diego).
        - **Specialties:** **Endocrinology, Internal Medicine, and Family Medicine** only. Cardiology is excluded so the
          cohort stays diabetes- and GLP-1–relevant and strata stay balanced.
        - **Prescribing gate:** **Medicare Part D 2022** aggregates show meaningful **diabetes-related** prescribing
          (script volume and mix thresholds in the cohort builder). **Part D 2023** is required on the same NPIs so we
          can attach **revealed** post-window measures (e.g., tirzepatide claims where identifiable).
        - **Engagement:** **Open Payments (2022)** is merged for pharma exposure tiers; see breakdown tables below.

        **What this is not**

        This is a **purposive, Part D–scoped** slice—not a national probability sample of U.S. physicians. Readouts are
        **descriptive baselines** for this cohort; they are not causal estimates of launch effects.
        """
    )
    if cohort_df is None:
        st.info(
            f"No cohort file at `{COHORT_PATH.relative_to(PROJECT_ROOT)}`. "
            "Run `python scripts/06_tirzepatide_simulation_cohort.py` to build "
            "`tirzepatide_simulation_cohort_100.tsv` (large inputs; first run can take tens of minutes)."
        )
        return

    st.caption(
        "**What this block is:** a quick **QC / segment mix** view of the real cohort file the simulation draws from. "
        "**Why it matters for the business story:** it shows *who* sits behind the GLP-1 / tirzepatide readouts—"
        "specialty mix, priority cities, and adoption-style tags—so stakeholders do not confuse a purposive slice "
        "with “all U.S. doctors.”"
    )
    n = len(cohort_df)
    st.subheader("Loaded cohort snapshot")
    m1, m2, m3 = st.columns(3)
    m1.metric("Doctors in cohort file", n)
    if "has_tirzepatide_2023" in cohort_df.columns:
        rate = float(cohort_df["has_tirzepatide_2023"].mean())
        m2.metric("Share with any tirzepatide (Part D 2023)", f"{rate:.0%}")
    else:
        m2.metric("Share with any tirzepatide (Part D 2023)", "—")
    if "part_d_rows_2023" in cohort_df.columns:
        m3.metric("Typical Part D rows per doctor (2023, median)", f"{cohort_df['part_d_rows_2023'].median():.0f}")
    else:
        m3.metric("Typical Part D rows per doctor (2023, median)", "—")

    st.caption(
        "**How to read the three numbers:** cohort **size**, **early adoption signal** (any tirzepatide in the window "
        "we can see in Part D), and **data richness** (how many detail rows we see per doctor in 2023)—a proxy for "
        "how noisy segment splits might be."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Specialty mix**")
        if "specialty" in cohort_df.columns:
            s_counts = cohort_df["specialty"].value_counts().reset_index()
            s_counts.columns = ["Specialty", "Doctors"]
            st.caption("**What / why:** count of doctors by **board-style specialty**—helps you judge whether the "
                       "story is PCP-heavy vs endocrine-heavy.")
            st.dataframe(s_counts, hide_index=True, use_container_width=True)
    with c2:
        st.markdown("**Priority metros**")
        if "geo_cluster" in cohort_df.columns:
            vc = cohort_df["geo_cluster"].value_counts().reset_index()
            vc.columns = ["_raw", "Doctors"]
            vc["City / region (from data bucket)"] = vc["_raw"].map(_pretty_geo_cluster)
            vc = vc[["City / region (from data bucket)", "Doctors"]]
            st.caption("**What / why:** each **planning bucket** (e.g. Los Angeles, CA) is how we grouped practices for "
                       "geo balance—not a street address.")
            st.dataframe(vc, hide_index=True, use_container_width=True)
    with c3:
        st.markdown("**Adoption style tags**")
        if "adoption_archetype" in cohort_df.columns:
            vc = cohort_df["adoption_archetype"].value_counts().reset_index()
            vc.columns = ["_raw", "Doctors"]
            vc["Adoption style (model tag)"] = vc["_raw"].map(_pretty_archetype)
            vc = vc[["Adoption style (model tag)", "Doctors"]]
            st.caption("**What / why:** **internal segment labels** derived from 2022 prescribing—useful for comparing "
                       "simulated attitudes to **revealed** 2023 behavior by segment.")
            st.dataframe(vc, hide_index=True, use_container_width=True)

    with st.expander("Data sources & limitations"):
        st.markdown(
            """
            - **Medicare Part D** annual files define prescribing visibility; non-Medicare channels are out of scope.
            - **LLM survey** rows in this app come from `artifacts/demo/` unless you re-run the batch with an API key.
            - Full narrative: `docs/target_report.md`.
            """
        )


def _render_method_comparison_from_summary(mc: dict) -> None:
    st.subheader("Simulated survey: rich profile vs segment-card prompt")
    st.caption(
        "**What this is:** side-by-side **answer counts** for each survey item—**Method A** (full Part D + Sunshine "
        "profile in the prompt) vs **Method B** (exec-style segment card: specialty, home city/state, site line, "
        "archetype + Sunshine band, geo bucket—**no** claim-level tables). "
        "**Why it matters commercially:** if the two stories diverge sharply, **scenario outputs depend on how much "
        "grounding you pay for**; if they align, a lighter briefing may be enough for early workshops."
    )
    if not mc:
        st.info("No saved method comparison in `summary.json` yet—run a batch so the demo bundle includes counts.")
        return
    for qid, dists in mc.items():
        title = _question_short_title(qid)
        labels = _option_labels_for_question(qid)
        ma = dists.get("method_a_distribution") or {}
        mb = dists.get("method_b_distribution") or {}
        with st.expander(title):
            st.caption(f"Internal id: `{qid}` — use the tables below, not raw option codes, when presenting.")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Method A — data-room style profile**")
                st.dataframe(_dist_counts_to_df(ma, labels), hide_index=True, use_container_width=True)
            with c2:
                st.markdown("**Method B — segment card (ChatGPT-style briefing)**")
                st.dataframe(_dist_counts_to_df(mb, labels), hide_index=True, use_container_width=True)


def _render_survey_agreement_block(metrics: dict) -> None:
    surv = metrics.get("survey") or {}
    pq = surv.get("per_question") or {}
    if not pq:
        return
    st.subheader("Agreement: same doctor, two prompt styles")
    st.caption(
        "**What this is:** for each survey item, **Cohen's κ** summarizes how often **Method A and Method B** pick "
        "the **same answer** for the **same doctor** (paired rows). **JS / TV** describe how different the **overall** "
        "answer mix is between methods. "
        "**Why it matters:** low agreement means **your workshop narrative flips** depending on whether you used a "
        "heavy claims brief or a light segment card—high agreement means the extra data may not change the headline "
        "for that question."
    )
    rows = []
    for qid, v in pq.items():
        rows.append(
            {
                "Survey item": _question_short_title(qid),
                "Paired doctors": v.get("n_paired"),
                "Cohen κ (A vs B)": None if v.get("cohen_kappa") is None else round(float(v["cohen_kappa"]), 4),
                "Scenario spread (JS)": None
                if v.get("js_method_ab_marginal") is None
                else round(float(v["js_method_ab_marginal"]), 4),
                "Max category shift (TV)": None
                if v.get("tv_method_ab_marginal") is None
                else round(float(v["tv_method_ab_marginal"]), 4),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    stab = surv.get("stability")
    if stab:
        st.markdown(f"**Read of average agreement:** {stab}")
    st.caption(
        "**κ (kappa):** 0–1 would be strong alignment on categories; negative values mean **less overlap than chance** "
        "on these labels—often a sign the two prompts push the model to **different plausible worlds**. "
        "**JS** = Jensen–Shannon (0 = identical mix). **TV** = total variation (0–1; share of answers that would need "
        "to move to match the other method's histogram)."
    )


def _render_distribution_quality_block(metrics: dict) -> None:
    dq = metrics.get("distribution_quality")
    if not isinstance(dq, dict):
        return
    st.subheader("Distribution quality pillar (aggregate)")
    pillar = dq.get("pillar", "")
    st.caption(
        (f"**Eval intent:** {pillar} " if pillar else "")
        + "**Business translation:** this pillar is **not** “did we replicate a human panel?”—it asks whether **two "
        "grounding budgets** (full profile vs segment card) produce **the same market-shape story** when you tally "
        "simulated choices."
    )
    c1, c2 = st.columns(2)
    mjs = dq.get("mean_js_method_ab")
    mtv = dq.get("mean_tv_method_ab")
    c1.metric("Avg scenario spread (JS, across items)", f"{mjs:.4f}" if mjs is not None else "—")
    c2.metric("Avg max category shift (TV, across items)", f"{mtv:.4f}" if mtv is not None else "—")


def _render_behavioral_alignment_block(metrics: dict) -> None:
    ba = metrics.get("behavioral_alignment")
    if not isinstance(ba, dict) or not ba.get("per_question"):
        st.subheader("Behavioral check vs Medicare-derived hints")
        st.caption(
            "**What this would be:** how often simulated answers line up with **simple rules built from the same "
            "doctor's claims row** (pseudo-labels—not a human truth panel). "
            "**Why it matters:** if you pitch “grounded personas,” this is a **sanity check** that the model is not "
            "systematically contradicting **observable prescribing cues**."
        )
        st.info("No `behavioral_alignment` block in this metrics file—usually because the cohort TSV was missing "
                "when `eval.run_eval` ran, or the bundle is a placeholder.")
        return
    st.subheader("Behavioral check vs Medicare-derived hints")
    note = ba.get("note", "")
    rules_v = ba.get("rules_version", "")
    mfa = ba.get("method_for_alignment", "")
    st.caption(
        f"**What this is:** match rate between **simulated answers** (here: **{mfa}**) and **claims-derived "
        f"pseudo-labels** (rules version **{rules_v}**). {note} "
        "**Why it matters commercially:** it is a **facing test** for “did we stay in the zip code of what Medicare "
        "says about this prescriber?”—not proof of clinical accuracy."
    )
    m_acc = ba.get("mean_accuracy_over_labeled_questions")
    st.metric("Average match rate across labeled items", f"{m_acc:.3f}" if m_acc is not None else "—")
    rows = []
    for qid, v in ba["per_question"].items():
        rows.append(
            {
                "Survey item": _question_short_title(qid),
                "Labeled doctor-answers": v.get("n_labeled"),
                "Match rate": None if v.get("accuracy") is None else round(float(v["accuracy"]), 4),
                "Shape gap vs hints (JS)": None
                if v.get("js_divergence_marginal") is None
                else round(float(v["js_divergence_marginal"]), 4),
                "Max shift vs hints (TV)": None
                if v.get("tv_distance_marginal") is None
                else round(float(v["tv_distance_marginal"]), 4),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    with st.expander("Technical: predicted vs pseudo-label counts (per item)"):
        st.caption("Use for debugging segment skew—not for executive readouts.")
        for qid, v in ba["per_question"].items():
            st.markdown(f"**{_question_short_title(qid)}** (`{qid}`)")
            c1, c2 = st.columns(2)
            labels = _option_labels_for_question(qid)
            with c1:
                st.markdown("*Simulated (method above)*")
                st.json(v.get("pred_distribution") or {})
            with c2:
                st.markdown("*Pseudo-labels from claims rules*")
                st.json(v.get("pseudo_label_distribution") or {})


def _render_persona_coherence_block(metrics: dict) -> None:
    pc = metrics.get("persona_coherence")
    if not isinstance(pc, dict):
        return
    st.subheader("Persona coherence (cross-item rules)")
    st.caption(
        f"**What this is:** cheap **logic checks** across answers for the same simulated doctor (rules "
        f"`{pc.get('rules_version', '')}`). {pc.get('note', '')} "
        "**Why it matters:** even if marginals look fine, **internally inconsistent stories** read as “bad bots” to "
        "medical reviewers and undermine workshop trust."
    )
    n_blocks = pc.get("n_method_blocks_checked")
    n_viol = pc.get("n_violations")
    vr = pc.get("violation_rate_per_method_block")
    c1, c2, c3 = st.columns(3)
    c1.metric("Method blocks checked", int(n_blocks) if n_blocks is not None else "—")
    c2.metric("Rule violations found", int(n_viol) if n_viol is not None else "—")
    c3.metric("Violation rate (per block)", f"{float(vr):.3f}" if vr is not None else "—")
    sample = pc.get("violations_sample") or []
    if sample:
        st.markdown("**Sample violations (up to 50)**")
        st.dataframe(pd.DataFrame(sample), hide_index=True, use_container_width=True)


def _render_instrument_health_block(metrics: dict) -> None:
    ih = metrics.get("instrument_health")
    if not isinstance(ih, dict):
        return
    st.subheader("Run & instrument health")
    st.caption(
        "**What this is:** **pipeline QA**—parse coverage, missing cells, API errors, latency—straight from the "
        "response JSONL. **Why it matters:** before you interpret κ or segment charts, you need to know whether the "
        "run was **complete, slow, or partially broken** (e.g. missing answers for one method)."
    )
    lat = ih.get("latency_ms") or {}
    summary_rows = [
        {"Metric": "Survey rows in file", "Value": ih.get("n_jsonl_rows")},
        {"Metric": "v2 (one row per doctor) rows", "Value": ih.get("n_v2_rows")},
        {"Metric": "Legacy flat rows", "Value": ih.get("n_legacy_rows")},
        {"Metric": "Flattened answer cells", "Value": ih.get("n_flat_cells")},
        {"Metric": "Cells with errors", "Value": ih.get("flat_cells_with_error")},
        {"Metric": "Cells missing a chosen option", "Value": ih.get("flat_cells_missing_option")},
        {"Metric": "v2 missing per-question cells", "Value": ih.get("v2_missing_question_cells")},
        {"Metric": "v2 survey-level API/parse errors", "Value": ih.get("v2_survey_level_errors")},
        {"Metric": "LLM calls with latency logged", "Value": lat.get("n_calls_with_latency")},
        {"Metric": "Mean latency (ms)", "Value": None if lat.get("mean") is None else round(float(lat["mean"]), 1)},
        {"Metric": "Median latency (ms)", "Value": lat.get("p50")},
        {"Metric": "Max latency (ms)", "Value": lat.get("max")},
        {"Metric": "Question ↔ claims map (YAML)", "Value": ih.get("claims_map_file")},
    ]
    st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)
    sn = ih.get("schema_notes")
    if sn:
        with st.expander("Schema notes"):
            st.write(sn)


def _render_eval_coverage_sidebar(metrics: dict) -> None:
    """Confirm each eval bundle section is present (parity with eval.metrics.compute_metrics_bundle)."""
    with st.expander("Eval bundle coverage (what this file contains)"):
        st.caption("Each row ties to **`eval/metrics.py` → `compute_metrics_bundle`**.")
        rows = [
            {"Block": "survey", "Present": bool(metrics.get("survey")), "Role": "A vs B agreement + per-item stats"},
            {
                "Block": "distribution_quality",
                "Present": bool(metrics.get("distribution_quality")),
                "Role": "Aggregate JS/TV summary for A vs B marginals",
            },
            {"Block": "persona_coherence", "Present": bool(metrics.get("persona_coherence")), "Role": "Cross-item rule violations"},
            {"Block": "instrument_health", "Present": bool(metrics.get("instrument_health")), "Role": "Parse/coverage/latency QA"},
            {
                "Block": "behavioral_alignment",
                "Present": bool(metrics.get("behavioral_alignment")),
                "Role": "Match to claims-derived pseudo-labels (needs cohort at eval time)",
            },
            {"Block": "run_manifest", "Present": bool(metrics.get("run_manifest")), "Role": "Pinned batch settings (if saved)"},
            {"Block": "eval_meta", "Present": bool(metrics.get("eval_meta")), "Role": "Paths + options used when metrics were built"},
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def main() -> None:
    load_local_dotenv(override=False)
    st.set_page_config(page_title="Tirzepatide Adoption Simulation", layout="wide")

    with st.sidebar:
        st.header("LLM smoke re-run")
        st.caption("Session-only; not saved. Used when you trigger Advanced → live API re-run.")
        smoke_provider = st.selectbox(
            "LLM provider",
            ["together", "openai"],
            index=0,
            help="together uses the native Together SDK; openai uses the OpenAI Python client (optional base URL).",
        )
        smoke_model = st.text_input(
            "Model",
            value=DEFAULT_TOGETHER_MODEL,
            help="Together model id by default; use e.g. gpt-4o-mini with OpenAI provider.",
        )
        smoke_temp = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
        smoke_base_url = st.text_input(
            "API base URL (optional)",
            value="",
            help="Only for OpenAI provider: e.g. https://api.together.xyz/v1 for OpenAI-compatible endpoints.",
        )

    summary = _load_json(SUMMARY_PATH)

    st.title("Tirzepatide adoption simulation")
    st.caption("Novo's 6-week decision problem (June 2022) — Medicare Part D–scoped physician POC")

    cohort_df = _read_cohort_tsv()

    st.header("About")
    st.markdown(
        """
        **Problem.** At GLP-1 launch speed, brand teams need **segment- and geography-aware**
        hypotheses faster than traditional surveys—while staying **anchored to observed prescribing**.

        **Cohort.** ~100 physicians (after filters) in **six priority metros**, **Endocrinology /
        Internal Medicine / Family Medicine only**, with **2022→2023 Part D** linked for **revealed**
        tirzepatide/GLP-1 patterns. See `docs/target_report.md` for scope conditions.

        **Methods.** **Method A** = LLM persona with **full Part D + Sunshine “data room”** context in the prompt.
        **Method B** = **exec-style segment card** (specialty, home city/state, registry site line, adoption-style tag,
        Sunshine band, priority-metro bucket)—**not** claim-by-claim tables. **Eval** = agreement between those two
        elicitation styles, **distribution** checks, **coherence** rules, **instrument health**, and optional
        **claims-hint alignment**—plus **empirical** adoption baselines from Part D (not causal).
        """
    )

    st.header("Sample description")
    _render_sample_description(cohort_df)

    if summary.get("offline_seed"):
        st.warning(
            "Demo bundle uses **offline deterministic seed** data (`--offline-seed-demo`), not real LLM calls. "
            "Run `python -m simulation.run_batch --limit-npis 10 --save-as-demo-bundle` with "
            "`TOGETHER_API_KEY` (or `--provider openai` and `OPENAI_API_KEY`) for authentic responses."
        )
    elif summary.get("is_placeholder"):
        st.info(
            "Demo bundle is a **placeholder**. Run `python -m simulation.run_batch "
            "--limit-npis 10 --save-as-demo-bundle` with `TOGETHER_API_KEY` set (default provider), "
            "or `--provider openai` with `OPENAI_API_KEY`, "
            "or use `--offline-seed-demo` if you only need a runnable pipeline without an API."
        )

    st.header("Results")
    metrics_options = _discover_metrics_files()
    if not metrics_options:
        st.warning("No `metrics.json` found. Run `make eval` or `python -m eval.run_eval`.")
        metrics: dict = {}
        metrics_path_used: Path | None = None
    else:
        labels = [x[0] for x in metrics_options]
        default_ix = 0
        for i, (_, p) in enumerate(metrics_options):
            if p.resolve() == DEFAULT_METRICS_PATH.resolve():
                default_ix = i
                break
        pick = st.selectbox(
            "Choose eval snapshot",
            range(len(labels)),
            format_func=lambda i: labels[i],
            index=default_ix,
            help="Pick which `metrics.json` to render. Demo default is the bundled file; other folders are prior runs.",
        )
        _, metrics_path_used = metrics_options[pick]
        metrics = _load_json(metrics_path_used)
        st.caption(
            f"**Loaded file:** `{metrics_path_used.relative_to(PROJECT_ROOT)}` — **what / why:** pick which "
            "frozen eval bundle you are presenting (e.g. latest smoke test vs checked-in demo)."
        )

    _render_eval_coverage_sidebar(metrics)

    if metrics.get("error"):
        st.error(
            f"**This metrics file is incomplete:** `{metrics.get('error')}`. "
            "Run `python -m simulation.run_batch …` then `make eval` so charts below fill in."
        )

    st.subheader("Executive snapshot")
    st.caption(
        "**What this row is:** a **one-glance** view of run size, **cross-prompt agreement** (κ), **shape gap** "
        "between methods (JS), **facing vs claims hints** (when present), **story consistency** (coherence violation "
        "rate), and **data completeness** (missing cells). **Why:** executives should see **trust + stability** "
        "signals before diving into tables."
    )
    surv = metrics.get("survey", {}) if metrics else {}
    dq = metrics.get("distribution_quality") if metrics else {}
    ba = metrics.get("behavioral_alignment") if metrics else {}
    pc = metrics.get("persona_coherence") if metrics else {}
    ih = metrics.get("instrument_health") if metrics else {}

    kappa = surv.get("method_agreement_kappa_mean")
    mjs = dq.get("mean_js_method_ab") if isinstance(dq, dict) else None
    m_acc = ba.get("mean_accuracy_over_labeled_questions") if isinstance(ba, dict) else None
    vr = pc.get("violation_rate_per_method_block") if isinstance(pc, dict) else None
    miss = ih.get("flat_cells_missing_option") if isinstance(ih, dict) else None

    e1, e2, e3, e4, e5, e6 = st.columns(6)
    e1.metric("Doctors in demo summary", summary.get("n_npis", "—"))
    e2.metric("Survey items in battery", summary.get("n_questions", "—"))
    e3.metric("Avg κ (A vs B)", f"{kappa:.3f}" if kappa is not None else "—")
    e4.metric("Avg scenario spread (JS)", f"{mjs:.4f}" if mjs is not None else "—")
    e5.metric("Avg match vs claims hints", f"{m_acc:.3f}" if m_acc is not None else "—")
    e6.metric("Coherence violation rate", f"{vr:.3f}" if vr is not None else "—")
    if miss is not None:
        st.metric("Missing answer cells (run health)", int(miss))

    rm = metrics.get("run_manifest")
    if isinstance(rm, dict) and rm:
        with st.expander("Run configuration (pinned batch settings)"):
            st.caption(
                "**What / why:** exact CLI/model/cohort choices for **reproducibility**—attach to any screenshot "
                "shared internally."
            )
            st.json(rm)

    em = metrics.get("eval_meta")
    if isinstance(em, dict) and em:
        with st.expander("Eval build metadata"):
            st.caption("**What / why:** which responses path and alignment method were used when `run_eval` produced "
                       "this JSON.")
            st.json(em)

    st.markdown("---")
    actual = summary.get("adoption_by_archetype_actual") or _cohort_adoption_by_archetype(cohort_df)
    if actual:
        st.subheader("Revealed adoption by segment tag (Medicare Part D 2023)")
        st.caption(
            "**What this chart is:** among doctors in the cohort, the **share with any tirzepatide** visible in "
            "**Part D 2023**, broken out by **adoption-style tag** from 2022 data. **Why it matters:** it is the "
            "**empirical baseline** you might compare to **simulated** early-launch posture—**descriptive only**, "
            "not proof that messaging caused prescribing."
        )
        archetypes = list(actual.keys())
        archetypes_pretty = [_pretty_archetype(a) for a in archetypes]
        rates = [actual[a]["rate"] for a in archetypes]
        fig = go.Figure(
            data=[
                go.Bar(
                    x=archetypes_pretty,
                    y=rates,
                    marker_color="#2E5077",
                    name="Actual (Part D 2023)",
                )
            ]
        )
        fig.update_layout(
            title="Share with any tirzepatide claims (Part D 2023), by adoption-style tag",
            yaxis_title="Share of doctors in segment",
            xaxis_title="Adoption-style tag (from 2022 prescribing)",
            template="plotly_white",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No cohort or summary data for adoption-by-archetype chart.")

    _render_method_comparison_from_summary(summary.get("method_comparison") or {})

    _render_survey_agreement_block(metrics)
    _render_distribution_quality_block(metrics)
    _render_behavioral_alignment_block(metrics)
    _render_persona_coherence_block(metrics)
    _render_instrument_health_block(metrics)

    st.header("Reasoning examples")
    st.caption(
        "**What this is:** a few **verbatim rationales** pulled from flattened `sample_responses.jsonl`. "
        "**Why it matters:** executives often trust **one believable quote** more than a wall of metrics—use to "
        "illustrate *how* the model is arguing, not as evidence of real physician speech."
    )
    lines: list[str] = []
    if SAMPLE_JSONL.is_file():
        with open(SAMPLE_JSONL, encoding="utf-8") as fh:
            lines = [ln.strip() for ln in fh if ln.strip()]
    if not lines:
        st.write("No `sample_responses.jsonl` lines yet.")
    else:
        for i, line in enumerate(lines[:5]):
            r = json.loads(line)
            q_label = _question_short_title(str(r.get("question_id", "")))
            with st.expander(f"Doctor ID {r.get('npi')} — {q_label} ({r.get('method')})"):
                st.code(r.get("parsed_option") or r.get("error") or "", language="text")
                st.write(r.get("reasoning") or "(no rationale captured)")

    st.header("Advanced")
    live = st.checkbox("Re-run with live API (session only; key not saved)")
    if live:
        key_label = "TOGETHER_API_KEY" if smoke_provider == "together" else "OPENAI_API_KEY"
        api_key = st.text_input(key_label, type="password", help="Used only in this browser session.")
        if st.button("Smoke re-run (5 NPIs, all questions, both methods)"):
            if not api_key:
                st.error("Enter an API key.")
            else:
                env = {**os.environ}
                if smoke_provider == "together":
                    env["TOGETHER_API_KEY"] = api_key
                else:
                    env["OPENAI_API_KEY"] = api_key
                default_model = DEFAULT_TOGETHER_MODEL if smoke_provider == "together" else "gpt-4o-mini"
                cmd = [
                    sys.executable,
                    "-m",
                    "simulation.run_batch",
                    "--provider",
                    smoke_provider,
                    "--run-id",
                    "streamlit_smoke",
                    "--persona-variant",
                    "ab",
                    "--concurrency",
                    "4",
                    "--limit-npis",
                    "5",
                    "--model",
                    smoke_model.strip() or default_model,
                    "--temperature",
                    str(smoke_temp),
                    "--output-dir",
                    str(PROJECT_ROOT / "data" / "output" / "runs" / "streamlit_smoke"),
                ]
                if smoke_base_url.strip():
                    cmd.extend(["--base-url", smoke_base_url.strip()])
                try:
                    with st.spinner("Running batch…"):
                        proc = subprocess.run(
                            cmd,
                            cwd=str(PROJECT_ROOT),
                            env=env,
                            capture_output=True,
                            text=True,
                            timeout=600,
                        )
                    st.text(proc.stdout)
                    if proc.stderr:
                        st.text(proc.stderr)
                    if proc.returncode != 0:
                        st.warning("Batch returned non-zero; showing demo artifacts as fallback.")
                except subprocess.TimeoutExpired:
                    st.error("Timed out after 10 minutes.")
                except Exception as exc:  # noqa: BLE001
                    st.error(str(exc))

    st.divider()
    repo_hint = os.environ.get("DEMO_REPO_URL", "").strip()
    repo_line = (
        f"Repository: {repo_hint}"
        if repo_hint
        else "Repository: set `DEMO_REPO_URL` in the environment to show a link here (optional)."
    )
    st.caption(
        f"{repo_line} Setup: `pip install -r requirements.txt` then `streamlit run streamlit_app.py` "
        "or `make demo`. **Keys:** `TOGETHER_API_KEY` for the default Together SDK, or `--provider openai` with "
        "`OPENAI_API_KEY` (optional `--base-url`; see `.env.example`). **Limitations:** Medicare Part D only; "
        "annual files; "
        "purposive cohort—not a national probability sample."
    )


if __name__ == "__main__":
    main()
