"""Tirzepatide adoption simulation — offline-first demo (Streamlit)."""

from __future__ import annotations

import html
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

# -----------------------------------------------------------------------------
# Paths and defaults
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
QUESTIONS_YAML = PROJECT_ROOT / "simulation" / "questions.yaml"
SUMMARY_PATH = PROJECT_ROOT / "artifacts" / "demo" / "summary.json"
SAMPLE_JSONL = PROJECT_ROOT / "artifacts" / "demo" / "sample_responses.jsonl"
DEFAULT_METRICS_PATH = PROJECT_ROOT / "artifacts" / "demo" / "metrics.json"
RUNS_ROOT = PROJECT_ROOT / "data" / "output" / "runs"
ARTIFACT_RUNS_DIR = PROJECT_ROOT / "artifacts" / "runs"
COHORT_PATH = PROJECT_ROOT / "data" / "output" / "tirzepatide_simulation_cohort_100.tsv"
DEFAULT_TOGETHER_MODEL = "zai-org/GLM-5.1"

# Short labels for expander headers (Streamlit truncates long titles in the collapsed row).
_EXPANDER_LABELS_BY_QID: dict[str, str] = {
    "f_q1_tirzepatide_12m": "Q1 — Tirzepatide use outlook (12 months, Medicare Part D)",
    "f_q2_glp1_trajectory": "Q2 — GLP-1 share trajectory (program year into 2023)",
    "f_q3_branded_trajectory": "Q3 — Branded vs generic oral diabetes mix (Part D)",
    "f_q4_diabetes_mix_trajectory": "Q4 — Diabetes-related share of Part D prescribing",
    "f_q5_panel_scale_trajectory": "Q5 — Overall Part D activity scale vs 2022 baseline",
    "f_q6_molecule_breadth_trajectory": "Q6 — Breadth of diabetes-relevant molecules (Part D)",
}


# -----------------------------------------------------------------------------
# Shared UI primitives (used across sections)
# -----------------------------------------------------------------------------


def _inject_page_styles() -> None:
    """Subtle typography and spacing for long-form explanations (Streamlit markdown is paragraph-based)."""
    st.markdown(
        """
        <style>
          /* Slightly calmer reading width on ultra-wide monitors */
          .block-container { max-width: 1200px; }
          /* Lift default prose so paragraphs/lists read closer to header scale */
          div[data-testid="stMarkdownContainer"] {
            font-size: 1.07rem;
            line-height: 1.62;
          }
          div[data-testid="stMarkdownContainer"] p {
            font-size: inherit;
            line-height: inherit;
          }
          div[data-testid="stMarkdownContainer"] li {
            line-height: inherit;
            margin-bottom: 0.35rem;
          }
          /* Info / warning callouts use the same markdown nodes */
          div[data-testid="stAlert"] div[data-testid="stMarkdownContainer"] {
            font-size: 1.05rem;
            line-height: 1.58;
          }
          .ps-callout {
            font-size: 1.02rem;
            line-height: 1.58;
            color: #1f2937;
            background: #f8fafc;
            border-left: 3px solid #2E5077;
            border-radius: 6px;
            padding: 0.75rem 1rem 0.85rem 1rem;
            margin: 0 0 0.9rem 0;
          }
          .ps-callout p { margin: 0 0 0.55rem 0; }
          .ps-callout p:last-child { margin-bottom: 0; }
          .ps-muted {
            font-size: 0.98rem;
            line-height: 1.55;
            color: #4b5563;
            margin: 0.15rem 0 0.85rem 0;
          }
          h2 { margin-top: 1.25rem; }
          h3 { margin-top: 0.85rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _callout_md(body_markdown: str) -> None:
    """Render a shaded callout; ``body_markdown`` must be trusted (static app copy)."""
    st.markdown(f'<div class="ps-callout">{body_markdown}</div>', unsafe_allow_html=True)


def _muted_md(body_markdown: str) -> None:
    """Smaller supporting line(s); ``body_markdown`` is trusted static copy."""
    st.markdown(f'<div class="ps-muted">{body_markdown}</div>', unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Survey YAML helpers (questions, expander labels, option text)
# -----------------------------------------------------------------------------


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


def _expander_label(qid: str) -> str:
    return _EXPANDER_LABELS_BY_QID.get(qid, _question_short_title(qid, max_len=96))


def _option_labels_for_question(qid: str) -> dict[str, str]:
    q = _question_by_id(qid)
    if not q:
        return {}
    return {o.option_id: o.label for o in q.options}


# -----------------------------------------------------------------------------
# Small formatters and table builders
# -----------------------------------------------------------------------------


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


def _dist_counts_to_df(dist: dict[str, int] | None, labels: dict[str, str]) -> pd.DataFrame:
    if not dist:
        return pd.DataFrame(columns=["Answer choice", "Simulated count"])
    rows = [
        {"Answer choice": labels.get(k, k), "Simulated count": int(v)}
        for k, v in sorted(dist.items(), key=lambda x: -x[1])
    ]
    return pd.DataFrame(rows)


def _distribution_from_summary_entry(dists: dict) -> dict[str, int]:
    """Normalize new ``simulated_distributions`` and legacy ``method_comparison`` shapes."""
    if "distribution" in dists:
        raw = dists["distribution"]
        return {str(k): int(v) for k, v in raw.items()} if isinstance(raw, dict) else {}
    ma = dists.get("method_a_distribution")
    return {str(k): int(v) for k, v in ma.items()} if isinstance(ma, dict) else {}


# -----------------------------------------------------------------------------
# File I / O and metrics discovery
# -----------------------------------------------------------------------------


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


# =============================================================================
# Report layout: functions below follow on-page order (top → bottom).
# Sidebar is listed first because Streamlit executes it before the main column.
# =============================================================================


def _render_sidebar_smoke_settings() -> tuple[str, str, float, str]:
    """Optional live batch controls (values consumed later in Advanced)."""
    with st.sidebar:
        st.header("Optional live model smoke test")
        st.caption(
            "Runs only if you use **Advanced → Re-run with live API**. Settings apply to this browser session only "
            "and are not saved."
        )
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
    return smoke_provider, smoke_model, smoke_temp, smoke_base_url


def _render_title_block() -> None:
    st.title("Simulating Physician Reactions to a GLP-1 Launch")
    st.caption(
        "A proof-of-concept for LLM-based simulation of high-value professional audiences, "
        "grounded in revealed prescribing behavior from Medicare Part D and CMS Open Payments."
    )


def _render_about_section() -> None:
    st.header("What this demonstrates")
    st.markdown(
        """
        Traditional market research asks physicians what they *say* they would do. This simulation
        asks what they would do based on what they *actually* do.

        Each persona is a real physician whose prescribing profile (Medicare Part D 2022) and
        industry financial relationships (CMS Open Payments 2022) are encoded into an LLM prompt.
        The simulation asks forward-looking questions — e.g., "will you adopt tirzepatide?" — and
        we evaluate answers against what actually happened in the physician's Part D 2023 claims,
        which the model never sees.

        **Why this matters commercially.** Pharma brand teams currently spend USD 200–500K per physician
        survey and USD 15–25K per physician per day for advisory boards. A validated simulation lets
        them pre-test messaging, explore sensitive hypotheticals, and iterate on launch strategy at
        a fraction of the cost — with responses grounded in each physician's real practice, not
        demographic stereotypes.

        **Why this matters methodologically.** Standard LLM persona approaches condition on
        demographics alone and achieve 61–67% distribution accuracy on opinion surveys (per
        [Artificial Societies' January 2026 eval report](https://societies.io)). By grounding
        personas in revealed behavioral data — actual prescribing patterns, financial relationships,
        practice context — this demo tests whether distributional fidelity improves, and whether
        the qualitative reasoning becomes clinically plausible enough to substitute for live
        advisory input.

        **Current status: proof-of-concept, not production.** Simulation quality is not yet at the
        level required for commercial deployment. The current pipeline uses a single year of
        prescribing data with a general-purpose LLM and no social network modeling. The
        [improvement roadmap](#improvement-roadmap) below describes concrete next steps — several
        already underway — that we expect to materially improve both distributional accuracy and
        qualitative fidelity.
        """
    )


def _render_sample_description(cohort_df: pd.DataFrame | None) -> None:
    st.header("Sample description")
    st.markdown(
        """
        **The cohort: 100 real physicians across six metros.** Endocrinologists, internists, and family medicine
        physicians in Houston, Los Angeles,
        New York, Miami, Dallas, and San Diego — selected because they have Medicare Part D
        prescribing data in both 2022 (persona input) and 2023 (evaluation holdout), and
        because these metros concentrate enough physicians in overlapping health systems to
        form a natural professional network.

        Each persona is built from:
        - **Prescribing profile** (Part D 2022): drug mix, volumes, therapeutic focus, branded vs. generic share
        - **Industry relationships** (Open Payments 2022): payment amounts, companies, payment types (consulting, speaking, research, meals)
        - **Demographics** (NPPES): specialty, gender, geography, organizational affiliation

        **What the model never sees:** Part D 2023 claims data, including whether the physician adopted
        tirzepatide. This is the holdout used for evaluation.
        """
    )
    if cohort_df is None:
        st.info(
            f"No cohort file at `{COHORT_PATH.relative_to(PROJECT_ROOT)}`. "
            "Run `python scripts/06_tirzepatide_simulation_cohort.py` to build "
            "`tirzepatide_simulation_cohort_100.tsv` (large inputs; first run can take tens of minutes)."
        )
        return

    _callout_md(
        "<p><strong>What this block is.</strong> A quality-control view of the real cohort file the simulation draws "
        "from: who is in the sample by specialty, geography bucket, and segment tags.</p>"
        "<p><strong>Why it matters for the business story.</strong> It grounds the tirzepatide / GLP-1 readouts in a "
        "specific slice of prescribers so stakeholders do not mistake a purposive cohort for “all U.S. doctors.”</p>"
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

    _muted_md(
        "<strong>How to read the three numbers.</strong><br/>"
        "1) Cohort size.<br/>"
        "2) Early adoption signal: share with any tirzepatide claims visible in Medicare Part D for 2023 "
        "(the administrative window we can observe).<br/>"
        "3) Data richness: typical count of Part D detail rows per doctor in 2023 (more rows usually means stabler "
        "segment breakdowns)."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("##### Specialty mix")
        if "specialty" in cohort_df.columns:
            s_counts = cohort_df["specialty"].value_counts().reset_index()
            s_counts.columns = ["Specialty", "Doctors"]
            _muted_md(
                "<strong>What this shows.</strong> Counts of doctors by board-style specialty.<br/>"
                "<strong>Why it matters.</strong> Shows whether the narrative leans primary care (for example, "
                "internal or family medicine) versus endocrinology-heavy."
            )
            st.dataframe(s_counts, hide_index=True, use_container_width=True)
    with c2:
        st.markdown("##### Priority metros")
        if "geo_cluster" in cohort_df.columns:
            vc = cohort_df["geo_cluster"].value_counts().reset_index()
            vc.columns = ["_raw", "Doctors"]
            vc["City / region (from data bucket)"] = vc["_raw"].map(_pretty_geo_cluster)
            vc = vc[["City / region (from data bucket)", "Doctors"]]
            _muted_md(
                "<strong>What this shows.</strong> Planning buckets used to balance geography (for example, "
                "<em>Los Angeles, CA</em>), not street-level locations.<br/>"
                "<strong>Why it matters.</strong> Confirms which markets drive the sample mix."
            )
            st.dataframe(vc, hide_index=True, use_container_width=True)
    with c3:
        st.markdown("##### Adoption style tags")
        if "adoption_archetype" in cohort_df.columns:
            vc = cohort_df["adoption_archetype"].value_counts().reset_index()
            vc.columns = ["_raw", "Doctors"]
            vc["Adoption style (model tag)"] = vc["_raw"].map(_pretty_archetype)
            vc = vc[["Adoption style (model tag)", "Doctors"]]
            _muted_md(
                "<strong>What this shows.</strong> Internal segment labels derived from 2022 prescribing patterns.<br/>"
                "<strong>Why it matters.</strong> Lets you compare simulated June-2022-forward attitudes with "
                "<strong>revealed</strong> 2023 prescribing outcomes within each segment."
            )
            st.dataframe(vc, hide_index=True, use_container_width=True)

    with st.expander("Data sources and limitations"):
        st.markdown(
            """
            - **Medicare Part D** is Medicare’s outpatient prescription drug benefit; annual claims files define what
              prescribing we can see here. Non-Medicare channels are out of scope.

            - **Simulated survey rows** shown in this app default to the packaged `artifacts/demo/` bundle unless you
              re-run the batch with an API key.

            - **Full methodology narrative:** `docs/target_report.md`.
            """
        )


def _render_demo_bundle_banner(summary: dict) -> None:
    """Offline seed vs placeholder notices and developer commands."""
    if summary.get("offline_seed"):
        st.info(
            "You are viewing the **packaged offline demo**: responses were generated with a **fixed deterministic "
            "seed** so the story is repeatable. **No live large language model (LLM) API calls** are made for this "
            "bundle."
        )
        with st.expander("Developers: regenerate with a live model or refresh the demo bundle"):
            st.markdown(
                "Run the batch module with a provider API key, then rebuild metrics. Example command (copy into "
                "your terminal from the project root):"
            )
            st.code("python -m simulation.run_batch --limit-npis 10 --save-as-demo-bundle", language="bash")
            st.markdown(
                "- Set **`TOGETHER_API_KEY`** for the default Together provider, **or**  \n"
                "- Use **`--provider openai`** with **`OPENAI_API_KEY`** for OpenAI-compatible endpoints."
            )
    elif summary.get("is_placeholder"):
        st.info(
            "The demo bundle on disk is still a **placeholder** (no finalized survey output). Generate a real bundle "
            "with a model API key, or use the offline seed flag if you only need a runnable pipeline."
        )
        with st.expander("Developers: commands to build a real or offline demo bundle"):
            st.code("python -m simulation.run_batch --limit-npis 10 --save-as-demo-bundle", language="bash")
            st.markdown(
                "Use **`TOGETHER_API_KEY`** with the default provider, **`--provider openai`** with "
                "**`OPENAI_API_KEY`**, or **`--offline-seed-demo`** for a deterministic run without an API."
            )


def _render_results_metrics_selector() -> dict:
    """Results header, metrics file picker, and eval bundle coverage table."""
    st.header("Results")
    metrics_options = _discover_metrics_files()
    if not metrics_options:
        st.warning("No `metrics.json` found. Run `make eval` or `python -m eval.run_eval`.")
        metrics: dict = {}
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
        rel = html.escape(str(metrics_path_used.relative_to(PROJECT_ROOT)))
        _muted_md(
            f"<strong>Loaded metrics file:</strong> <code>{rel}</code><br/>"
            "<strong>Why this control exists.</strong> Choose which saved evaluation snapshot you are presenting—for "
            "example, the checked-in demo versus a local smoke run."
        )

    _render_eval_coverage_sidebar(metrics)

    if metrics.get("error"):
        st.error(
            f"**This metrics file is incomplete:** `{metrics.get('error')}`. "
            "Run `python -m simulation.run_batch …` then `make eval` so charts below fill in."
        )

    return metrics


def _render_eval_coverage_sidebar(metrics: dict) -> None:
    """Confirm each eval bundle section is present (parity with eval.metrics.compute_metrics_bundle)."""
    with st.expander("Eval bundle coverage (what this file contains)"):
        _muted_md(
            "Each row mirrors a block emitted by <code>compute_metrics_bundle</code> in <code>eval/metrics.py</code>."
        )
        rows = [
            {
                "Block": "survey_marginals",
                "Present": bool(metrics.get("survey_marginals")),
                "Role": "Per-item simulated answer histograms (method_a)",
            },
            {
                "Block": "distribution_quality",
                "Present": bool(metrics.get("distribution_quality")),
                "Role": "JS/TV: simulated vs hold-out pseudo marginals",
            },
            {"Block": "persona_coherence", "Present": bool(metrics.get("persona_coherence")), "Role": "Cross-item rule violations"},
            {"Block": "instrument_health", "Present": bool(metrics.get("instrument_health")), "Role": "Parse/coverage/latency QA"},
            {
                "Block": "behavioral_alignment",
                "Present": bool(metrics.get("behavioral_alignment")),
                "Role": "Per-NPI match vs hold-out pseudo-labels (needs cohort at eval time)",
            },
            {"Block": "run_manifest", "Present": bool(metrics.get("run_manifest")), "Role": "Pinned batch settings (if saved)"},
            {"Block": "eval_meta", "Present": bool(metrics.get("eval_meta")), "Role": "Paths + options used when metrics were built"},
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def _render_executive_snapshot(summary: dict, metrics: dict) -> None:
    st.subheader("Executive snapshot")
    _callout_md(
        "<p><strong>One-line summary.</strong> Are the simulated forward-looking answers compatible "
        "with what these physicians actually did in their 2023 prescribing data?</p>"
        "<p>Jensen–Shannon divergence measures distributional shape match (lower = better). "
        "Exact match rate compares individual physician predictions to claims-derived pseudo-labels. "
        "Coherence checks flag contradictions within a single simulated physician's response set.</p>"
    )
    dq = metrics.get("distribution_quality") if metrics else {}
    ba = metrics.get("behavioral_alignment") if metrics else {}
    pc = metrics.get("persona_coherence") if metrics else {}
    ih = metrics.get("instrument_health") if metrics else {}

    mjs = dq.get("mean_js_sim_vs_holdout") if isinstance(dq, dict) else None
    m_acc = ba.get("mean_accuracy_over_labeled_questions") if isinstance(ba, dict) else None
    vr = pc.get("violation_rate_per_method_block") if isinstance(pc, dict) else None
    miss_rate = ih.get("missing_answer_cell_rate") if isinstance(ih, dict) else None
    miss_cells = ih.get("v2_missing_question_cells") if isinstance(ih, dict) else None
    miss_expected = ih.get("v2_expected_answer_cells") if isinstance(ih, dict) else None
    if (
        miss_rate is None
        and isinstance(ih, dict)
        and miss_cells is not None
        and ih.get("n_v2_rows") is not None
        and summary.get("n_questions")
    ):
        exp = int(ih["n_v2_rows"]) * int(summary["n_questions"])
        if exp > 0:
            miss_rate = float(miss_cells) / float(exp)
            if miss_expected is None:
                miss_expected = exp

    e1, e2, e3, e4, e5 = st.columns(5)
    e1.metric("Doctors in demo summary", summary.get("n_npis", "—"))
    e2.metric(
        "Survey items in battery",
        summary.get("n_questions", "—"),
        help="Count of forward-looking survey items in the battery for this run.",
    )
    e3.metric(
        "Mean Jensen–Shannon vs hold-out pseudo",
        f"{mjs:.4f}" if mjs is not None else "—",
        help="Mean Jensen–Shannon divergence between simulated distributions and hold-out pseudo marginals.",
    )
    e4.metric(
        "Mean exact match vs claims-based hints",
        f"{m_acc:.3f}" if m_acc is not None else "—",
        help="Average per-item exact agreement between simulated answers and pseudo-labels from later Part D fields.",
    )
    e5.metric(
        "Coherence violation rate",
        f"{vr:.3f}" if vr is not None else "—",
        help="Cross-item rule violations per simulated physician block (see Persona coherence section).",
    )
    if miss_rate is not None:
        raw = ""
        if miss_cells is not None and miss_expected is not None:
            raw = f" Raw counts: {miss_cells} / {miss_expected} cells."
        st.metric(
            "Missing answers (instrument health)",
            f"{float(miss_rate):.1%}",
            help=(
                "Share of expected v2 survey cells (doctors × survey items in the battery) without a parsed "
                "option_id."
                + raw
                + " Includes failed joint calls where `method_a` is empty (truncation, parse errors, or empty model "
                "content)."
            ),
        )


def _render_run_provenance_expanders(metrics: dict) -> None:
    rm = metrics.get("run_manifest")
    if isinstance(rm, dict) and rm:
        with st.expander("Run configuration (pinned batch settings)"):
            _muted_md(
                "<strong>What this is.</strong> Exact command-line, model, and cohort choices used to produce the "
                "bundle.<br/><strong>Why it matters.</strong> Attach to screenshots for reproducible internal review."
            )
            st.json(rm)

    em = metrics.get("eval_meta")
    if isinstance(em, dict) and em:
        with st.expander("Eval build metadata"):
            _muted_md(
                "<strong>What this is.</strong> Paths and options used when <code>python -m eval.run_eval</code> "
                "built this metrics file.<br/><strong>Why it matters.</strong> Confirms which response file and "
                "cohort snapshot the numbers refer to."
            )
            st.json(em)


def _render_revealed_adoption_chart(summary: dict, cohort_df: pd.DataFrame | None) -> None:
    st.markdown("---")
    actual = summary.get("adoption_by_archetype_actual") or _cohort_adoption_by_archetype(cohort_df)
    if actual:
        st.subheader("Revealed adoption by segment tag (Medicare Part D 2023)")
        _callout_md(
            "<p><strong>What this shows.</strong> Among the 100 cohort physicians, the share who actually "
            "prescribed tirzepatide in Medicare Part D 2023, broken out by behavioral segment tags derived "
            "from their 2022 prescribing patterns.</p>"
            "<p><strong>Why it matters.</strong> This is the empirical reality the simulation is trying to "
            "approximate. If the simulated adoption distribution matches these observed rates, the personas "
            "are capturing real variation in physician behavior — not just echoing population averages.</p>"
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
            height=440,
            margin=dict(l=60, r=40, t=70, b=120),
            xaxis=dict(tickangle=-28, automargin=True),
            yaxis=dict(automargin=True, range=[0, 1.05]),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No cohort or summary data for adoption-by-archetype chart.")


def _render_simulated_distributions_from_summary(summary: dict) -> None:
    st.subheader("Simulated survey (answer counts)")
    _callout_md(
        "<p><strong>What this shows.</strong> How the simulated physicians answered each survey question. "
        "Compare these distributions to the revealed adoption chart above — alignment suggests the "
        "personas are behaviorally calibrated.</p>"
    )
    mc = summary.get("simulated_distributions") or summary.get("method_comparison") or {}
    if not mc:
        st.info("No saved distributions in `summary.json` yet—run a batch with `--save-as-demo-bundle`.")
        return
    for qid, dists in mc.items():
        labels = _option_labels_for_question(qid)
        counts = _distribution_from_summary_entry(dists if isinstance(dists, dict) else {})
        with st.expander(_expander_label(qid)):
            q_obj = _question_by_id(qid)
            if q_obj:
                st.markdown(f"**Full survey wording.** {q_obj.text.strip()}")
            st.markdown(
                f"**Internal identifier (for logs):** `{html.escape(qid)}` — use the human-readable labels in the "
                "table below when presenting, not raw option codes."
            )
            st.dataframe(_dist_counts_to_df(counts, labels), hide_index=True, use_container_width=True)


def _render_distribution_quality_block(metrics: dict) -> None:
    dq = metrics.get("distribution_quality")
    if not isinstance(dq, dict):
        return
    st.subheader("Hold-out distribution match (simulated vs Medicare-derived pseudo marginals)")
    _callout_md(
        "<p><strong>What this measures.</strong> How closely the shape of simulated answer distributions "
        "matches distributions implied by actual 2023 Medicare Part D claims for the same physicians.</p>"
        "<p><strong>Jensen–Shannon divergence:</strong> 0 = identical distributions. Values below 0.05 "
        "indicate strong shape alignment. <strong>Total variation distance:</strong> similar "
        "interpretation, different scale.</p>"
    )
    c1, c2 = st.columns(2)
    mjs = dq.get("mean_js_sim_vs_holdout")
    mtv = dq.get("mean_tv_sim_vs_holdout")
    c1.metric(
        "Mean Jensen–Shannon vs hold-out pseudo",
        f"{mjs:.4f}" if mjs is not None else "—",
        help="Jensen–Shannon divergence versus hold-out pseudo marginals (0 = identical shape).",
    )
    c2.metric(
        "Mean total variation vs hold-out pseudo",
        f"{mtv:.4f}" if mtv is not None else "—",
        help="Total variation distance versus hold-out pseudo marginals (0 = identical).",
    )


def _render_behavioral_alignment_block(metrics: dict) -> None:
    st.subheader("Hold-out alignment (June 2022 forward vs later Part D)")
    ba = metrics.get("behavioral_alignment")
    if not isinstance(ba, dict) or not ba.get("per_question"):
        st.info(
            "This metrics file has no behavioral-alignment block. That usually means the cohort tab-separated file "
            "(`tirzepatide_simulation_cohort_100.tsv`) was missing when evaluation ran, or the bundle is a placeholder."
        )
        return
    _callout_md(
        "<p><strong>What this measures.</strong> For each survey question, did the simulated physician's "
        "answer match what they actually did in 2023? This is the strongest test: individual-level "
        "prediction accuracy, not just distributional match.</p>"
        "<p>Pseudo-labels are derived from Part D 2023 claims using deterministic rules (e.g., "
        "'physician prescribed tirzepatide in 2023 → ground truth answer is Yes'). The model never "
        "saw these labels.</p>"
    )
    m_acc = ba.get("mean_accuracy_over_labeled_questions")
    st.metric(
        "Average match rate across labeled survey items",
        f"{m_acc:.3f}" if m_acc is not None else "—",
        help="Share of simulated answers that exactly match pseudo-labels built from later claims fields.",
    )
    rows = []
    for qid, v in ba["per_question"].items():
        rows.append(
            {
                "Survey item": _question_short_title(qid),
                "Labeled doctor-answers": v.get("n_labeled"),
                "Match rate": None if v.get("accuracy") is None else round(float(v["accuracy"]), 4),
                "Jensen–Shannon shape gap vs claims hints": None
                if v.get("js_divergence_marginal") is None
                else round(float(v["js_divergence_marginal"]), 4),
                "Total variation vs claims hints": None
                if v.get("tv_distance_marginal") is None
                else round(float(v["tv_distance_marginal"]), 4),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    with st.expander("Technical: predicted vs pseudo-label counts (per item)"):
        st.caption("For debugging segment skew and rule behavior—not intended as an executive readout.")
        for qid, v in ba["per_question"].items():
            st.markdown(f"**{_question_short_title(qid)}** (`{qid}`)")
            c1, c2 = st.columns(2)
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
    _callout_md(
        "<p><strong>What this checks.</strong> Do a single physician's answers contradict each other? "
        "For example, if a persona says they have not adopted tirzepatide but also says their GLP-1 "
        "prescribing increased significantly, that is a coherence violation.</p>"
        "<p>Low violation rates indicate the personas maintain internally consistent clinical "
        "narratives — a prerequisite for the qualitative outputs to be trustworthy.</p>"
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
    st.subheader("Run and instrument health")
    _callout_md(
        "<p><strong>What this checks.</strong> Did the simulation pipeline run cleanly? Parse failures, "
        "missing answers, and API errors are tallied here. Interpret all results above in light of "
        "these numbers — high error rates invalidate the metrics.</p>"
    )
    lat = ih.get("latency_ms") or {}
    summary_rows = [
        {"Metric": "Survey rows in file", "Value": ih.get("n_jsonl_rows")},
        {"Metric": "Schema v2 rows (one row per doctor)", "Value": ih.get("n_v2_rows")},
        {"Metric": "Legacy flat rows", "Value": ih.get("n_legacy_rows")},
        {"Metric": "Flattened answer cells", "Value": ih.get("n_flat_cells")},
        {"Metric": "Cells with errors", "Value": ih.get("flat_cells_with_error")},
        {
            "Metric": "Missing answer share (v2 expected cells)",
            "Value": None
            if ih.get("missing_answer_cell_rate") is None
            else f"{float(ih['missing_answer_cell_rate']):.1%}",
        },
        {"Metric": "Cells missing a chosen option (count)", "Value": ih.get("flat_cells_missing_option")},
        {"Metric": "Schema v2 missing per-question cells", "Value": ih.get("v2_missing_question_cells")},
        {"Metric": "Schema v2 expected answer cells", "Value": ih.get("v2_expected_answer_cells")},
        {"Metric": "Schema v2 survey-level API or parse errors", "Value": ih.get("v2_survey_level_errors")},
        {"Metric": "Large language model (LLM) calls with latency logged", "Value": lat.get("n_calls_with_latency")},
        {"Metric": "Mean latency (ms)", "Value": None if lat.get("mean") is None else round(float(lat["mean"]), 1)},
        {"Metric": "Median latency (ms)", "Value": lat.get("p50")},
        {"Metric": "Max latency (ms)", "Value": lat.get("max")},
        {"Metric": "Question-to-claims map file (YAML)", "Value": ih.get("claims_map_file")},
    ]
    st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)
    sn = ih.get("schema_notes")
    if sn:
        with st.expander("Schema notes"):
            st.write(sn)


def _render_reasoning_examples_section() -> None:
    st.header("Reasoning examples")
    _callout_md(
        "<p><strong>Why this section matters most.</strong> Distribution accuracy proves the personas "
        "are calibrated. But the commercial value is here: qualitative reasoning that a pharma "
        "medical affairs team can actually use. Each excerpt shows how a specific physician — with "
        "their specific prescribing history and practice context — reasons through a clinical "
        "decision. No survey or regression produces this.</p>"
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


def _render_advanced_live_rerun(
    smoke_provider: str,
    smoke_model: str,
    smoke_temp: float,
    smoke_base_url: str,
) -> None:
    st.header("Advanced")
    live = st.checkbox("Re-run with live API (session only; key not saved)")
    if live:
        key_label = "TOGETHER_API_KEY" if smoke_provider == "together" else "OPENAI_API_KEY"
        api_key = st.text_input(key_label, type="password", help="Used only in this browser session.")
        if st.button("Smoke re-run (5 NPIs, all questions, production persona)"):
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
                    "production",
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


def _render_platform_positioning() -> None:
    st.header("Beyond physicians: where behavioral persona simulation creates value")
    st.markdown(
        """
        This demo simulates physicians reacting to a drug launch. The methodology — build personas
        from revealed behavioral data, validate against held-out outcomes, generate qualitative
        reasoning — applies wherever three conditions hold:

        1. **The target audience is a small, identifiable group of professionals** whose collective
           reaction to a stimulus carries significant economic consequences.
        2. **Rich revealed-preference data is publicly or commercially available** to ground
           each persona in observable behavior, not demographic stereotypes.
        3. **The decision-maker cannot easily survey the audience directly** — because the group
           is too small, too expensive to reach, the question is too sensitive, or the scenario
           is hypothetical.
        """
    )

    st.markdown("##### Candidate markets for behavioral persona simulation")

    market_data = [
        {
            "Market": "Pharma launch strategy",
            "Target audience": "Physician KOLs (100–500)",
            "Stimulus tested": "Drug launch narrative, safety signal, formulary change",
            "Persona data sources": "Medicare Part D, Open Payments, PubMed",
            "Current alternative": "Advisory boards ($15–25K/physician/day), physician surveys ($200–500K, 6–8 weeks)",
            "What simulation adds": "Unlimited hypothetical testing, zero leakage risk, grounded in actual prescribing",
        },
        {
            "Market": "Litigation strategy",
            "Target audience": "Jury pool (12–50 per venue)",
            "Stimulus tested": "Trial narrative, witness framing, damages presentation",
            "Persona data sources": "Voter rolls, census demographics, social media (where public)",
            "Current alternative": "Mock jury consultants ($50–200K per engagement, 1–2 runs)",
            "What simulation adds": "Unlimited narrative iterations, precise venue-demographic matching",
        },
        {
            "Market": "Regulatory strategy",
            "Target audience": "FDA Advisory Committee members (15–20)",
            "Stimulus tested": "Clinical data presentation, risk-benefit framing",
            "Persona data sources": "PubMed publications, prior AdCom votes (FDA transcripts), institutional affiliations",
            "Current alternative": "Regulatory consultant intuition, prior vote pattern analysis",
            "What simulation adds": "Named-persona deliberation simulation with documented reasoning",
        },
        {
            "Market": "Activist defense / investor relations",
            "Target audience": "Institutional shareholders (50–200)",
            "Stimulus tested": "Proxy statement, restructuring narrative, ESG messaging",
            "Persona data sources": "13F filings (holdings), proxy vote records, earnings call transcripts",
            "Current alternative": "Expert networks ($1K/hr), IR advisory firms ($500K–2M per engagement)",
            "What simulation adds": "Predicted vote distribution with individual-level reasoning",
        },
        {
            "Market": "Judicial analytics",
            "Target audience": "Federal appellate judges (3-judge panels)",
            "Stimulus tested": "Legal arguments, case framing, motion strategy",
            "Persona data sources": "CourtListener (opinions, affiliations, financial disclosures), SCDB voting records",
            "Current alternative": "Pre/Dicta (85% accuracy on motion outcomes, no qualitative reasoning)",
            "What simulation adds": "Simulated judicial reasoning in the judge's own voice; panel deliberation dynamics",
        },
    ]

    st.dataframe(
        pd.DataFrame(market_data),
        hide_index=True,
        use_container_width=True,
    )

    st.markdown(
        """
        **The common thread.** In each market, the existing alternative either gives you a number
        without reasoning (Pre/Dicta's 85% probability, a survey's 64% agree) or gives you
        qualitative depth without behavioral grounding (an expert network call, a mock jury's
        gut reactions). Behavioral persona simulation combines both: distributional predictions
        validated against observed outcomes, plus qualitative reasoning grounded in each
        individual's professional record.

        **What makes this an Artificial Societies problem.** The pharma demo here uses
        individual personas reacting independently. The full value emerges when personas interact:
        a KOL's reaction propagates through their co-authorship network; a panel of three judges
        deliberates toward a joint opinion; institutional shareholders observe each other's proxy
        votes. These are the social influence dynamics that Artificial Societies' platform is
        built to model — applied to audiences where behavioral grounding makes the simulation
        trustworthy enough for high-stakes decisions.
        """
    )

    st.markdown('<a id="improvement-roadmap"></a>', unsafe_allow_html=True)
    st.subheader("Improvement roadmap")
    st.markdown(
        """
        Current simulation quality reflects a proof-of-concept built on a single year of
        prescribing data, a general-purpose LLM, and no inter-persona dynamics. Four concrete
        improvements target the main sources of error:
        """
    )

    roadmap_data = [
        {
            "Improvement": "Multi-year prescribing history",
            "Status": "Data pulled (2013–2021 Part D for all 100 NPIs)",
            "Expected impact": "Personas currently see one year of prescribing. A decade of "
            "history reveals trajectory — early adopter vs. conservative, growing vs. shrinking "
            "panel, specialty evolution. This is the single richest enrichment available and "
            "directly parallels how Artificial Societies builds personas from longitudinal "
            "social media behavior.",
            "What changes in the prompt": "Prescribing trend summary (e.g., 'your GLP-1 share "
            "grew from 5% in 2015 to 29% in 2022') replaces the current single-year snapshot.",
        },
        {
            "Improvement": "Hybrid ML + LLM pipeline",
            "Status": "Planned",
            "Expected impact": "A classifier (logistic regression or random forest) trained on "
            "structured features predicts the behavioral outcome — the thing classifiers are "
            "good at. The LLM then generates a rationale conditioned on that prediction — the "
            "thing LLMs are good at. The classifier handles distributional accuracy; the LLM "
            "handles qualitative depth and follow-up interviewing on out-of-distribution or "
            "speculative scenarios that no classifier can answer.",
            "What changes in the prompt": "The LLM prompt includes 'a statistical model predicts "
            "you would answer X with Y% confidence' as a calibration anchor, then generates "
            "reasoning consistent with that prediction.",
        },
        {
            "Improvement": "Social network dynamics",
            "Status": "Planned",
            "Expected impact": "Physicians in the same health system, same specialty, or same "
            "geography influence each other's practice patterns. Edges can be drawn from shared "
            "organizational affiliation (NPI registry), co-authorship (PubMed API), geographic "
            "proximity, and shared industry relationships (Open Payments). A multi-round "
            "simulation — where each persona sees peer responses before answering — would capture "
            "institutional cascade effects (e.g., a department chair's early adoption accelerating "
            "colleagues' uptake).",
            "What changes in the prompt": "Round 2 adds: 'colleagues in your practice have "
            "responded as follows: [peer responses]. Given your professional context and your "
            "peers' positions, how do you respond?'",
        },
        {
            "Improvement": "Frontier models",
            "Status": "Planned",
            "Expected impact": "The current version of the demo uses a top-tier open-source LLM(GLM 5.1) "
            "but we might achieve better results with a frontier model like Claude Opus 4.6 or GPT 5.4 Pro."
            "This hypothesis is still to be tested.",
        },
    ]

    st.dataframe(
        pd.DataFrame(roadmap_data),
        hide_index=True,
        use_container_width=True,
    )

    st.markdown(
        """
        **The key hypothesis these improvements test:** persona fidelity scales with the
        richness of the behavioral data grounding each persona. If multi-year prescribing
        history and a hybrid ML anchor materially improve distributional accuracy over the
        current single-year LLM-only baseline, that validates the core Artificial Societies
        thesis — that grounding personas in real human behavior, not demographic templates,
        is what makes simulation trustworthy for high-stakes decisions.
        """
    )


def _render_footer() -> None:
    st.divider()
    repo_hint = os.environ.get("DEMO_REPO_URL", "").strip()
    repo_line = (
        f"Repository: {repo_hint}"
        if repo_hint
        else "Repository: set `DEMO_REPO_URL` in the environment to show a link here (optional)."
    )
    _muted_md(
        "<strong>Methodology overview.</strong> Personas are built from CMS Medicare Part D (prescribing) "
        "and Open Payments (industry financial relationships) for calendar year 2022. Survey questions "
        "elicit forward-looking clinical judgments. Evaluation compares simulated responses to pseudo-labels "
        "derived from Part D 2023 claims — data the model never sees. Full methodology: "
        "<code>docs/target_report.md</code>."
    )
    _muted_md(
        f"{html.escape(repo_line)}<br/><br/>"
        "<strong>Local setup (developers).</strong> "
        "<code>pip install -r requirements.txt</code>, then <code>streamlit run streamlit_app.py</code> or "
        "<code>make demo</code>. "
        "Environment variables: <code>TOGETHER_API_KEY</code> for the default Together SDK, or OpenAI-compatible "
        "credentials with <code>--provider openai</code> (see <code>.env.example</code>).<br/><br/>"
        "<strong>Limitations.</strong> Medicare Part D only; annual files; purposive cohort—not a national "
        "probability sample."
    )


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------


def main() -> None:
    load_local_dotenv(override=False)
    st.set_page_config(page_title="Tirzepatide Adoption Simulation", layout="wide")
    _inject_page_styles()

    smoke_provider, smoke_model, smoke_temp, smoke_base_url = _render_sidebar_smoke_settings()

    summary = _load_json(SUMMARY_PATH)
    cohort_df = _read_cohort_tsv()

    _render_title_block()
    _render_about_section()
    _render_sample_description(cohort_df)
    _render_demo_bundle_banner(summary)

    metrics = _render_results_metrics_selector()
    _render_executive_snapshot(summary, metrics)
    _render_run_provenance_expanders(metrics)
    _render_revealed_adoption_chart(summary, cohort_df)
    _render_simulated_distributions_from_summary(summary)
    _render_distribution_quality_block(metrics)
    _render_behavioral_alignment_block(metrics)
    _render_persona_coherence_block(metrics)
    _render_instrument_health_block(metrics)
    _render_reasoning_examples_section()
    _render_platform_positioning()
    _render_advanced_live_rerun(smoke_provider, smoke_model, smoke_temp, smoke_base_url)
    _render_footer()


if __name__ == "__main__":
    main()
