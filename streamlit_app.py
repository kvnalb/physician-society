"""Tirzepatide adoption simulation — offline-first demo (Streamlit)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from simulation.env_bootstrap import load_local_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
SUMMARY_PATH = PROJECT_ROOT / "artifacts" / "demo" / "summary.json"
SAMPLE_JSONL = PROJECT_ROOT / "artifacts" / "demo" / "sample_responses.jsonl"
DEFAULT_METRICS_PATH = PROJECT_ROOT / "artifacts" / "demo" / "metrics.json"
RUNS_ROOT = PROJECT_ROOT / "data" / "output" / "runs"
ARTIFACT_RUNS_DIR = PROJECT_ROOT / "artifacts" / "runs"
COHORT_PATH = PROJECT_ROOT / "data" / "output" / "tirzepatide_simulation_cohort_100.tsv"
DEFAULT_TOGETHER_MODEL = "zai-org/GLM-5.1"


def _load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _discover_metrics_files() -> list[tuple[str, Path]]:
    """(label, path) for eval metrics JSON; deduped by resolved path."""
    found: list[tuple[str, Path]] = []
    if DEFAULT_METRICS_PATH.is_file():
        found.append(("artifacts/demo/metrics.json (default demo)", DEFAULT_METRICS_PATH))
    if RUNS_ROOT.is_dir():
        for d in sorted(RUNS_ROOT.iterdir()):
            if not d.is_dir():
                continue
            mp = d / "metrics.json"
            if mp.is_file():
                found.append((f"data/output/runs/{d.name}/metrics.json", mp))
    if ARTIFACT_RUNS_DIR.is_dir():
        for mp in sorted(ARTIFACT_RUNS_DIR.glob("*metrics*.json")):
            if mp.is_file():
                found.append((f"artifacts/runs/{mp.name}", mp))
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
        - **Engagement:** **Open Payments (2022)** is merged for pharma exposure tiers; see cohort columns for details.

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

    n = len(cohort_df)
    st.subheader("Loaded cohort snapshot")
    m1, m2, m3 = st.columns(3)
    m1.metric("Physicians in TSV", n)
    if "has_tirzepatide_2023" in cohort_df.columns:
        rate = float(cohort_df["has_tirzepatide_2023"].mean())
        m2.metric("Any tirzepatide claims (Part D 2023)", f"{rate:.0%}")
    else:
        m2.metric("Any tirzepatide claims (Part D 2023)", "—")
    if "part_d_rows_2023" in cohort_df.columns:
        m3.metric("Median Part D rows / NPI (2023)", f"{cohort_df['part_d_rows_2023'].median():.0f}")
    else:
        m3.metric("Median Part D rows / NPI (2023)", "—")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Specialty**")
        if "specialty" in cohort_df.columns:
            s_counts = cohort_df["specialty"].value_counts().reset_index()
            s_counts.columns = ["specialty", "n"]
            st.dataframe(s_counts, hide_index=True, use_container_width=True)
    with c2:
        st.markdown("**Metro cluster**")
        if "geo_cluster" in cohort_df.columns:
            vc = cohort_df["geo_cluster"].value_counts().reset_index()
            vc.columns = ["geo_cluster", "n"]
            st.dataframe(vc, hide_index=True, use_container_width=True)
    with c3:
        st.markdown("**Adoption archetype**")
        if "adoption_archetype" in cohort_df.columns:
            vc = cohort_df["adoption_archetype"].value_counts().reset_index()
            vc.columns = ["adoption_archetype", "n"]
            st.dataframe(vc, hide_index=True, use_container_width=True)

    with st.expander("Data sources & limitations"):
        st.markdown(
            """
            - **Medicare Part D** annual files define prescribing visibility; non-Medicare channels are out of scope.
            - **LLM survey** rows in this app come from `artifacts/demo/` unless you re-run the batch with an API key.
            - Full narrative: `docs/target_report.md`.
            """
        )


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

        **Methods.** **Method A** = LLM persona with **rich** administrative + prescribing context;
        **Method B** = **minimal** context. **Eval** = Cohen's κ on paired A/B answers plus **empirical**
        adoption baselines from claims (not causal).
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
        metrics = {}
        metrics_path_used: Path | None = None
    else:
        labels = [x[0] for x in metrics_options]
        default_ix = 0
        for i, (_, p) in enumerate(metrics_options):
            if p.resolve() == DEFAULT_METRICS_PATH.resolve():
                default_ix = i
                break
        pick = st.selectbox(
            "Eval metrics snapshot",
            range(len(labels)),
            format_func=lambda i: labels[i],
            index=default_ix,
            help="Each batch run can write `data/output/runs/<run_id>/metrics.json` next to `responses__<model>.jsonl`. "
            "`make eval` refreshes `artifacts/demo/metrics.json`.",
        )
        _, metrics_path_used = metrics_options[pick]
        metrics = _load_json(metrics_path_used)
        st.caption(f"Loaded: `{metrics_path_used.relative_to(PROJECT_ROOT)}`")

    ba = metrics.get("behavioral_alignment") if metrics else None

    rm = metrics.get("run_manifest")
    if isinstance(rm, dict) and rm:
        with st.expander("Run configuration (run_manifest.json)"):
            st.json(rm)

    ba = metrics.get("behavioral_alignment")
    if isinstance(ba, dict) and ba.get("per_question"):
        m_acc = ba.get("mean_accuracy_over_labeled_questions")
        st.metric(
            "Behavioral alignment (mean acc. vs pseudo-labels)",
            f"{m_acc:.3f}" if m_acc is not None else "—",
            help="Claims-derived pseudo-labels; see eval/behavioral_labels.py.",
        )

    dq = metrics.get("distribution_quality") if metrics else None
    if isinstance(dq, dict):
        mjs = dq.get("mean_js_method_ab")
        st.metric(
            "Distribution: mean JS (A vs B marginals)",
            f"{mjs:.4f}" if mjs is not None else "—",
            help="Per-question Jensen–Shannon between method_a and method_b histograms (not human panel).",
        )

    pc = metrics.get("persona_coherence") if metrics else None
    if isinstance(pc, dict) and pc.get("violation_rate_per_method_block") is not None:
        vr = pc["violation_rate_per_method_block"]
        st.metric(
            "Coherence: rule violation rate",
            f"{vr:.3f}" if vr is not None else "—",
            help="Cross-item consistency rules in eval/coherence_rules.py (heuristic).",
        )

    ih = metrics.get("instrument_health") if metrics else None
    if isinstance(ih, dict) and ih.get("flat_cells_missing_option") is not None:
        miss = ih.get("flat_cells_missing_option")
        st.metric("Run health: missing option cells", int(miss) if miss is not None else "—")

    col1, col2, col3 = st.columns(3)
    col1.metric("NPIs in summary", summary.get("n_npis", "—"))
    col2.metric("Survey questions", summary.get("n_questions", "—"))
    kappa = metrics.get("survey", {}).get("method_agreement_kappa_mean")
    col3.metric("Mean κ (A vs B)", f"{kappa:.3f}" if kappa is not None else "—")

    actual = summary.get("adoption_by_archetype_actual") or _cohort_adoption_by_archetype(cohort_df)
    if actual:
        archetypes = list(actual.keys())
        rates = [actual[a]["rate"] for a in archetypes]
        fig = go.Figure(
            data=[
                go.Bar(
                    x=archetypes,
                    y=rates,
                    marker_color="#2E5077",
                    name="Actual (Part D 2023)",
                )
            ]
        )
        fig.update_layout(
            title="Tirzepatide adoption rate by archetype (Medicare Part D 2023, cohort)",
            yaxis_title="Share with any tirzepatide claims",
            template="plotly_white",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No cohort or summary data for adoption-by-archetype chart.")

    st.subheader("Method comparison (distributions)")
    mc = summary.get("method_comparison") or {}
    if not mc:
        st.write("No method comparison yet (run batch with API key).")
    else:
        for qid, dists in mc.items():
            st.markdown(f"**{qid}**")
            st.json(dists)

    if metrics.get("survey", {}).get("stability"):
        st.markdown(f"**Stability:** {metrics['survey']['stability']}")

    if isinstance(ba, dict) and ba.get("per_question"):
        st.subheader("Behavioral alignment (per question)")
        st.dataframe(
            [
                {
                    "question_id": qid,
                    "n_labeled": v.get("n_labeled"),
                    "accuracy": v.get("accuracy"),
                    "js_divergence_marginal": v.get("js_divergence_marginal"),
                    "tv_distance_marginal": v.get("tv_distance_marginal"),
                }
                for qid, v in ba["per_question"].items()
            ],
            hide_index=True,
            use_container_width=True,
        )

    surv = metrics.get("survey", {}) if metrics else {}
    pq = surv.get("per_question") or {}
    if pq and any("js_method_ab_marginal" in v for v in pq.values()):
        st.subheader("Method A vs B — marginal divergence (per question)")
        st.dataframe(
            [
                {
                    "question_id": qid,
                    "n_paired": v.get("n_paired"),
                    "js_method_ab": v.get("js_method_ab_marginal"),
                    "tv_method_ab": v.get("tv_method_ab_marginal"),
                }
                for qid, v in pq.items()
            ],
            hide_index=True,
            use_container_width=True,
        )

    if isinstance(ih, dict) and ih:
        with st.expander("Instrument / run health"):
            st.json(ih)

    if isinstance(pc, dict) and pc.get("violations_sample"):
        with st.expander("Persona coherence — sample violations"):
            st.dataframe(pc["violations_sample"], hide_index=True, use_container_width=True)

    st.header("Reasoning examples")
    lines = []
    if SAMPLE_JSONL.is_file():
        with open(SAMPLE_JSONL, encoding="utf-8") as fh:
            lines = [ln.strip() for ln in fh if ln.strip()]
    if not lines:
        st.write("No `sample_responses.jsonl` lines yet.")
    else:
        for i, line in enumerate(lines[:5]):
            r = json.loads(line)
            with st.expander(f"NPI {r.get('npi')} — {r.get('question_id')} ({r.get('method')})"):
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
