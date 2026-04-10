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

PROJECT_ROOT = Path(__file__).resolve().parent
SUMMARY_PATH = PROJECT_ROOT / "artifacts" / "demo" / "summary.json"
SAMPLE_JSONL = PROJECT_ROOT / "artifacts" / "demo" / "sample_responses.jsonl"
METRICS_PATH = PROJECT_ROOT / "artifacts" / "demo" / "metrics.json"
COHORT_PATH = PROJECT_ROOT / "data" / "output" / "tirzepatide_simulation_cohort_100.tsv"


def _load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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
    st.set_page_config(page_title="Tirzepatide Adoption Simulation", layout="wide")
    summary = _load_json(SUMMARY_PATH)
    metrics = _load_json(METRICS_PATH)

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

    if summary.get("is_placeholder"):
        st.info(
            "Demo bundle is a **placeholder**. Run `python -m simulation.run_batch "
            "--limit-npis 10 --save-as-demo-bundle` with `OPENAI_API_KEY` set to populate "
            "`artifacts/demo/summary.json`."
        )

    st.header("Results")
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
        api_key = st.text_input("OPENAI_API_KEY", type="password", help="Used only in this browser session.")
        if st.button("Smoke re-run (5 NPIs, all questions, both methods)"):
            if not api_key:
                st.error("Enter an API key.")
            else:
                env = {**os.environ, "OPENAI_API_KEY": api_key}
                cmd = [
                    sys.executable,
                    "-m",
                    "simulation.run_batch",
                    "--limit-npis",
                    "5",
                    "--output-dir",
                    str(PROJECT_ROOT / "data" / "output" / "runs" / "streamlit_smoke"),
                ]
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
    st.caption(
        "GitHub: link your repo here. Setup: `pip install -r requirements.txt` then "
        "`streamlit run streamlit_app.py`. **Limitations:** Medicare Part D only; annual files; "
        "purposive cohort—not a national probability sample."
    )


if __name__ == "__main__":
    main()
