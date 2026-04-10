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


def _cohort_adoption_by_archetype() -> dict:
    if not COHORT_PATH.is_file():
        return {}
    df = pd.read_csv(COHORT_PATH, sep="\t", low_memory=False)
    if "adoption_archetype" not in df.columns or "has_tirzepatide_2023" not in df.columns:
        return {}
    g = df.groupby("adoption_archetype")["has_tirzepatide_2023"].agg(["mean", "count"])
    out = {}
    for idx, row in g.iterrows():
        out[str(idx)] = {"rate": float(row["mean"]), "n": int(row["count"])}
    return out


def main() -> None:
    st.set_page_config(page_title="Tirzepatide Adoption Simulation", layout="wide")
    summary = _load_json(SUMMARY_PATH)
    metrics = _load_json(METRICS_PATH)

    st.title("Tirzepatide adoption simulation")
    st.caption("Novo's 6-week decision problem (June 2022) — Medicare Part D–scoped physician POC")

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

    actual = summary.get("adoption_by_archetype_actual") or _cohort_adoption_by_archetype()
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
