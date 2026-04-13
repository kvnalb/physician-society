**Note: Old streamlit link has been deprecated: click on new link in the sidebar.**

# Physician Society

Research-oriented **physician persona simulation** for a Medicare Part D–scoped cohort: synthetic survey responses from a structured LLM persona (2022-only administrative context in the prompt), **forward-looking** items framed as **mid-2022** (including tirzepatide / Mounjaro launch context), and **evaluation** against hold-out-style signals derived from later Part D fields in the cohort file. A **Streamlit** app presents the story using a small checked-in demo bundle; the full methodology narrative lives in [`docs/target_report.md`](docs/target_report.md).

## What you get in this repository

| Area | Role |
|------|------|
| [`scripts/06_tirzepatide_simulation_cohort.py`](scripts/06_tirzepatide_simulation_cohort.py) | Build the purposive ~100-physician cohort TSV (Part D 2022–2023, six metros, selected specialties). Requires large CMS extracts under `data/raw/` (gitignored). |
| [`simulation/run_batch.py`](simulation/run_batch.py) | Run the multi-question survey over the cohort; writes versioned JSONL under `data/output/runs/<run-id>/`, cache, and `run_manifest.json`. |
| [`simulation/`](simulation/) | Persona construction, prompts, question YAML, response schema (v2: one row per NPI, `method_a` survey block). |
| [`eval/`](eval/) | Metrics bundle: survey marginals, behavioral alignment vs pseudo-labels, distribution distance, persona coherence, instrument health. See [`eval/README.md`](eval/README.md). |
| [`streamlit_app.py`](streamlit_app.py) | Offline-first demo UI: cohort snapshot, results, eval blocks, optional live smoke batch. |
| [`artifacts/demo/`](artifacts/demo/) | Checked-in **`summary.json`**, **`metrics.json`**, and **`sample_responses.jsonl`** so the UI runs without regenerating LLM output. |

## Requirements

- **Python 3.11+**
- **Dependencies:** `pip install -r requirements.txt` (or `make setup` to create `.venv` and install).
- **LLM batch runs:** Together (`TOGETHER_API_KEY`) or OpenAI-compatible client (`OPENAI_API_KEY`, optional `--base-url`). Not required if you only open the Streamlit demo that reads `artifacts/demo/`.
- **Cohort rebuild:** multi-gigabyte CMS CSVs and patience (chunked I/O; see project conventions in `.cursor/rules` if you use Cursor).

## Quick start (Streamlit only)

Use the packaged demo artifacts—no API keys, no cohort rebuild.

```bash
make setup          # or: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
make demo           # http://localhost:8501
```

Equivalent: `.venv/bin/python -m streamlit run streamlit_app.py` from the repo root.

## Full pipeline (cohort → batch → eval → UI)

Typical order for a clean machine that will run inference:

```bash
make setup
# Place CMS extracts under data/raw/ per cohort script expectations, then:
make run-tirzepatide-cohort    # writes data/output/tirzepatide_simulation_cohort_100.tsv (large; gitignored if regenerated)

# One multi-question call per NPI (default); writes e.g. data/output/runs/latest/responses__<model>.jsonl
python -m simulation.run_batch --save-as-demo-bundle

make eval    # metrics from latest run → artifacts/demo/metrics.json (+ run sidecar when applicable)
make demo
```

**Deterministic smoke (no API key):** `python -m simulation.run_batch --offline-seed-demo --limit-npis 10` produces synthetic rows for CI or UI plumbing; use `--write-demo-bundle` if you intentionally want to overwrite `artifacts/demo/` from that seed.

**Re-evaluate or refresh the demo bundle without new LLM calls:** after editing eval or parsing logic, point at an existing JSONL and rebuild `artifacts/demo/` (summary, sample JSONL, and metrics):

```bash
make refresh-demo
# same as: python -m simulation.refresh_demo_from_responses
```

**Metrics only (faster):** `make eval` runs [`eval/run_eval.py`](eval/run_eval.py), which resolves `responses.jsonl` vs `responses__<model>.jsonl` using `run_manifest.json` when present.

## Make targets

| Target | Purpose |
|--------|---------|
| `make setup` | Create `.venv` and install `requirements.txt` |
| `make run-tirzepatide-cohort` | Build cohort TSV from CMS inputs |
| `make run-tirzepatide-cohort-dry` | Dry-run cohort builder |
| `make demo` | Streamlit app |
| `make eval` | Compute `artifacts/demo/metrics.json` from latest run responses |
| `make refresh-demo` | Rebuild `artifacts/demo/summary.json`, `sample_responses.jsonl`, and `metrics.json` from existing JSONL (no API) |
| `make report-html` | Static HTML report → `docs/build/demo_report.html` |
| `make smoke-batch` | Short LLM batch (5 NPIs), default settings |
| `make docker-build` / `make docker-run` | Container image; Streamlit on port **8501**, `./data` mounted |
| `make clean-venv` | Remove `.venv` |

Legacy exploratory scripts under `archive/legacy/` are optional; see [`archive/legacy/README.md`](archive/legacy/README.md).

## API keys and environment

1. Copy [`.env.example`](.env.example) to **`.env`** in the repo root.
2. Set `TOGETHER_API_KEY` for the default Together SDK path, or use `--provider openai` with `OPENAI_API_KEY` (and optional base URL for OpenAI-compatible gateways).
3. Do not commit `.env`. `simulation.env_bootstrap.load_local_dotenv()` loads it at startup for `run_batch` and Streamlit (`override=False`, so existing shell variables win).

**Streamlit:** optional live re-run uses sidebar provider, model, temperature, and base URL; keys are entered in the session only.

**Footer link (optional):** `DEMO_REPO_URL=https://github.com/OWNER/REPO` for the Streamlit footer.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -p "test_*.py"
```

## Stability and reporting

- **Retest / prompt order sensitivity:** [`docs/retest_stability.md`](docs/retest_stability.md), [`scripts/compare_runs_stability.py`](scripts/compare_runs_stability.py)
- **Shuffle survey block order in batch:** `python -m simulation.run_batch --shuffle-questions --shuffle-seed <int> ...`
- **Planned backlog** (virtual interview UI, and similar): section 8 in [`docs/target_report.md`](docs/target_report.md)

## Limitations (read before citing numbers)

The cohort is **purposive and Part D–scoped**, not a national probability sample of U.S. physicians. Evaluation uses **rules-based pseudo-labels** from administrative fields, not human survey ground truth. Readouts are **descriptive and associational**, not causal estimates of promotional or launch effects.

## Docker

```bash
make docker-build
make docker-run
```

Opens Streamlit at `http://localhost:8501` with `./data` available inside the container.
