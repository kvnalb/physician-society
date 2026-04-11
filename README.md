# Physician Society

Tirzepatide / GLP-1 **physician persona simulation** demo: Medicare Part D–grounded cohort, LLM survey (methods A vs B), Cohen’s κ, and Streamlit readouts. Full narrative: [`docs/target_report.md`](docs/target_report.md).

## Quick path (target demo)

```bash
make setup
# Build cohort (long; needs CMS extracts under data/raw/ — gitignored)
make run-tirzepatide-cohort
# LLM batch → writes data/output/runs/latest/responses.jsonl
python -m simulation.run_batch --limit-npis 10 --save-as-demo-bundle
# Optional: no API key — deterministic bundle for CI / UI smoke (not real LLM output)
# python -m simulation.run_batch --offline-seed-demo --limit-npis 10
make eval
make demo
```

`make eval` writes **`artifacts/demo/metrics.json`** and, when `data/output/runs/latest/` exists, also **`data/output/runs/latest/metrics.json`**, embedding **`run_manifest.json`** from that run when present. Use **Eval metrics snapshot** in Streamlit to compare runs.

Static HTML bundle (Jinja): `make report-html` → open `docs/build/demo_report.html`.

## API keys

Copy [`.env.example`](.env.example) to `.env`. Supported patterns:

- **Together (default):** `TOGETHER_API_KEY`. Batch inference uses the native Together SDK (`from together import Together`). Pass `--model` with a Together model id (default on CLI: `meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo`).
- **OpenAI or OpenAI-compatible:** `--provider openai` with `OPENAI_API_KEY`, or `TOGETHER_API_KEY` plus `--base-url https://api.together.xyz/v1` for the OpenAI-compatible client. See [`simulation/llm_client.py`](simulation/llm_client.py).

Streamlit **Advanced → live re-run** uses the sidebar provider / model / temperature / optional base URL; keys are session-only.

Optional: `DEMO_REPO_URL=https://github.com/OWNER/REPO` to show a link in the Streamlit footer.

## Make targets

| Target | Purpose |
|--------|---------|
| `make run-tirzepatide-cohort` | Build `data/output/tirzepatide_simulation_cohort_100.tsv` (gitignored when regenerated) |
| `make demo` | Streamlit UI |
| `make eval` | Latest `responses.jsonl` → `artifacts/demo/metrics.json` + run sidecar `runs/latest/metrics.json` |
| `make report-html` | `docs/build/demo_report.html` from narrative + demo JSON |
| `make smoke-batch` | Short batch (5 NPIs) |
| `make legacy-run-select-org` | Archived exploratory script (see below) |

## Docker

```bash
make docker-build
make docker-run
```

Runs Streamlit on **port 8501** (`http://localhost:8501`). Mounts `./data` into the container.

## Legacy scaffold

Earlier data-prep and stubs live under [`archive/legacy/`](archive/legacy/README.md) (`01`–`05` scripts, empty simulation stubs). They are **not** required for the tirzepatide demo path.

## Local venv (manual)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
