# Physician Society

Project scaffold for physician persona simulation and evaluation.

## Local setup (venv)

```bash
make setup
make run-select-org
```

Equivalent manual commands:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/01_select_organization.py
```

## Docker setup

```bash
make docker-build
make docker-run
```

Notes:
- `docker-run` mounts your local `data/` directory into the container at `/app/data`.
- Outputs are written to `data/output/` on your host machine.

## Tirzepatide LLM simulation (demo)

1. Build cohort TSV: `make run-tirzepatide-cohort` (long run; use `--use-cache` after first full pass).
2. Offline narrative: `docs/target_report.md`.
3. Streamlit UI (loads `artifacts/demo/` by default): `make demo` or `streamlit run streamlit_app.py`.
4. Batch survey (needs `OPENAI_API_KEY`): `python -m simulation.run_batch --limit-npis 10 --save-as-demo-bundle`.
5. Metrics from responses: `make eval` (expects `data/output/runs/latest/responses.jsonl`).

See `.env.example` for API key layout.
