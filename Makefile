PYTHON ?= python3
VENV_DIR ?= .venv
VENV_PYTHON := $(VENV_DIR)/bin/python
VENV_PIP := $(VENV_DIR)/bin/pip

.PHONY: venv install setup legacy-run-select-org legacy-run-group-locations legacy-run-sample-pharma legacy-run-sample-pharma-dry run-tirzepatide-cohort run-tirzepatide-cohort-dry demo eval refresh-demo smoke-batch report-html docker-build docker-run clean-venv

venv:
	$(PYTHON) -m venv $(VENV_DIR)

install: venv
	$(VENV_PIP) install --upgrade pip
	$(VENV_PIP) install -r requirements.txt

setup: install

# Legacy scaffold (see archive/legacy/README.md)
legacy-run-select-org:
	$(VENV_PYTHON) archive/legacy/scripts/01_select_organization.py

legacy-run-group-locations:
	$(VENV_PYTHON) -u archive/legacy/scripts/02_group_npis_by_practice_location.py

legacy-run-sample-pharma:
	$(VENV_PYTHON) -u archive/legacy/scripts/05_sample_pharma_sales_cohort.py

legacy-run-sample-pharma-dry:
	$(VENV_PYTHON) -u archive/legacy/scripts/05_sample_pharma_sales_cohort.py --dry-run

run-tirzepatide-cohort:
	$(VENV_PYTHON) -u scripts/06_tirzepatide_simulation_cohort.py

run-tirzepatide-cohort-dry:
	$(VENV_PYTHON) -u scripts/06_tirzepatide_simulation_cohort.py --dry-run

demo:
	$(VENV_PYTHON) -m streamlit run streamlit_app.py

eval:
	$(VENV_PYTHON) -m eval.run_eval \
		--responses-file data/output/runs/latest/responses.jsonl \
		--output artifacts/demo/metrics.json

# Rebuild artifacts/demo/summary.json, sample_responses.jsonl, and metrics.json from an existing JSONL (no API).
refresh-demo:
	$(VENV_PYTHON) -m simulation.refresh_demo_from_responses

report-html:
	$(VENV_PYTHON) docs/build_report.py

smoke-batch:
	$(VENV_PYTHON) -m simulation.run_batch --limit-npis 5

docker-build:
	docker build -t physician-society:latest .

docker-run:
	docker run --rm -it -p 8501:8501 -v "$$(pwd)/data:/app/data" physician-society:latest

clean-venv:
	rm -rf $(VENV_DIR)
