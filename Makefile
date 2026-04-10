PYTHON ?= python3
VENV_DIR ?= .venv
VENV_PYTHON := $(VENV_DIR)/bin/python
VENV_PIP := $(VENV_DIR)/bin/pip

.PHONY: venv install setup run-select-org run-group-locations run-sample-pharma run-sample-pharma-dry run-tirzepatide-cohort run-tirzepatide-cohort-dry demo eval smoke-batch docker-build docker-run clean-venv

venv:
	$(PYTHON) -m venv $(VENV_DIR)

install: venv
	$(VENV_PIP) install --upgrade pip
	$(VENV_PIP) install -r requirements.txt

setup: install

run-select-org:
	$(VENV_PYTHON) scripts/01_select_organization.py

run-group-locations:
	$(VENV_PYTHON) -u scripts/02_group_npis_by_practice_location.py

run-sample-pharma:
	$(VENV_PYTHON) -u scripts/05_sample_pharma_sales_cohort.py

run-sample-pharma-dry:
	$(VENV_PYTHON) -u scripts/05_sample_pharma_sales_cohort.py --dry-run

run-tirzepatide-cohort:
	$(VENV_PYTHON) -u scripts/06_tirzepatide_simulation_cohort.py

run-tirzepatide-cohort-dry:
	$(VENV_PYTHON) -u scripts/06_tirzepatide_simulation_cohort.py --dry-run

demo:
	$(VENV_PYTHON) -m streamlit run streamlit_app.py

eval:
	$(VENV_PYTHON) -m eval.run_eval

smoke-batch:
	$(VENV_PYTHON) -m simulation.run_batch --limit-npis 5

docker-build:
	docker build -t physician-society:latest .

docker-run:
	docker run --rm -it -v "$$(pwd)/data:/app/data" physician-society:latest python scripts/01_select_organization.py

clean-venv:
	rm -rf $(VENV_DIR)
