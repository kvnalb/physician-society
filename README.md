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
