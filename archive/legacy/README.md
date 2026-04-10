# Legacy scaffold (archived)

These files are **not** on the active tirzepatide demo path. They were kept for reference and optional local data-prep workflows.

**Active demo (review this first):**

- Cohort: `scripts/06_tirzepatide_simulation_cohort.py` → `data/output/tirzepatide_simulation_cohort_100.tsv` (gitignored when regenerated)
- LLM batch: `python -m simulation.run_batch`
- UI: `streamlit run streamlit_app.py` or `make demo`
- Narrative: `docs/target_report.md`

**This folder**

| Path | Notes |
|------|--------|
| `scripts/01_select_organization.py` | Exploratory CMS schema / org selection; writes `data/output/schema_report.txt` if run |
| `scripts/02_group_npis_by_practice_location.py` | Builds large location TSVs (gitignored outputs) |
| `scripts/02_build_personas.py`, `03_…`, `04_…` | Stubs |
| `scripts/05_sample_pharma_sales_cohort.py` | Pharma-sales sampling (depends on 02 outputs) |
| `simulation/` | Empty stubs (`run.py`, `methods.py`, …); real code lives in repo-root `simulation/` |
| `dashboard/build.py` | Stub |
| `outputs/schema_report.txt` | Copy of exploratory schema report from an older run |

**Running archived scripts:** from repo root, paths resolve to the real `data/` directory (`PROJECT_ROOT` is set three levels up from `archive/legacy/scripts/`).

Makefile targets `legacy-*` invoke these files.
