"""Select a large healthcare organization with rich linked data coverage.

This script discovers raw source files, inspects schemas, and can either run
targeted descriptive analysis (default) to inform filter choices, or the full
selection and enrichment pipeline when requested.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from itertools import chain
from pathlib import Path
from typing import Dict, Iterator, Optional, Sequence, Set, Tuple

try:
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency. Create/activate a virtualenv and install requirements:\n"
        "  python3 -m venv .venv\n"
        "  source .venv/bin/activate\n"
        "  pip install -r requirements.txt"
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
SCHEMA_REPORT_PATH = OUTPUT_DIR / "schema_report.txt"
CANDIDATE_CSV_PATH = OUTPUT_DIR / "candidate_orgs.csv"
DESCRIPTIVE_REPORT_PATH = OUTPUT_DIR / "descriptive_analysis.txt"

# Step 0 discovery variables (populated dynamically from data/raw)
DISCOVERED_CSV_FILES: list[Path] = []
DISCOVERED_XLSX_FILES: list[Path] = []
NPI_REGISTRY_PATH: Optional[Path] = None
PARTD_2023_PATH: Optional[Path] = None
OPEN_PAYMENTS_PATHS: list[Path] = []

CHUNK_SIZE = 100_000
TOP_ORGS_LIMIT = 50
MIN_PHYSICIANS_PER_ORG = 150


def normalize_header(header: str) -> str:
    """Normalize a header into a lowercase alphanumeric token string."""
    return "".join(ch for ch in header.lower().strip() if ch.isalnum())


def find_column(columns: Sequence[str], candidate_names: Sequence[str]) -> Optional[str]:
    """Find a column by trying exact and normalized matches."""
    normalized_columns = {normalize_header(col): col for col in columns}
    for candidate in candidate_names:
        if candidate in columns:
            return candidate
        match = normalized_columns.get(normalize_header(candidate))
        if match:
            return match
    return None


def format_size(size_bytes: int) -> str:
    """Convert bytes into a human-readable file size string."""
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(size_bytes)
    unit_idx = 0
    while size >= 1024 and unit_idx < len(units) - 1:
        size /= 1024
        unit_idx += 1
    return f"{size:.1f}{units[unit_idx]}"


def print_chunk_progress(step_label: str, chunk_idx: int) -> None:
    """Print periodic chunk progress updates."""
    if chunk_idx % 10 == 0:
        print(f"[{step_label}] Processing chunk {chunk_idx}/~? ...")


def chunked_csv_reader(path: Path, chunksize: int = CHUNK_SIZE) -> Iterator[pd.DataFrame]:
    """Yield CSV chunks with UTF-8 first, then latin-1 fallback."""
    for encoding in ("utf-8", "latin-1"):
        try:
            yield from pd.read_csv(path, chunksize=chunksize, low_memory=False, encoding=encoding)
            return
        except UnicodeDecodeError:
            print(f"[WARN] Encoding {encoding} failed for {path.name}; trying fallback.")
            continue
    raise UnicodeDecodeError("utf-8", b"", 0, 1, f"Unable to decode {path}")


def clean_str_series(series: pd.Series) -> pd.Series:
    """Return stripped string values with NA normalized."""
    return series.astype("string").str.strip().fillna("")


def clean_npi_series(series: pd.Series) -> pd.Series:
    """Normalize NPI values as stripped strings and remove invalid placeholders."""
    cleaned = clean_str_series(series)
    cleaned = cleaned.replace({"": pd.NA, "nan": pd.NA, "<NA>": pd.NA, "None": pd.NA})
    return cleaned


def entity_type_is_individual(series: pd.Series) -> pd.Series:
    """Return mask for NPPES Entity Type Code == 1 (handles float 1.0 from CSV)."""
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric == 1


def discover_files() -> None:
    """Step 0: Discover raw files, print paths and sizes, and classify key inputs."""
    global DISCOVERED_CSV_FILES
    global DISCOVERED_XLSX_FILES
    global NPI_REGISTRY_PATH
    global PARTD_2023_PATH
    global OPEN_PAYMENTS_PATHS

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    discovered = sorted(
        [p for p in RAW_DATA_DIR.rglob("*") if p.is_file() and p.suffix.lower() in {".csv", ".xlsx"}],
        key=lambda p: str(p),
    )
    DISCOVERED_CSV_FILES = [p for p in discovered if p.suffix.lower() == ".csv"]
    DISCOVERED_XLSX_FILES = [p for p in discovered if p.suffix.lower() == ".xlsx"]

    print("=== STEP 0: FILE DISCOVERY ===")
    if not discovered:
        raise FileNotFoundError(f"No CSV/XLSX files found under {RAW_DATA_DIR}")

    for path in discovered:
        rel = path.relative_to(PROJECT_ROOT)
        print(f"{rel}  ({format_size(path.stat().st_size)})")

    # Heuristic identification of key files without hardcoding exact names.
    # Prioritize full NPPES provider file (`npidata`) over endpoint metadata files.
    for path in DISCOVERED_CSV_FILES:
        name = path.name.lower()
        rel = str(path).lower()
        if NPI_REGISTRY_PATH is None and "npidata" in name:
            NPI_REGISTRY_PATH = path

    for path in DISCOVERED_CSV_FILES:
        name = path.name.lower()
        rel = str(path).lower()
        if NPI_REGISTRY_PATH is None and "nppes" in rel and "endpoint" not in name:
            NPI_REGISTRY_PATH = path
        if PARTD_2023_PATH is None and (
            ("part_d" in rel or "dpr" in rel or "prescribers" in rel)
            and ("2023" in rel or "dy23" in rel)
        ):
            PARTD_2023_PATH = path
        if "op_dtl" in name and ("pgyr2022" in rel or "pgyr2023" in rel):
            OPEN_PAYMENTS_PATHS.append(path)

    OPEN_PAYMENTS_PATHS = sorted(OPEN_PAYMENTS_PATHS, key=lambda p: str(p))

    if NPI_REGISTRY_PATH is None:
        raise FileNotFoundError("Could not identify NPI registry CSV from discovered files.")
    if PARTD_2023_PATH is None:
        raise FileNotFoundError("Could not identify Part D 2023 CSV from discovered files.")
    if not OPEN_PAYMENTS_PATHS:
        raise FileNotFoundError("Could not identify Open Payments CSV files from discovered files.")

    print("\nSelected key files:")
    print(f"- NPI registry: {NPI_REGISTRY_PATH.relative_to(PROJECT_ROOT)}")
    print(f"- Part D 2023: {PARTD_2023_PATH.relative_to(PROJECT_ROOT)}")
    print("- Open Payments files:")
    for p in OPEN_PAYMENTS_PATHS:
        print(f"  - {p.relative_to(PROJECT_ROOT)}")


def write_schema_report() -> None:
    """Step 1: Inspect schemas and write a schema report."""
    print("\n=== STEP 1: SCHEMA INSPECTION ===")
    lines: list[str] = []

    for path in DISCOVERED_CSV_FILES:
        lines.append(f"\n### {path.relative_to(PROJECT_ROOT)}")
        try:
            sample_df = pd.read_csv(path, nrows=5, low_memory=False)
        except UnicodeDecodeError:
            sample_df = pd.read_csv(path, nrows=5, low_memory=False, encoding="latin-1")

        lines.append("Columns and dtypes:")
        for col in sample_df.columns:
            lines.append(f"- {col}: {sample_df[col].dtype}")
        lines.append("First 5 rows:")
        lines.append(sample_df.to_string(index=False))
        print(f"Inspected: {path.relative_to(PROJECT_ROOT)}")

    for path in DISCOVERED_XLSX_FILES:
        lines.append(f"\n### {path.relative_to(PROJECT_ROOT)}")
        sample_df = pd.read_excel(path, nrows=5)
        lines.append("Columns and dtypes:")
        for col in sample_df.columns:
            lines.append(f"- {col}: {sample_df[col].dtype}")
        lines.append("First 5 rows:")
        lines.append(sample_df.to_string(index=False))
        print(f"Inspected: {path.relative_to(PROJECT_ROOT)}")

    SCHEMA_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Schema report saved: {SCHEMA_REPORT_PATH.relative_to(PROJECT_ROOT)}")


def run_descriptive_analysis() -> None:
    """Summarize NPPES, Part D, and Open Payments without selection filters."""
    print("\n=== DESCRIPTIVE ANALYSIS (no org filters) ===")

    lines: list[str] = []
    lines.append("=== DESCRIPTIVE DATA ANALYSIS ===\n")
    lines.append("Purpose: inform filter choices before organization selection.\n")

    # --- NPPES ---
    lines.append("\n--- NPPES registry (full chunked pass) ---\n")
    npi_reader = chunked_csv_reader(NPI_REGISTRY_PATH)
    first = next(npi_reader)
    npi_col, entity_type_col, org_col, _, _, _ = infer_npi_registry_columns(first.columns)

    total_rows = 0
    entity_raw_counts: Counter[str] = Counter()
    type1_rows = 0
    type1_valid_npi = 0
    type1_org_nonempty = 0
    type1_org_empty = 0
    org_sizes: Counter[str] = Counter()

    for idx, chunk in enumerate(chain([first], npi_reader), start=1):
        print_chunk_progress("Descriptive NPPES", idx)
        total_rows += len(chunk)

        raw_et = chunk[entity_type_col]
        for v in raw_et.fillna("__NA__").astype(str).str.strip():
            entity_raw_counts[v] += 1

        is_t1 = entity_type_is_individual(raw_et)
        chunk[npi_col] = clean_npi_series(chunk[npi_col])
        chunk[org_col] = clean_str_series(chunk[org_col])

        t1 = chunk[is_t1]
        type1_rows += len(t1)
        if not t1.empty:
            vn = t1[npi_col].notna()
            type1_valid_npi += int(vn.sum())
            org_filled = t1[org_col] != ""
            type1_org_nonempty += int((vn & org_filled).sum())
            type1_org_empty += int((vn & ~org_filled).sum())
            sub = t1[vn & org_filled]
            for org in sub[org_col].astype(str).str.strip():
                org_sizes[org] += 1

    lines.append(f"Total rows: {total_rows:,}\n")
    lines.append(f"Entity Type Code raw value counts (top 15 by frequency):\n")
    for val, c in entity_raw_counts.most_common(15):
        lines.append(f"  {repr(val)}: {c:,}\n")
    lines.append(f"\nRows with Entity Type Code == 1 (numeric): {type1_rows:,}\n")
    lines.append(f"Among type-1 rows with valid NPI: {type1_valid_npi:,}\n")
    lines.append(f"  With non-empty organization name: {type1_org_nonempty:,}\n")
    lines.append(f"  With empty organization name: {type1_org_empty:,}\n")
    if type1_valid_npi:
        pct = 100.0 * type1_org_nonempty / type1_valid_npi
        lines.append(f"  Share of type-1 (valid NPI) with org name: {pct:.2f}%\n")

    n_orgs = len(org_sizes)
    lines.append(f"\nDistinct organization names (type-1, valid NPI, non-empty org): {n_orgs:,}\n")
    if org_sizes:
        sizes = list(org_sizes.values())
        lines.append(f"Physicians per org (row counts; min / max / mean): {min(sizes)} / {max(sizes)} / {sum(sizes)/len(sizes):.2f}\n")
        thresholds = [10, 50, 100, 150, 200, 500]
        lines.append("Organizations at or above size threshold (physician row counts):\n")
        for t in thresholds:
            n = sum(1 for s in sizes if s >= t)
            lines.append(f"  >= {t}: {n:,} orgs\n")
        lines.append("\nTop 40 organizations by physician row count (type-1, org name present):\n")
        for org, cnt in org_sizes.most_common(40):
            safe = org.replace("\n", " ")[:120]
            lines.append(f"  {cnt:6d}  {safe}\n")

    # --- Part D ---
    lines.append("\n--- Part D 2023 (chunked row and NPI presence) ---\n")
    partd_reader = chunked_csv_reader(PARTD_2023_PATH)
    partd_first = next(partd_reader)
    partd_npi_col = infer_partd_npi_column(partd_first.columns)
    pd_rows = 0
    pd_npi_nonnull = 0
    for idx, chunk in enumerate(chain([partd_first], partd_reader), start=1):
        print_chunk_progress("Descriptive Part D", idx)
        pd_rows += len(chunk)
        s = clean_npi_series(chunk[partd_npi_col])
        pd_npi_nonnull += int(s.notna().sum())
    lines.append(f"Total rows: {pd_rows:,}\n")
    lines.append(f"Rows with non-null NPI (after normalize): {pd_npi_nonnull:,}\n")
    if pd_rows:
        lines.append(f"Share with NPI: {100.0 * pd_npi_nonnull / pd_rows:.2f}%\n")

    # --- Open Payments ---
    lines.append("\n--- Open Payments files (per file: rows, NPI presence) ---\n")
    for pay_path in OPEN_PAYMENTS_PATHS:
        rel = pay_path.relative_to(PROJECT_ROOT)
        pay_reader = chunked_csv_reader(pay_path)
        p_first = next(pay_reader)
        pay_npi_col = infer_open_payments_npi_column(p_first.columns)
        pr, pn = 0, 0
        for idx, chunk in enumerate(chain([p_first], pay_reader), start=1):
            print_chunk_progress(f"Descriptive OP {pay_path.name}", idx)
            pr += len(chunk)
            s = clean_npi_series(chunk[pay_npi_col])
            pn += int(s.notna().sum())
        lines.append(f"{rel}\n")
        lines.append(f"  rows: {pr:,}; non-null NPI: {pn:,}")
        if pr:
            lines.append(f" ({100.0 * pn / pr:.2f}%)\n")
        else:
            lines.append("\n")

    report = "".join(lines)
    DESCRIPTIVE_REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nDescriptive report saved: {DESCRIPTIVE_REPORT_PATH.relative_to(PROJECT_ROOT)}")


def infer_npi_registry_columns(columns: Sequence[str]) -> Tuple[str, str, str, Optional[str], Optional[str], Optional[str]]:
    """Infer required NPI registry columns across schema variants."""
    npi_col = find_column(
        columns,
        ["NPI", "npi", "Provider NPI", "Provider_NPI"],
    )
    entity_type_col = find_column(
        columns,
        ["Entity Type Code", "Entity_Type_Code", "entity_type_code"],
    )
    org_col = find_column(
        columns,
        [
            "Provider Organization Name (Legal Business Name)",
            "Provider Organization Name",
            "Provider_Organization_Name",
            "Provider Organization Name (Legal Business Name)_1",
        ],
    )
    taxonomy_code_col = find_column(
        columns,
        [
            "Healthcare Provider Taxonomy Code_1",
            "Healthcare Provider Taxonomy Code 1",
            "Provider Taxonomy Code_1",
            "Taxonomy Code_1",
        ],
    )
    taxonomy_desc_col = find_column(
        columns,
        [
            "Healthcare Provider Primary Taxonomy Switch_1",
            "Healthcare Provider Taxonomy Group_1",
            "Healthcare Provider Taxonomy Description_1",
            "Taxonomy Description_1",
        ],
    )
    credential_col = find_column(
        columns,
        [
            "Provider Credential Text",
            "Provider Credential Text_1",
            "Provider Credentials",
        ],
    )

    missing = [name for name, col in [("NPI", npi_col), ("Entity Type Code", entity_type_col), ("Organization Name", org_col)] if col is None]
    if missing:
        raise ValueError(f"Missing required NPI columns: {missing}")

    return npi_col, entity_type_col, org_col, taxonomy_code_col, taxonomy_desc_col, credential_col


def infer_partd_npi_column(columns: Sequence[str]) -> str:
    """Infer the NPI column name for Part D files."""
    npi_col = find_column(columns, ["NPI", "npi", "Prscrbr_NPI", "Provider NPI"])
    if npi_col is None:
        raise ValueError("Could not identify NPI column in Part D file.")
    return npi_col


def infer_open_payments_npi_column(columns: Sequence[str]) -> str:
    """Infer the NPI column name for Open Payments files."""
    npi_col = find_column(
        columns,
        [
            "Covered_Recipient_NPI",
            "Covered Recipient NPI",
            "Physician_NPI",
            "NPI",
        ],
    )
    if npi_col is None:
        raise ValueError("Could not identify NPI column in Open Payments file.")
    return npi_col


def specialty_label(row: pd.Series, taxonomy_code_col: Optional[str], taxonomy_desc_col: Optional[str], credential_col: Optional[str]) -> str:
    """Build a readable specialty label from available NPI fields."""
    code = ""
    desc = ""
    credential = ""
    if taxonomy_code_col and taxonomy_code_col in row:
        code = str(row[taxonomy_code_col]).strip()
    if taxonomy_desc_col and taxonomy_desc_col in row:
        desc = str(row[taxonomy_desc_col]).strip()
    if credential_col and credential_col in row:
        credential = str(row[credential_col]).strip()

    if desc and desc.lower() not in {"nan", "none", "<na>"}:
        return desc
    if credential and credential.lower() not in {"nan", "none", "<na>"}:
        return f"Credential: {credential}"
    if code and code.lower() not in {"nan", "none", "<na>"}:
        return f"Taxonomy {code}"
    return "Unknown"


def build_candidate_organizations() -> Tuple[pd.DataFrame, Dict[str, Set[str]], Dict[str, str], Dict[str, str]]:
    """Step 2: Build candidate organizations from NPI data."""
    print("\n=== STEP 2: FIND CANDIDATE ORGANIZATIONS ===")
    org_to_npis: Dict[str, Set[str]] = defaultdict(set)
    npi_to_org: Dict[str, str] = {}
    npi_to_specialty: Dict[str, str] = {}

    first_chunk = next(chunked_csv_reader(NPI_REGISTRY_PATH))
    npi_col, entity_type_col, org_col, taxonomy_code_col, taxonomy_desc_col, credential_col = infer_npi_registry_columns(first_chunk.columns)

    # Re-run from beginning now that schema is inferred.
    for idx, chunk in enumerate(chunked_csv_reader(NPI_REGISTRY_PATH), start=1):
        print_chunk_progress("Step 2", idx)

        chunk[org_col] = clean_str_series(chunk[org_col])
        chunk[npi_col] = clean_npi_series(chunk[npi_col])

        is_individual = entity_type_is_individual(chunk[entity_type_col])
        filtered = chunk[is_individual & (chunk[org_col] != "") & chunk[npi_col].notna()].copy()
        if filtered.empty:
            continue

        for _, row in filtered.iterrows():
            org = str(row[org_col]).strip()
            npi = str(row[npi_col]).strip()
            org_to_npis[org].add(npi)
            npi_to_org[npi] = org
            npi_to_specialty[npi] = specialty_label(row, taxonomy_code_col, taxonomy_desc_col, credential_col)

    rows = [{"org_name": org, "physician_count": len(npis)} for org, npis in org_to_npis.items() if len(npis) >= MIN_PHYSICIANS_PER_ORG]
    if not rows:
        raise RuntimeError(
            f"No organizations met physician_count >= {MIN_PHYSICIANS_PER_ORG}. "
            "Run without --full-pipeline to generate descriptive_analysis.txt, "
            "or lower MIN_PHYSICIANS_PER_ORG in the script."
        )
    candidates = pd.DataFrame(rows).sort_values("physician_count", ascending=False).head(TOP_ORGS_LIMIT).reset_index(drop=True)

    top30 = candidates.head(30).iloc[::-1]
    plt.figure(figsize=(12, 10))
    sns.barplot(data=top30, x="physician_count", y="org_name", color="#377eb8")
    plt.title("Top 30 Healthcare Organizations by Physician Count")
    plt.xlabel("Physician Count")
    plt.ylabel("Organization")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "org_physician_counts.png", dpi=200)
    plt.close()

    candidates.to_csv(CANDIDATE_CSV_PATH, index=False)
    print(f"Saved initial candidates: {CANDIDATE_CSV_PATH.relative_to(PROJECT_ROOT)}")

    top_orgs = set(candidates["org_name"])
    top_org_to_npis = {org: npis for org, npis in org_to_npis.items() if org in top_orgs}
    return candidates, top_org_to_npis, npi_to_org, npi_to_specialty


def add_partd_coverage(candidates: pd.DataFrame, top_org_to_npis: Dict[str, Set[str]], npi_to_org: Dict[str, str]) -> Tuple[pd.DataFrame, Dict[str, Set[str]]]:
    """Step 3: Add Part D physician coverage by organization."""
    print("\n=== STEP 3: ENRICH WITH PART D COVERAGE ===")
    org_to_partd_npis: Dict[str, Set[str]] = {org: set() for org in top_org_to_npis}
    target_npis = set(npi_to_org.keys())

    first_chunk = next(chunked_csv_reader(PARTD_2023_PATH))
    partd_npi_col = infer_partd_npi_column(first_chunk.columns)

    for idx, chunk in enumerate(chunked_csv_reader(PARTD_2023_PATH), start=1):
        print_chunk_progress("Step 3", idx)
        chunk[partd_npi_col] = clean_npi_series(chunk[partd_npi_col])
        npi_values = set(chunk[partd_npi_col].dropna().astype(str))
        matches = npi_values & target_npis
        if not matches:
            continue
        for npi in matches:
            org = npi_to_org.get(npi)
            if org in org_to_partd_npis:
                org_to_partd_npis[org].add(npi)

    partd_counts = {org: len(npis) for org, npis in org_to_partd_npis.items()}
    candidates["partd_physician_count"] = candidates["org_name"].map(partd_counts).fillna(0).astype(int)
    candidates["partd_coverage_pct"] = (candidates["partd_physician_count"] / candidates["physician_count"] * 100).round(2)

    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=candidates, x="physician_count", y="partd_coverage_pct", s=90)
    for _, row in candidates.head(10).iterrows():
        plt.text(row["physician_count"], row["partd_coverage_pct"], str(row["org_name"])[:40], fontsize=8)
    plt.title("Organization Size vs Part D Data Coverage")
    plt.xlabel("Total Physicians")
    plt.ylabel("Part D Coverage (%)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "org_partd_coverage.png", dpi=200)
    plt.close()

    return candidates, org_to_partd_npis


def add_open_payments_coverage(candidates: pd.DataFrame, top_org_to_npis: Dict[str, Set[str]], npi_to_org: Dict[str, str]) -> Tuple[pd.DataFrame, Dict[str, Set[str]]]:
    """Step 4: Add Open Payments physician coverage by organization."""
    print("\n=== STEP 4: ENRICH WITH OPEN PAYMENTS COVERAGE ===")
    org_to_payment_npis: Dict[str, Set[str]] = {org: set() for org in top_org_to_npis}
    target_npis = set(npi_to_org.keys())

    for payments_path in OPEN_PAYMENTS_PATHS:
        print(f"Scanning Open Payments file: {payments_path.relative_to(PROJECT_ROOT)}")
        first_chunk = next(chunked_csv_reader(payments_path))
        payments_npi_col = infer_open_payments_npi_column(first_chunk.columns)

        for idx, chunk in enumerate(chunked_csv_reader(payments_path), start=1):
            print_chunk_progress(f"Step 4 ({payments_path.name})", idx)
            chunk[payments_npi_col] = clean_npi_series(chunk[payments_npi_col])
            npi_values = set(chunk[payments_npi_col].dropna().astype(str))
            matches = npi_values & target_npis
            if not matches:
                continue
            for npi in matches:
                org = npi_to_org.get(npi)
                if org in org_to_payment_npis:
                    org_to_payment_npis[org].add(npi)

    payment_counts = {org: len(npis) for org, npis in org_to_payment_npis.items()}
    candidates["payments_physician_count"] = candidates["org_name"].map(payment_counts).fillna(0).astype(int)
    candidates["payments_coverage_pct"] = (candidates["payments_physician_count"] / candidates["physician_count"] * 100).round(2)

    top15 = candidates.head(15).copy()
    top15 = top15.iloc[::-1]
    y_positions = range(len(top15))
    bar_h = 0.25

    plt.figure(figsize=(14, 10))
    plt.barh([y - bar_h for y in y_positions], top15["physician_count"], height=bar_h, label="Total physicians")
    plt.barh(y_positions, top15["partd_physician_count"], height=bar_h, label="With Part D data")
    plt.barh([y + bar_h for y in y_positions], top15["payments_physician_count"], height=bar_h, label="With Open Payments data")
    plt.yticks(list(y_positions), top15["org_name"])
    plt.title("Data Coverage by Organization (Top 15)")
    plt.xlabel("Physician Count")
    plt.ylabel("Organization")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "org_data_coverage.png", dpi=200)
    plt.close()

    return candidates, org_to_payment_npis


def run_specialty_diversity_check(
    candidates: pd.DataFrame,
    top_org_to_npis: Dict[str, Set[str]],
    org_to_partd_npis: Dict[str, Set[str]],
    org_to_payment_npis: Dict[str, Set[str]],
    npi_to_specialty: Dict[str, str],
) -> None:
    """Step 5: Print specialty diversity summaries and build composition chart."""
    print("\n=== STEP 5: SPECIALTY DIVERSITY CHECK ===")
    top5 = candidates.sort_values("partd_physician_count", ascending=False).head(5)
    specialty_mix_rows: list[dict[str, object]] = []

    for _, row in top5.iterrows():
        org = str(row["org_name"])
        org_npis = top_org_to_npis.get(org, set())
        specialty_counts: Counter[str] = Counter(npi_to_specialty.get(npi, "Unknown") for npi in org_npis)
        distinct_specialties = sum(1 for _, c in specialty_counts.items() if c > 0)

        print(f"\nOrganization: {org}")
        print(f"- Total physicians with Part D data: {len(org_to_partd_npis.get(org, set()))}")
        payments_count = len(org_to_payment_npis.get(org, set()))
        payments_pct = (payments_count / max(len(org_npis), 1)) * 100
        print(f"- Open Payments coverage: {payments_count} ({payments_pct:.2f}%)")
        print(f"- Distinct specialties: {distinct_specialties}")
        print("- Top 10 specialties:")
        for specialty, count in specialty_counts.most_common(10):
            print(f"  - {specialty}: {count}")

        top5_specs = specialty_counts.most_common(5)
        top5_names = {name for name, _ in top5_specs}
        other_count = sum(count for name, count in specialty_counts.items() if name not in top5_names)
        for spec_name, count in top5_specs:
            specialty_mix_rows.append({"org_name": org, "specialty": spec_name, "count": count})
        if other_count > 0:
            specialty_mix_rows.append({"org_name": org, "specialty": "Other", "count": other_count})

    mix_df = pd.DataFrame(specialty_mix_rows)
    if not mix_df.empty:
        pivot_df = mix_df.pivot_table(index="org_name", columns="specialty", values="count", aggfunc="sum", fill_value=0)
        pivot_df = pivot_df.loc[top5["org_name"]]
        pivot_df.plot(kind="barh", stacked=True, figsize=(14, 8), colormap="tab20")
        plt.title("Specialty Mix: Top 5 Candidate Organizations")
        plt.xlabel("Physician Count")
        plt.ylabel("Organization")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "org_specialty_mix.png", dpi=200)
        plt.close()


def print_recommendation(
    candidates: pd.DataFrame,
    top_org_to_npis: Dict[str, Set[str]],
    org_to_partd_npis: Dict[str, Set[str]],
    org_to_payment_npis: Dict[str, Set[str]],
    npi_to_specialty: Dict[str, str],
) -> None:
    """Step 6: Print final structured recommendation and output manifest."""
    print("\n=== ORGANIZATION SELECTION RECOMMENDATION ===")
    ranked = candidates.sort_values(
        by=["partd_physician_count", "payments_physician_count", "physician_count"],
        ascending=False,
    ).reset_index(drop=True)

    labels = ["Top pick", "Runner-up", "Runner-up 2"]
    for label, (_, row) in zip(labels, ranked.head(3).iterrows()):
        org = str(row["org_name"])
        org_npis = top_org_to_npis.get(org, set())
        partd_count = len(org_to_partd_npis.get(org, set()))
        payments_count = len(org_to_payment_npis.get(org, set()))
        partd_pct = (partd_count / max(len(org_npis), 1)) * 100
        pay_pct = (payments_count / max(len(org_npis), 1)) * 100
        specialty_counts = Counter(npi_to_specialty.get(npi, "Unknown") for npi in org_npis)
        top5_specs = specialty_counts.most_common(5)

        print(f"\n{label}: {org}")
        print(f"- Total physicians in NPI: {len(org_npis)}")
        print(f"- With Part D prescribing data: {partd_count} ({partd_pct:.2f}%)")
        print(f"- With any Open Payments record: {payments_count} ({pay_pct:.2f}%)")
        print(f"- Distinct specialties represented: {len(specialty_counts)}")
        print(f"- Top 5 specialties: {top5_specs}")
        print(
            "- Why this org: Strong physician scale with high overlap across prescribing and "
            "Open Payments sources, enabling realistic network simulation."
        )

    print("\nData quality concerns:")
    print("- Source schemas vary by release year; dynamic column inference was applied.")
    print("- Some files may require latin-1 encoding fallback.")
    print("- Missing or malformed NPIs are excluded from joins after normalization.")

    print("\nFiles created:")
    created_files = [
        CANDIDATE_CSV_PATH,
        SCHEMA_REPORT_PATH,
        FIGURES_DIR / "org_physician_counts.png",
        FIGURES_DIR / "org_partd_coverage.png",
        FIGURES_DIR / "org_data_coverage.png",
        FIGURES_DIR / "org_specialty_mix.png",
    ]
    for file_path in created_files:
        print(f"- {file_path.relative_to(PROJECT_ROOT)}")


def main() -> None:
    """Discover files, schema report, descriptive analysis; optionally full selection pipeline."""
    parser = argparse.ArgumentParser(description="NPPES / Part D / Open Payments organization analysis.")
    parser.add_argument(
        "--full-pipeline",
        action="store_true",
        help="Run filtered org selection, Part D/Open Payments enrichment, figures, and recommendation.",
    )
    args = parser.parse_args()

    sns.set_theme(style="whitegrid")
    discover_files()
    write_schema_report()
    run_descriptive_analysis()

    if not args.full_pipeline:
        print("\nDone (descriptive only). Use --full-pipeline to run selection and enrichment after choosing filters.")
        return

    candidates, top_org_to_npis, npi_to_org, npi_to_specialty = build_candidate_organizations()
    candidates, org_to_partd_npis = add_partd_coverage(candidates, top_org_to_npis, npi_to_org)
    candidates, org_to_payment_npis = add_open_payments_coverage(candidates, top_org_to_npis, npi_to_org)
    candidates.to_csv(CANDIDATE_CSV_PATH, index=False)
    run_specialty_diversity_check(candidates, top_org_to_npis, org_to_partd_npis, org_to_payment_npis, npi_to_specialty)
    print_recommendation(candidates, top_org_to_npis, org_to_partd_npis, org_to_payment_npis, npi_to_specialty)
    print(f"\nFinal enriched candidates saved: {CANDIDATE_CSV_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
