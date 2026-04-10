"""Select co-located prescribers with strong Part D and Open Payments signal.

Uses outputs from 02_group_npis_by_practice_location.py (location counts + NPI map),
then scores NPIs for pharma sales relevance: prescribing volume and breadth (Part D),
plus industry-facing activity (Open Payments). Prefers a single practice-address
cluster (proxy for one site/hospital); if fewer than target N qualify, fills from the
next largest locations (ranked by NPI count — larger sites expected to carry more KOLs).

Use --dry-run to validate the pipeline without reading every row (capped map scan,
first N chunks of Part D and Open Payments).

Why upstream files are large: npi_practice_location.tsv has one row per type-1 NPI
with a valid address (~millions of rows). practice_location_npi_counts.tsv has one
row per distinct normalized address (~millions of locations, long keys).
"""

from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import DefaultDict, Dict, Iterator, List, Optional, Sequence, Set, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
LOCATION_COUNTS_PATH = OUTPUT_DIR / "practice_location_npi_counts.tsv"
NPI_LOCATION_MAP_PATH = OUTPUT_DIR / "npi_practice_location.tsv"
SAMPLE_OUTPUT_TSV = OUTPUT_DIR / "sample_100_pharma_sales_npis.tsv"
SAMPLE_REPORT_PATH = OUTPUT_DIR / "sample_100_pharma_sales_report.txt"
SAMPLE_OUTPUT_DRY_TSV = OUTPUT_DIR / "sample_pharma_sales_npis_dry_run.tsv"
SAMPLE_REPORT_DRY_PATH = OUTPUT_DIR / "sample_pharma_sales_report_dry_run.txt"

CHUNK_SIZE = 100_000

# Typical MD-facing specialties (light boost; volume remains primary).
PHYSICIAN_FRIENDLY_TYPES = frozenset(
    {
        "Internal Medicine",
        "Family Practice",
        "Hospitalist",
        "Cardiology",
        "Psychiatry",
        "Neurology",
        "Endocrinology",
        "Gastroenterology",
        "Pulmonary Disease",
        "Hematology",
        "Oncology",
        "Infectious Disease",
        "Nephrology",
        "Rheumatology",
        "Dermatology",
        "Urology",
        "Anesthesiology",
        "Emergency Medicine",
        "Pulmonary Critical Care",
    }
)


def normalize_header(header: str) -> str:
    return "".join(ch for ch in header.lower().strip() if ch.isalnum())


def find_column(columns: Sequence[str], candidate_names: Sequence[str]) -> Optional[str]:
    normalized_columns = {normalize_header(col): col for col in columns}
    for candidate in candidate_names:
        if candidate in columns:
            return candidate
        match = normalized_columns.get(normalize_header(candidate))
        if match:
            return match
    return None


def print_chunk_progress(step_label: str, chunk_idx: int) -> None:
    if chunk_idx % 10 == 0:
        print(f"[{step_label}] Processing chunk {chunk_idx}/~? ...")


def chunked_csv_reader(path: Path, chunksize: int = CHUNK_SIZE) -> Iterator[pd.DataFrame]:
    for encoding in ("utf-8", "latin-1"):
        try:
            yield from pd.read_csv(path, chunksize=chunksize, low_memory=False, encoding=encoding)
            return
        except UnicodeDecodeError:
            print(f"[WARN] Encoding {encoding} failed for {path.name}; trying fallback.")
            continue
    raise UnicodeDecodeError("utf-8", b"", 0, 1, f"Unable to decode {path}")


def discover_part_d_2023_path() -> Path:
    for path in sorted(RAW_DATA_DIR.rglob("*.csv"), key=lambda p: str(p)):
        rel = str(path).lower()
        name = path.name.lower()
        if "endpoint" in name or "npidata" in name:
            continue
        if ("part_d" in rel or "dpr" in rel or "prescribers" in rel) and (
            "2023" in rel or "dy23" in rel
        ):
            return path
    raise FileNotFoundError("Could not find Part D 2023 CSV under data/raw")


def discover_open_payments_paths() -> List[Path]:
    paths: List[Path] = []
    for path in sorted(RAW_DATA_DIR.rglob("*.csv"), key=lambda p: str(p)):
        name = path.name.lower()
        rel = str(path).lower()
        if "op_dtl" in name and ("pgyr2022" in rel or "pgyr2023" in rel):
            paths.append(path)
    if not paths:
        raise FileNotFoundError("Could not find Open Payments OP_DTL CSVs under data/raw")
    return sorted(paths, key=lambda p: str(p))


def infer_partd_columns(columns: Sequence[str]) -> Tuple[str, str, Optional[str], Optional[str], Optional[str]]:
    npi_col = find_column(columns, ["Prscrbr_NPI", "NPI", "npi"])
    gnrc_col = find_column(columns, ["Gnrc_Name", "Generic Name"])
    clms_col = find_column(columns, ["Tot_Clms"])
    cost_col = find_column(columns, ["Tot_Drug_Cst", "Tot_Drug_Cost"])
    type_col = find_column(columns, ["Prscrbr_Type", "Prescriber_Type"])
    if npi_col is None or gnrc_col is None:
        raise ValueError("Could not resolve Part D columns for NPI / drug name.")
    return npi_col, gnrc_col, clms_col, cost_col, type_col


def infer_open_payments_npi_column(columns: Sequence[str]) -> str:
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


def npi_str_from_cell(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        s = str(value).strip()
        return "".join(c for c in s if c.isdigit())


@dataclass
class PartDAgg:
    rows: int = 0
    drugs: Set[str] = field(default_factory=set)
    claim_units: float = 0.0
    drug_cost: float = 0.0
    specialty_rows: Counter[str] = field(default_factory=Counter)


def read_top_location_keys(counts_path: Path, k: int) -> List[Tuple[str, int]]:
    """Read first k rows (file is sorted descending by count)."""
    lines = counts_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return []
    out: List[Tuple[str, int]] = []
    for line in lines[1 : k + 1]:
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        key, cnt_s = parts[0], parts[1]
        out.append((key, int(cnt_s)))
    return out


def load_npis_by_location(
    map_path: Path,
    keys: Set[str],
    max_lines: Optional[int] = None,
    max_npi_per_key: Optional[int] = None,
) -> Dict[str, List[str]]:
    """Single pass: location_key -> list of NPIs (order preserved).

    If max_lines or max_npi_per_key is set, may stop early (dry-run / faster pools).
    """
    result: Dict[str, List[str]] = defaultdict(list)
    n_lines = 0
    with map_path.open(encoding="utf-8") as handle:
        handle.readline()
        for line in handle:
            n_lines += 1
            if max_lines is not None and n_lines > max_lines:
                break
            line = line.rstrip("\n")
            if not line:
                continue
            tab = line.find("\t")
            if tab < 0:
                continue
            npi = line[:tab].strip()
            key = line[tab + 1 :]
            if key in keys:
                if max_npi_per_key is None or len(result[key]) < max_npi_per_key:
                    result[key].append(npi)
            if max_npi_per_key is not None and keys:
                if all(
                    k in result and len(result[k]) >= max_npi_per_key for k in keys
                ):
                    break
    return {k: result[k] for k in keys if k in result}


def aggregate_part_d(
    path: Path,
    candidates: Set[str],
    max_chunks: Optional[int] = None,
) -> Dict[str, PartDAgg]:
    """Chunked pass; only updates NPIs in candidates. max_chunks limits rows read (dry-run)."""
    reader = chunked_csv_reader(path)
    first = next(reader)
    npi_c, gnrc_c, clms_c, cost_c, type_c = infer_partd_columns(first.columns)
    acc: DefaultDict[str, PartDAgg] = defaultdict(PartDAgg)

    def consume(chunk: pd.DataFrame) -> None:
        npi_series = chunk[npi_c].map(npi_str_from_cell)
        mask = npi_series.isin(candidates)
        if not mask.any():
            return
        sub = chunk.loc[mask].copy()
        sub["_npi"] = npi_series[mask]
        for npi, g in sub.groupby("_npi", sort=False):
            a = acc[npi]
            a.rows += len(g)
            for raw in g[gnrc_c].dropna().astype(str):
                t = raw.strip()
                if t and t.lower() != "nan":
                    a.drugs.add(t)
            if clms_c is not None and clms_c in g.columns:
                a.claim_units += pd.to_numeric(g[clms_c], errors="coerce").fillna(0).sum()
            if cost_c is not None and cost_c in g.columns:
                a.drug_cost += pd.to_numeric(g[cost_c], errors="coerce").fillna(0).sum()
            if type_c is not None and type_c in g.columns:
                for raw in g[type_c].dropna().astype(str):
                    a.specialty_rows[raw.strip()] += 1

    consume(first)
    chunks_done = 1
    if max_chunks is not None and chunks_done >= max_chunks:
        return dict(acc)
    for idx, chunk in enumerate(reader, start=2):
        if max_chunks is not None and chunks_done >= max_chunks:
            break
        print_chunk_progress("Part D", idx)
        consume(chunk)
        chunks_done += 1

    return dict(acc)


def aggregate_open_payments(
    paths: List[Path],
    candidates: Set[str],
    max_chunks_per_file: Optional[int] = None,
) -> Dict[str, int]:
    """Total Open Payments rows per NPI (any payment type)."""
    counts: DefaultDict[str, int] = defaultdict(int)
    for pay_path in paths:
        reader = chunked_csv_reader(pay_path)
        first = next(reader)
        npi_c = infer_open_payments_npi_column(first.columns)

        def consume(chunk: pd.DataFrame) -> None:
            npi_series = chunk[npi_c].map(npi_str_from_cell)
            mask = npi_series.isin(candidates)
            if not mask.any():
                return
            for npi, c in npi_series[mask].value_counts().items():
                counts[str(npi)] += int(c)

        consume(first)
        chunks_done = 1
        if max_chunks_per_file is not None and chunks_done >= max_chunks_per_file:
            continue
        for idx, chunk in enumerate(reader, start=2):
            if max_chunks_per_file is not None and chunks_done >= max_chunks_per_file:
                break
            print_chunk_progress(f"Open Payments {pay_path.name}", idx)
            consume(chunk)
            chunks_done += 1
    return dict(counts)


def dominant_specialty(agg: PartDAgg) -> str:
    if not agg.specialty_rows:
        return ""
    return agg.specialty_rows.most_common(1)[0][0]


def specialty_boost(agg: PartDAgg) -> float:
    """Small tilt toward physician-heavy panels (optional nuance for sales targeting)."""
    dom = dominant_specialty(agg)
    if not dom:
        return 0.0
    if dom in PHYSICIAN_FRIENDLY_TYPES:
        return 0.25
    if dom in ("Nurse Practitioner", "Physician Assistant"):
        return -0.05
    return 0.0


def pharma_sales_score(agg: PartDAgg, op_rows: int) -> float:
    nd = len(agg.drugs)
    cost = max(agg.drug_cost, 0.0)
    return (
        2.0 * math.log1p(agg.rows)
        + 1.5 * math.log1p(nd)
        + 0.4 * math.log1p(cost)
        + 0.3 * math.log1p(agg.claim_units)
        + 2.0 * math.log1p(op_rows)
        + (1.2 if op_rows > 0 else 0.0)
        + specialty_boost(agg)
    )


def select_sample(
    location_order: List[Tuple[str, int]],
    npis_by_location: Dict[str, List[str]],
    part_d: Dict[str, PartDAgg],
    op_counts: Dict[str, int],
    target_n: int,
    min_part_d_rows: int,
) -> Tuple[List[Dict[str, object]], List[str]]:
    """Greedy fill from largest locations first."""
    selected: List[Dict[str, object]] = []
    sources: List[str] = []

    for loc_key, loc_cnt in location_order:
        if len(selected) >= target_n:
            break
        npis = npis_by_location.get(loc_key, [])
        before_n = len(selected)
        scored: List[Tuple[float, str, PartDAgg, int]] = []
        for npi in npis:
            agg = part_d.get(npi)
            if agg is None or agg.rows < min_part_d_rows:
                continue
            opr = op_counts.get(npi, 0)
            s = pharma_sales_score(agg, opr)
            scored.append((s, npi, agg, opr))
        scored.sort(key=lambda x: -x[0])
        need = target_n - len(selected)
        for s, npi, agg, opr in scored[:need]:
            selected.append(
                {
                    "npi": npi,
                    "practice_location_key": loc_key,
                    "part_d_rows": agg.rows,
                    "part_d_distinct_generics": len(agg.drugs),
                    "part_d_tot_drug_cost": round(agg.drug_cost, 2),
                    "open_payments_rows": opr,
                    "dominant_prscrbr_type": dominant_specialty(agg),
                    "pharma_sales_score": round(s, 4),
                }
            )
        if len(selected) > before_n:
            sources.append(loc_key)

    return selected, sources


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample co-located prescribers with strong Part D + Open Payments signal."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Fast smoke test: only top-ranked locations by NPI count, capped map scan, "
            "first N chunks of Part D and Open Payments. Does not read every row."
        ),
    )
    parser.add_argument(
        "--top-locations",
        type=int,
        default=None,
        help="Top N sites from practice_location_npi_counts.tsv (default: 3 dry-run, 15 full).",
    )
    parser.add_argument(
        "--target-n",
        type=int,
        default=None,
        help="Target cohort size (default: 30 dry-run, 100 full).",
    )
    parser.add_argument(
        "--min-part-d-rows",
        type=int,
        default=None,
        help="Min Part D rows to qualify (default: 2 dry-run, 8 full).",
    )
    parser.add_argument(
        "--dry-run-part-d-chunks",
        type=int,
        default=6,
        help="Dry-run only: max Part D chunks to read (~100k rows each).",
    )
    parser.add_argument(
        "--dry-run-op-chunks",
        type=int,
        default=2,
        help="Dry-run only: max chunks per Open Payments file.",
    )
    parser.add_argument(
        "--dry-run-map-max-lines",
        type=int,
        default=1_200_000,
        help="Dry-run only: stop reading NPI map after this many data lines (after header).",
    )
    parser.add_argument(
        "--dry-run-max-npi-per-location",
        type=int,
        default=500,
        help="Dry-run only: cap NPIs per location key (stop map scan early when all keys full).",
    )
    args = parser.parse_args()
    dry = args.dry_run

    if dry:
        top_n = args.top_locations if args.top_locations is not None else 3
        target_n = args.target_n if args.target_n is not None else 30
        min_row = args.min_part_d_rows if args.min_part_d_rows is not None else 2
    else:
        top_n = args.top_locations if args.top_locations is not None else 15
        target_n = args.target_n if args.target_n is not None else 100
        min_row = args.min_part_d_rows if args.min_part_d_rows is not None else 8

    if not LOCATION_COUNTS_PATH.is_file():
        raise SystemExit(
            f"Missing {LOCATION_COUNTS_PATH}. Run scripts/02_group_npis_by_practice_location.py first."
        )
    if not NPI_LOCATION_MAP_PATH.is_file():
        raise SystemExit(
            f"Missing {NPI_LOCATION_MAP_PATH}. Run scripts/02_group_npis_by_practice_location.py "
            "without --summary-only."
        )

    top_locs = read_top_location_keys(LOCATION_COUNTS_PATH, top_n)
    if not top_locs:
        raise SystemExit("No rows in practice_location_npi_counts.tsv.")

    key_set = {k for k, _ in top_locs}
    print(f"Loading NPI lists for top {len(key_set)} locations from map file...")
    if dry:
        print(
            f"  [dry-run] map max_lines={args.dry_run_map_max_lines:,}, "
            f"max_npi_per_key={args.dry_run_max_npi_per_location}, "
            f"Part D chunks<={args.dry_run_part_d_chunks}, OP chunks/file<={args.dry_run_op_chunks}"
        )
        npis_by_loc = load_npis_by_location(
            NPI_LOCATION_MAP_PATH,
            key_set,
            max_lines=args.dry_run_map_max_lines,
            max_npi_per_key=args.dry_run_max_npi_per_location,
        )
    else:
        npis_by_loc = load_npis_by_location(NPI_LOCATION_MAP_PATH, key_set)
    total_pool = sum(len(v) for v in npis_by_loc.values())
    print(f"Total NPIs in pool (top {top_n} sites): {total_pool:,}")

    candidates: Set[str] = set()
    for npi_list in npis_by_loc.values():
        candidates.update(npi_list)
    print(f"Distinct NPIs: {len(candidates):,}")

    part_d_path = discover_part_d_2023_path()
    op_paths = discover_open_payments_paths()
    print(f"Part D file: {part_d_path.relative_to(PROJECT_ROOT)}")
    print(f"Open Payments files: {len(op_paths)}")

    print("Aggregating Part D...")
    part_d = aggregate_part_d(
        part_d_path,
        candidates,
        max_chunks=args.dry_run_part_d_chunks if dry else None,
    )
    print(f"NPIs with any Part D row in file: {len(part_d):,}")

    print("Aggregating Open Payments...")
    op_counts = aggregate_open_payments(
        op_paths,
        candidates,
        max_chunks_per_file=args.dry_run_op_chunks if dry else None,
    )

    location_order = list(top_locs)
    selected, source_keys = select_sample(
        location_order,
        npis_by_loc,
        part_d,
        op_counts,
        target_n=target_n,
        min_part_d_rows=min_row,
    )

    selected.sort(key=lambda r: -float(r["pharma_sales_score"]))
    for i, row in enumerate(selected, start=1):
        row["cohort_rank"] = i

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(selected)
    out_path = SAMPLE_OUTPUT_DRY_TSV if dry else SAMPLE_OUTPUT_TSV
    report_path = SAMPLE_REPORT_DRY_PATH if dry else SAMPLE_REPORT_PATH
    out_df.to_csv(out_path, sep="\t", index=False)

    report_lines = [
        "=== PHARMA SALES–RELEVANT COHORT SAMPLE ===\n",
        f"Mode: {'DRY-RUN (partial data; pipeline check only)' if dry else 'FULL'}\n",
        f"Target size: {target_n}; selected: {len(selected)}\n",
        f"Top locations considered: {top_n}; min Part D rows: {min_row}\n",
        "\nRationale (for a pharma field / access team):\n",
        "- Locations are ranked by NPI count (larger sites → more KOL density in expectation).\n",
        "- Part D: row count proxies prescribing activity; distinct generics proxy breadth.\n",
        "- Tot_Drug_Cst (Part D) proxies economic scale of prescribing (CMS limitations apply).\n",
        "- Open Payments: Sunshine exposure indicates prior industry touchpoints.\n",
        "- Practice location key groups NPIs at the same normalized address (same site).\n",
        "\nLocations contributing to this sample (in order tried):\n",
    ]
    for k in source_keys:
        report_lines.append(f"  - {k[:120]}{'...' if len(k) > 120 else ''}\n")

    if len(selected) < target_n:
        report_lines.append(
            f"\nWARNING: Only {len(selected)} NPIs met criteria. "
            "Increase --top-locations, lower --min-part-d-rows, or widen the location pool.\n"
        )
    if dry:
        report_lines.append(
            "\nDry-run does not scan all Part D / Open Payments rows; "
            "scores are incomplete for production use.\n"
        )

    report_text = "".join(report_lines)
    report_path.write_text(report_text, encoding="utf-8")

    print(report_text)
    print(f"Wrote: {out_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote: {report_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
