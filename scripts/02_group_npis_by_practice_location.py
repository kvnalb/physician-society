"""Group NPPES type-1 providers by normalized practice address for network sampling.

Streams the NPI registry in chunks, builds a stable practice-location key from
practice address fields, counts NPIs per location (sorted descending), and
optionally writes an NPI-to-location map for downstream rich-data selection.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Iterator, Optional, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
LOCATION_COUNTS_PATH = OUTPUT_DIR / "practice_location_npi_counts.tsv"
NPI_LOCATION_MAP_PATH = OUTPUT_DIR / "npi_practice_location.tsv"

CHUNK_SIZE = 100_000


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


def discover_npidata_path() -> Path:
    """Return path to NPPES npidata CSV under data/raw."""
    candidates: list[Path] = []
    for path in sorted(RAW_DATA_DIR.rglob("*.csv"), key=lambda p: str(p)):
        name = path.name.lower()
        if "npidata" in name:
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"No npidata CSV found under {RAW_DATA_DIR}")
    return candidates[0]


def entity_type_is_individual(series: pd.Series) -> pd.Series:
    """Return mask for NPPES Entity Type Code == 1 (handles float 1.0 from CSV)."""
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric == 1


def normalize_location_token(value: object) -> str:
    """Uppercase, strip, collapse internal whitespace for address matching."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "<na>"}:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.upper()


def normalize_zip_code(value: object) -> str:
    """Extract up to 5 leading digits for US ZIP (handles float postal codes)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if pd.isna(value):
            return ""
        digits = str(int(float(value)))
    else:
        digits = "".join(c for c in str(value).strip() if c.isdigit())
    if len(digits) >= 5:
        return digits[:5]
    if len(digits) >= 3:
        return digits.zfill(5) if len(digits) == 4 else digits
    return ""


def practice_location_key_series(
    line1: pd.Series,
    city: pd.Series,
    state: pd.Series,
    zip5: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Vectorized key; returns (key series, boolean mask of valid rows)."""
    st2 = state.where(state.str.len() == 2, pd.NA)
    mask = (
        (line1 != "")
        & (city != "")
        & st2.notna()
        & (zip5 != "")
    )
    key = line1 + "|" + city + "|" + st2.fillna("") + "|" + zip5
    key = key.where(mask, pd.NA)
    return key, mask


def infer_nppes_columns(columns: Sequence[str]) -> tuple[str, str, str, str, str, str]:
    """Resolve NPI, entity type, and practice address column names."""
    npi_col = find_column(columns, ["NPI", "npi"])
    entity_col = find_column(columns, ["Entity Type Code", "Entity_Type_Code"])
    line1_col = find_column(
        columns,
        ["Provider First Line Business Practice Location Address"],
    )
    city_col = find_column(
        columns,
        ["Provider Business Practice Location Address City Name"],
    )
    state_col = find_column(
        columns,
        ["Provider Business Practice Location Address State Name"],
    )
    zip_col = find_column(
        columns,
        ["Provider Business Practice Location Address Postal Code"],
    )
    missing = [
        name
        for name, col in [
            ("NPI", npi_col),
            ("Entity Type Code", entity_col),
            ("Practice line 1", line1_col),
            ("Practice city", city_col),
            ("Practice state", state_col),
            ("Practice ZIP", zip_col),
        ]
        if col is None
    ]
    if missing:
        raise ValueError(f"Missing NPPES columns: {missing}")
    assert npi_col and entity_col and line1_col and city_col and state_col and zip_col
    return npi_col, entity_col, line1_col, city_col, state_col, zip_col


def clean_npi_series(series: pd.Series) -> pd.Series:
    """Normalize NPI as stripped string; invalid -> NA."""
    s = series.astype("string").str.strip()
    s = s.replace({"": pd.NA, "nan": pd.NA, "<NA>": pd.NA, "None": pd.NA})
    return s


def run_grouping(summary_only: bool) -> None:
    """Single streamed pass: count NPIs per location; optionally write NPI map."""
    npidata_path = discover_npidata_path()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"NPPES file: {npidata_path.relative_to(PROJECT_ROOT)}")
    print(f"Chunk size: {CHUNK_SIZE:,}")

    reader = chunked_csv_reader(npidata_path)
    first = next(reader)
    cols = infer_nppes_columns(first.columns)
    npi_c, et_c, l1_c, cy_c, st_c, zp_c = cols

    location_counts: Counter[str] = Counter()
    eligible_rows = 0
    type1_rows = 0

    map_file = None
    if not summary_only:
        map_file = open(NPI_LOCATION_MAP_PATH, "w", encoding="utf-8")
        map_file.write("npi\tpractice_location_key\n")

    def process_chunk(chunk: pd.DataFrame, chunk_idx: int) -> None:
        nonlocal eligible_rows, type1_rows
        print_chunk_progress("Practice locations", chunk_idx)
        is_t1 = entity_type_is_individual(chunk[et_c])
        type1_rows += int(is_t1.sum())
        chunk[npi_c] = clean_npi_series(chunk[npi_c])

        l1 = chunk[l1_c].map(normalize_location_token).astype("string")
        cy = chunk[cy_c].map(normalize_location_token).astype("string")
        st = chunk[st_c].map(normalize_location_token).astype("string")
        zp = chunk[zp_c].map(normalize_zip_code).astype("string")
        ploc, loc_ok = practice_location_key_series(l1, cy, st, zp)

        row_ok = is_t1 & chunk[npi_c].notna() & loc_ok
        eligible_rows += int(row_ok.sum())
        if not row_ok.any():
            return

        keys_series = ploc[row_ok]
        counts = keys_series.value_counts()
        location_counts.update(counts.to_dict())

        if map_file is not None:
            out = pd.DataFrame(
                {
                    "npi": chunk.loc[row_ok, npi_c].astype(str),
                    "practice_location_key": keys_series.astype(str),
                }
            )
            out.to_csv(map_file, sep="\t", header=False, index=False, lineterminator="\n")

    process_chunk(first, 1)
    for idx, chunk in enumerate(reader, start=2):
        process_chunk(chunk, idx)

    if map_file is not None:
        map_file.close()

    sorted_locs = location_counts.most_common()
    LOCATION_COUNTS_PATH.write_text(
        "practice_location_key\tnpi_count\n"
        + "\n".join(f"{k}\t{c}" for k, c in sorted_locs)
        + "\n",
        encoding="utf-8",
    )

    print(f"\nType-1 rows seen: {type1_rows:,}")
    print(f"Type-1 rows with valid NPI + full practice address key: {eligible_rows:,}")
    print(f"Distinct practice locations: {len(location_counts):,}")
    print(f"Location counts (sorted): {LOCATION_COUNTS_PATH.relative_to(PROJECT_ROOT)}")
    if not summary_only:
        print(f"NPI to location map: {NPI_LOCATION_MAP_PATH.relative_to(PROJECT_ROOT)}")
    print("\nTop 25 locations by NPI count:")
    for key, c in sorted_locs[:25]:
        print(f"  {c:6d}  {key[:100]}{'...' if len(key) > 100 else ''}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Group NPPES type-1 NPIs by normalized practice address."
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only write practice_location_npi_counts.tsv (skip npi_practice_location.tsv).",
    )
    args = parser.parse_args()
    run_grouping(summary_only=args.summary_only)


if __name__ == "__main__":
    main()
