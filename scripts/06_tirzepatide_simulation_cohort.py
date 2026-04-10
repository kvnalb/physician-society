"""Build a stratified physician cohort for tirzepatide (2022→2023) LLM simulation.

Implements the user's Step 1–6 logic against NPPES, Part D 2022, and Open Payments
2022. The final cohort is restricted to NPIs that also appear in Part D 2023 so
2023 metrics are available as ground truth; those columns are merged onto each row.

Fast re-runs: use --use-cache after the first full pass; caches live under
data/processed/ (gitignored). Clear with --refresh-cache.
"""

from __future__ import annotations

import argparse
import gzip
import pickle
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterator, List, Optional, Sequence, Set, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"

CHUNK_SIZE = 100_000

# Step 1: high-diabetes metros (state ∩ city). Six cities only — balances pool size
# with even-ish coverage across CA / TX / NY / FL (drops Phoenix, Austin, Tampa, SF).
# After changing TARGET_* or specialty filters, rebuild caches: --refresh-cache
TARGET_STATES = {"CA", "TX", "NY", "FL"}
TARGET_CITIES = {
    "HOUSTON",
    "LOS ANGELES",
    "NEW YORK",
    "MIAMI",
    "DALLAS",
    "SAN DIEGO",
}

# Diabetes-focused prescribers only (exclude Cardiology and other NUCC buckets).
DIABETES_SPECIALTIES = frozenset({"Endocrinology", "Internal Medicine", "Family Medicine"})

ALLOWED_CREDENTIALS = {"MD", "DO", "MBBS", "M.D.", "D.O."}


def normalize_city(s: object) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    t = str(s).strip().upper()
    t = re.sub(r"\s+", " ", t)
    return t


def normalize_header(header: str) -> str:
    return "".join(ch for ch in header.lower().strip() if ch.isalnum())


def find_column(columns: Sequence[str], names: Sequence[str]) -> Optional[str]:
    norm = {normalize_header(c): c for c in columns}
    for n in names:
        if n in columns:
            return n
        m = norm.get(normalize_header(n))
        if m:
            return m
    return None


def chunked_csv_reader(path: Path, chunksize: int = CHUNK_SIZE) -> Iterator[pd.DataFrame]:
    for encoding in ("utf-8", "latin-1"):
        try:
            yield from pd.read_csv(path, chunksize=chunksize, low_memory=False, encoding=encoding)
            return
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("utf-8", b"", 0, 1, str(path))


def npi_str(v: object) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    try:
        return str(int(float(v)))
    except (TypeError, ValueError):
        return "".join(c for c in str(v) if c.isdigit())


def discover_npidata() -> Path:
    for p in sorted(RAW_DATA_DIR.rglob("*.csv"), key=lambda x: str(x)):
        if "npidata" in p.name.lower():
            return p
    raise FileNotFoundError("npidata CSV not found")


def discover_part_d_2022() -> Path:
    for p in sorted(RAW_DATA_DIR.rglob("*.csv"), key=lambda x: str(x)):
        rel = str(p).lower()
        if "endpoint" in rel or "npidata" in rel:
            continue
        if "2022" in rel and ("part_d" in rel or "prescribers" in rel or "medicare_part_d" in rel):
            return p
    raise FileNotFoundError("Part D 2022 CSV not found")


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


def discover_open_payments_2022() -> List[Path]:
    out: List[Path] = []
    for p in sorted(RAW_DATA_DIR.rglob("*.csv"), key=lambda x: str(x)):
        rel = str(p).lower()
        if "op_dtl" in p.name.lower() and "pgyr2022" in rel:
            out.append(p)
    if not out:
        raise FileNotFoundError("Open Payments 2022 OP_DTL CSVs not found")
    return sorted(out, key=str)


def taxonomy_specialty_label(code: object) -> str:
    """Map NUCC taxonomy code to coarse specialty (order matters)."""
    if code is None or (isinstance(code, float) and pd.isna(code)):
        return "Unknown"
    c = str(code).strip().upper()
    if not c:
        return "Unknown"
    if c.startswith("207RE"):
        return "Endocrinology"
    if c.startswith("207RA"):
        return "Cardiology"
    if c.startswith("207Q"):
        return "Family Medicine"
    if c.startswith("207R"):
        return "Internal Medicine"
    return "Other"


def credential_ok(text: object) -> bool:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return False
    t = str(text).strip().upper().replace(".", "")
    for a in ALLOWED_CREDENTIALS:
        if a.upper().replace(".", "") in t or t in a.upper():
            return True
    return bool(re.search(r"\b(MD|DO|MBBS)\b", t))


def is_active_deactivation(deact: object) -> bool:
    if deact is None or (isinstance(deact, float) and pd.isna(deact)):
        return True
    s = str(deact).strip().lower()
    return s in {"", "nan", "nat", "none"}


def gnrc_lower(s: object) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    return str(s).lower()


def is_diabetes_row(gnrc: str, brnd: str) -> bool:
    g = gnrc_lower(gnrc)
    b = gnrc_lower(brnd)
    blob = g + " " + b
    keys = (
        "diabet",
        "metformin",
        "insulin",
        "glucose",
        "glp-1",
        "glp1",
        "semaglutide",
        "tirzepatide",
        "liraglutide",
        "dulaglutide",
        "exenatide",
        "lixisenatide",
        "canagliflozin",
        "dapagliflozin",
        "empagliflozin",
        "sitagliptin",
        "linagliptin",
        "alogliptin",
        "saxagliptin",
        "pioglitazone",
        "rosiglitazone",
        "acarbose",
        "miglitol",
        "repaglinide",
        "nateglinide",
        "glimepiride",
        "glipizide",
        "glyburide",
        "tolbutamide",
        "chlorpropamide",
    )
    return any(k in blob for k in keys)


def is_glp1_row(gnrc: str, brnd: str) -> bool:
    blob = gnrc_lower(gnrc) + " " + gnrc_lower(brnd)
    keys = (
        "semaglutide",
        "tirzepatide",
        "liraglutide",
        "dulaglutide",
        "exenatide",
        "lixisenatide",
        "albiglutide",
        "glp-1",
        "glp1",
    )
    return any(k in blob for k in keys)


def is_tirzepatide_row(gnrc: str, brnd: str) -> bool:
    blob = gnrc_lower(gnrc) + " " + gnrc_lower(brnd)
    return "tirzepatide" in blob


def is_branded_row(brnd: object, gnrc: object) -> bool:
    b = gnrc_lower(brnd)
    g = gnrc_lower(gnrc)
    if not b or b == g:
        return False
    return True


def cache_path(name: str) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    return PROCESSED_DIR / f"{name}.pkl.gz"


def save_pickle_gz(path: Path, obj: Any) -> None:
    with gzip.open(path, "wb", compresslevel=6) as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle_gz(path: Path) -> Any:
    with gzip.open(path, "rb") as handle:
        return pickle.load(handle)


@dataclass
class PartDAgg:
    total_claims: float = 0.0
    total_benes: float = 0.0
    diabetes_claims: float = 0.0
    glp1_claims: float = 0.0
    tirz_claims: float = 0.0
    branded_rows: int = 0
    total_rows: int = 0
    diabetes_drugs: Set[str] = field(default_factory=set)


def part_d_ground_truth_metrics(a: PartDAgg) -> Dict[str, Any]:
    """Per-NPI Part D aggregates as 2023 ground-truth columns (same shape as 2022)."""
    tclaims = max(a.total_claims, 1e-9)
    diab_share = a.diabetes_claims / tclaims
    glp1_p = a.glp1_claims / max(a.diabetes_claims, 1e-9)
    branded_share = a.branded_rows / max(a.total_rows, 1)
    div = len(a.diabetes_drugs)
    return {
        "claims_2023": a.total_claims,
        "beneficiaries_2023": a.total_benes,
        "diabetes_share_2023": diab_share,
        "glp1_penetration_2023": glp1_p,
        "branded_share_2023": branded_share,
        "drug_diversity_2023": div,
        "has_tirzepatide_2023": 1 if a.tirz_claims > 0 else 0,
        "tirzepatide_claims_2023": a.tirz_claims,
        "part_d_rows_2023": a.total_rows,
    }


def merge_part_d_2023_ground_truth(
    master: pd.DataFrame,
    part_d_23: Dict[str, PartDAgg],
    min_rows_2023: int,
    min_beneficiaries_2023: float,
) -> pd.DataFrame:
    """Keep only NPIs with Part D 2023 presence (rows + beneficiaries) and append GT columns."""
    if master.empty:
        return master
    out_rows: List[Dict[str, Any]] = []
    for _, r in master.iterrows():
        npi = str(r["npi"])
        agg = part_d_23.get(npi)
        if not agg or agg.total_rows < min_rows_2023:
            continue
        if agg.total_benes < min_beneficiaries_2023:
            continue
        out_rows.append({**r.to_dict(), **part_d_ground_truth_metrics(agg)})
    return pd.DataFrame(out_rows)


def step1_nppes(
    path: Path,
    max_chunks: Optional[int],
) -> pd.DataFrame:
    """Filter NPPES to target geo + taxonomy + credentials + active."""
    reader = chunked_csv_reader(path)
    first = next(reader)
    cols = first.columns.tolist()
    npi_c = find_column(cols, ["NPI"])
    et_c = find_column(cols, ["Entity Type Code"])
    tax_c = find_column(cols, ["Healthcare Provider Taxonomy Code_1"])
    cred_c = find_column(cols, ["Provider Credential Text"])
    deact_c = find_column(cols, ["NPI Deactivation Date"])
    city_c = find_column(cols, ["Provider Business Practice Location Address City Name"])
    state_c = find_column(cols, ["Provider Business Practice Location Address State Name"])
    line1_c = find_column(cols, ["Provider First Line Business Practice Location Address"])
    sex_c = find_column(cols, ["Provider Sex Code"])

    if not all([npi_c, et_c, tax_c, cred_c, deact_c, city_c, state_c]):
        raise ValueError("Missing required NPPES columns.")

    rows: List[dict[str, Any]] = []
    chunks_done = 1

    print(
        f"  [NPPES] Scanning {path.name} (~10GB+; often 15–40+ min on a laptop). "
        "Progress prints every 10 chunks — not frozen.",
        flush=True,
    )

    def consume(chunk: pd.DataFrame) -> None:
        et = pd.to_numeric(chunk[et_c], errors="coerce")
        mask = et == 1
        chunk = chunk.loc[mask].copy()
        if chunk.empty:
            return
        chunk["_city"] = chunk[city_c].map(normalize_city)
        chunk["_state"] = chunk[state_c].map(lambda x: str(x).strip().upper() if pd.notna(x) else "")
        geo = chunk["_state"].isin(TARGET_STATES) & chunk["_city"].isin(TARGET_CITIES)
        chunk = chunk.loc[geo]
        if chunk.empty:
            return
        chunk["_tax"] = chunk[tax_c].map(taxonomy_specialty_label)
        spec_ok = chunk["_tax"].isin(DIABETES_SPECIALTIES)
        chunk = chunk.loc[spec_ok]
        if chunk.empty:
            return
        chunk["_cred_ok"] = chunk[cred_c].map(credential_ok)
        chunk = chunk.loc[chunk["_cred_ok"]]
        if chunk.empty:
            return
        chunk["_active"] = chunk[deact_c].map(is_active_deactivation)
        chunk = chunk.loc[chunk["_active"]]
        if chunk.empty:
            return
        cols_for_records = [npi_c, tax_c, cred_c, "_tax", "_city", "_state"]
        if line1_c:
            cols_for_records.append(line1_c)
        if sex_c:
            cols_for_records.append(sex_c)
        for rec in chunk[cols_for_records].to_dict("records"):
            npi = npi_str(rec[npi_c])
            if not npi:
                continue
            org = ""
            if line1_c:
                org = str(rec.get(line1_c, "") or "").strip()[:120]
            rows.append(
                {
                    "npi": npi,
                    "specialty": rec["_tax"],
                    "taxonomy_code": str(rec[tax_c]).strip() if pd.notna(rec.get(tax_c)) else "",
                    "organization_name": org or "INDIVIDUAL",
                    "city": rec["_city"].title() if rec["_city"] else "",
                    "state": rec["_state"],
                    "gender": str(rec[sex_c]).strip() if sex_c and pd.notna(rec.get(sex_c)) else "",
                    "credentials": str(rec[cred_c]).strip() if pd.notna(rec.get(cred_c)) else "",
                }
            )

    consume(first)
    print(
        f"  [NPPES] chunk {chunks_done} done, matched rows so far: {len(rows):,}",
        flush=True,
    )
    if max_chunks and chunks_done >= max_chunks:
        return pd.DataFrame(rows)
    for idx, chunk in enumerate(reader, start=2):
        if max_chunks and chunks_done >= max_chunks:
            break
        consume(chunk)
        chunks_done += 1
        if chunks_done % 10 == 0 or chunks_done <= 5:
            print(
                f"  [NPPES] chunk {chunks_done} done, matched rows so far: {len(rows):,}",
                flush=True,
            )
        if max_chunks and chunks_done >= max_chunks:
            break

    return pd.DataFrame(rows)


def step2_part_d_aggregate(
    path: Path,
    npi_allow: Set[str],
    max_chunks: Optional[int],
    *,
    progress_label: str = "Part D",
) -> Dict[str, PartDAgg]:
    reader = chunked_csv_reader(path)
    first = next(reader)
    c = first.columns
    npi_c = find_column(c, ["Prscrbr_NPI"])
    gnrc_c = find_column(c, ["Gnrc_Name"])
    brnd_c = find_column(c, ["Brnd_Name"])
    clms_c = find_column(c, ["Tot_Clms"])
    benes_c = find_column(c, ["Tot_Benes"])
    if not all([npi_c, gnrc_c, brnd_c, clms_c, benes_c]):
        raise ValueError("Missing Part D columns.")

    agg: DefaultDict[str, PartDAgg] = defaultdict(PartDAgg)
    chunks_done = 1
    print(
        f"  [{progress_label}] Scanning {path.name} (large file; progress every 10 chunks)…",
        flush=True,
    )

    def consume(chunk: pd.DataFrame) -> None:
        chunk["_npi"] = chunk[npi_c].map(npi_str)
        chunk = chunk[chunk["_npi"].isin(npi_allow)]
        if chunk.empty:
            return
        cols = ["_npi", gnrc_c, brnd_c, clms_c, benes_c]
        for rec in chunk[cols].to_dict("records"):
            n = rec["_npi"]
            a = agg[n]
            clm = float(rec[clms_c]) if pd.notna(rec[clms_c]) else 0.0
            ben = float(rec[benes_c]) if pd.notna(rec[benes_c]) else 0.0
            gn = rec[gnrc_c]
            br = rec[brnd_c]
            a.total_claims += clm
            a.total_benes += ben
            a.total_rows += 1
            if is_branded_row(br, gn):
                a.branded_rows += 1
            if is_diabetes_row(str(gn), str(br)):
                a.diabetes_claims += clm
                dg = gnrc_lower(gn)
                if dg:
                    a.diabetes_drugs.add(dg[:80])
            if is_glp1_row(str(gn), str(br)):
                a.glp1_claims += clm
            if is_tirzepatide_row(str(gn), str(br)):
                a.tirz_claims += clm

    consume(first)
    print(
        f"  [{progress_label}] chunk {chunks_done}, NPIs in agg so far: {len(agg):,}",
        flush=True,
    )
    if max_chunks and chunks_done >= max_chunks:
        return dict(agg)
    for _, chunk in enumerate(reader, start=2):
        if max_chunks and chunks_done >= max_chunks:
            break
        consume(chunk)
        chunks_done += 1
        if chunks_done % 10 == 0 or chunks_done <= 5:
            print(
                f"  [{progress_label}] chunk {chunks_done}, NPIs in agg so far: {len(agg):,}",
                flush=True,
            )
        if max_chunks and chunks_done >= max_chunks:
            break
    return dict(agg)


def step4_open_payments(
    paths: List[Path],
    npi_allow: Set[str],
    max_chunks: Optional[int],
) -> pd.DataFrame:
    """Per-NPI: total $, Novo Nordisk $, Eli Lilly $, research flag (from RSRCH file)."""
    totals: DefaultDict[str, float] = defaultdict(float)
    novo: DefaultDict[str, float] = defaultdict(float)
    lilly: DefaultDict[str, float] = defaultdict(float)
    research_any: Set[str] = set()

    for p in paths:
        is_rsch = "rsrch" in p.name.lower()
        reader = chunked_csv_reader(p)
        first = next(reader)
        cols = first.columns
        npi_c = find_column(cols, ["Covered_Recipient_NPI"])
        amt_c = find_column(cols, ["Total_Amount_of_Payment_USDollars"])
        mfg_c = find_column(
            cols,
            ["Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_Name", "Submitting_Applicable_Manufacturer_or_Applicable_GPO_Name"],
        )
        if not npi_c or not amt_c or not mfg_c:
            continue
        chunks_done = 1

        def consume(chunk: pd.DataFrame) -> None:
            chunk["_npi"] = chunk[npi_c].map(npi_str)
            chunk = chunk[chunk["_npi"].isin(npi_allow)]
            if chunk.empty:
                return
            for _, r in chunk.iterrows():
                n = r["_npi"]
                amt = float(r[amt_c]) if pd.notna(r[amt_c]) else 0.0
                mfg = str(r[mfg_c]).upper() if pd.notna(r[mfg_c]) else ""
                totals[n] += amt
                if "NOVO" in mfg or "NORDISK" in mfg:
                    novo[n] += amt
                if "LILLY" in mfg or "ELI LILLY" in mfg:
                    lilly[n] += amt
                if is_rsch and amt > 0:
                    research_any.add(n)

        consume(first)
        if max_chunks and chunks_done >= max_chunks:
            continue
        for _, chunk in enumerate(reader, start=2):
            if max_chunks and chunks_done >= max_chunks:
                break
            consume(chunk)
            chunks_done += 1
            if max_chunks and chunks_done >= max_chunks:
                break

    rows = []
    for n in totals:
        rows.append(
            {
                "npi": n,
                "total_payments_2022": totals[n],
                "novo_nordisk_payments": novo.get(n, 0.0),
                "eli_lilly_payments": lilly.get(n, 0.0),
                "has_research_payments": 1 if n in research_any else 0,
            }
        )
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["npi", "total_payments_2022", "novo_nordisk_payments", "eli_lilly_payments", "has_research_payments"]
    )


def classify_archetype(row: pd.Series) -> str:
    glp1_p = row["glp1_penetration_2022"]
    div = row["drug_diversity_2022"]
    branded = row["branded_share_2022"]
    spec = str(row["specialty"])

    if (
        glp1_p >= 0.20
        and div >= 15
        and branded >= 0.35
    ):
        return "Early_Adopter_Specialist"
    if 0.10 <= glp1_p <= 0.19 and 10 <= div <= 14:
        return "Mainstream_Specialist"
    if (
        ("Internal Medicine" in spec or "Family Medicine" in spec)
        and 0.05 <= glp1_p <= 0.15
        and div < 10
    ):
        return "Conservative_PCP"
    if glp1_p < 0.05 and branded < 0.25:
        return "Laggard"
    return "Unclassified"


def pharma_tier(row: pd.Series) -> str:
    t = row.get("total_payments_2022", 0.0) or 0.0
    novo = row.get("novo_nordisk_payments", 0.0) or 0.0
    lilly = row.get("eli_lilly_payments", 0.0) or 0.0
    if t >= 5000 and (novo > 0 or lilly > 0):
        return "High_Pharma_Engagement"
    if 1000 <= t <= 4999:
        return "Medium_Engagement"
    return "Low_Engagement"


def build_master_frame(
    nppes: pd.DataFrame,
    part_d: Dict[str, PartDAgg],
    op: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for _, n in nppes.iterrows():
        npi = n["npi"]
        if npi not in part_d:
            continue
        a = part_d[npi]
        if a.diabetes_claims < 50:
            continue
        tclaims = max(a.total_claims, 1e-9)
        diab_share = a.diabetes_claims / tclaims
        if diab_share < 0.15:
            continue
        if a.total_benes < 30:
            continue
        glp1_p = a.glp1_claims / max(a.diabetes_claims, 1e-9)
        branded_share = a.branded_rows / max(a.total_rows, 1)
        div = len(a.diabetes_drugs)

        op_row = op[op["npi"] == npi]
        if op_row.empty:
            tp, novo, lilly, res = 0.0, 0.0, 0.0, 0
        else:
            tp = float(op_row["total_payments_2022"].iloc[0])
            novo = float(op_row["novo_nordisk_payments"].iloc[0])
            lilly = float(op_row["eli_lilly_payments"].iloc[0])
            res = int(op_row["has_research_payments"].iloc[0])

        rows.append(
            {
                **n.to_dict(),
                "claims_2022": a.total_claims,
                "beneficiaries_2022": a.total_benes,
                "diabetes_share_2022": diab_share,
                "glp1_penetration_2022": glp1_p,
                "branded_share_2022": branded_share,
                "drug_diversity_2022": div,
                "has_tirzepatide_2022": 1 if a.tirz_claims > 0 else 0,
                "total_payments_2022": tp,
                "novo_nordisk_payments": novo,
                "eli_lilly_payments": lilly,
                "has_research_payments": res,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["adoption_archetype"] = df.apply(classify_archetype, axis=1)
    df["pharma_engagement_tier"] = df.apply(pharma_tier, axis=1)
    df["geo_cluster"] = df["state"] + "_" + df["city"].str.replace(" ", "_", regex=False)
    df["network_group"] = df.apply(
        lambda r: (
            r["organization_name"]
            if r["organization_name"]
            and "PRIVATE" not in r["organization_name"].upper()
            and "SOLO" not in r["organization_name"].upper()
            else f"{r['city']}_Independent"
        ),
        axis=1,
    )
    return df


def stratified_sample_100(df: pd.DataFrame) -> pd.DataFrame:
    """Approximate the SQL stratification; fill up to 100 rows."""
    if df.empty:
        return df
    df = df.copy()
    df = df[df["claims_2022"] >= 100]
    if df.empty:
        return df

    df["rank_within"] = df.groupby(["geo_cluster", "adoption_archetype", "specialty"])[
        "glp1_penetration_2022"
    ].rank(ascending=False, method="first")

    picked: List[Any] = []
    seen: Set[str] = set()

    def take_stratum(mask: pd.Series, max_from_stratum: int) -> None:
        sub = df.loc[mask].sort_values(
            ["glp1_penetration_2022", "branded_share_2022"], ascending=False
        )
        taken = 0
        for idx in sub.index:
            if len(picked) >= 100:
                return
            if taken >= max_from_stratum:
                return
            n = str(df.at[idx, "npi"])
            if n in seen:
                continue
            picked.append(idx)
            seen.add(n)
            taken += 1

    m1 = (
        (df["adoption_archetype"] == "Early_Adopter_Specialist")
        & (df["specialty"] == "Endocrinology")
        & (df["rank_within"] <= 2)
        & (df["pharma_engagement_tier"].isin(["High_Pharma_Engagement", "Medium_Engagement"]))
    )
    take_stratum(m1, 30)

    m2 = (
        (df["adoption_archetype"] == "Mainstream_Specialist")
        & (df["specialty"].isin(["Endocrinology", "Internal Medicine"]))
        & (df["rank_within"] <= 2)
    )
    take_stratum(m2, 25)

    m3 = (df["adoption_archetype"] == "Conservative_PCP") & (df["rank_within"] <= 3)
    take_stratum(m3, 25)

    m4 = (df["adoption_archetype"] == "Laggard") & (df["rank_within"] <= 1)
    take_stratum(m4, 15)

    m5 = (
        (df["pharma_engagement_tier"] == "High_Pharma_Engagement")
        & ((df["novo_nordisk_payments"] > 3000) | (df["eli_lilly_payments"] > 1500))
        & (df["rank_within"] == 1)
    )
    take_stratum(m5, 5)

    if len(picked) < 100:
        rest = df.index.difference(picked)
        for idx in (
            df.loc[rest]
            .sort_values(["glp1_penetration_2022", "branded_share_2022"], ascending=False)
            .index
        ):
            if len(picked) >= 100:
                break
            n = str(df.at[idx, "npi"])
            if n not in seen:
                picked.append(idx)
                seen.add(n)

    out = df.loc[picked].sort_values(
        ["geo_cluster", "adoption_archetype", "glp1_penetration_2022"],
        ascending=[True, True, False],
    )
    return out.head(100)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tirzepatide simulation cohort (2022 baseline).")
    parser.add_argument("--dry-run", action="store_true", help="Only first chunks of each file.")
    parser.add_argument("--dry-run-chunks", type=int, default=8)
    parser.add_argument("--use-cache", action="store_true", help="Load step caches from data/processed if present.")
    parser.add_argument("--refresh-cache", action="store_true", help="Rebuild caches (ignore existing).")
    parser.add_argument(
        "--min-part-d-rows-2023",
        type=int,
        default=1,
        metavar="N",
        help="Require at least N Part D 2023 detail rows per NPI for ground truth (default: 1).",
    )
    parser.add_argument(
        "--min-beneficiaries-2023",
        type=float,
        default=1.0,
        metavar="X",
        help="Require Part D 2023 Tot_Benes sum >= X per NPI (default: 1; use to drop zero-bene rows).",
    )
    args = parser.parse_args()
    max_c = args.dry_run_chunks if args.dry_run else None

    npi_path = discover_npidata()
    pd22 = discover_part_d_2022()
    pd23 = discover_part_d_2023_path()
    op22 = discover_open_payments_2022()

    # v2 suffix invalidates pickles when metro/specialty/GT filters change
    c1 = cache_path("tirz_s1_nppes_v2")
    c2 = cache_path("tirz_s2_partd_agg_v2")

    if args.use_cache and not args.refresh_cache and c1.is_file() and c2.is_file():
        print("Loading caches from data/processed/ …")
        nppes = load_pickle_gz(c1)
        part_d = load_pickle_gz(c2)
    else:
        print("Step 1: NPPES filter …")
        nppes = step1_nppes(npi_path, max_c)
        print(f"  NPPES rows after filter: {len(nppes):,}")
        allow = set(nppes["npi"].astype(str))
        print("Step 2: Part D 2022 aggregate …")
        part_d = step2_part_d_aggregate(pd22, allow, max_c, progress_label="Part D 2022")
        print(f"  NPIs with Part D rows: {len(part_d):,}")
        if not args.dry_run:
            save_pickle_gz(c1, nppes)
            save_pickle_gz(c2, part_d)
            print(f"  Wrote caches: {c1.name}, {c2.name}")

    print("Step 4: Open Payments 2022 …")
    allow = set(nppes["npi"].astype(str))
    op_df = step4_open_payments(op22, allow, max_c)
    print(f"  NPIs with payment rows: {len(op_df):,}")

    print("Step 5–6: Merge 2022, archetypes …")
    master = build_master_frame(nppes, part_d, op_df)
    print(f"  After prescribing + diabetes filters (2022): {len(master):,}")

    cohort_npis = set(master["npi"].astype(str))
    print("Part D 2023: aggregate for cohort candidates (ground truth) …")
    part_d_23 = step2_part_d_aggregate(pd23, cohort_npis, max_c, progress_label="Part D 2023")
    print(f"  NPIs with Part D 2023 rows (among cohort candidates): {len(part_d_23):,}")
    master = merge_part_d_2023_ground_truth(
        master,
        part_d_23,
        args.min_part_d_rows_2023,
        args.min_beneficiaries_2023,
    )
    print(
        f"  After Part D 2023 GT (min_rows={args.min_part_d_rows_2023}, "
        f"min_benes={args.min_beneficiaries_2023}): {len(master):,}"
    )
    if master.empty:
        raise SystemExit(
            "No NPIs left after Part D 2023 ground-truth filter. "
            "Check data/raw, or lower --min-part-d-rows-2023 / --min-beneficiaries-2023."
        )

    print("Stratified sample …")
    sample = stratified_sample_100(master)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_tsv = OUTPUT_DIR / "tirzepatide_simulation_cohort_100.tsv"
    sample.to_csv(out_tsv, sep="\t", index=False)
    print(f"Wrote {out_tsv.relative_to(PROJECT_ROOT)} (n={len(sample)})")

    rep = OUTPUT_DIR / "tirzepatide_simulation_cohort_report.txt"
    rep.write_text(
        "Tirzepatide LLM simulation cohort\n"
        f"Mode: {'DRY-RUN' if args.dry_run else 'FULL'}\n"
        f"Final sample size: {len(sample)}\n"
        "\nFast re-runs (this script):\n"
        "- After one FULL run, caches are written to data/processed/: "
        "tirz_s1_nppes_v2.pkl.gz, tirz_s2_partd_agg_v2.pkl.gz\n"
        "- Cohort: six target metros (CA/TX/NY/FL); specialties Endo/IM/FM only (no cardiology).\n"
        "- Re-run with: --use-cache (skip NPPES + Part D 2022 rescans; Open Payments + Part D 2023 still stream)\n"
        "- --refresh-cache forces rebuild\n"
        "- Final sample: Part D 2023 rows + min beneficiaries (see flags); TSV includes 2023 GT columns.\n"
        "- --min-part-d-rows-2023 (default 1), --min-beneficiaries-2023 (default 1.0).\n"
        "\nOther speedups (general):\n"
        "- DuckDB or Polars scan CSV with SQL/predicate pushdown\n"
        "- Export hot columns to Parquet after first pass; read Parquet only\n"
        "- Restrict usecols= in pandas to needed fields only\n"
        "\n",
        encoding="utf-8",
    )
    print(f"Wrote {rep.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
