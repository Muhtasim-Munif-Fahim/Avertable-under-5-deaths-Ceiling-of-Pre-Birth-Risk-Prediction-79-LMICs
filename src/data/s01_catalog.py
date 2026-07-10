"""
00_catalog_datasets.py — DHS Dataset Cataloguing
=================================================

Scans the raw DHS download directory, parses standardized DHS filenames,
and produces a structured catalog of all available Birth Recode datasets.

DHS Filename Convention
-----------------------
  XXBRYYZZ.DTA
  XX   = 2-letter country code (ISO-like, DHS-specific)
  BR   = Birth Recode (file type)
  YY   = DHS phase number (e.g., 7 = DHS-VII, 81 = DHS-VIII wave 1)
  ZZ   = suffix (DT = Stata dataset)

Outputs
-------
  - dataset_catalog.parquet : Structured catalog with metadata
  - dataset_catalog.csv     : Human-readable copy
  - catalog_summary.txt     : Summary statistics

Decision References
-------------------
  - D002: India state-level files flagged as is_subnational=True
  - D004: Lists variables available per dataset for extraction planning

References
----------
  - DHS Recode Manual (v7): https://dhsprogram.com/publications/publication-dhsg4-dhs-questionnaires-and-manuals.cfm
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

import pyreadstat

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.paths import RAW_DHS_DIR, CATALOG_DIR, CATALOG_PARQUET, CATALOG_CSV, LOG_DIR

LOG_FILE = LOG_DIR / "00_catalog.log"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# India state-level 2-letter prefixes (Decision D002)
# These are sub-national files from NFHS-2 (IABR42)
INDIA_STATE_PREFIXES = {
    "AP", "AR", "AS", "BH", "DL", "GJ", "GO", "HP", "HR", "JM",
    "KA", "KE", "MG", "MH", "MN", "MP", "MZ", "NA", "OR", "PJ",
    "RJ", "SK", "TN", "TR", "UP", "WB",
}

# DHS country code → ISO-3166 name mapping (most common BR countries)
# Source: DHS Recode Manual Appendix
DHS_COUNTRY_NAMES: dict[str, str] = {
    "AF": "Afghanistan", "AL": "Albania", "AM": "Armenia",
    "AO": "Angola", "AZ": "Azerbaijan", "BD": "Bangladesh",
    "BF": "Burkina Faso", "BJ": "Benin", "BO": "Bolivia",
    "BR": "Brazil", "BU": "Burundi", "CD": "DR Congo",
    "CF": "Central African Republic", "CG": "Congo",
    "CI": "Cote d'Ivoire", "CM": "Cameroon", "CO": "Colombia",
    "DR": "Dominican Republic", "EC": "Ecuador", "EG": "Egypt",
    "ES": "El Salvador", "ET": "Ethiopia", "GA": "Gabon",
    "GH": "Ghana", "GM": "Gambia", "GN": "Guinea",
    "GU": "Guatemala", "GY": "Guyana", "HN": "Honduras",
    "HT": "Haiti", "IA": "India", "ID": "Indonesia",
    "JO": "Jordan", "KE": "Kenya", "KH": "Cambodia",
    "KK": "Kazakhstan", "KM": "Comoros", "KY": "Kyrgyz Republic",
    "LB": "Liberia", "LK": "Sri Lanka", "LS": "Lesotho",
    "MA": "Morocco", "MB": "Moldova", "MD": "Madagascar",
    "ML": "Mali", "MM": "Myanmar", "MR": "Mauritania",
    "MV": "Maldives", "MW": "Malawi", "MX": "Mexico",
    "MZ": "Mozambique", "NC": "Nicaragua", "NG": "Nigeria",
    "NI": "Niger", "NM": "Namibia", "NP": "Nepal",
    "OS": "Nigeria (Ondo State)", "PE": "Peru", "PG": "Papua New Guinea",
    "PH": "Philippines", "PK": "Pakistan", "PY": "Paraguay",
    "RW": "Rwanda", "SD": "Sudan", "SL": "Sierra Leone",
    "SN": "Senegal", "ST": "Sao Tome and Principe",
    "SZ": "Eswatini", "TD": "Chad", "TG": "Togo",
    "TH": "Thailand", "TJ": "Tajikistan", "TL": "Timor-Leste",
    "TN": "Tunisia", "TR": "Turkey", "TT": "Trinidad and Tobago",
    "TZ": "Tanzania", "UA": "Ukraine", "UG": "Uganda",
    "UZ": "Uzbekistan", "VN": "Vietnam", "YE": "Yemen",
    "ZA": "South Africa", "ZM": "Zambia", "ZW": "Zimbabwe",
}

# WHO regions for analytical grouping
DHS_TO_WHO_REGION: dict[str, str] = {
    # Sub-Saharan Africa
    "AO": "SSA", "BF": "SSA", "BJ": "SSA", "BU": "SSA",
    "CD": "SSA", "CF": "SSA", "CG": "SSA", "CI": "SSA",
    "CM": "SSA", "ET": "SSA", "GA": "SSA", "GH": "SSA",
    "GM": "SSA", "GN": "SSA", "KE": "SSA", "KM": "SSA",
    "LB": "SSA", "LS": "SSA", "MD": "SSA", "ML": "SSA",
    "MR": "SSA", "MW": "SSA", "MZ": "SSA", "NI": "SSA",
    "NM": "SSA", "NG": "SSA", "OS": "SSA", "RW": "SSA",
    "SD": "SSA", "SL": "SSA", "SN": "SSA", "ST": "SSA",
    "SZ": "SSA", "TD": "SSA", "TG": "SSA", "TZ": "SSA",
    "UG": "SSA", "ZA": "SSA", "ZM": "SSA", "ZW": "SSA",
    # South / Southeast Asia
    "AF": "SSEA", "BD": "SSEA", "IA": "SSEA", "ID": "SSEA",
    "KH": "SSEA", "LK": "SSEA", "MM": "SSEA", "MV": "SSEA",
    "NP": "SSEA", "PG": "SSEA", "PH": "SSEA", "PK": "SSEA",
    "TH": "SSEA", "TL": "SSEA", "VN": "SSEA",
    # Europe & Central Asia
    "AL": "ECA", "AM": "ECA", "AZ": "ECA", "KK": "ECA",
    "KY": "ECA", "TJ": "ECA", "TR": "ECA", "UA": "ECA",
    "UZ": "ECA", "MB": "ECA",  # MB = Moldova (Europe & Central Asia)
    # Latin America & Caribbean
    "BO": "LAC", "BR": "LAC", "CO": "LAC", "DR": "LAC",
    "EC": "LAC", "ES": "LAC", "GU": "LAC", "GY": "LAC",
    "HN": "LAC", "HT": "LAC", "MX": "LAC", "NC": "LAC",
    "PE": "LAC", "PY": "LAC", "TT": "LAC",
    # Middle East & North Africa
    "EG": "MENA", "JO": "MENA", "MA": "MENA", "TN": "MENA",
    "YE": "MENA",
}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging() -> logging.Logger:
    """Set up file + console logging."""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("catalog")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(str(LOG_FILE), encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


# ---------------------------------------------------------------------------
# Filename Parsing
# ---------------------------------------------------------------------------

def parse_dhs_filename(folder_name: str) -> dict:
    """
    Parse a DHS dataset folder name into structured metadata.

    Parameters
    ----------
    folder_name : str
        e.g. "BDBR81DT"

    Returns
    -------
    dict
        Keys: country_code, file_type, phase_raw, suffix,
              country_name, who_region, is_subnational
    """
    # Standard pattern: XXBRYYZZ where XX=country, BR=type, YY=phase, ZZ=suffix
    name = folder_name.upper()

    country_code = name[:2]
    file_type = name[2:4]  # Should be "BR"
    phase_suffix = name[4:]  # e.g., "81DT", "7BDT", "3ADT"

    # Extract phase: everything before "DT" suffix
    phase_raw = phase_suffix.replace("DT", "")

    # Determine DHS phase number (first digit)
    dhs_phase = int(phase_raw[0]) if phase_raw and phase_raw[0].isdigit() else 0

    # Check if this is an India state-level file (Decision D002)
    is_subnational = country_code in INDIA_STATE_PREFIXES

    # For sub-national files, the "country" is India
    effective_country = "IA" if is_subnational else country_code

    return {
        "folder_name": folder_name,
        "country_code": country_code,
        "effective_country": effective_country,
        "file_type": file_type,
        "phase_raw": phase_raw,
        "dhs_phase": dhs_phase,
        "suffix": "DT",
        "country_name": DHS_COUNTRY_NAMES.get(
            effective_country, f"Unknown ({country_code})"
        ),
        "who_region": DHS_TO_WHO_REGION.get(effective_country, "Unknown"),
        "is_subnational": is_subnational,
    }


def find_dta_file(folder_path: Path) -> Optional[Path]:
    """Find the .DTA file inside a dataset folder."""
    dta_files = list(folder_path.glob("*.DTA"))
    if not dta_files:
        # Try lowercase
        dta_files = list(folder_path.glob("*.dta"))
    return dta_files[0] if dta_files else None


def get_dta_metadata(dta_path: Path) -> dict:
    """
    Read just the metadata from a DTA file (no data loaded).

    Returns
    -------
    dict
        Keys: n_rows, n_cols, file_size_mb, survey_year, column_names
    """
    try:
        # Read just 1 row to get metadata quickly
        _, meta = pyreadstat.read_dta(
            str(dta_path), metadataonly=True
        )
        # Read first row to get survey year from v007
        df_head, _ = pyreadstat.read_dta(
            str(dta_path),
            row_limit=1,
            usecols=["v007"] if "v007" in meta.column_names else None,
        )
        survey_year = (
            int(df_head["v007"].iloc[0])
            if "v007" in df_head.columns and len(df_head) > 0
            else None
        )
        return {
            "n_rows": meta.number_rows,
            "n_cols": meta.number_columns,
            "file_size_mb": round(dta_path.stat().st_size / (1024**2), 1),
            "survey_year": survey_year,
            "column_names_sample": meta.column_names[:10],
            "has_v005": "v005" in meta.column_names,
            "has_b5": "b5" in meta.column_names,
            "has_b7": "b7" in meta.column_names,
        }
    except Exception as e:
        return {
            "n_rows": None,
            "n_cols": None,
            "file_size_mb": round(dta_path.stat().st_size / (1024**2), 1),
            "survey_year": None,
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Catalog all DHS datasets."""
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("DHS Dataset Cataloguing")
    logger.info("=" * 60)
    logger.info("Source: %s", RAW_DHS_DIR)

    # Scan all subdirectories
    folders = sorted([
        d for d in RAW_DHS_DIR.iterdir()
        if d.is_dir() and "BR" in d.name.upper()
    ])
    logger.info("Found %d BR dataset folders", len(folders))

    records = []
    for i, folder in enumerate(folders, 1):
        parsed = parse_dhs_filename(folder.name)
        dta_path = find_dta_file(folder)

        if dta_path is None:
            logger.warning("[%d/%d] %s — no .DTA file found",
                           i, len(folders), folder.name)
            parsed["dta_path"] = None
            parsed["dta_found"] = False
            records.append(parsed)
            continue

        parsed["dta_path"] = str(dta_path)
        parsed["dta_found"] = True

        # Read metadata
        meta = get_dta_metadata(dta_path)
        parsed.update(meta)

        status = "SUB-NAT" if parsed["is_subnational"] else "OK"
        logger.info(
            "[%d/%d] %s — %s | %s rows | year=%s | %s",
            i, len(folders), folder.name, status,
            f"{parsed.get('n_rows', '?'):,}" if parsed.get('n_rows') else "?",
            parsed.get("survey_year", "?"),
            parsed.get("country_name", "?"),
        )

        records.append(parsed)

    # Build catalog DataFrame
    catalog = pd.DataFrame(records)

    # Save
    out_parquet = CATALOG_PARQUET
    out_csv = CATALOG_CSV
    catalog.to_parquet(out_parquet, index=False)
    catalog.to_csv(out_csv, index=False)
    logger.info("Catalog saved: %s", out_parquet)

    # Summary
    total = len(catalog)
    national = catalog[~catalog["is_subnational"]]
    subnational = catalog[catalog["is_subnational"]]
    with_data = catalog[catalog["dta_found"]]
    total_births = with_data["n_rows"].sum()

    summary_lines = [
        "=" * 60,
        "CATALOG SUMMARY",
        "=" * 60,
        f"Total folders:       {total}",
        f"National datasets:   {len(national)}",
        f"Sub-national (excl): {len(subnational)}",
        f"With .DTA files:     {len(with_data)}",
        f"Total births (est):  {total_births:,.0f}" if total_births else "Total births: unknown",
        "",
        "By WHO Region (national only):",
    ]

    region_counts = national.groupby("who_region").agg(
        n_surveys=("folder_name", "count"),
        n_countries=("effective_country", "nunique"),
        total_births=("n_rows", "sum"),
    ).sort_values("n_surveys", ascending=False)

    for region, row in region_counts.iterrows():
        births_str = f"{row['total_births']:,.0f}" if pd.notna(row['total_births']) else "?"
        summary_lines.append(
            f"  {str(region):6s}: {int(row['n_surveys']):3d} surveys, "
            f"{int(row['n_countries']):2d} countries, "
            f"{births_str} births"
        )

    summary_lines.extend(["", "By Decade (national only):"])
    national_with_year = national[national["survey_year"].notna()].copy()
    national_with_year["decade"] = (
        national_with_year["survey_year"] // 10 * 10
    ).astype(int).astype(str) + "s"
    decade_counts = national_with_year.groupby("decade").agg(
        n_surveys=("folder_name", "count"),
    )
    for decade, row in decade_counts.iterrows():
        summary_lines.append(f"  {decade}: {row['n_surveys']} surveys")

    summary_text = "\n".join(summary_lines)
    logger.info("\n%s", summary_text)

    summary_path = CATALOG_DIR / "catalog_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    logger.info("Done.")


if __name__ == "__main__":
    main()
