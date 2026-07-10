"""
s03_harmonize.py - Global DHS Feature Harmonization
====================================================

Reads all national DHS Birth Recode Parquet files and produces a single
harmonized analysis dataset with consistent variable coding across all
countries and survey phases.

Design Decisions Implemented
-----------------------------
  D001: Under-5 mortality focus (b5==0 & b7<60)
  D002: India sub-national files excluded (in conversion step)
  D006: Primary scope 2000-2024; flag 1990s for supplement
  D007: Deduplicate re-released surveys (with programmatic guardrail)
  D008: Exclude Nigeria Ondo State (OS)
  D009: Compute birth_year; define is_recent_birth by time
  D010: Keep raw + categorical features; missingness indicators
  D011: Clamp edu_level 4-6 to 3 (Higher)
  D012: Negative birth intervals to NaN (twin coding artifacts)
  D018: Single canonical output (global_analysis.parquet)

Expert Review Fixes
-------------------
  R1-Flaw1: Birth year computed from b2; years_before_survey added
  R1-Flaw2: Maternal age at birth computed (v012 - years_before_survey)
  R1-Flaw3: is_recent_birth defined by time, not data availability
  R2-Flaw1: Nepal 2-digit year → Bikram Sambat conversion
  R2-Flaw2: Ethiopia calendar detection expanded for v007 > 2000
  R2-Flaw3: Survey deduplication logic
  R2-Flaw4: OS exclusion
  R3-Flaw1: b8/b9 excluded from features
  R3-Flaw2: Both raw and categorical features retained
  R3-Flaw3: Missingness indicators created
  R3-Flaw4: sample_weight kept but flagged as regression-only

References
----------
  - DHS Recode Manual v7 (ICF, 2018)
  - Mosley & Chen (1984) proximate determinants framework
  - Rutstein & Johnson (2004) DHS wealth index
  - Yaya et al. (2025), J Global Health - PAF methodology
"""

import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Add project root to path for config imports
sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent.parent)
)
from config.paths import (
    PARQUET_DIR, HARMONIZED_DIR, LOG_DIR,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Paths imported from config.paths
OUTPUT_DIR = HARMONIZED_DIR.parent.parent

# Surveys to SKIP: re-releases of existing data (D007)
# Keep the version with more rows, or the latest release
SKIP_SURVEYS: set[str] = {
    # Peru 2004: PEBR51DT and PEBR5ADT both 89,220 rows. Keep 5A.
    "PEBR51DT",
    # Dom. Rep. 2007: DRBR5ADT has 4,385 rows (a sub-sample). Keep 52.
    "DRBR5ADT",
    # Dom. Rep. 2013: DRBR6ADT has 4,313 rows (a sub-sample). Keep 61.
    "DRBR6ADT",
    # Nigeria Ondo State: sub-national (D008)
    "OSBR01DT",
}

# Country codes with non-Gregorian calendars
# Afghanistan: Solar Hijri → Gregorian = +621
# Ethiopia: Ethiopian → Gregorian = +8 (approx)
# Nepal: Bikram Sambat → Gregorian = -57 (approx)
NON_GREGORIAN = {"AF", "ET", "NP"}

# DHS country code → name mapping
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
    "PE": "Peru", "PG": "Papua New Guinea",
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

DHS_TO_WHO_REGION: dict[str, str] = {
    "AO": "SSA", "BF": "SSA", "BJ": "SSA", "BU": "SSA",
    "CD": "SSA", "CF": "SSA", "CG": "SSA", "CI": "SSA",
    "CM": "SSA", "ET": "SSA", "GA": "SSA", "GH": "SSA",
    "GM": "SSA", "GN": "SSA", "KE": "SSA", "KM": "SSA",
    "LB": "SSA", "LS": "SSA", "MD": "SSA", "ML": "SSA",
    "MR": "SSA", "MW": "SSA", "MZ": "SSA", "NI": "SSA",
    "NM": "SSA", "NG": "SSA", "RW": "SSA", "SD": "SSA",
    "SL": "SSA", "SN": "SSA", "ST": "SSA", "SZ": "SSA",
    "TD": "SSA", "TG": "SSA", "TZ": "SSA", "UG": "SSA",
    "ZA": "SSA", "ZM": "SSA", "ZW": "SSA",
    "AF": "SSEA", "BD": "SSEA", "IA": "SSEA", "ID": "SSEA",
    "KH": "SSEA", "LK": "SSEA", "MM": "SSEA", "MV": "SSEA",
    "NP": "SSEA", "PG": "SSEA", "PH": "SSEA", "PK": "SSEA",
    "TH": "SSEA", "TL": "SSEA", "VN": "SSEA",
    "AL": "ECA", "AM": "ECA", "AZ": "ECA", "KK": "ECA",
    "KY": "ECA", "TJ": "ECA", "TR": "ECA", "UA": "ECA",
    "UZ": "ECA", "MB": "ECA",  # MB = Moldova (Europe & Central Asia)
    "BO": "LAC", "BR": "LAC", "CO": "LAC", "DR": "LAC",
    "EC": "LAC", "ES": "LAC", "GU": "LAC", "GY": "LAC",
    "HN": "LAC", "HT": "LAC", "MX": "LAC", "NC": "LAC",
    "PE": "LAC", "PY": "LAC", "TT": "LAC",
    "EG": "MENA", "JO": "MENA", "MA": "MENA", "TN": "MENA",
    "YE": "MENA",
}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging() -> logging.Logger:
    """Configure file + console logging."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / "02_harmonize.log"
    logger = logging.getLogger("harmonize")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(str(log_file), encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    import io
    # Wrap stdout to handle Unicode on Windows (cp1252 can't encode → etc.)
    safe_stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace",
    )
    ch = logging.StreamHandler(safe_stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


# ---------------------------------------------------------------------------
# Calendar Normalization (R2-Flaw1, R2-Flaw2)
# ---------------------------------------------------------------------------

def normalize_survey_year(
    v007: float, country_code: str
) -> Optional[int]:
    """
    Convert DHS v007 (year of interview) to Gregorian year.

    Handles four calendar systems:
      - Gregorian (most countries): no change
      - Ethiopian: offset +8
      - Nepali Bikram Sambat: offset -57
      - Afghan Solar Hijri: offset +621
      - 2-digit legacy years: +1900 or +2000

    Parameters
    ----------
    v007 : float
        Raw v007 value from DHS.
    country_code : str
        2-letter DHS country code.

    Returns
    -------
    int or None
        Normalized Gregorian year.
    """
    if pd.isna(v007):
        return None

    v = int(v007)

    # Nepal: Bikram Sambat calendar
    # Full BS years (2052-2080) AND 2-digit BS years (52-79)
    if country_code == "NP":
        if v > 2000:
            return v - 57  # e.g., 2078 -> 2021
        elif 40 <= v < 100:
            # 2-digit BS: 52 -> 2052 BS -> 1995 AD
            return v + 2000 - 57
        elif v < 40:
            return v + 2000  # unlikely edge case

    # Ethiopia: Ethiopian calendar
    # Ethiopian years range ~1984-2016 in our data
    # v007 values: 1992, 1997, 2000, 2003, 2005, 2008, 2011, 2016
    if country_code == "ET":
        if v < 2020:  # All Ethiopian v007 < 2020
            return v + 8

    # Afghanistan: Solar Hijri calendar
    if country_code == "AF":
        if v < 1500:
            return v + 621  # e.g., 1394 -> 2015

    # Standard 2-digit years (80s-90s surveys)
    if v < 100:
        return v + 1900 if v > 10 else v + 2000

    # Standard Gregorian
    return v


# Note: a per-row birth-year normalizer is not needed; the vectorized
# logic in process_single_survey() applies the calendar offsets directly
# to the b2 column. Audit check #6 in s03c_audit.py guards the invariant
# `years_before_survey == survey_year - birth_year` end-to-end.


# ---------------------------------------------------------------------------
# Feature Engineering Functions
# ---------------------------------------------------------------------------

def construct_outcome(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct died_under5 outcome variable.

    Rule: b5 == 0 (not alive) AND b7 < 60 (died before 60 months)
    If b5 == 0 and b7 is NaN -> outcome is NaN (unknown)
    If b5 == 0 and b7 >= 60 -> died_under5 = 0 (died AFTER age 5)
    """
    df["died_under5"] = np.where(
        df["b5"] == 0,
        np.where(
            df["b7"].notna() & (df["b7"] < 60),
            1,
            np.where(df["b7"].notna() & (df["b7"] >= 60), 0, np.nan),
        ),
        0,  # b5 == 1 -> alive -> did not die under 5
    )
    return df


def construct_child_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create child-level features from birth record variables."""
    # child_male: b4=1 -> male, b4=2 -> female
    df["child_male"] = (df["b4"] == 1).astype(int)

    # multiple_birth: b0 > 0 means twin/triplet
    df["multiple_birth"] = (df["b0"] > 0).astype(int)

    # birth_order: raw bord (continuous)
    df["birth_order"] = df["bord"]

    # parity_cat: categorical grouping
    df["parity_cat"] = pd.cut(
        df["bord"],
        bins=[0, 1, 3, float("inf")],
        labels=["First born", "2-3", "4+"],
        right=True,
    )

    # birth_interval_months: raw b11 (continuous, NaN=first birth)
    df["birth_interval_months"] = df["b11"].copy()
    # D012: negative intervals are twin-coding artifacts → NaN
    df.loc[df["birth_interval_months"] < 0, "birth_interval_months"] = np.nan

    # birth_interval_cat: categorical (consumes the cleaned values above)
    bim = df["birth_interval_months"]
    conditions = [
        bim.isna(),       # First birth (no preceding interval)
        bim < 24,         # Short interval
        bim.between(24, 47, inclusive="both"),
        bim >= 48,        # Long interval
    ]
    choices = ["First birth", "<24 months", "24-47 months", "48+ months"]
    df["birth_interval_cat"] = np.select(
        conditions, choices, default="_missing"
    )
    df.loc[
        df["birth_interval_cat"] == "_missing",
        "birth_interval_cat",
    ] = np.nan

    return df


def construct_maternal_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Create maternal demographic features."""
    # maternal_age: raw v012 (age at survey)
    df["maternal_age_survey"] = df["v012"]

    # maternal_age_at_birth: computed (R1-Flaw2 fix)
    # = survey_age - (survey_year - birth_year)
    # = v012 - years_before_survey
    df["maternal_age_at_birth"] = np.where(
        df["years_before_survey"].notna(),
        df["v012"] - df["years_before_survey"],
        np.nan,
    )
    # Clamp to plausible range [10, 55]
    df.loc[
        (df["maternal_age_at_birth"] < 10)
        | (df["maternal_age_at_birth"] > 55),
        "maternal_age_at_birth",
    ] = np.nan

    # mat_age_cat: based on age at birth
    conditions = [
        df["maternal_age_at_birth"] < 20,
        df["maternal_age_at_birth"].between(20, 34, inclusive="both"),
        df["maternal_age_at_birth"] >= 35,
    ]
    choices = ["<20", "20-34", "35+"]
    df["mat_age_cat"] = np.select(
        conditions, choices, default="_missing"
    )
    df.loc[df["mat_age_cat"] == "_missing", "mat_age_cat"] = np.nan

    # edu_level: v106 (0=None, 1=Primary, 2=Secondary, 3=Higher)
    df["edu_level"] = df["v106"].copy()
    df.loc[df["edu_level"] >= 8, "edu_level"] = np.nan  # 8/9 = DK/missing
    # D011: clamp 4/5/6 (post-secondary sub-categories used by some surveys) → 3
    df.loc[df["edu_level"] > 3, "edu_level"] = 3

    edu_map = {0: "None", 1: "Primary", 2: "Secondary", 3: "Higher"}
    df["edu_level_cat"] = df["edu_level"].map(edu_map)

    # currently_married: v501 in [1=married, 2=living together]
    df["currently_married"] = np.where(
        df["v501"].isin([1, 2]), 1,
        np.where(df["v501"].notna(), 0, np.nan),
    )

    return df


def construct_ses_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create socioeconomic status features."""
    # wealth_quintile: v190 (1-5), keep raw
    df["wealth_quintile"] = df.get("v190", pd.Series(
        np.nan, index=df.index
    )).copy()

    # wealth_cat: 3 categories
    wealth_map = {1: "Poor", 2: "Poor", 3: "Middle", 4: "Rich", 5: "Rich"}
    df["wealth_cat"] = df["wealth_quintile"].map(wealth_map)

    # urban: v025=1 -> urban, v025=2 -> rural
    df["urban"] = np.where(
        df["v025"] == 1, 1,
        np.where(df["v025"] == 2, 0, np.nan),
    )

    return df


def construct_healthcare_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create healthcare utilization features.

    These apply only to recent births (m-variables are NaN for
    non-recent births BY DESIGN in DHS, not due to missing data).
    """
    # --- ANC visits: m14 ---
    anc = df.get("m14", pd.Series(np.nan, index=df.index)).copy()
    anc = anc.where(anc < 98)  # 98=DK -> NaN
    df["anc_visits"] = anc

    conditions = [
        anc == 0,
        anc.between(1, 3, inclusive="both"),
        anc.between(4, 7, inclusive="both"),
        anc >= 8,
    ]
    choices = ["None", "1-3", "4-7", "8+"]
    df["anc_visits_cat"] = np.select(
        conditions, choices, default="_missing"
    )
    df.loc[
        df["anc_visits_cat"] == "_missing", "anc_visits_cat"
    ] = np.nan

    # --- Facility delivery: m15 (R2-Flaw5 fix) ---
    m15 = df.get("m15", pd.Series(np.nan, index=df.index)).copy()
    # Detect coding scheme: 1-digit (old) vs 2-digit (standard)
    max_m15 = m15.max() if m15.notna().any() else np.nan
    if pd.notna(max_m15) and max_m15 < 10:
        # 1-digit coding: 1=home, 2+=facility
        df["facility_delivery"] = np.where(
            m15.notna(), (m15 >= 2).astype(float), np.nan
        )
    else:
        # 2-digit coding: tens digit 2,3,4 = facility
        # 11-13=home, 21-26=public, 31-36=private, 41-46=NGO
        # 96=other (not facility)
        df["facility_delivery"] = np.where(
            m15.notna(),
            np.where(
                (m15 >= 20) & (m15 < 96), 1.0, 0.0
            ),
            np.nan,
        )

    # --- Caesarean: m17 ---
    m17 = df.get("m17", pd.Series(np.nan, index=df.index)).copy()
    df["caesarean"] = np.where(
        m17.notna(), (m17 == 1).astype(float), np.nan
    )

    # --- Tetanus protection: m1 ---
    m1 = df.get("m1", pd.Series(np.nan, index=df.index)).copy()
    df["tetanus_protected"] = np.where(
        m1.notna() & (m1 < 8),  # 8=DK
        (m1 >= 1).astype(float),
        np.nan,
    )

    return df


def construct_birth_size_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Create birth size and weight features."""
    # birth_size: m18 (1-5 scale, 8=DK)
    m18 = df.get("m18", pd.Series(np.nan, index=df.index)).copy()
    df["birth_size"] = m18.where(m18 <= 5)  # 8=DK -> NaN

    # small_at_birth: m18 >= 4 (smaller than average or very small)
    df["small_at_birth"] = np.where(
        df["birth_size"].notna(),
        (df["birth_size"] >= 4).astype(float),
        np.nan,
    )

    # birth_weight_kg: m19 is in GRAMS (despite DHS label)
    m19 = df.get("m19", pd.Series(np.nan, index=df.index)).copy()
    df["birth_weight_kg"] = np.where(
        m19.notna() & (m19 < 9996),  # 9996=not weighed, 9998=DK
        m19 / 1000.0,
        np.nan,
    )
    # Plausibility: 0.5-6.5 kg
    df.loc[
        (df["birth_weight_kg"] < 0.5) | (df["birth_weight_kg"] > 6.5),
        "birth_weight_kg",
    ] = np.nan

    return df


def construct_household_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Create household-level features."""
    # hh_members: v136
    df["hh_members"] = df.get(
        "v136", pd.Series(np.nan, index=df.index)
    ).copy()

    # maternal_bmi: v445 / 100 (coded as BMI * 100)
    v445 = df.get("v445", pd.Series(np.nan, index=df.index)).copy()
    df["maternal_bmi"] = np.where(
        v445.notna() & (v445 < 9000),  # 9998=flagged
        v445 / 100.0,
        np.nan,
    )
    # Plausibility: 12-60
    df.loc[
        (df["maternal_bmi"] < 12) | (df["maternal_bmi"] > 60),
        "maternal_bmi",
    ] = np.nan

    return df


def construct_survey_design_vars(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Create survey design variables for weighted analysis."""
    # Sample weight: v005 has 6 implied decimal places
    df["sample_weight"] = df["v005"] / 1_000_000

    # PSU: prefer v021, fall back to v001
    if "v021" in df.columns and df["v021"].notna().any():
        df["psu"] = df["v021"]
    else:
        df["psu"] = df["v001"]

    # Stratum: prefer v023, fall back to v022
    if "v023" in df.columns and df["v023"].notna().any():
        df["stratum"] = df["v023"]
    elif "v022" in df.columns:
        df["stratum"] = df["v022"]
    else:
        df["stratum"] = np.nan

    return df


def add_missingness_indicators(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create binary missingness indicators for ML (D010d).

    Only for variables with meaningful missingness patterns.
    """
    miss_vars = [
        "wealth_quintile", "anc_visits", "facility_delivery",
        "caesarean", "tetanus_protected", "birth_weight_kg",
        "maternal_bmi", "small_at_birth",
    ]
    for var in miss_vars:
        if var in df.columns:
            df[f"{var}_missing"] = df[var].isna().astype(int)
    return df


# ---------------------------------------------------------------------------
# Single-Survey Processing
# ---------------------------------------------------------------------------

# Raw DHS columns consumed by the harmonization pipeline.
# Reading only these (instead of 200+ raw columns) keeps large
# surveys like India NFHS-4/5 (~1.3M rows) within memory limits.
_RAW_COLS_NEEDED = {
    # Identity / survey design
    "v000", "v007", "v005", "v001", "v021", "v022", "v023",
    # Outcome
    "b5", "b7",
    # Birth record
    "b2", "b4", "b0", "bord", "b11",
    # Maternal demographics
    "v012", "v106", "v501",
    # SES
    "v190", "v025",
    # Healthcare utilization
    "m14", "m15", "m17", "m1",
    # Birth size / weight
    "m18", "m19",
    # Household
    "v136", "v445",
}


def process_single_survey(
    parquet_path: Path, logger: logging.Logger
) -> Optional[pd.DataFrame]:
    """
    Read and harmonize a single survey Parquet file.

    Returns
    -------
    pd.DataFrame or None
        Harmonized DataFrame, or None on failure.
    """
    folder = parquet_path.stem  # e.g., "BDBR81DT"

    try:
        # Read only the columns we need — critical for large surveys
        # (India NFHS ~1.3M rows; reading all 200+ DHS cols OOMs)
        import pyarrow.parquet as _pq_local
        file_cols = set(_pq_local.read_schema(str(parquet_path)).names)
        cols_to_read = list(_RAW_COLS_NEEDED & file_cols)
        df = pd.read_parquet(parquet_path, columns=cols_to_read)
        n_raw = len(df)

        # Extract country code from v000 or filename
        if "v000" in df.columns and df["v000"].notna().any():
            cc_raw = str(df["v000"].dropna().iloc[0])
            country_code = cc_raw[:2].upper()
        else:
            country_code = folder[:2].upper()

        # Normalize survey year (R2-Flaw1, R2-Flaw2)
        raw_year = df["v007"].mode().iloc[0] if "v007" in df.columns else None
        survey_year = normalize_survey_year(raw_year, country_code)
        if survey_year is None:
            logger.warning("  %s: Could not determine survey year", folder)
            return None

        # Normalize birth year - VECTORIZED (R1-Flaw1)
        if "b2" in df.columns:
            b2 = df["b2"].copy()
            by = b2.copy()
            # Apply same calendar logic vectorized
            if country_code == "NP":
                by = np.where(b2 > 2000, b2 - 57,
                     np.where((b2 >= 40) & (b2 < 100), b2 + 2000 - 57,
                     np.where(b2 < 40, b2 + 2000, b2)))
            elif country_code == "ET":
                by = np.where(b2 < 2020, b2 + 8, b2)
            elif country_code == "AF":
                by = np.where(b2 < 1500, b2 + 621, b2)
            else:
                by = np.where((b2 > 10) & (b2 < 100), b2 + 1900,
                     np.where(b2 <= 10, b2 + 2000, b2))
            df["birth_year"] = pd.array(by, dtype="Float64")
        else:
            df["birth_year"] = np.nan

        # Birth-year sanity guard
        by_arr = df["birth_year"].dropna()
        if len(by_arr) and (by_arr.min() < 1950 or by_arr.max() > survey_year + 1):
            logger.warning(
                "  %s: birth_year out of range [%s, %s] for survey_year=%d",
                folder, int(by_arr.min()), int(by_arr.max()), survey_year,
            )

        # Compute years_before_survey (R1-Flaw1)
        df["years_before_survey"] = survey_year - df["birth_year"]

        # is_recent_birth: within 5 years of survey (R1-Flaw3)
        df["is_recent_birth"] = np.where(
            df["years_before_survey"].notna(),
            (df["years_before_survey"] <= 5).astype(int),
            np.nan,
        )

        # Add metadata
        df["country_code"] = country_code
        df["country_name"] = DHS_COUNTRY_NAMES.get(
            country_code, f"Unknown ({country_code})"
        )
        df["who_region"] = DHS_TO_WHO_REGION.get(country_code, "Other")
        df["survey_year"] = survey_year
        df["_source_file"] = folder

        # --- Feature Engineering ---
        df = construct_outcome(df)
        df = construct_child_features(df)
        df = construct_maternal_features(df)
        df = construct_ses_features(df)
        df = construct_healthcare_features(df)
        df = construct_birth_size_features(df)
        df = construct_household_features(df)
        df = construct_survey_design_vars(df)
        df = add_missingness_indicators(df)

        # --- Select output columns ---
        output_cols = [
            # Metadata
            "country_code", "country_name", "who_region",
            "survey_year", "birth_year", "years_before_survey",
            "is_recent_birth", "_source_file",
            # Survey design
            "sample_weight", "psu", "stratum",
            # Outcome
            "died_under5",
            # Child factors (raw + categorical)
            "child_male", "multiple_birth",
            "birth_order", "parity_cat",
            "birth_interval_months", "birth_interval_cat",
            # Maternal factors (raw + categorical)
            "maternal_age_survey", "maternal_age_at_birth",
            "mat_age_cat",
            "edu_level", "edu_level_cat",
            "currently_married",
            # SES
            "wealth_quintile", "wealth_cat", "urban",
            # Healthcare
            "anc_visits", "anc_visits_cat",
            "facility_delivery", "caesarean",
            "tetanus_protected",
            # Birth size
            "birth_size", "small_at_birth", "birth_weight_kg",
            # Household
            "hh_members", "maternal_bmi",
            # Missingness indicators
            "wealth_quintile_missing", "anc_visits_missing",
            "facility_delivery_missing", "caesarean_missing",
            "tetanus_protected_missing", "birth_weight_kg_missing",
            "maternal_bmi_missing", "small_at_birth_missing",
        ]

        # Keep only columns that exist
        available = [c for c in output_cols if c in df.columns]
        df_out = df[available].copy()

        # Compute U5MR for validation
        u5mr = (
            df_out["died_under5"].sum()
            / df_out["died_under5"].notna().sum()
            * 1000
        ) if df_out["died_under5"].notna().sum() > 0 else np.nan

        logger.info(
            "  %s (%s %d): %s births, U5MR=%.0f/1000, %d cols",
            folder,
            country_code,
            survey_year,
            f"{n_raw:,}",
            u5mr if pd.notna(u5mr) else 0,
            len(available),
        )

        # Validation: flag implausible U5MR
        if pd.notna(u5mr) and (u5mr < 5 or u5mr > 500):
            logger.warning(
                "    WARNING: U5MR=%.0f/1000 outside plausible range "
                "[5, 500] for %s %d",
                u5mr, country_code, survey_year,
            )

        return df_out

    except Exception as e:
        logger.error("  %s: FAILED - %s", folder, str(e))
        return None


# ---------------------------------------------------------------------------
# Re-release detection guardrail
# ---------------------------------------------------------------------------

def check_rerelease_coverage(
    catalog_path: Path, logger: logging.Logger,
) -> None:
    """
    Scan the dataset catalog for potential re-releases (same country,
    same phase, same or very close survey year) and assert they are
    covered by SKIP_SURVEYS.

    DHS phases span multiple distinct surveys (e.g., Bangladesh Phase 7
    includes both BDHS 2014 and BDHS 2017-18). Multiple files in the
    same phase with DIFFERENT survey years are separate surveys, not
    re-releases. We only flag files whose survey_year values are within
    1 year of each other, which is the actual re-release pattern
    (e.g., PEBR51DT and PEBR5ADT both 2004).
    """
    cat = pd.read_parquet(catalog_path)
    cat = cat[cat["file_type"] == "BR"].copy()

    # Within each (country, phase), cluster files by survey_year proximity
    n_flagged = 0
    for (cc, phase), grp in cat.groupby(["country_code", "dhs_phase"]):
        if len(grp) < 2:
            continue
        # Sort by survey_year; group files whose years are within 1 of each other
        rows = grp.sort_values("survey_year")[
            ["folder_name", "survey_year", "n_rows"]
        ].to_dict("records")
        # Build clusters of files with similar years
        clusters: list[list[dict]] = []
        for r in rows:
            placed = False
            yr = r["survey_year"]
            if pd.notna(yr):
                for cl in clusters:
                    # Exact same year = likely re-release (e.g., PE 51/5A both 2004)
                    # Different years = separate annual surveys (e.g., Peru ENDES)
                    if any(
                        pd.notna(c["survey_year"])
                        and c["survey_year"] == yr
                        for c in cl
                    ):
                        cl.append(r)
                        placed = True
                        break
            if not placed:
                clusters.append([r])
        # Only flag clusters with >1 file (same-year duplicates)
        for cl in clusters:
            if len(cl) < 2:
                continue
            stems = {c["folder_name"] for c in cl}
            skipped = stems & SKIP_SURVEYS
            kept = stems - SKIP_SURVEYS
            if len(kept) > 1:
                msg = (
                    f"UNACCOUNTED RE-RELEASE: country={cc}, "
                    f"phase={phase}, files={sorted(stems)}, "
                    f"only {sorted(skipped)} in SKIP_SURVEYS, "
                    f"{sorted(kept)} all kept"
                )
                logger.error(msg)
                raise ValueError(msg)
            elif len(kept) == 0:
                logger.warning(
                    "  Re-release %s phase %s: ALL skipped (%s)",
                    cc, phase, sorted(stems),
                )
            else:
                logger.info(
                    "  Re-release %s phase %s: keep %s, skip %s",
                    cc, phase, sorted(kept), sorted(skipped),
                )
                n_flagged += 1

    logger.info(
        "Re-release check complete: %d groups verified", n_flagged
    )


# ---------------------------------------------------------------------------
# Dtype optimization (ported from s03b_optimize_parquet.py)
# ---------------------------------------------------------------------------

# Columns to convert to pandas Categorical
_CATEGORICAL_COLS = [
    "country_code", "country_name", "who_region",
    "_source_file", "parity_cat", "birth_interval_cat",
    "mat_age_cat", "edu_level_cat", "wealth_cat",
    "anc_visits_cat",
]

# Columns to downcast to Int8 (0/1 binary or small range)
_INT8_COLS = [
    "child_male", "multiple_birth", "in_primary_scope",
    "wealth_quintile_missing", "anc_visits_missing",
    "facility_delivery_missing", "caesarean_missing",
    "tetanus_protected_missing", "birth_weight_kg_missing",
    "maternal_bmi_missing", "small_at_birth_missing",
]

# Columns to downcast to Int16
_INT16_COLS = ["birth_order", "survey_year"]

# Float columns safe to store as float32
_FLOAT32_COLS = [
    "birth_year", "years_before_survey", "is_recent_birth",
    "sample_weight", "psu", "stratum",
    "died_under5", "birth_interval_months",
    "maternal_age_survey", "maternal_age_at_birth",
    "edu_level", "currently_married",
    "wealth_quintile", "urban",
    "anc_visits", "facility_delivery", "caesarean",
    "tetanus_protected", "birth_size", "small_at_birth",
    "birth_weight_kg", "hh_members", "maternal_bmi",
]


def _apply_dtype_optimization(df: pd.DataFrame) -> pd.DataFrame:
    """
    Memory-optimize dtypes in-place.

    Mirrors the logic previously in s03b_optimize_parquet.py but
    applied at write time inside the canonical pipeline so there
    is no separate "v2" file.
    """
    for col in _CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")

    for col in _INT8_COLS:
        if col in df.columns:
            # Use pandas nullable Int8 to preserve NaN
            df[col] = df[col].astype("Int8")

    for col in _INT16_COLS:
        if col in df.columns:
            df[col] = df[col].astype("Int16")

    for col in _FLOAT32_COLS:
        if col in df.columns:
            df[col] = df[col].astype(np.float32)

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run full harmonization pipeline."""
    import gc
    import shutil
    import pyarrow.parquet as pq

    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("DHS Global Harmonization Pipeline")
    logger.info("=" * 60)
    t0 = time.time()

    # Discover parquet files
    all_files = sorted(PARQUET_DIR.glob("*.parquet"))
    all_files = [f for f in all_files if not f.name.startswith("_")]
    logger.info("Found %d parquet files", len(all_files))

    # --- Re-release guardrail (D007) ---
    from config.paths import CATALOG_PARQUET
    if CATALOG_PARQUET.exists():
        logger.info("Running re-release coverage check...")
        check_rerelease_coverage(CATALOG_PARQUET, logger)
    else:
        logger.warning(
            "Catalog not found at %s — skipping re-release check",
            CATALOG_PARQUET,
        )

    # --- Build exclusion table ---
    exclusion_records: list[dict] = []
    for stem in sorted(SKIP_SURVEYS):
        reason = "D008_subnational" if stem == "OSBR01DT" else "D007_rereleased"
        exclusion_records.append({
            "survey_id": stem,
            "country_code": stem[:2],
            "reason": reason,
            "stage": "pre_harmonize",
        })

    # Apply exclusions (D007, D008)
    files = [f for f in all_files if f.stem not in SKIP_SURVEYS]
    skipped = len(all_files) - len(files)
    logger.info(
        "After exclusions (D007/D008): %d files (%d skipped: %s)",
        len(files), skipped, sorted(SKIP_SURVEYS),
    )

    # Incremental processing: write each to temp dir, then merge
    temp_dir = HARMONIZED_DIR / "_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    success = 0
    failed = 0

    for i, fpath in enumerate(files, 1):
        logger.info("[%d/%d] Processing %s", i, len(files), fpath.stem)
        result = process_single_survey(fpath, logger)
        if result is not None:
            temp_path = temp_dir / f"{fpath.stem}.parquet"
            result.to_parquet(
                temp_path, engine="pyarrow",
                compression="snappy", index=False,
            )
            del result
            gc.collect()
            success += 1
        else:
            failed += 1
            # Track the failure reason for the exclusion table
            exclusion_records.append({
                "survey_id": fpath.stem,
                "country_code": fpath.stem[:2],
                "reason": "harmonize_failed",
                "stage": "harmonize",
            })

    logger.info("")
    logger.info("Processed: %d success, %d failed", success, failed)

    # --- Write exclusion table ---
    from config.paths import TABLES_DIR
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    excl_df = pd.DataFrame(exclusion_records)
    excl_path = TABLES_DIR / "table_s0_excluded_surveys.csv"
    excl_df.to_csv(excl_path, index=False)
    logger.info(
        "Wrote %d exclusions to %s", len(excl_df), excl_path,
    )

    # ---------------------------------------------------------------
    # Write canonical parquet using streaming ParquetWriter.
    # This uses O(1) memory regardless of dataset size — we write
    # one survey at a time as a separate row group.
    #
    # The schema is unified first (metadata-only pass), handling:
    #   - null-typed columns (all-NaN in some surveys) → float64
    #   - dictionary<string> columns → plain string
    #   - int64 vs float64 mismatches → float64 (via promote)
    # ---------------------------------------------------------------
    import pyarrow as pa

    logger.info("Building unified schema from %d temp files...", success)
    temp_files = sorted(temp_dir.glob("*.parquet"))

    def _normalize_schema(schema: "pa.Schema") -> "pa.Schema":
        """Normalize types for cross-file unification:
        - dictionary<string> → string
        - any integer type → float64 (avoids int64 vs double conflicts)
        - null stays null (pa.unify_schemas promotes null → other type)
        """
        fields = []
        for field in schema:
            if pa.types.is_dictionary(field.type):
                fields.append(pa.field(field.name, pa.string()))
            elif pa.types.is_integer(field.type):
                fields.append(pa.field(field.name, pa.float64()))
            else:
                fields.append(field)
        return pa.schema(fields)

    def _conform_table(t: "pa.Table", target: "pa.Schema") -> "pa.Table":
        """Conform table t to target schema: decode dicts, cast nulls, add missing cols."""
        cols = {}
        for field in target:
            if field.name in t.schema.names:
                col = t.column(field.name)
                if pa.types.is_dictionary(col.type):
                    col = col.combine_chunks().dictionary_decode()
                if col.type == pa.null():
                    col = pa.nulls(len(t), type=field.type)
                elif col.type != field.type:
                    try:
                        col = col.cast(field.type)
                    except Exception:
                        col = pa.nulls(len(t), type=field.type)
            else:
                col = pa.nulls(len(t), type=field.type)
            cols[field.name] = col
        return pa.table(cols, schema=target)

    # Pass 1 (metadata only): build unified schema
    raw_schemas = [_normalize_schema(pq.read_schema(str(f))) for f in temp_files]
    unified_schema = pa.unify_schemas(raw_schemas, promote_options="default")

    # Add in_primary_scope column to schema (int8)
    unified_schema = unified_schema.append(pa.field("in_primary_scope", pa.int8()))

    logger.info(
        "Unified schema: %d columns", len(unified_schema)
    )

    # Pass 2: stream-write one file per survey row group
    out_path = HARMONIZED_DIR / "global_analysis.parquet"
    total_rows = 0
    n_primary = 0
    n_supplement = 0

    with pq.ParquetWriter(str(out_path), schema=unified_schema,
                          compression="snappy") as writer:
        for tf in temp_files:
            t = pq.read_table(str(tf))
            t = _conform_table(t, unified_schema.remove(
                unified_schema.get_field_index("in_primary_scope")
            ))
            # Attach D006 scope flag
            sy_col = t.column("survey_year").to_pylist()
            scope = pa.array(
                [1 if 2000 <= y <= 2024 else 0 for y in sy_col],
                type=pa.int8(),
            )
            t = t.append_column(pa.field("in_primary_scope", pa.int8()), scope)
            writer.write_table(t)
            total_rows += len(t)
            n_primary += int(scope.to_pylist().count(1))
            del t, scope, sy_col
            gc.collect()

    size_mb = out_path.stat().st_size / (1024**2)
    logger.info(
        "Saved: %s (%.1f MB, %s rows)",
        out_path, size_mb, f"{total_rows:,}",
    )
    n_supplement = total_rows - n_primary
    logger.info(
        "Temporal scope: %s primary (2000-2024), %s supplementary (pre-2000)",
        f"{n_primary:,}", f"{n_supplement:,}",
    )

    # Verify row count via metadata (no full read)
    meta = pq.read_metadata(str(out_path))
    assert meta.num_rows == total_rows, (
        f"Row count mismatch: written {meta.num_rows} vs expected {total_rows}"
    )
    logger.info("Verified: %s rows on disk", f"{meta.num_rows:,}")

    # --- Quick validation (pyarrow only, no full pandas load) ---
    # Full invariant checks are in s03c_audit.py.
    logger.info("")
    logger.info("=" * 60)
    logger.info("QUICK VALIDATION (pyarrow column stats)")
    logger.info("=" * 60)

    import pyarrow.compute as pc

    # Survey year range (numeric column, cheap)
    sy_col = pq.read_table(str(out_path), columns=["survey_year"]).column(0)
    yr_min = pc.min(sy_col).as_py()
    yr_max = pc.max(sy_col).as_py()
    logger.info("  Survey year range: %d - %d", yr_min, yr_max)
    if yr_min < 1980 or yr_max > 2026:
        logger.warning("  FAIL: Year range outside [1980, 2026]!")
    del sy_col

    # Unique countries (string column dictionary-encoded by parquet is cheap)
    cc_col = pq.read_table(str(out_path), columns=["country_code"]).column(0)
    n_countries = pc.count_distinct(cc_col).as_py()
    logger.info("  Unique countries: %d", n_countries)
    del cc_col

    # Overall U5MR from numeric column
    d5_col = pq.read_table(str(out_path), columns=["died_under5"]).column(0)
    n_deaths = pc.sum(d5_col).as_py() or 0
    n_valid = pc.count(d5_col).as_py() or 1
    overall_u5mr = n_deaths / n_valid * 1000
    logger.info("  Overall U5MR (all years): %.1f per 1000", overall_u5mr)
    del d5_col

    logger.info("  Total births: %s", f"{total_rows:,}")
    logger.info("  Primary scope (2000-2024): %s births", f"{n_primary:,}")
    logger.info("")
    logger.info("  Full checks: run  python -m src.data.s03c_audit")
    gc.collect()

    # Cleanup temp files
    shutil.rmtree(temp_dir, ignore_errors=True)
    logger.info("Cleaned up temp directory")

    elapsed = time.time() - t0
    logger.info("")
    logger.info("Total time: %.0f seconds", elapsed)
    logger.info("Done.")


if __name__ == "__main__":
    main()
