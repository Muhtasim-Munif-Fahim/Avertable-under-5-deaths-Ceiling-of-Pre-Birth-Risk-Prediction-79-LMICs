"""
s03_mics_harmonize.py - MICS Survey Harmonization
==================================================

Reads one MICS country zip file, extracts bh.sav (birth history) and
wm.sav (women), maps MICS variables to the project's standardized column
schema, and returns a DataFrame ready to append to global_analysis.parquet.

MICS-specific coding notes
--------------------------
- BH5: 1=alive (YES), 2=dead (NO)
- BH9C: age at death in MIXED units; BH9U gives the unit:
    1=days, 2=months, 3=years, 9=DK. Must convert to months before use.
- birthint: ordinal category (0=first birth, 1=<2yr, 2=2yr, 3=3yr, 4=4+yr)
    NOT raw months. Map to midpoints for the continuous column.
- magebrt: ordinal category (1=<20, 2=20-34, 3=35+), NOT raw age.
    Map to midpoints for the continuous column.
- wmweight: already at analysis scale (mean ~1.0); do NOT divide by 1e6.
- welevel: 0=None, 1=Primary, 2=Lower secondary, 3=Upper secondary+,
    9=Missing/DK. Collapse 3->2 to match DHS edu_level schema.

Design decisions
----------------
- Healthcare variables (ANC, delivery, tetanus) are excluded from the
  main output; they belong in a separate restricted-birth sub-dataset
  matching the DHS pattern in s03_harmonize.py.
- hl.sav is read only for household size; absent zip -> household_size=NaN.
- pyreadstat requires a real file path; we extract to a NamedTemporaryFile
  and clean up immediately after reading.
"""
from __future__ import annotations

import io
import os
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.paths import RNG_SEED  # noqa: F401  (imported for callers)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(RNG_SEED)


def _read_sav_from_zip(zf: zipfile.ZipFile, member: str) -> pd.DataFrame:
    """
    Extract one .sav member to a temp file, read with pyreadstat, clean up.
    pyreadstat does not accept BytesIO -- it needs a real filesystem path.
    """
    import pyreadstat

    data = zf.read(member)
    with tempfile.NamedTemporaryFile(suffix=".sav", delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        last_exc = None
        for encoding in (None, "LATIN1", "cp1251"):
            try:
                if encoding is None:
                    df, _meta = pyreadstat.read_sav(tmp_path)
                else:
                    df, _meta = pyreadstat.read_sav(tmp_path, encoding=encoding)
                return df
            except Exception as exc:
                last_exc = exc
        raise last_exc
    finally:
        os.unlink(tmp_path)


def _find_member(zf: zipfile.ZipFile, suffix: str) -> Optional[str]:
    """Return the first zip member whose basename matches suffix (case-insensitive)."""
    for name in zf.namelist():
        if name.split("/")[-1].lower() == suffix.lower():
            return name
    return None


# ---------------------------------------------------------------------------
# BH9C: convert mixed-unit age-at-death to months
# ---------------------------------------------------------------------------

def _age_at_death_months(df: pd.DataFrame) -> pd.Series:
    """
    Convert BH9C + BH9U to a single months column.

    BH9U: 1=days, 2=months, 3=years, 9=DK/missing.
    BH9C is only meaningful when BH5==2 (child died).
    Days are converted via 1/30.4375 (average month length).
    """
    raw = df["BH9C"].copy()
    # Replace MICS missing sentinels before arithmetic
    raw = raw.where(raw >= 0)

    unit = df["BH9U"] if "BH9U" in df.columns else pd.Series(np.nan, index=df.index)

    months = np.where(
        unit == 1, raw / 30.4375,     # days -> months
        np.where(
            unit == 2, raw,            # already months
            np.where(
                unit == 3, raw * 12.0, # years -> months
                np.nan,                # DK / missing unit
            ),
        ),
    )
    # Only meaningful for dead children; set alive children to NaN
    months = np.where(df["BH5"] == 2, months, np.nan)
    return pd.Series(months, index=df.index, dtype="float64")


# ---------------------------------------------------------------------------
# Variable mapping functions
# ---------------------------------------------------------------------------

def _map_child_vars(df: pd.DataFrame) -> pd.DataFrame:
    """Map MICS child-level BH variables to the standardized schema."""
    # child_male: BH3=1(BOY)->1, BH3=2(GIRL)->0, 9(NR)->NaN
    df["child_male"] = np.where(
        df["BH3"] == 1, 1.0,
        np.where(df["BH3"] == 2, 0.0, np.nan),
    )

    # multiple_birth: BH2=1(Single)->0, BH2=2(Multiple)->1, 9->NaN
    df["multiple_birth"] = np.where(
        df["BH2"] == 1, 0.0,
        np.where(df["BH2"] == 2, 1.0, np.nan),
    )

    # birth_order: brthord is already integer order
    df["birth_order"] = df["brthord"].where(df["brthord"] > 0)

    # parity_cat: matches DHS schema bins [0,1], [2,3], [4+]
    bo = df["birth_order"]
    df["parity_cat"] = np.select(
        [bo == 1, bo.between(2, 3, inclusive="both"), bo >= 4],
        ["First born", "2-3", "4+"],
        default=None,
    )
    df["parity_cat"] = pd.Categorical(
        df["parity_cat"],
        categories=["First born", "2-3", "4+"],
        ordered=False,
    )

    # birth_interval_months: MICS6 has birthint as ordinal (0=first, 1=<2yr, 2=2yr, 3=3yr, 4=4+yr).
    # MICS5 (and some older surveys) has birthint as raw months (e.g., 24, 36, ...).
    # Detect which encoding by checking the non-null max: ordinal max==4; continuous can be >>4.
    interval_midpoints = {0.0: np.nan, 1.0: 18.0, 2.0: 24.0, 3.0: 36.0, 4.0: 60.0}
    raw_int = df["birthint"].copy()
    raw_int = raw_int.where(raw_int >= 0)   # drop negative sentinels
    if raw_int.dropna().empty:
        df["birth_interval_months"] = np.nan
    elif raw_int.max() <= 4.0:
        # MICS6 ordinal: map to midpoints
        df["birth_interval_months"] = raw_int.map(interval_midpoints).astype("float64")
    else:
        # MICS5 continuous months: use directly. Cap at 120 to drop implausible values.
        df["birth_interval_months"] = raw_int.where(raw_int <= 120).astype("float64")

    # birth_interval_cat: matches DHS categorical labels
    bim = df["birth_interval_months"]
    df["birth_interval_cat"] = np.select(
        [
            bim.isna() & (df["birth_order"] == 1),
            bim < 24,
            bim.between(24, 47, inclusive="both"),
            bim >= 48,
        ],
        ["First birth", "<24 months", "24-47 months", "48+ months"],
        default=None,
    )
    # Rows where birth_order>1 but interval is NaN get NaN category
    mask_missing_int = bim.isna() & (df["birth_order"] > 1)
    df.loc[mask_missing_int, "birth_interval_cat"] = None
    df["birth_interval_cat"] = df["birth_interval_cat"].astype("object")

    return df


def _map_maternal_vars(df: pd.DataFrame) -> pd.DataFrame:
    """Map MICS maternal variables to the standardized schema."""
    # maternal_age_at_birth: magebrt is an ordinal category in MICS
    # (1=<20, 2=20-34, 3=35+). Map to midpoints for the continuous column.
    # These midpoints roughly match the DHS construct (v012 - years_before_survey).
    age_midpoints = {1.0: 17.0, 2.0: 27.0, 3.0: 37.0}
    raw_age = df["magebrt"].where(df["magebrt"] > 0)
    df["maternal_age_at_birth"] = raw_age.map(age_midpoints).astype("float64")

    # mat_age_cat: matches DHS labels
    mab = df["maternal_age_at_birth"]
    df["mat_age_cat"] = np.select(
        [mab < 20, mab.between(20, 34, inclusive="both"), mab >= 35],
        ["<20", "20-34", "35+"],
        default=None,
    )
    df["mat_age_cat"] = df["mat_age_cat"].astype("object")

    # maternal_age_survey: not available in BH file -> NaN
    # (MICS wm.sav has WAGE but it's not birth-history-level data)
    df["maternal_age_survey"] = np.nan

    # edu_level: welevel 0=None, 1=Primary, 2=Lower sec, 3=Upper sec+, 9=DK
    # Collapse upper secondary (3) into Secondary (2) to match DHS D011 schema.
    edu_raw = df["welevel"].copy()
    edu_raw = edu_raw.where(edu_raw != 9.0)   # 9 = DK/missing -> NaN
    edu_raw = edu_raw.where(edu_raw >= 0.0)    # negative sentinels -> NaN
    edu_raw = edu_raw.clip(upper=3.0)          # cap at 3 (Higher); already collapsed above
    # 3 in welevel = Upper secondary+ -> map to 2 (Secondary) for DHS parity
    edu_raw = edu_raw.where(edu_raw != 3.0, other=2.0)
    df["edu_level"] = edu_raw.astype("float64")

    edu_map = {0.0: "None", 1.0: "Primary", 2.0: "Secondary"}
    df["edu_level_cat"] = df["edu_level"].map(edu_map).astype("object")

    # currently_married: MSTATUS 1=married/in union, 2=formerly, 3=never
    df["currently_married"] = np.where(
        df["MSTATUS"] == 1.0, 1.0,
        np.where(df["MSTATUS"].isin([2.0, 3.0]), 0.0, np.nan),
    )

    return df


def _map_ses_vars(df: pd.DataFrame) -> pd.DataFrame:
    """Map MICS SES variables to the standardized schema."""
    # wealth_quintile: windex5 is 1-5, same coding as DHS v190
    wq = df["windex5"].copy()
    wq = wq.where((wq >= 1.0) & (wq <= 5.0))
    df["wealth_quintile"] = wq.astype("float64")

    wealth_map = {1.0: "Poor", 2.0: "Poor", 3.0: "Middle", 4.0: "Rich", 5.0: "Rich"}
    df["wealth_cat"] = df["wealth_quintile"].map(wealth_map).astype("object")

    # urban: HH6=1(Urban)->1, HH6=2(Rural)->0
    df["urban"] = np.where(
        df["HH6"] == 1.0, 1.0,
        np.where(df["HH6"] == 2.0, 0.0, np.nan),
    )

    return df


def _map_survey_design_vars(df: pd.DataFrame) -> pd.DataFrame:
    """Map MICS survey design variables to the standardized schema."""
    # sample_weight: wmweight is already at analysis scale (mean ~1.0)
    # Unlike DHS v005 (6 implied decimal places), MICS wmweight needs no rescaling.
    df["sample_weight"] = df["wmweight"].where(df["wmweight"] > 0).astype("float64")

    # psu: HH1 is the cluster identifier (primary sampling unit)
    df["psu"] = df["HH1"].astype("float64")

    # stratum: not directly available in BH file; use HH6 (urban/rural) as proxy
    # so weighted analyses can at least distinguish strata
    df["stratum"] = df["HH6"].astype("float64")

    # cluster_id: alias for psu (kept for explicit reference in sensitivity analyses)
    df["cluster_id"] = df["psu"]

    return df


def _construct_outcome(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct died_under5 and age_at_death_months from MICS variables.

    BH5=1 (YES/alive), BH5=2 (NO/dead).
    BH9C + BH9U give age at death in mixed units.
    died_under5=1 only if dead AND age_at_death < 60 months.
    """
    df["age_at_death_months"] = _age_at_death_months(df)

    dead = (df["BH5"] == 2)
    age_ok = df["age_at_death_months"].notna() & (df["age_at_death_months"] < 60.0)

    df["died_under5"] = np.where(
        dead & age_ok, 1.0,
        np.where(
            dead & df["age_at_death_months"].notna() & (df["age_at_death_months"] >= 60.0),
            0.0,
            np.where(
                dead & df["age_at_death_months"].isna(),
                np.nan,    # dead but unknown age -> unknown outcome
                0.0,       # alive -> did not die under 5
            ),
        ),
    )

    return df


def _construct_temporal_vars(
    df: pd.DataFrame,
    survey_year: int,
    doi_cmc_col: str = "WDOI",
) -> pd.DataFrame:
    """
    Compute birth_year, years_before_survey, is_recent_birth from CMC dates.

    CMC (Century Month Code) = (year - 1900) * 12 + month.
    Birth year from BH4C: year = 1900 + floor((BH4C - 1) / 12).

    We intentionally align ``years_before_survey`` to the DHS canonical
    schema: integer survey-year minus integer birth-year. That keeps the
    pooled parquet consistent with the downstream audit and sensitivity
    analyses that operate on the harmonized DHS definition.
    """
    dob_cmc = df["BH4C"].where(df["BH4C"] > 0)
    doi_cmc = df[doi_cmc_col].where(df[doi_cmc_col] > 0) if doi_cmc_col in df.columns else pd.Series(np.nan, index=df.index)

    # Convert CMC to calendar year
    df["birth_year"] = (1900.0 + np.floor((dob_cmc - 1.0) / 12.0)).astype("float64")
    df["dob_cmc"] = dob_cmc.astype("float64")
    df["doi_cmc"] = doi_cmc.astype("float64")

    # Match DHS harmonization: integer survey-year minus integer birth-year.
    df["years_before_survey"] = (float(survey_year) - df["birth_year"]).astype("float64")

    # is_recent_birth: born within 5 years before interview
    df["is_recent_birth"] = np.where(
        df["years_before_survey"].notna(),
        (df["years_before_survey"] <= 5.0).astype("float64"),
        np.nan,
    )

    return df


def _add_missingness_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Binary missingness indicators for ML — matches DHS pattern in s03_harmonize.py."""
    miss_vars = [
        "wealth_quintile", "birth_weight_kg",
        "maternal_bmi", "small_at_birth",
        # Healthcare vars not in MICS main output -> set as always-missing
        "anc_visits", "facility_delivery", "caesarean", "tetanus_protected",
    ]
    for var in miss_vars:
        col = df[var] if var in df.columns else pd.Series(np.nan, index=df.index)
        df[f"{var}_missing"] = col.isna().astype(int)
    return df


# ---------------------------------------------------------------------------
# Household size from hl.sav
# ---------------------------------------------------------------------------

def _read_household_size(zf: zipfile.ZipFile) -> Optional[pd.DataFrame]:
    """
    Count rows per HH1+HH2 in hl.sav as household_size.
    Returns a DataFrame with columns [HH1, HH2, household_size], or None.
    """
    member = _find_member(zf, "hl.sav")
    if member is None:
        return None
    try:
        hl = _read_sav_from_zip(zf, member)
        if "HH1" not in hl.columns or "HH2" not in hl.columns:
            return None
        sizes = (
            hl.groupby(["HH1", "HH2"], sort=False)
            .size()
            .reset_index(name="household_size")
        )
        return sizes
    except Exception as exc:
        print(f"[MICS] Warning: could not read hl.sav for household size: {exc}")
        return None


# ---------------------------------------------------------------------------
# Main harmonization function
# ---------------------------------------------------------------------------

# Canonical output column order, matching global_analysis.parquet schema.
_OUTPUT_COLS = [
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
    # Healthcare (excluded from MICS main output; NaN placeholders for schema parity)
    "anc_visits", "anc_visits_cat",
    "facility_delivery", "caesarean", "tetanus_protected",
    # Birth size (not in MICS BH file; NaN placeholders)
    "birth_size", "small_at_birth", "birth_weight_kg",
    # Household
    "hh_members", "maternal_bmi",
    # Missingness indicators
    "wealth_quintile_missing", "anc_visits_missing",
    "facility_delivery_missing", "caesarean_missing",
    "tetanus_protected_missing", "birth_weight_kg_missing",
    "maternal_bmi_missing", "small_at_birth_missing",
    # Scope flag
    "in_primary_scope",
    # MICS-specific provenance
    "data_source", "country_iso3", "cluster_id",
]


def harmonize_mics_survey(
    zip_path: str,
    country_iso3: str,
    country_name: str,
    survey_year: int,
    who_region: str,
) -> pd.DataFrame:
    """
    Read one MICS zip file and return a DataFrame with the standardized column schema.

    Parameters
    ----------
    zip_path : str
        Path to the MICS country zip file.
    country_iso3 : str
        ISO 3166-1 alpha-3 country code (e.g., "SLE").
    country_name : str
        Human-readable country name from the catalog.
    survey_year : int
        Survey year from the catalog (e.g., 2017).
    who_region : str
        WHO/project region code (e.g., "SSA", "SSEA", "LAC", "MENA", "ECA").

    Returns
    -------
    pd.DataFrame
        Harmonized DataFrame with standardized columns.
        Returns an empty DataFrame (with correct columns) if bh.sav is missing.
    """
    empty = pd.DataFrame(columns=_OUTPUT_COLS)

    if not os.path.exists(zip_path):
        print(f"[MICS] ERROR: zip not found: {zip_path}")
        return empty

    with zipfile.ZipFile(zip_path, "r") as zf:
        # --- Locate bh.sav (required) ---
        bh_member = _find_member(zf, "bh.sav")
        if bh_member is None:
            print(f"[MICS] WARNING: bh.sav not found in {zip_path}")
            return empty

        print(f"[MICS] Reading {bh_member} ...")
        bh = _read_sav_from_zip(zf, bh_member)
        print(f"[MICS] BH rows: {len(bh):,}  cols: {len(bh.columns)}")

        # --- Merge wm.sav for any columns missing from BH ---
        # In MICS6 the BH file ships with WM-level covariates pre-merged.
        # We attempt a merge anyway so the code is robust to older MICS files
        # where the BH file does not carry WM columns.
        wm_member = _find_member(zf, "wm.sav")
        _wm_merge_cols = [
            "HH1", "HH2", "WM3",
            "welevel", "windex5", "HH6", "MSTATUS", "wmweight",
            "WDOI", "WDOB", "HH7", "HH7A",
        ]
        wm_cols_missing = [c for c in _wm_merge_cols[3:] if c not in bh.columns]

        if wm_cols_missing and wm_member is not None:
            print(f"[MICS] Merging wm.sav for columns: {wm_cols_missing}")
            wm = _read_sav_from_zip(zf, wm_member)
            # MICS6 has both WM3 and LN; MICS5 has only LN. Pick what's in both.
            if "WM3" in bh.columns and "WM3" in wm.columns:
                line_key = "WM3"
            elif "LN" in bh.columns and "LN" in wm.columns:
                line_key = "LN"
            else:
                line_key = None
            if line_key is None:
                print("[MICS] WARNING: no common line-number key (WM3/LN) between BH and WM -- skipping merge")
            else:
                wm_key = ["HH1", "HH2", line_key]
                merge_cols = [c for c in wm_cols_missing if c in wm.columns]
                if merge_cols:
                    bh = bh.merge(
                        wm[wm_key + merge_cols].drop_duplicates(subset=wm_key),
                        on=wm_key,
                        how="left",
                    )
                    print(f"[MICS] After WM merge (key={line_key}): {len(bh):,} rows")

        # --- Household size from hl.sav ---
        hl_sizes = _read_household_size(zf)

    # --- Variable mapping ---
    df = bh.copy()

    df = _construct_temporal_vars(df, survey_year=survey_year)
    df = _construct_outcome(df)
    df = _map_child_vars(df)
    df = _map_maternal_vars(df)
    df = _map_ses_vars(df)
    df = _map_survey_design_vars(df)

    # --- Household size ---
    if hl_sizes is not None:
        df = df.merge(hl_sizes, on=["HH1", "HH2"], how="left")
        df["hh_members"] = df["household_size"].astype("float64")
        df = df.drop(columns=["household_size"])
    else:
        df["hh_members"] = np.nan

    # --- Columns not available in MICS BH file; NaN for schema parity ---
    for col in ["anc_visits", "facility_delivery", "caesarean", "tetanus_protected",
                "birth_size", "small_at_birth", "birth_weight_kg", "maternal_bmi"]:
        df[col] = np.nan

    df["anc_visits_cat"] = np.nan

    # --- Missingness indicators ---
    df = _add_missingness_indicators(df)

    # --- Metadata / provenance columns ---
    df["country_code"] = country_iso3          # use ISO3 as the code for MICS rows
    df["country_name"] = country_name
    df["country_iso3"] = country_iso3
    df["who_region"] = who_region
    df["survey_year"] = float(survey_year)
    df["data_source"] = "MICS"
    df["_source_file"] = os.path.basename(zip_path)

    # --- D006 scope flag: 2000-2024 ---
    df["in_primary_scope"] = int(2000 <= survey_year <= 2024)

    # --- Select and order output columns ---
    available = [c for c in _OUTPUT_COLS if c in df.columns]
    df_out = df[available].copy()

    # Validation summary
    n_total = len(df_out)
    n_died = df_out["died_under5"].sum()
    n_valid = df_out["died_under5"].notna().sum()
    u5mr = (n_died / n_valid * 1000) if n_valid > 0 else np.nan

    print(
        f"[MICS] {country_iso3} {survey_year}: "
        f"{n_total:,} births, U5MR={u5mr:.0f}/1000, "
        f"{len(available)} cols"
    )

    if n_valid > 0 and (u5mr < 5 or u5mr > 500):
        print(
            f"[MICS] WARNING: U5MR={u5mr:.0f}/1000 outside plausible range [5, 500]"
        )

    return df_out


# ---------------------------------------------------------------------------
# __main__ smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ZIP = (
        r"J:\Survey Research\03 Temporal NAS\expand_global"
        r"\data\raw\mics\Sierra Leone MICS6 Datasets.zip"
    )

    print("=" * 60)
    print("MICS Harmonization Smoke Test")
    print(f"File: {ZIP}")
    print("=" * 60)

    result = harmonize_mics_survey(
        zip_path=ZIP,
        country_iso3="SLE",
        country_name="Sierra Leone",
        survey_year=2017,
        who_region="SSA",
    )

    print()
    print(f"Output shape: {result.shape}")
    print()
    print("Dtypes:")
    print(result.dtypes.to_string())
    print()
    print("died_under5 value_counts:")
    print(result["died_under5"].value_counts(dropna=False).sort_index())
    print()
    print("Sample 3 rows (key columns):")
    key_cols = [
        "country_iso3", "survey_year", "died_under5",
        "age_at_death_months", "child_male", "multiple_birth",
        "birth_order", "birth_interval_months", "maternal_age_at_birth",
        "edu_level", "wealth_quintile", "urban",
        "sample_weight", "psu", "hh_members",
        "birth_year", "years_before_survey", "in_primary_scope",
    ]
    avail_key = [c for c in key_cols if c in result.columns]
    print(result[avail_key].head(3).to_string())
    print()
    print("Column coverage vs canonical schema:")
    from config.paths import GLOBAL_PARQUET
    import pyarrow.parquet as pq
    canon_cols = set(pq.read_schema(str(GLOBAL_PARQUET)).names)
    out_cols = set(result.columns)
    missing_from_output = canon_cols - out_cols - {"age_at_death_months", "cluster_id", "data_source", "country_iso3", "dob_cmc", "doi_cmc"}
    extra_in_output = out_cols - canon_cols
    print(f"  Canonical cols: {len(canon_cols)}")
    print(f"  Output cols:    {len(out_cols)}")
    if missing_from_output:
        print(f"  Missing (in canonical but not output): {sorted(missing_from_output)}")
    if extra_in_output:
        print(f"  Extra (MICS-only): {sorted(extra_in_output)}")
    print()
    print("Done.")
