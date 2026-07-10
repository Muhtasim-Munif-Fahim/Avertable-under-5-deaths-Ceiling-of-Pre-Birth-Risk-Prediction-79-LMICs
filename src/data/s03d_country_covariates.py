"""
s03d_country_covariates.py — Country-level WDI covariates
=========================================================

Downloads three World Bank WDI indicators for the 68 DHS countries in
the primary analysis scope (2000-2024) and aligns each to the nearest
available WDI observation relative to each survey year.

Indicators:
  NY.GDP.PCAP.PP.KD  →  gdp_per_capita_ppp  (PPP constant 2021 USD)
  SH.XPD.CHEX.GD.ZS  →  health_exp_pct_gdp  (% of GDP)
  SH.MED.PHYS.ZS     →  physicians_per_1000  (per 1,000 people)

Temporal alignment:
  Exact match preferred. Then backward-fill (carry last known value
  forward): up to 3 years for GDP / health expenditure, up to 5 years
  for physicians (sparse series). Remaining gaps kept as NaN.

Outputs:
  data/02_harmonized/country_covariates.parquet
  output/tables/table_s_wdi_coverage.csv

Run:
  cd "J:/Survey Research/03 Temporal NAS/study1_expand_global"
  python -m src.data.s03d_country_covariates
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import world_bank_data as wb

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.paths import COUNTRY_COVARIATES, GLOBAL_PARQUET, TABLES_DIR

TABLES_DIR.mkdir(parents=True, exist_ok=True)

INDICATORS = {
    "gdp_per_capita_ppp": "NY.GDP.PCAP.PP.KD",
    "health_exp_pct_gdp": "SH.XPD.CHEX.GD.ZS",
    "physicians_per_1000": "SH.MED.PHYS.ZS",
}

# Max years to carry last known value forward per indicator
FILL_LIMIT = {
    "gdp_per_capita_ppp": 3,
    "health_exp_pct_gdp": 3,
    "physicians_per_1000": 5,
}

YEAR_MIN = 2000
YEAR_MAX = 2024
COVERAGE_PATH = TABLES_DIR / "table_s_wdi_coverage.csv"
BATCH_SIZE = 15   # max countries per WDI API call (avoids 403 on long URLs)
BATCH_DELAY = 2   # seconds between batches to avoid rate-limiting
MAX_RETRIES = 3   # retry attempts per batch on 403/5xx

# DHS uses non-standard 2-letter country codes that differ from ISO2.
# This mapping overrides the automatic lookup for all diverging cases.
# MICS rows (added 2026-05-21) carry 3-letter ISO3 codes — also mapped here.
DHS_TO_ISO2 = {
    "BU": "BI",  # Burundi (DHS=BU, ISO2=BI)
    "DR": "DO",  # Dominican Republic (DHS=DR, ISO2=DO)
    "GU": "GT",  # Guatemala (DHS=GU, ISO2=GT; GU in ISO2 = Guam)
    "IA": "IN",  # India (DHS=IA, ISO2=IN)
    "KY": "KG",  # Kyrgyz Republic (DHS=KY, ISO2=KG; KY in ISO2 = Cayman Islands)
    "LB": "LR",  # Liberia (DHS=LB, ISO2=LR; LB in ISO2 = Lebanon)
    "MB": "MD",  # Moldova (DHS=MB, ISO2=MD)
    "MD": "MG",  # Madagascar (DHS=MD, ISO2=MG; MD in ISO2 = Moldova)
    "NC": "NI",  # Nicaragua (DHS=NC, ISO2=NI; NC in ISO2 = New Caledonia)
    "NI": "NE",  # Niger (DHS=NI, ISO2=NE; NI in ISO2 = Nicaragua)
    "NM": "NA",  # Namibia (DHS=NM, ISO2=NA)
    # MICS countries (3-letter ISO3 -> ISO2)
    "DZA": "DZ", "CAF": "CF", "GNB": "GW", "KIR": "KI", "LAO": "LA",
    "MNG": "MN", "PSE": "PS", "SDN": "SD", "TUN": "TN", "UZB": "UZ",
    "VUT": "VU",
}


def load_survey_pairs() -> pd.DataFrame:
    """Return unique (country_code, country_name, survey_year) for primary scope."""
    print("Loading survey pairs from primary scope...")
    pf = pq.ParquetFile(str(GLOBAL_PARQUET))
    frames = []
    for batch in pf.iter_batches(
        batch_size=500_000,
        columns=["country_code", "country_name", "survey_year", "in_primary_scope"],
    ):
        df = batch.to_pandas()
        df = df[df["in_primary_scope"] == 1]
        frames.append(
            df[["country_code", "country_name", "survey_year"]].drop_duplicates()
        )
    pairs = (
        pd.concat(frames)
        .drop_duplicates()
        .sort_values(["country_code", "survey_year"])
        .reset_index(drop=True)
    )
    pairs["survey_year"] = pairs["survey_year"].astype(int)
    print(
        f"  {len(pairs)} survey-year pairs, "
        f"{pairs['country_code'].nunique()} countries"
    )
    return pairs


def build_dhs_to_wdi_iso2(dhs_codes: list) -> dict:
    """
    Map each DHS country code to the ISO2 code used by the WDI API.

    DHS uses non-standard 2-letter codes that conflict with ISO2 for
    11 countries (see DHS_TO_ISO2). All others are passed through as-is.
    Returns {dhs_code: wdi_iso2}.
    """
    return {code: DHS_TO_ISO2.get(code, code) for code in dhs_codes}


def build_wdi_iso2_to_name(wdi_iso2_codes: list) -> dict:
    """Return {wdi_iso2: wdi_country_name} looked up from the WDI countries table."""
    print("Building WDI ISO2->name mapping...")
    wdi_countries = wb.get_countries()[["iso2Code", "name"]]
    mapping = {}
    missing = []
    for code in set(wdi_iso2_codes):
        row = wdi_countries[wdi_countries["iso2Code"] == code]
        if row.empty:
            missing.append(code)
        else:
            mapping[code] = row.iloc[0]["name"]
    if missing:
        print(f"  WARNING: No WDI entry for ISO2 code(s): {missing}")
    print(f"  Mapped {len(mapping)}/{len(set(wdi_iso2_codes))} ISO2 codes")
    return mapping


def fetch_indicator(
    col_name: str,
    indicator_code: str,
    wdi_name_to_dhs: dict,
) -> pd.DataFrame:
    """
    Fetch one WDI indicator for ALL countries over YEAR_MIN:YEAR_MAX,
    then filter to the DHS study countries.

    Fetching globally (no country filter) avoids 403 errors that occur
    when certain ISO2 codes (e.g. NA for Namibia) appear in batch URLs.
    Returns DataFrame with columns [country_code, year, <col_name>]
    where country_code is the DHS code.
    """
    print(f"  Fetching {indicator_code} ({col_name}) [global]...")
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            series = wb.get_series(
                indicator_code,
                date=f"{YEAR_MIN}:{YEAR_MAX}",
            )
            df = series.reset_index()
            df = df.rename(columns={"Year": "year", indicator_code: col_name})
            df["year"] = df["year"].astype(int)
            # Keep only rows for our study countries
            df["country_code"] = df["Country"].map(wdi_name_to_dhs)
            df = df.dropna(subset=["country_code"])
            n_valid = df[col_name].notna().sum()
            print(
                f"    {n_valid} non-null observations "
                f"across {df['country_code'].nunique()} countries"
            )
            return df[["country_code", "year", col_name]].copy()
        except Exception as exc:
            if attempt < MAX_RETRIES:
                print(f"  Retry {attempt}/{MAX_RETRIES}...")
                time.sleep(BATCH_DELAY * attempt)
            else:
                print(f"  ERROR: {exc}")
                return pd.DataFrame(columns=["country_code", "year", col_name])


def align_country(
    country_code: str,
    country_name: str,
    survey_years: list,
    ind_dfs: dict,
) -> tuple:
    """
    For one country, align each indicator to each survey year.

    Returns (result_rows, coverage_rows).
    """
    result_rows = []
    coverage_rows = []
    all_years = list(range(YEAR_MIN, YEAR_MAX + 1))

    for sy in survey_years:
        row = {
            "country_code": country_code,
            "country_name": country_name,
            "survey_year": sy,
        }
        for col_name in INDICATORS:
            df_ind = ind_dfs[col_name]
            country_data = df_ind[df_ind["country_code"] == country_code]
            limit = FILL_LIMIT[col_name]

            if country_data.empty:
                row[col_name] = np.nan
                status = "no_wdi_data"
            else:
                yr_map = dict(zip(country_data["year"], country_data[col_name]))
                full = pd.Series(
                    [yr_map.get(y, np.nan) for y in all_years],
                    index=all_years,
                    dtype=float,
                )
                full = full.ffill(limit=limit)
                val = full.get(sy, np.nan)
                row[col_name] = val

                has_exact = sy in yr_map and pd.notna(yr_map.get(sy))
                if has_exact:
                    status = "exact"
                elif pd.notna(val):
                    status = "filled"
                else:
                    status = "missing"

            coverage_rows.append({
                "country_code": country_code,
                "country_name": country_name,
                "survey_year": sy,
                "indicator": col_name,
                "status": status,
            })

        result_rows.append(row)

    return result_rows, coverage_rows


def main() -> None:
    print("=" * 60)
    print("s03d: Country-level WDI covariates")
    print("=" * 60)

    pairs = load_survey_pairs()
    dhs_codes = pairs["country_code"].unique().tolist()

    # Build DHS code -> WDI ISO2 mapping
    dhs_to_wdi_iso2 = build_dhs_to_wdi_iso2(dhs_codes)
    wdi_iso2_codes = list(set(dhs_to_wdi_iso2.values()))

    # Build WDI ISO2 -> country name, then invert to name -> DHS code
    wdi_iso2_to_name = build_wdi_iso2_to_name(wdi_iso2_codes)
    # wdi_name -> dhs_code (for tagging fetched rows with the DHS code)
    wdi_name_to_dhs = {
        wdi_iso2_to_name[iso2]: dhs
        for dhs, iso2 in dhs_to_wdi_iso2.items()
        if iso2 in wdi_iso2_to_name
    }

    print("\nFetching WDI indicators...")
    ind_dfs = {}
    for col_name, indicator_code in INDICATORS.items():
        ind_dfs[col_name] = fetch_indicator(col_name, indicator_code, wdi_name_to_dhs)
        time.sleep(BATCH_DELAY)

    print("\nAligning to survey years...")
    all_result_rows = []
    all_coverage_rows = []
    for country_code, grp in pairs.groupby("country_code"):
        country_name = grp["country_name"].iloc[0]
        survey_years = grp["survey_year"].tolist()
        res, cov = align_country(
            country_code, country_name, survey_years, ind_dfs
        )
        all_result_rows.extend(res)
        all_coverage_rows.extend(cov)

    result_df = pd.DataFrame(all_result_rows)
    cov_df = pd.DataFrame(all_coverage_rows)

    # Coverage summary
    print("\nCoverage summary (survey-year pairs):")
    summary = (
        cov_df.groupby(["indicator", "status"])
        .size()
        .unstack(fill_value=0)
    )
    print(summary.to_string())

    print("\nMissing rates:")
    for col in INDICATORS:
        n = result_df[col].isna().sum()
        pct = n / len(result_df) * 100
        print(f"  {col}: {n}/{len(result_df)} missing ({pct:.1f}%)")

    # Save parquet
    COUNTRY_COVARIATES.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(result_df, preserve_index=False)
    pq.write_table(table, str(COUNTRY_COVARIATES))
    print(f"\nSaved: {COUNTRY_COVARIATES}")
    print(f"  Rows: {len(result_df)}, Columns: {result_df.shape[1]}")

    # Save coverage report
    cov_df.to_csv(COVERAGE_PATH, index=False)
    print(f"Saved: {COVERAGE_PATH}")

    print("\nSample output:")
    print(result_df.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
