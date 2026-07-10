"""
Per-country x survey-decade recorded under-five mortality (survey-weighted).

Single reusable artifact consumed by:
  - journal_tables.build_T1()  -> per-country mortality-trend column (first vs
    last surveyed decade, %change)
  - journal_tables.build_S6()  -> 79-country x decade U5M grid (eTable)

Method is IDENTICAL to s04_descriptive_tables.table_s3() (the source of the
abstract's weighted decade trend 111.2 -> 74.8 -> 62.6): decade is binned on
survey_year, and U5M is the sample-weighted rate per 1,000 via
weighted_rate_per1000. This guarantees T1's summary-row trend reproduces the
published headline exactly and that country rows pool to it.

Output: output/tables/table_s_country_decade_u5m.csv
Columns: country_code, who_region, decade, n_births, u5mr_per1000

Run: python -m src.descriptive.s04b_country_decade
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from src.utils.io import load_primary, weighted_rate_per1000

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "output" / "tables" / "table_s_country_decade_u5m.csv"

DECADE_BINS = [1999, 2009, 2019, 2025]
DECADE_LABELS = ["2000-2009", "2010-2019", "2020-2024"]


def build() -> pd.DataFrame:
    df = load_primary(
        columns=["country_code", "who_region", "survey_year",
                 "died_under5", "sample_weight"],
        scope="primary",
    )
    df["decade"] = pd.cut(df["survey_year"], bins=DECADE_BINS, labels=DECADE_LABELS)

    rows = []
    for (cc, dec), sub in df.groupby(["country_code", "decade"], observed=True):
        if len(sub) == 0:
            continue
        rows.append({
            "country_code": cc,
            "who_region": sub["who_region"].iloc[0],
            "decade": str(dec),
            "n_births": int(len(sub)),
            "u5mr_per1000": round(
                weighted_rate_per1000(
                    sub["died_under5"].values, sub["sample_weight"].values
                ), 1),
        })
    out = pd.DataFrame(rows).sort_values(["country_code", "decade"]).reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    # Sanity: pooled decade rates should reproduce 111.2 / 74.8 / 62.6
    glob = []
    for dec in DECADE_LABELS:
        s = df[df["decade"] == dec]
        glob.append((dec, len(s),
                     round(weighted_rate_per1000(
                         s["died_under5"].values, s["sample_weight"].values), 1)))
    print(f"Wrote {OUT} ({out['country_code'].nunique()} countries, {len(out)} country-decades)")
    for dec, n, r in glob:
        print(f"  GLOBAL {dec}: n={n:,}  weighted U5M/1k={r}")
    n_trend = (out.groupby("country_code")["decade"].nunique() >= 2).sum()
    print(f"  countries with >=2 surveyed decades (trend computable): {n_trend}")
    return out


if __name__ == "__main__":
    build()
