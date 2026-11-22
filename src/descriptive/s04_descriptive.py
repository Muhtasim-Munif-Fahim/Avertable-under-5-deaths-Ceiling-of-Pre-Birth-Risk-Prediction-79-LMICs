"""s04_descriptive.py — v2 descriptive tables on the exact cohort.

v2 port of the manuscript-cited descriptive tables (S1 survey inventory,
S2 characteristics by region, S3 mortality by decade x region). Follows
Analysis Plan v2:

  * Mortality is the exact >=60-month complete-cohort risk (died_under5_complete
    on u5_binary_eligible == 1), survey-wave-normalised weights — NOT the legacy
    all-birth recorded-event proportion, which §4 forbids labelling "U5MR".
  * Full data (no 5% sample). Cohort/weights match the phase2 publication.

country_name is not in the v2 candidate; it is joined from the legacy canonical
code->name map. hh_members is absent from the candidate and omitted (flagged).

Output: output/phase4/manuscript/table_s{1,2,3}.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from config.paths import HARMONIZED_DIR, OUTPUT_DIR, PROJECT_ROOT
from src.inferential.phase2_analysis import _positive_normalized_weights, _read_pilot

OUTDIR = OUTPUT_DIR / "phase4" / "manuscript"
# country_code -> country_name, extracted once from the harmonised DHS/MICS base (2026-09-24)
COUNTRY_NAMES = PROJECT_ROOT / "data" / "catalog" / "country_names.csv"


def _country_names() -> dict[str, str]:
    tbl = pd.read_csv(COUNTRY_NAMES)
    tbl = tbl.dropna().drop_duplicates("country_code")
    return dict(zip(tbl["country_code"].astype(str), tbl["country_name"].astype(str)))


def _wavg(x: np.ndarray, w: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    return float(np.average(x[m], weights=w[m])) if m.any() else float("nan")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    frame = _read_pilot(HARMONIZED_DIR / "global_analysis_v2_candidate.parquet")
    w_all = _positive_normalized_weights(frame)
    # Exact >=60-month complete cohort (primary estimand population, §4).
    elig = pd.to_numeric(frame["died_under5_complete"], errors="coerce").notna() & w_all.notna()
    d = frame.loc[elig].copy()
    w = w_all.loc[elig].to_numpy(float)
    y = pd.to_numeric(d["died_under5_complete"], errors="coerce").to_numpy(float)
    names = _country_names()

    # ---- Table S1: survey inventory (exact-cohort risk per survey) ----
    d1 = d.assign(_w=w, _y=y)
    rows = []
    for src, g in d1.groupby("_source_file"):
        gw, gy = g["_w"].to_numpy(), g["_y"].to_numpy()
        cc = str(g["country_code"].iloc[0])
        rows.append({
            "survey_id": src, "country_code": cc,
            "country_name": names.get(cc, cc),
            "who_region": g["who_region"].iloc[0],
            "survey_year": int(pd.to_numeric(g["survey_year"], errors="coerce").iloc[0]),
            "n_births_exact": len(g), "n_deaths": int(gy.sum()),
            "u5_risk_per1000_exact": round(_wavg(gy, gw) * 1000, 1),
        })
    s1 = pd.DataFrame(rows).sort_values(["who_region", "country_code", "survey_year"])
    s1.to_csv(OUTDIR / "table_s1_survey_inventory.csv", index=False)
    print(f"[S1] {len(s1)} surveys, {int(s1['n_births_exact'].sum()):,} exact births")

    # ---- Table S2: weighted characteristics by WHO region ----
    male = pd.to_numeric(d["child_male"], errors="coerce").to_numpy()
    mult = pd.to_numeric(d["multiple_birth"], errors="coerce").to_numpy()
    bord = pd.to_numeric(d["birth_order"], errors="coerce").to_numpy()
    bint = pd.to_numeric(d["birth_interval_months"], errors="coerce").to_numpy()
    mage = pd.to_numeric(d["maternal_age_at_birth"], errors="coerce").to_numpy()
    edu = pd.to_numeric(d["edu_level"], errors="coerce").to_numpy()
    urb = pd.to_numeric(d["urban"], errors="coerce").to_numpy()
    mar = pd.to_numeric(d["currently_married"], errors="coerce").to_numpy()
    region = d["who_region"].astype(str).to_numpy()

    def region_row(mask):
        ww = w[mask]
        return {
            "n_births": int(mask.sum()),
            "u5_risk_per1000": round(_wavg(y[mask], ww) * 1000, 1),
            "male_pct": round(_wavg(male[mask], ww) * 100, 1),
            "multiple_birth_pct": round(_wavg(mult[mask], ww) * 100, 1),
            "mean_birth_order": round(_wavg(bord[mask], ww), 2),
            "mean_birth_interval_mo": round(_wavg(bint[mask], ww), 1),
            "mean_maternal_age": round(_wavg(mage[mask], ww), 1),
            "no_education_pct": round(_wavg((edu[mask] == 0).astype(float), ww) * 100, 1),
            "urban_pct": round(_wavg(urb[mask], ww) * 100, 1),
            "married_pct": round(_wavg(mar[mask], ww) * 100, 1),
        }

    s2_rows = []
    for reg in sorted(pd.unique(region)):
        r = region_row(region == reg)
        r = {"region": reg, **r}
        s2_rows.append(r)
    s2_rows.append({"region": "All", **region_row(np.ones(len(d), bool))})
    s2 = pd.DataFrame(s2_rows)
    s2.to_csv(OUTDIR / "table_s2_characteristics.csv", index=False)
    print(f"[S2] {len(s2)} region rows")

    # ---- Table S3: exact-cohort mortality by decade x region ----
    syear = pd.to_numeric(d["survey_year"], errors="coerce").to_numpy()
    decade = np.select(
        [(syear >= 2000) & (syear <= 2009), (syear >= 2010) & (syear <= 2019), syear >= 2020],
        ["2000-2009", "2010-2019", "2020-2024"], default="Other")
    s3_rows = []
    for reg in sorted(pd.unique(region)) + ["All"]:
        for dec in ["2000-2009", "2010-2019", "2020-2024"]:
            m = (decade == dec) & ((region == reg) if reg != "All" else np.ones(len(d), bool))
            if m.sum() < 100:
                continue
            s3_rows.append({
                "region": reg, "decade": dec, "n_births": int(m.sum()),
                "n_deaths": int(y[m].sum()),
                "u5_risk_per1000": round(_wavg(y[m], w[m]) * 1000, 1),
            })
    s3 = pd.DataFrame(s3_rows)
    s3.to_csv(OUTDIR / "table_s3_mortality_trends.csv", index=False)
    print(f"[S3] {len(s3)} region x decade rows")

    # Overall descriptive risk (exact cohort, survey-weighted)
    overall = _wavg(y, w) * 1000
    print(f"[overall] exact-cohort survey-weighted U5 risk = {overall:.1f} per 1000 "
          f"(n={len(d):,}, deaths={int(y.sum()):,})")
    print(f"[note] hh_members absent from v2 candidate — omitted from S2.")


if __name__ == "__main__":
    main()
