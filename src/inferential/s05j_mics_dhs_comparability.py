"""
s05j_mics_dhs_comparability.py -- MICS-vs-DHS comparability analysis
=====================================================================

Supplement analysis N1 (2026-06 rewrite): quantifies how the 16 MICS
surveys (11 MICS-only countries) compare with the 199 DHS surveys on
covariate distributions and the recorded under-5 mortality outcome.

Because the merge rule is geographic-gaps-only (no country contributes
both programmes), DHS-vs-MICS differences conflate programme effects
with country effects. This analysis therefore makes no causal claim;
it documents the magnitude of compositional difference a reader should
expect, using survey-weighted means/proportions and standardized mean
differences (SMD, weighted; Austin 2009 binary formula for proportions).

Outputs:
  output/tables/table_s_mics_dhs_comparability.csv

Run:  python -m src.inferential.s05j_mics_dhs_comparability
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.print_util import print
from config.paths import TABLES_DIR
from src.utils.io import load_systematic_sample, weighted_mean

TABLES_DIR.mkdir(parents=True, exist_ok=True)

# 5% systematic sample, aligned with s05b/s05c/s05d (decision D015)
SAMPLE_STEP = 20

COLS = [
    "died_under5", "country_code", "who_region", "_source_file",
    "sample_weight", "survey_year",
    "child_male", "multiple_birth", "birth_order",
    "birth_interval_months", "maternal_age_at_birth",
    "edu_level", "currently_married", "wealth_quintile",
    "urban", "hh_members",
    "in_primary_scope",
]

# (label, column, kind) -- kind: "binary" uses proportion SMD,
# "continuous" uses pooled-SD SMD.
VARIABLES = [
    ("Recorded U5M (per 1,000)", "died_under5", "binary"),
    ("Male child (%)", "child_male", "binary"),
    ("Multiple birth (%)", "multiple_birth", "binary"),
    ("Birth order (mean)", "birth_order", "continuous"),
    ("Preceding interval, months (mean)", "birth_interval_months", "continuous"),
    ("Maternal age at birth, years (mean)", "maternal_age_at_birth", "continuous"),
    ("No maternal education (%)", "edu_none", "binary"),
    ("Secondary+ maternal education (%)", "edu_sec_plus", "binary"),
    ("Currently married (%)", "currently_married", "binary"),
    ("Poorest wealth quintile (%)", "wq_poorest", "binary"),
    ("Urban residence (%)", "urban", "binary"),
    ("Household members (mean)", "hh_members", "continuous"),
]


def weighted_var(x: np.ndarray, w: np.ndarray) -> float:
    """Weighted variance with the same NaN handling as weighted_mean."""
    mask = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if mask.sum() < 2:
        return np.nan
    m = np.average(x[mask], weights=w[mask])
    return float(np.average((x[mask] - m) ** 2, weights=w[mask]))


def smd(x1, w1, x2, w2, kind: str) -> float:
    """Weighted standardized mean difference (DHS minus MICS)."""
    m1, m2 = weighted_mean(x1, w1), weighted_mean(x2, w2)
    if kind == "binary":
        denom = np.sqrt((m1 * (1 - m1) + m2 * (1 - m2)) / 2.0)
    else:
        denom = np.sqrt((weighted_var(x1, w1) + weighted_var(x2, w2)) / 2.0)
    if not np.isfinite(denom) or denom == 0:
        return np.nan
    return float((m1 - m2) / denom)


def main():
    print("=" * 60)
    print("MICS-vs-DHS comparability (N1)")
    print("=" * 60)

    df = load_systematic_sample(COLS, step=SAMPLE_STEP, scope="primary")
    df["is_mics"] = df["_source_file"].astype(str).str.contains(
        "MICS", case=False, na=False)

    # Derived indicator columns
    edu = df["edu_level"].values
    df["edu_none"] = np.where(np.isnan(edu), np.nan, (edu == 0).astype(float))
    df["edu_sec_plus"] = np.where(np.isnan(edu), np.nan, (edu >= 2).astype(float))
    wq = df["wealth_quintile"].values
    df["wq_poorest"] = np.where(np.isnan(wq), np.nan, (wq == 1).astype(float))

    dhs = df[~df["is_mics"]]
    mics = df[df["is_mics"]]
    print(f"  DHS rows:  {len(dhs):,} ({dhs['country_code'].nunique()} countries)")
    print(f"  MICS rows: {len(mics):,} ({mics['country_code'].nunique()} countries)")

    rows = []
    for label, col, kind in VARIABLES:
        x1, w1 = dhs[col].values.astype(float), dhs["sample_weight"].values
        x2, w2 = mics[col].values.astype(float), mics["sample_weight"].values
        m1, m2 = weighted_mean(x1, w1), weighted_mean(x2, w2)
        d = smd(x1, w1, x2, w2, kind)
        scale = 1000.0 if "per 1,000" in label else (100.0 if kind == "binary" else 1.0)
        rows.append({
            "variable": label,
            "dhs_value": round(m1 * scale, 2) if np.isfinite(m1) else np.nan,
            "mics_value": round(m2 * scale, 2) if np.isfinite(m2) else np.nan,
            "smd_dhs_minus_mics": round(d, 3) if np.isfinite(d) else np.nan,
            "n_dhs": int(np.isfinite(x1).sum()),
            "n_mics": int(np.isfinite(x2).sum()),
        })
        print(f"  {label:<42} DHS={rows[-1]['dhs_value']:>8}  "
              f"MICS={rows[-1]['mics_value']:>8}  SMD={rows[-1]['smd_dhs_minus_mics']}")

    out = pd.DataFrame(rows)
    out_csv = TABLES_DIR / "table_s_mics_dhs_comparability.csv"
    out.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    # Per-MICS-country mini-summary (context for small Pacific surveys)
    per_country = (
        mics.groupby(["country_code", "who_region"], observed=True)
        .agg(n_births=("died_under5", "size"),
             u5m_per1000=("died_under5", lambda s: round(s.mean() * 1000, 1)),
             survey_years=("survey_year",
                           lambda s: "/".join(str(int(v)) for v in sorted(s.unique()))))
        .reset_index()
        .sort_values(["who_region", "country_code"])
    )
    out2 = TABLES_DIR / "table_s_mics_country_summary.csv"
    per_country.to_csv(out2, index=False)
    print(f"Saved: {out2}")


if __name__ == "__main__":
    main()
