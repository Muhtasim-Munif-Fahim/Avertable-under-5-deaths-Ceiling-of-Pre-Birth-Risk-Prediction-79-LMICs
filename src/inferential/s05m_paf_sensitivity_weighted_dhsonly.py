"""
s05m_paf_sensitivity_weighted_dhsonly.py -- PAF robustness sensitivities
========================================================================

Reviewer sensitivities (Lancet GH review 2026-06-13), REPORT-ONLY, NON-DESTRUCTIVE:
  - R2.2: survey-weighted g-computation PAFs (vs the canonical unweighted ones).
  - R1.3: DHS-only PAFs (drop the 16 MICS surveys) to test the MICS case-mix.

It re-uses the canonical g-computation recipe from s05c_counterfactual_paf.py
(same 5% systematic sample, same FORMULA, same whole-population counterfactual
shift). The ONLY differences per scenario are (a) freq_weights on the GLM +
weighted predictive margins (R2.2) and (b) a DHS-only row filter (R1.3).

It does NOT modify s05c or overwrite table_3_gcomp_pafs.csv. Point estimates
only (no bootstrap) -- the decision-relevant question is whether weighting /
MICS-exclusion moves the point PAFs, especially birth spacing.

Output: output/tables/table_s_paf_sensitivity_weighted_dhsonly.csv
Run:    python -m src.inferential.s05m_paf_sensitivity_weighted_dhsonly
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.genmod.families import Binomial

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.paths import TABLES_DIR
from src.utils.io import load_systematic_sample
from src.inferential._encoding import encode_cats as _encode_cats

# Copied verbatim from s05c (kept standalone to avoid side effects).
RISK_FACTORS = [
    ("edu_cat", "Education < Secondary -> Secondary+", "Secondary"),
    ("wq_cat", "Wealth < Richest -> Richest", "Richest"),
    ("urban_cat", "Rural -> Urban", "Urban"),
    ("ma_cat", "Maternal age != 20-34 -> 20-34", "20-34"),
    ("bi_cat", "Birth interval != 24-47mo -> 24-47mo", "24-47 mo"),
    ("marital_cat", "Unmarried -> Married", "Married"),
    ("birth_type", "Multiple -> Singleton", "Singleton"),
    ("sex_cat", "Male -> Female", "Female"),
]
FORMULA = (
    "died_under5 ~ sex_cat + birth_type + bo_cat + bi_cat "
    "+ ma_cat + edu_cat + wq_cat + marital_cat + urban_cat "
    "+ decade"
)


def load_df():
    cols = [
        "died_under5", "country_code", "who_region", "_source_file",
        "sample_weight", "survey_year",
        "child_male", "multiple_birth", "birth_order",
        "birth_interval_months", "maternal_age_at_birth",
        "edu_level", "currently_married", "wealth_quintile",
        "urban", "in_primary_scope",
    ]
    df = load_systematic_sample(cols, step=20, scope="primary")
    df["is_mics"] = df["_source_file"].astype(str).str.contains(
        "MICS", case=False, na=False)
    _encode_cats(df)
    return df


def fit_glm(data, weights=None):
    if weights is None:
        return sm.GLM.from_formula(FORMULA, data=data, family=Binomial()).fit()
    w = np.asarray(pd.Series(weights).fillna(1.0).clip(lower=1e-9), dtype=float)
    return sm.GLM.from_formula(
        FORMULA, data=data, family=Binomial(), freq_weights=w).fit()


def gcomp_paf(df, col, cf_val, model, weights=None):
    p_fac = np.asarray(model.predict(df), dtype=float)
    df_cf = df.copy()
    cats = df[col].cat.categories.tolist()
    df_cf[col] = pd.Categorical([cf_val] * len(df_cf), categories=cats)
    p_cf = np.asarray(model.predict(df_cf), dtype=float)
    if weights is None:
        r_fac, r_cf = p_fac.mean(), p_cf.mean()
    else:
        w = np.asarray(pd.Series(weights).fillna(1.0).clip(lower=1e-9), dtype=float)
        r_fac, r_cf = np.average(p_fac, weights=w), np.average(p_cf, weights=w)
    return 0.0 if r_fac == 0 else (r_fac - r_cf) / r_fac


def run_scenario(df, label, weighted, dhs_only):
    sub = df[~df["is_mics"]] if dhs_only else df
    w = sub["sample_weight"] if weighted else None
    model = fit_glm(sub, weights=w)
    out = {}
    for col, _, cf_val in RISK_FACTORS:
        out[col] = round(gcomp_paf(sub, col, cf_val, model, weights=w) * 100, 1)
    out["_n"] = len(sub)
    out["_countries"] = sub["country_code"].nunique()
    print(f"  {label:22} n={out['_n']:,} ({out['_countries']} countries): "
          f"edu={out['edu_cat']}, spacing={out['bi_cat']}, wealth={out['wq_cat']}")
    return out


def main():
    print("=" * 64)
    print("PAF sensitivities: survey-weighted (R2.2) + DHS-only (R1.3)")
    print("=" * 64)
    df = load_df()
    print(f"Loaded {len(df):,} rows "
          f"({(~df['is_mics']).sum():,} DHS / {df['is_mics'].sum():,} MICS)\n")

    scenarios = {
        "canonical (unwt,all)": run_scenario(df, "canonical (unwt,all)", False, False),
        "weighted (R2.2)":      run_scenario(df, "weighted (R2.2)", True, False),
        "DHS-only (R1.3)":      run_scenario(df, "DHS-only (R1.3)", False, True),
    }

    rows = []
    for col, label, _ in RISK_FACTORS:
        rows.append({
            "risk_factor": label,
            "PAF_canonical": scenarios["canonical (unwt,all)"][col],
            "PAF_weighted": scenarios["weighted (R2.2)"][col],
            "PAF_dhs_only": scenarios["DHS-only (R1.3)"][col],
        })
    res = pd.DataFrame(rows)
    res["delta_weighted"] = (res["PAF_weighted"] - res["PAF_canonical"]).round(1)
    res["delta_dhs_only"] = (res["PAF_dhs_only"] - res["PAF_canonical"]).round(1)

    print("\n--- PAF comparison (point estimates, %) ---")
    print(res.to_string(index=False))

    out = TABLES_DIR / "table_s_paf_sensitivity_weighted_dhsonly.csv"
    res.to_csv(out, index=False)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
