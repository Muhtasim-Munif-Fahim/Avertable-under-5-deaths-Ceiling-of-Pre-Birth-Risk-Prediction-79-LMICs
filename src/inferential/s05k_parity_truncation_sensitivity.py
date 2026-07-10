"""
s05k_parity_truncation_sensitivity.py -- MICS parity-truncation sensitivity
============================================================================

Supplement analysis N5 (2026-06 rewrite): MICS surveys truncate birth
order at 4+, compressing the upper parity tail relative to DHS. This
script quantifies the impact of that truncation on the Model-3 adjusted
odds ratios by refitting the primary GEE specification on DHS-only rows
(where fine parity is observed) under two parity codings:

  Spec A (fine, DHS native):    1st / 2nd-3rd / 4th-5th / 6th+
  Spec B (MICS-style truncated): 1st / 2nd-3rd / 4th+

If the truncation mattered for the headline estimates, the education,
interval, and other covariate aORs would shift between specs. Backs the
manuscript limitation claim that estimates are direction-preserved.

Outputs:
  output/tables/table_s_parity_truncation.csv

Run:  python -m src.inferential.s05k_parity_truncation_sensitivity
"""
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.cov_struct import Exchangeable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.print_util import print
from config.paths import TABLES_DIR
from src.utils.io import load_systematic_sample
from src.inferential._encoding import encode_cats

TABLES_DIR.mkdir(parents=True, exist_ok=True)

COLS = [
    "died_under5", "country_code", "who_region", "_source_file",
    "sample_weight", "survey_year",
    "child_male", "multiple_birth", "birth_order",
    "birth_interval_months", "maternal_age_at_birth",
    "edu_level", "currently_married", "wealth_quintile",
    "urban",
    "in_primary_scope",
]


def fit_spec(df: pd.DataFrame, bo_col: str, spec_name: str) -> pd.DataFrame:
    """Fit the Model-3 GEE spec with the given birth-order column."""
    formula = (f"died_under5 ~ sex_cat + birth_type "
               f"+ {bo_col} + bi_cat "
               f"+ ma_cat + edu_cat + wq_cat "
               f"+ marital_cat + urban_cat "
               f"+ decade")
    print(f"  Fitting {spec_name}...")
    try:
        model = smf.gee(formula, groups="country_code", data=df,
                        family=Binomial(), cov_struct=Exchangeable())
        result = model.fit(maxiter=100)
        method = "GEE"
    except Exception as e:  # same fallback as s05b
        print(f"    GEE failed ({e}), falling back to GLM+cluster SEs")
        model = smf.glm(formula, data=df, family=Binomial())
        result = model.fit(cov_type="cluster",
                           cov_kwds={"groups": df["country_code"]})
        method = "GLM+ClusterSE"
    print(f"    {method} converged")

    conf = result.conf_int()
    rows = []
    for name in result.params.index:
        if name == "Intercept":
            continue
        rows.append({
            "spec": spec_name,
            "term": name,
            "aOR": float(np.exp(result.params[name])),
            "ci_lo": float(np.exp(conf.loc[name, 0])),
            "ci_hi": float(np.exp(conf.loc[name, 1])),
            "p": float(result.pvalues[name]),
        })
    return pd.DataFrame(rows)


def main():
    print("=" * 60)
    print("Parity-truncation sensitivity (N5), DHS-only cohort")
    print("=" * 60)

    df = load_systematic_sample(COLS, step=20, scope="primary")
    df = df[~df["_source_file"].astype(str).str.contains(
        "MICS", case=False, na=False)].copy()
    print(f"  DHS-only rows: {len(df):,} "
          f"({df['country_code'].nunique()} countries)")

    encode_cats(df)

    # MICS-style truncated parity: 1st / 2nd-3rd / 4th+
    bo = df["birth_order"].values
    bo_t = np.full(len(df), "1st", dtype=object)
    bo_t[(bo >= 2) & (bo <= 3)] = "2nd-3rd"
    bo_t[bo >= 4] = "4th+"
    df["bo_cat_trunc"] = pd.Categorical(
        bo_t, categories=["1st", "2nd-3rd", "4th+"])

    df = df.sort_values("country_code").reset_index(drop=True)

    res_fine = fit_spec(df, "bo_cat", "A: fine parity (DHS native)")
    gc.collect()
    res_trunc = fit_spec(df, "bo_cat_trunc", "B: truncated parity (MICS-style)")
    gc.collect()

    out = pd.concat([res_fine, res_trunc], ignore_index=True)

    # Side-by-side comparison for shared (non-parity) terms
    shared_a = res_fine[~res_fine["term"].str.startswith("bo_cat")].set_index("term")
    shared_b = res_trunc[~res_trunc["term"].str.startswith("bo_cat")].set_index("term")
    comp = shared_a[["aOR", "ci_lo", "ci_hi"]].join(
        shared_b[["aOR", "ci_lo", "ci_hi"]],
        lsuffix="_fine", rsuffix="_trunc")
    comp["abs_diff_aOR"] = (comp["aOR_fine"] - comp["aOR_trunc"]).abs()
    comp["rel_diff_pct"] = (comp["abs_diff_aOR"] / comp["aOR_fine"] * 100).round(2)
    max_term = comp["rel_diff_pct"].idxmax()
    print(f"\n  Max relative aOR shift among shared terms: "
          f"{comp.loc[max_term, 'rel_diff_pct']:.2f}% ({max_term})")

    out_csv = TABLES_DIR / "table_s_parity_truncation.csv"
    out.to_csv(out_csv, index=False)
    comp.reset_index().to_csv(
        TABLES_DIR / "table_s_parity_truncation_comparison.csv", index=False)
    print(f"\nSaved: {out_csv}")
    print(f"Saved: {TABLES_DIR / 'table_s_parity_truncation_comparison.csv'}")


if __name__ == "__main__":
    main()
