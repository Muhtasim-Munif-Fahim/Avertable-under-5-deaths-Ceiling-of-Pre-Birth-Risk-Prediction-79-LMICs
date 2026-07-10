"""
s05n_gee_sensitivity_weighted_dhsonly.py -- GEE aOR robustness sensitivities
============================================================================

Reviewer sensitivities (Lancet GH review 2026-06-13), REPORT-ONLY, NON-DESTRUCTIVE:
  - R2.2: survey-weighted GEE for the headline adjusted ORs (Model 3).
  - R1.3: DHS-only GEE (drop the 16 MICS surveys).

Refits the PRIMARY reported model (Model 3) from s05b_multilevel_regression.py
under three scenarios and compares the headline aORs. The unweighted-all arm is
a sanity check against the published Table 2. Does NOT overwrite any canonical
table; writes only a new sensitivity CSV.

Output: output/tables/table_s_gee_sensitivity_weighted_dhsonly.csv
Run:    python -m src.inferential.s05n_gee_sensitivity_weighted_dhsonly
"""
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.paths import TABLES_DIR
from src.utils.io import load_systematic_sample
from src.inferential._encoding import encode_cats as _encode_cats

FORMULA_M3 = (
    "died_under5 ~ sex_cat + birth_type + bo_cat + bi_cat "
    "+ ma_cat + edu_cat + wq_cat + marital_cat + urban_cat + decade"
)

# Headline params to extract (statsmodels Treatment-coded names).
HEADLINE = {
    "edu_cat[T.Secondary]": "Education: Secondary",
    "edu_cat[T.Higher]": "Education: Higher",
    "bi_cat[T.<24 mo]": "Birth interval: <24 mo",
    "bi_cat[T.48+ mo]": "Birth interval: 48+ mo",
    "birth_type[T.Multiple]": "Multiple birth",
    "ma_cat[T.<20]": "Maternal age: <20",
    "bo_cat[T.6th+]": "Birth order: 6th+",
    "urban_cat[T.Urban]": "Residence: Urban",
    "wq_cat[T.Richest]": "Wealth: Richest",
}


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
    return df.sort_values("country_code").reset_index(drop=True)


def fit_gee(df, weighted):
    kw = {}
    if weighted:
        kw["weights"] = np.asarray(
            df["sample_weight"].fillna(1.0).clip(lower=1e-9), dtype=float)
    model = smf.gee(FORMULA_M3, groups="country_code", data=df,
                    family=Binomial(), cov_struct=Exchangeable(), **kw)
    return model.fit(maxiter=100)


def scenario(df, label, weighted, dhs_only):
    sub = df[~df["is_mics"]] if dhs_only else df
    sub = sub.sort_values("country_code").reset_index(drop=True)
    res = fit_gee(sub, weighted)
    ors = {}
    for p, name in HEADLINE.items():
        ors[name] = round(float(np.exp(res.params[p])), 3) if p in res.params.index else np.nan
    print(f"  {label:20} n={int(res.nobs):,} ({sub['country_code'].nunique()} ctr): "
          f"eduSec={ors['Education: Secondary']}, eduHi={ors['Education: Higher']}, "
          f"bi<24={ors['Birth interval: <24 mo']}, mult={ors['Multiple birth']}")
    return ors


def main():
    print("=" * 64)
    print("GEE Model 3 aOR sensitivities: weighted (R2.2) + DHS-only (R1.3)")
    print("=" * 64)
    df = load_df()
    print(f"Loaded {len(df):,} rows "
          f"({(~df['is_mics']).sum():,} DHS / {df['is_mics'].sum():,} MICS)\n")

    sc = {
        "canonical": scenario(df, "canonical (unwt,all)", False, False),
        "weighted": scenario(df, "weighted (R2.2)", True, False),
        "dhs_only": scenario(df, "DHS-only (R1.3)", False, True),
    }

    rows = []
    for p, name in HEADLINE.items():
        rows.append({
            "determinant": name,
            "aOR_canonical": sc["canonical"][name],
            "aOR_weighted": sc["weighted"][name],
            "aOR_dhs_only": sc["dhs_only"][name],
        })
    res = pd.DataFrame(rows)
    print("\n--- aOR comparison (Model 3) ---")
    print(res.to_string(index=False))
    out = TABLES_DIR / "table_s_gee_sensitivity_weighted_dhsonly.csv"
    res.to_csv(out, index=False)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
