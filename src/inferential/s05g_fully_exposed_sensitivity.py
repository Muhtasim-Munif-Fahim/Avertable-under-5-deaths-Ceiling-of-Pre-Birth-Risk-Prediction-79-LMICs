"""
s05g_fully_exposed_sensitivity.py
==================================

Sensitivity analysis restricted to the fully-exposed sub-cohort
(records with ``years_before_survey >= 5``). Under this restriction the
under-five outcome is uncensored: every child has been observed for at
least 60 completed months at the date of interview.

The script reports:
  (i)    descriptive proportion of births dying before age 5, overall
         and by WHO region (both unweighted and survey-weighted);
  (ii)   sequential GEE Model 3 adjusted odds ratios on the
         fully-exposed subset, for direct comparison with the main
         Model 3 (table_2_sequential_models.csv);
  (iii)  g-computation PAFs on the fully-exposed subset for the eight
         counterfactual interventions reported in Table 3 of the main
         manuscript, paired with the corresponding all-records PAFs
         from the same fit, so reviewers can see whether censoring at
         <60 months attenuates the cohort-level PAFs in the same
         direction as the OR-preservation check (eTable S22b);
  (iv)   a short summary table for inclusion as eTable S22.

Outputs:
  output/tables/table_s22_fully_exposed.csv          (descriptive)
  output/tables/table_s22b_fully_exposed_gee.csv     (ORs)
  output/tables/table_s22c_fully_exposed_pafs.csv    (PAFs)
"""

from __future__ import annotations
import os
import sys
import gc
import warnings
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.genmod.families import Binomial
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.print_util import print
from config.paths import GLOBAL_PARQUET, RNG_SEED, TABLES_DIR

OUT = TABLES_DIR / "table_s22_fully_exposed.csv"

# Same encoding as s05c so the GEE specification matches
from src.utils.io import load_systematic_sample
from src.inferential.s05c_counterfactual_paf import (
    _encode_cats, gcomp_paf, RISK_FACTORS, FORMULA,
)


def load_for_sensitivity() -> pd.DataFrame:
    cols = [
        "died_under5", "country_code", "who_region",
        "sample_weight", "survey_year", "years_before_survey",
        "child_male", "multiple_birth", "birth_order",
        "birth_interval_months", "maternal_age_at_birth",
        "edu_level", "currently_married", "wealth_quintile",
        "urban", "in_primary_scope",
    ]
    df = load_systematic_sample(cols, step=20, scope="primary")
    _encode_cats(df)
    return df

def descriptive_rates(df_full_all: pd.DataFrame) -> pd.DataFrame:
    """Per-region weighted and unweighted U5M proportions for the full cohort
    AND the fully-exposed subset. df_full_all is the 5% sample."""
    rows = []
    for label, sub in [
        ("All recorded births", df_full_all),
        ("Fully exposed (years_before_survey >= 5)",
         df_full_all[df_full_all["years_before_survey"] >= 5]),
        ("Definitively classified (death OR years_before_survey >= 5)",
         df_full_all[(df_full_all["died_under5"] == 1)
                     | (df_full_all["years_before_survey"] >= 5)]),
    ]:
        for reg in ["All"] + sorted(df_full_all["who_region"].dropna().unique()):
            s = sub if reg == "All" else sub[sub["who_region"] == reg]
            n = len(s)
            if n == 0:
                continue
            deaths = int(s["died_under5"].sum())
            unw = 1000.0 * deaths / n
            w = s["sample_weight"].fillna(1.0).clip(lower=0)
            wei = 1000.0 * float((s["died_under5"] * w).sum() / w.sum()) if w.sum() > 0 else float("nan")
            rows.append({
                "cohort": label,
                "region": reg,
                "n_records": n,
                "n_deaths": deaths,
                "u5m_per1000_unweighted": round(unw, 2),
                "u5m_per1000_survey_weighted": round(wei, 2),
            })
    return pd.DataFrame(rows)


# Model 3 formula (no recent-birth healthcare covariates) -- same as s05c
FORMULA_M3 = (
    "died_under5 ~ sex_cat + birth_type + bo_cat + bi_cat "
    "+ ma_cat + edu_cat + wq_cat + marital_cat + urban_cat + decade"
)


def gee_m3(df_sub: pd.DataFrame, label: str) -> pd.DataFrame:
    df_sub = df_sub.sort_values("country_code").reset_index(drop=True)
    print(f"  fitting GEE Model 3 on {label}: N={len(df_sub):,}")
    try:
        m = sm.GEE.from_formula(
            FORMULA_M3, groups="country_code", data=df_sub,
            family=Binomial(), cov_struct=sm.cov_struct.Exchangeable(),
        )
        r = m.fit(maxiter=80)
    except Exception as e:
        print(f"  GEE failed: {e}; falling back to cluster-robust logistic")
        m = sm.GLM.from_formula(FORMULA_M3, data=df_sub, family=Binomial())
        # Align cluster groups to the rows actually used (post NA-drop).
        groups = df_sub.loc[m.data.row_labels, "country_code"]
        r = m.fit(cov_type="cluster", cov_kwds={"groups": groups})
    params = r.params
    conf = r.conf_int()
    rows = []
    for name in params.index:
        if name == "Intercept":
            continue
        b = params[name]
        lo, hi = conf.loc[name]
        rows.append({
            "fit": label,
            "param": name,
            "aOR": round(np.exp(b), 3),
            "CI_lo": round(np.exp(lo), 3),
            "CI_hi": round(np.exp(hi), 3),
            "p": round(getattr(r, "pvalues", pd.Series()).get(name, np.nan), 4),
        })
    return pd.DataFrame(rows)


def main() -> None:
    print("=" * 60)
    print("Fully-exposed cohort sensitivity (eTable S22)")
    print("=" * 60)
    df = load_for_sensitivity()
    print(f"  5% sample (primary scope): {len(df):,}")

    desc = descriptive_rates(df)
    print("\n--- Descriptive rates ---")
    print(desc.to_string(index=False))

    # GEE on fully-exposed
    fe = df[df["years_before_survey"] >= 5].copy()
    print(f"\n--- Fully-exposed GEE (Model 3) ---")
    gee_fe = gee_m3(fe, "fully_exposed")

    # GEE on full sample for comparison (matches main M3)
    print(f"\n--- Full-sample GEE (Model 3) for comparison ---")
    gee_all = gee_m3(df, "all_records")

    # Combine into a single sheet
    OUT.parent.mkdir(parents=True, exist_ok=True)
    desc.to_csv(OUT, index=False)
    gee_compare = pd.merge(
        gee_all.rename(columns={"aOR": "aOR_all", "CI_lo": "CI_lo_all",
                                "CI_hi": "CI_hi_all", "p": "p_all"}),
        gee_fe.rename(columns={"aOR": "aOR_fe", "CI_lo": "CI_lo_fe",
                                "CI_hi": "CI_hi_fe", "p": "p_fe"}),
        on="param", how="outer",
    )
    gee_compare = gee_compare[["param", "aOR_all", "CI_lo_all", "CI_hi_all",
                                "aOR_fe", "CI_lo_fe", "CI_hi_fe"]]
    gee_out = OUT.with_name("table_s22b_fully_exposed_gee.csv")
    gee_compare.to_csv(gee_out, index=False)
    print(f"\nwrote {OUT}")
    print(f"wrote {gee_out}")
    print("\nGEE comparison (all vs fully-exposed):")
    print(gee_compare.to_string(index=False))

    # --- PAFs on all-records vs fully-exposed ---
    print("\n--- G-computation PAFs (all-records vs fully-exposed) ---")
    paf_out = OUT.with_name("table_s22c_fully_exposed_pafs.csv")
    paf_rows = []
    for label, sub in [
        ("all_records", df),
        ("fully_exposed", df[df["years_before_survey"] >= 5].copy()),
    ]:
        try:
            m_paf = sm.GLM.from_formula(FORMULA, data=sub, family=Binomial())
            r_paf = m_paf.fit()
        except Exception as e:
            print(f"  {label} PAF fit failed: {e}")
            continue
        for col, label_text, cf_val in RISK_FACTORS:
            try:
                paf = gcomp_paf(sub, col, cf_val, r_paf, FORMULA)
                paf_rows.append({
                    "cohort": label,
                    "risk_factor": label_text,
                    "variable": col,
                    "PAF": round(paf * 100, 1),
                })
            except Exception as e:
                paf_rows.append({
                    "cohort": label,
                    "risk_factor": label_text,
                    "variable": col,
                    "PAF": float("nan"),
                })
    paf_df = pd.DataFrame(paf_rows)
    # Wide format: paf_all_records vs paf_fully_exposed columns
    if not paf_df.empty:
        wide = paf_df.pivot_table(
            index=["risk_factor", "variable"],
            columns="cohort", values="PAF",
        ).reset_index()
        wide.columns.name = None
        wide["delta_pp"] = (
            wide.get("fully_exposed", pd.Series(dtype=float))
            - wide.get("all_records", pd.Series(dtype=float))
        ).round(1)
        wide.to_csv(paf_out, index=False)
        print(f"wrote {paf_out}")
        print(wide.to_string(index=False))


if __name__ == "__main__":
    main()
