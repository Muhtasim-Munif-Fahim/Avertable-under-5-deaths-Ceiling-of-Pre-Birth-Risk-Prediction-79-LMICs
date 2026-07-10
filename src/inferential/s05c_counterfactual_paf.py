"""
s05c_counterfactual_paf.py â€” Table 3: G-Computation PAFs
=========================================================

Computes Population Attributable Fractions using G-computation
(standardization / predictive margins). For each risk factor,
we predict mortality under factual conditions, then under a
counterfactual that shifts only the *worse-than-target* exposure
categories to the target level (rows already at or better than the
target -- e.g. higher education, protective 48+ month intervals --
are left unchanged), and compute:

  PAF = (R_factual - R_counterfactual) / R_factual

Bootstrap CIs by resampling countries (cluster bootstrap, D016).

Output: output/tables/table_3_gcomp_pafs.csv

References
----------
  - Hernan & Robins (2020) â€” Causal Inference: What If
  - Rockhill et al. (1998) â€” Use and misuse of PAFs, AJPH
  - Decision D014, D016 in DECISIONS.md
"""
import os
import sys
import gc
import warnings
from pathlib import Path
from typing import Any

# Prevent OpenBLAS threading issues when fitting 500 bootstrap GLMs
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.genmod.families import Binomial
from statsmodels.tools.sm_exceptions import ConvergenceWarning, PerfectSeparationError

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent.parent)
)
from src.utils.print_util import print
from config.paths import GLOBAL_PARQUET, RNG_SEED, TABLES_DIR
from src.utils.io import load_systematic_sample
from src.inferential._encoding import encode_cats as _encode_cats

TABLES_DIR.mkdir(parents=True, exist_ok=True)
SEED = RNG_SEED
N_BOOT = 500  # 500 cluster-bootstrap replicates for 95% CIs

# Counterfactual interventions: (categorical column, label, target level).
# Each PAF applies a WHOLE-POPULATION shift: the target (reference) level is
# assigned to every record, so off-target categories on BOTH sides move to the
# target (e.g. protective 48+ month intervals are moved to 24-47 as well as the
# short <24 month intervals). CF_WORSE_SOURCES (worse-than-target-only) was
# deliberately removed for the published estimates; do not re-add it.
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


def load_sample() -> pd.DataFrame:
    """Load 5% systematic sample (every 20th row) with full categorical encoding.

    Sampling rate aligned with s05b (GEE) and s05d (Wagstaff) per D015.
    """
    cols = [
        "died_under5", "country_code", "who_region",
        "sample_weight", "survey_year",
        "child_male", "multiple_birth", "birth_order",
        "birth_interval_months", "maternal_age_at_birth",
        "edu_level", "currently_married", "wealth_quintile",
        "urban",
        "in_primary_scope",
    ]
    df = load_systematic_sample(cols, step=20, scope="primary")
    _encode_cats(df)
    return df

def gcomp_paf(
    df: pd.DataFrame,
    col: str,
    counterfactual_val: str,
    model_result: Any,
    formula: str,
) -> float:
    """
    G-computation PAF for a single risk factor.

    Steps:
      1. Predict P(Y=1 | X_observed) for all individuals
      2. Create counterfactual data: set the exposure to its reference
         (target) level for the whole population
      3. Predict P(Y=1 | X_counterfactual)
      4. PAF = (mean_factual - mean_cf) / mean_factual

    Predictive margins are unweighted, consistent with the documented
    pooled population-averaged estimand (manuscript Methods; decision
    D015). This reproduces the published Table 3 PAFs.
    """
    # Factual predictions
    p_factual = model_result.predict(df)
    r_factual = float(np.mean(p_factual))

    # Counterfactual: set exposure to the reference level for everyone.
    df_cf = df.copy()
    orig_cats = df[col].cat.categories.tolist()
    df_cf[col] = pd.Categorical(
        [counterfactual_val] * len(df_cf), categories=orig_cats)

    p_cf = model_result.predict(df_cf)
    r_cf = float(np.mean(p_cf))

    if r_factual == 0:
        return 0.0
    return (r_factual - r_cf) / r_factual


def fit_paf_glm(formula: str, data: pd.DataFrame) -> Any:
    """Fit the PAF Binomial GLM unweighted.

    The estimand is the pooled, population-averaged association across the
    cohort rather than any single country's nationally representative rate,
    so the regression surface is fitted on unweighted birth-history records
    (manuscript Methods; decision D015). The g-computation margins
    (``gcomp_paf``) are likewise unweighted, so model and margins target the
    same population. Reported PAF uncertainty comes from the country-cluster
    bootstrap, not from model-based SEs.
    """
    return sm.GLM.from_formula(
        formula, data=data, family=Binomial(),
    ).fit()


def main() -> None:
    """Compute G-computation PAFs with bootstrap CIs."""
    print("=" * 60)
    print("Table 3: G-Computation PAFs")
    print("=" * 60)

    df = load_sample()
    print(f"  Sample: {len(df):,} rows")

    # Fit full model (survey-weighted; see fit_paf_glm)
    print("\nFitting full adjusted model (survey-weighted)...")
    try:
        result = fit_paf_glm(FORMULA, df)
        print(f"  Model fitted: {result.nobs} obs, "
              f"{len(result.params)} params")
    except Exception as e:
        print(f"  ERROR fitting model: {e}")
        return

    # Compute point estimate PAFs
    print("\nComputing point-estimate PAFs...")
    regions = ["All"] + sorted(df["who_region"].dropna().unique())
    rows = []

    for col, label, cf_val in RISK_FACTORS:
        row = {"risk_factor": label, "variable": col}

        for region in regions:
            if region == "All":
                sub = df
            else:
                sub = df[df["who_region"] == region]
            if len(sub) < 100:
                row[f"PAF_{region}"] = "N/A"
                continue

            paf = gcomp_paf(sub, col, cf_val, result, FORMULA)
            row[f"PAF_{region}"] = round(paf * 100, 1)

        rows.append(row)
        print(f"  {label}: PAF(All) = {row['PAF_All']:.1f}%")

    # Bootstrap CIs (All regions only, for speed)
    # Per D016: resample COUNTRY clusters with replacement (the documented
    # PSU fallback; PSU is not carried in the PAF sample). PSU-clustered
    # inference is reported separately for the GEE ORs in s05i.
    import time as _time
    print(f"\nBootstrapping CIs ({N_BOOT} reps, "
          f"cluster resampling)...")
    countries = df["country_code"].unique()
    n_countries = len(countries)

    # Pre-index country groups for fast resampling.
    # Cap each country at 5000 rows to prevent OOM when large
    # countries (India ~300K rows) are resampled multiple times.
    # Point estimates above use the full sample; this cap only
    # affects the variance estimation, which is robust to
    # within-cluster subsampling (Davison & Hinkley, 1997).
    MAX_PER_COUNTRY = 5000
    country_groups = {}
    for c in countries:
        cdf = df[df["country_code"] == c]
        if len(cdf) > MAX_PER_COUNTRY:
            cdf = cdf.sample(n=MAX_PER_COUNTRY, random_state=SEED)
        country_groups[c] = cdf
    boot_base_n = sum(len(v) for v in country_groups.values())
    print(f"  Bootstrap pool: {boot_base_n:,} rows "
          f"(capped at {MAX_PER_COUNTRY}/country)")

    boot_pafs = {
        col: [] for col, _, _ in RISK_FACTORS
    }

    np.random.seed(SEED)
    t_boot_start = _time.time()
    n_failed = 0
    for b in range(N_BOOT):
        # Cluster bootstrap: resample countries with replacement
        boot_countries = np.random.choice(
            countries, size=n_countries, replace=True
        )
        # Build bootstrap sample using pre-indexed groups
        df_boot = pd.concat(
            [country_groups[c] for c in boot_countries],
            ignore_index=True,
        )

        if len(df_boot) < 1000:
            n_failed += 1
            del df_boot
            continue

        try:
            r_boot = fit_paf_glm(FORMULA, df_boot)

            for col, _, cf_val in RISK_FACTORS:
                paf_b = gcomp_paf(
                    df_boot, col, cf_val, r_boot, FORMULA
                )
                boot_pafs[col].append(paf_b)
            del r_boot
        except (ValueError, np.linalg.LinAlgError, PerfectSeparationError):
            n_failed += 1

        del df_boot
        if (b + 1) % 50 == 0:
            gc.collect()
            elapsed = _time.time() - t_boot_start
            rate = (b + 1) / elapsed
            eta = (N_BOOT - b - 1) / rate if rate > 0 else 0
            print(f"    Bootstrap {b+1}/{N_BOOT} "
                  f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining, "
                  f"{n_failed} failed)")

    # Add CIs to results
    for i, (col, label, _) in enumerate(RISK_FACTORS):
        bvals = np.array(boot_pafs[col])
        if len(bvals) >= 10:
            ci_lo = np.percentile(bvals, 2.5) * 100
            ci_hi = np.percentile(bvals, 97.5) * 100
            rows[i]["CI_lo_All"] = round(ci_lo, 1)
            rows[i]["CI_hi_All"] = round(ci_hi, 1)
            rows[i]["PAF_All_formatted"] = (
                f"{rows[i]['PAF_All']:.1f} "
                f"({ci_lo:.1f}-{ci_hi:.1f})"
            )
        else:
            rows[i]["CI_lo_All"] = "N/A"
            rows[i]["CI_hi_All"] = "N/A"
            rows[i]["PAF_All_formatted"] = f"{rows[i]['PAF_All']:.1f}"

    result_df = pd.DataFrame(rows)

    out = TABLES_DIR / "table_3_gcomp_pafs.csv"
    result_df.to_csv(out, index=False)
    print(f"\nSaved: {out.name}")
    print(result_df[
        ["risk_factor", "PAF_All_formatted"]
    ].to_string(index=False))

    del df
    gc.collect()


if __name__ == "__main__":
    main()
