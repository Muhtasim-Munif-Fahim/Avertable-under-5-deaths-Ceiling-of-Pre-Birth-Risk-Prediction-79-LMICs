"""
s05b_multilevel_regression.py â€” Table 2: Sequential GEE Models
================================================================

Produces the core regression table with categorical ORs, reference
groups, and model comparison statistics. Uses GEE with exchangeable
correlation clustered on country (D013) with robust sandwich SEs.

Sequential models (Mosley-Chen framework):
  Model 1: Child factors only
  Model 2: + Maternal & SES factors
  Model 3: + Decade (temporal adjustment); primary reported model
  Model 4: + Country-level WDI covariates (GDP, health spend, physicians)
  Model 5: + Recent-birth healthcare variables (recent-birth subsample)

Between-country variation is captured by the country-level clustering of
the GEE (region is nested within country), so no separate region fixed
effect is included.

Fallback: If GEE fails, uses GLM + cluster-robust SEs (equivalent).

Output: output/tables/table_2_sequential_models.csv

References
----------
  - Liang & Zeger (1986) â€” GEE methodology
  - Mosley & Chen (1984) â€” Proximate determinants framework
  - Decision D013 in DECISIONS.md
"""
import os
import sys
import gc
import warnings
from pathlib import Path

# Prevent OpenBLAS from spawning worker threads, which causes LAPACK
# workspace allocation failures and segfaults when multiple GEE models
# are fitted sequentially in the same process (Windows + Python 3.13).
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent.parent)
)
from src.utils.print_util import print
from config.paths import (
    COUNTRY_COVARIATES, GLOBAL_PARQUET, RNG_SEED, TABLES_DIR,
)
from src.utils.io import load_systematic_sample
from src.inferential._encoding import encode_cats

TABLES_DIR.mkdir(parents=True, exist_ok=True)
PARQUET = GLOBAL_PARQUET

SEED = RNG_SEED


def load_sample() -> pd.DataFrame:
    """Load 5% systematic sample (every 20th row) with categorical encodings.

    Implements decision D015: 5% systematic sample of the canonical parquet,
    yielding ~512k records after the in_primary_scope filter. Sampling rate is
    aligned across s05b (GEE), s05c (PAF GLM), s05d (Wagstaff), s05g
    (fully-exposed) so headline tables share a denominator.
    """
    cols = [
        "died_under5", "country_code", "who_region",
        "sample_weight", "psu", "survey_year",
        "child_male", "multiple_birth", "birth_order",
        "birth_interval_months", "maternal_age_at_birth",
        "edu_level", "currently_married", "wealth_quintile",
        "urban",
        # Healthcare utilization (Model 5)
        "anc_visits", "facility_delivery", "caesarean",
        "tetanus_protected", "is_recent_birth",
        # Scope flag (D006)
        "in_primary_scope",
    ]
    df = load_systematic_sample(cols, step=20, scope="primary")

    print(f"  Sample: {len(df):,} rows, "
          f"{df['country_code'].nunique()} countries")

    # Standardised categorical encoding (shared with s05c / s05g)
    encode_cats(df)


    # ANC visits: ref = 4+ (WHO standard)
    anc = df["anc_visits"].values
    anc_cat = np.full(len(df), "4+ visits", dtype=object)
    anc_cat[np.isnan(anc)] = "Missing/DK"
    anc_cat[(~np.isnan(anc)) & (anc == 0)] = "None"
    anc_cat[(~np.isnan(anc)) & (anc >= 1) & (anc <= 3)] = "1-3 visits"
    df["anc_cat"] = pd.Categorical(
        anc_cat,
        categories=["4+ visits", "None", "1-3 visits", "Missing/DK"],
    )

    # Facility delivery: ref = Facility
    df["delivery_cat"] = pd.Categorical(
        np.where(
            df["facility_delivery"].fillna(-1) == 1,
            "Facility", "Home/Other"
        ),
        categories=["Facility", "Home/Other"],
    )

    # Caesarean: ref = No
    df["csection_cat"] = pd.Categorical(
        np.where(
            df["caesarean"].fillna(0) == 1,
            "Yes", "No"
        ),
        categories=["No", "Yes"],
    )

    # Tetanus: ref = Protected
    df["tetanus_cat"] = pd.Categorical(
        np.where(
            df["tetanus_protected"].fillna(0) == 1,
            "Protected", "Not protected"
        ),
        categories=["Protected", "Not protected"],
    )

    # Sort by group for GEE
    df = df.sort_values("country_code").reset_index(drop=True)
    return df

def join_wdi_covariates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join country-level WDI covariates onto the sample.

    Adds columns: log_gdp_ppp, health_exp_pct_gdp, physicians_per_1000.
    survey_year is matched exactly (no interpolation â€” alignment was
    already done in s03d with backward fill).
    Rows without WDI data get NaN and are excluded from Model 4 only.
    """
    wdi = pq.read_table(str(COUNTRY_COVARIATES)).to_pandas()
    wdi["survey_year"] = wdi["survey_year"].astype(int)
    wdi = wdi[["country_code", "survey_year",
               "gdp_per_capita_ppp", "health_exp_pct_gdp",
               "physicians_per_1000"]].copy()

    df["survey_year_int"] = df["survey_year"].astype(int)
    df = df.merge(wdi, left_on=["country_code", "survey_year_int"],
                  right_on=["country_code", "survey_year"],
                  how="left", suffixes=("", "_wdi"))
    df = df.drop(columns=["survey_year_wdi", "survey_year_int"],
                 errors="ignore")

    # Log-transform GDP per capita (right-skewed, covers 3 orders of magnitude)
    df["log_gdp_ppp"] = np.log(df["gdp_per_capita_ppp"].clip(lower=1))

    n_wdi = df["log_gdp_ppp"].notna().sum()
    pct = n_wdi / len(df) * 100
    print(f"  WDI coverage: {n_wdi:,}/{len(df):,} rows ({pct:.1f}%) "
          f"have all country covariates")
    return df


def fit_gee_model(
    df: pd.DataFrame,
    formula: str,
    model_name: str,
) -> dict:
    """
    Fit GEE model with fallback to GLM + robust SEs.

    Returns dict with coefficients, SEs, CIs, p-values.
    """
    print(f"  Fitting {model_name}...")

    try:
        model = smf.gee(
            formula,
            groups="country_code",
            data=df,
            family=Binomial(),
            cov_struct=Exchangeable(),
        )
        result = model.fit(maxiter=100)
        method = "GEE"
        print(f"    GEE converged ({method})")
    except Exception as e:
        print(f"    GEE failed ({e}), falling back to GLM...")
        model = smf.glm(
            formula, data=df, family=Binomial(),
        )
        result = model.fit(
            cov_type="cluster",
            cov_kwds={"groups": df["country_code"]},
        )
        method = "GLM+ClusterSE"
        print(f"    GLM with cluster-robust SEs fitted")

    # Extract results
    params = result.params
    conf = result.conf_int()
    pvals = result.pvalues

    records = []
    for name in params.index:
        if name == "Intercept":
            continue
        or_val = np.exp(params[name])
        ci_lo = np.exp(conf.loc[name, 0])
        ci_hi = np.exp(conf.loc[name, 1])
        p = pvals[name]
        records.append({
            "param": name,
            "OR": round(or_val, 3),
            "CI_lo": round(ci_lo, 3),
            "CI_hi": round(ci_hi, 3),
            "p": p,
            "model": model_name,
            "method": method,
        })

    # Model stats
    n_obs = int(result.nobs) if hasattr(result, 'nobs') else len(df)
    stats = {
        "n_obs": n_obs,
        "method": method,
    }
    if hasattr(result, 'llf') and result.llf is not None:
        stats["loglik"] = round(result.llf, 1)
    if hasattr(result, 'aic') and result.aic is not None:
        stats["aic"] = round(result.aic, 1)

    # Drop the fitted result object â€” it holds the full design matrix in memory.
    # We only need records and stats downstream.
    del result
    gc.collect()
    return {"records": records, "stats": stats}


def compute_null_icc(df: pd.DataFrame) -> float:
    """
    Approximate ICC from country-level mortality proportions.

    ICC = tau^2 / (tau^2 + pi^2/3)

    Uses country-level variance in mortality proportions as a proxy for
    the between-country variance component. This is an empirical
    approximation, not a formal GLMM variance component.
    """
    # Country-level mean mortality
    country_means = df.groupby("country_code")["died_under5"].mean()
    grand_mean = df["died_under5"].mean()

    # Between-country variance
    tau2 = country_means.var()
    # Within-country (logistic) variance
    sigma2 = (np.pi ** 2) / 3

    icc = tau2 / (tau2 + sigma2)
    return round(icc, 4)


def format_results(all_models: dict) -> pd.DataFrame:
    """Combine all model results into publication table."""
    # Variable label mapping
    label_map = {
        "sex_cat[T.Male]": ("Child sex", "Male"),
        "birth_type[T.Multiple]": ("Birth type", "Multiple"),
        "bo_cat[T.2nd-3rd]": ("Birth order", "2nd-3rd"),
        "bo_cat[T.4th-5th]": ("Birth order", "4th-5th"),
        "bo_cat[T.6th+]": ("Birth order", "6th+"),
        "bi_cat[T.First/missing]": ("Birth interval", "First/missing"),
        "bi_cat[T.<24 mo]": ("Birth interval", "<24 months"),
        "bi_cat[T.48+ mo]": ("Birth interval", "48+ months"),
        "ma_cat[T.<20]": ("Maternal age", "<20"),
        "ma_cat[T.35+]": ("Maternal age", "35+"),
        "edu_cat[T.Primary]": ("Education", "Primary"),
        "edu_cat[T.Secondary]": ("Education", "Secondary"),
        "edu_cat[T.Higher]": ("Education", "Higher"),
        "wq_cat[T.Poorer]": ("Wealth", "Poorer"),
        "wq_cat[T.Middle]": ("Wealth", "Middle"),
        "wq_cat[T.Richer]": ("Wealth", "Richer"),
        "wq_cat[T.Richest]": ("Wealth", "Richest"),
        "marital_cat[T.Married]": ("Marital status", "Married"),
        "urban_cat[T.Urban]": ("Residence", "Urban"),
        "decade[T.2010-2019]": ("Decade", "2010-2019"),
        "decade[T.2020-2024]": ("Decade", "2020-2024"),
        # WDI country-level (Model 4) â€” continuous, per-unit OR
        "log_gdp_ppp": ("GDP/capita (log PPP)", "per log unit"),
        "health_exp_pct_gdp": ("Health expenditure", "per 1% GDP"),
        "physicians_per_1000": ("Physicians", "per 1/1,000"),
        # Healthcare (Model 5)
        "anc_cat[T.None]": ("ANC visits", "None"),
        "anc_cat[T.1-3 visits]": ("ANC visits", "1-3 visits"),
        "anc_cat[T.Missing/DK]": ("ANC visits", "Missing/DK"),
        "delivery_cat[T.Home/Other]": ("Delivery place", "Home/Other"),
        "csection_cat[T.Yes]": ("Caesarean", "Yes"),
        "tetanus_cat[T.Not protected]": ("Tetanus", "Not protected"),
    }

    # Reference groups to insert
    ref_groups = [
        ("Child sex", "Female", "Ref"),
        ("Birth type", "Singleton", "Ref"),
        ("Birth order", "1st", "Ref"),
        ("Birth interval", "24-47 months", "Ref"),
        ("Maternal age", "20-34", "Ref"),
        ("Education", "None", "Ref"),
        ("Wealth", "Poorest", "Ref"),
        ("Marital status", "Not married", "Ref"),
        ("Residence", "Rural", "Ref"),
        ("Decade", "2000-2009", "Ref"),
        # Healthcare
        ("ANC visits", "4+ visits", "Ref"),
        ("Delivery place", "Facility", "Ref"),
        ("Caesarean", "No", "Ref"),
        ("Tetanus", "Protected", "Ref"),
    ]

    rows = []
    for variable, level, _ in ref_groups:
        row = {"variable": variable, "level": level}
        for model_name in all_models:
            row[f"aOR_{model_name}"] = "1.00 (Ref)"
            row[f"OR_{model_name}"] = 1.0
            row[f"CIlo_{model_name}"] = 1.0
            row[f"CIhi_{model_name}"] = 1.0
            row[f"p_{model_name}"] = np.nan
        rows.append(row)

        # Get levels for this variable
        for model_name, mdata in all_models.items():
            for rec in mdata["records"]:
                param = rec["param"]
                if param in label_map:
                    var, lev = label_map[param]
                    if var == variable:
                        # Check if already added
                        existing = [
                            r for r in rows
                            if r["variable"] == var and r["level"] == lev
                        ]
                        if not existing:
                            new_row = {"variable": var, "level": lev}
                            rows.append(new_row)

    # Now fill in the OR values
    for row in rows:
        if row.get(f"aOR_Model1") == "1.00 (Ref)":
            continue  # Reference group
        for model_name, mdata in all_models.items():
            col = f"aOR_{model_name}"
            if col in row:
                continue
            found = False
            for rec in mdata["records"]:
                if rec["param"] in label_map:
                    var, lev = label_map[rec["param"]]
                    if var == row["variable"] and lev == row["level"]:
                        sig = (
                            "***" if rec["p"] < 0.001
                            else "**" if rec["p"] < 0.01
                            else "*" if rec["p"] < 0.05
                            else ""
                        )
                        row[col] = (
                            f"{rec['OR']:.3f} "
                            f"({rec['CI_lo']:.3f}-"
                            f"{rec['CI_hi']:.3f}){sig}"
                        )
                        row[f"OR_{model_name}"] = rec["OR"]
                        row[f"CIlo_{model_name}"] = rec["CI_lo"]
                        row[f"CIhi_{model_name}"] = rec["CI_hi"]
                        row[f"p_{model_name}"] = rec["p"]
                        found = True
                        break
            if not found:
                row[col] = ""
                row[f"OR_{model_name}"] = np.nan
                row[f"CIlo_{model_name}"] = np.nan
                row[f"CIhi_{model_name}"] = np.nan
                row[f"p_{model_name}"] = np.nan

    # Continuous variables (WDI): appended as single rows, OR filled directly.
    # These have no reference category so they are not in ref_groups.
    continuous_params = [
        "log_gdp_ppp",
        "health_exp_pct_gdp",
        "physicians_per_1000",
    ]
    for param in continuous_params:
        if param not in label_map:
            continue
        var, lev = label_map[param]
        row = {"variable": var, "level": lev}
        for model_name, mdata in all_models.items():
            col = f"aOR_{model_name}"
            found = False
            for rec in mdata["records"]:
                if rec["param"] == param:
                    sig = (
                        "***" if rec["p"] < 0.001
                        else "**" if rec["p"] < 0.01
                        else "*" if rec["p"] < 0.05
                        else ""
                    )
                    row[col] = (
                        f"{rec['OR']:.3f} "
                        f"({rec['CI_lo']:.3f}-{rec['CI_hi']:.3f}){sig}"
                    )
                    row[f"OR_{model_name}"] = rec["OR"]
                    row[f"CIlo_{model_name}"] = rec["CI_lo"]
                    row[f"CIhi_{model_name}"] = rec["CI_hi"]
                    row[f"p_{model_name}"] = rec["p"]
                    found = True
                    break
            if not found:
                row[col] = ""
                row[f"OR_{model_name}"] = np.nan
                row[f"CIlo_{model_name}"] = np.nan
                row[f"CIhi_{model_name}"] = np.nan
                row[f"p_{model_name}"] = np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    """Run sequential GEE models."""
    print("=" * 60)
    print("Table 2: Sequential GEE Regression Models")
    print("=" * 60)

    df = load_sample()
    df = join_wdi_covariates(df)

    # Approximate null-model ICC
    icc = compute_null_icc(df)
    print(f"\n  Approximate null-model ICC (country): {icc:.4f} "
          f"({icc*100:.1f}% of variance between countries)")

    # Model 1: Child factors only
    f1 = ("died_under5 ~ sex_cat + birth_type "
          "+ bo_cat + bi_cat")
    m1 = fit_gee_model(df, f1, "Model1")
    gc.collect()

    # Model 2: + Maternal & SES
    f2 = ("died_under5 ~ sex_cat + birth_type "
          "+ bo_cat + bi_cat "
          "+ ma_cat + edu_cat + wq_cat "
          "+ marital_cat + urban_cat")
    m2 = fit_gee_model(df, f2, "Model2")
    gc.collect()

    # Model 3: + Decade (temporal adjustment)
    f3 = ("died_under5 ~ sex_cat + birth_type "
          "+ bo_cat + bi_cat "
          "+ ma_cat + edu_cat + wq_cat "
          "+ marital_cat + urban_cat "
          "+ decade")
    m3 = fit_gee_model(df, f3, "Model3")
    gc.collect()

    # Model 4: + Country-level structural factors (WDI)
    # Restricted to rows where all three WDI indicators are available.
    # log_gdp_ppp is log-transformed in join_wdi_covariates().
    df_wdi = df[
        df["log_gdp_ppp"].notna()
        & df["health_exp_pct_gdp"].notna()
        & df["physicians_per_1000"].notna()
    ].sort_values("country_code").reset_index(drop=True)
    print(f"  Model 4 (WDI): {len(df_wdi):,} rows, "
          f"{df_wdi['country_code'].nunique()} countries")
    f4 = ("died_under5 ~ sex_cat + birth_type "
          "+ bo_cat + bi_cat "
          "+ ma_cat + edu_cat + wq_cat "
          "+ marital_cat + urban_cat "
          "+ decade "
          "+ log_gdp_ppp + health_exp_pct_gdp + physicians_per_1000")
    m4 = fit_gee_model(df_wdi, f4, "Model4")

    # Model 5: + Healthcare utilization (recent births only)
    # ANC/delivery variables are collected only for the most recent birth
    # in each DHS interview. Restricting to is_recent_birth avoids
    # attributing historical births to contemporaneous care.
    df_recent = df[
        df.get("is_recent_birth", pd.Series(dtype=float)).fillna(0) == 1
    ].copy()
    if len(df_recent) < 5000:
        print("  WARNING: is_recent_birth too sparse, using full sample for Model 5")
        df_recent = df.copy()
    else:
        print(f"  Model 5 (healthcare): {len(df_recent):,} recent births")
    df_recent = df_recent.sort_values("country_code").reset_index(drop=True)
    f5 = ("died_under5 ~ sex_cat + birth_type "
          "+ bo_cat + bi_cat "
          "+ ma_cat + edu_cat + wq_cat "
          "+ marital_cat + urban_cat "
          "+ decade "
          "+ anc_cat + delivery_cat + csection_cat + tetanus_cat")
    m5 = fit_gee_model(df_recent, f5, "Model5")

    all_models = {"Model1": m1, "Model2": m2, "Model3": m3,
                  "Model4": m4, "Model5": m5}

    # Format and save
    table = format_results(all_models)
    out = TABLES_DIR / "table_2_sequential_models.csv"
    table.to_csv(out, index=False)
    print(f"\nSaved: {out.name}")
    print(f"  Rows: {len(table)}")

    # Print summary stats
    print("\n--- Model Summary ---")
    print(f"  Approximate ICC (null): {icc:.4f}")
    for name, mdata in all_models.items():
        s = mdata["stats"]
        print(f"  {name}: N={s['n_obs']:,}, "
              f"method={s['method']}")

    # Save model stats separately
    stats_rows = [{"metric": "ICC_null", "value": icc}]
    for name, mdata in all_models.items():
        for k, v in mdata["stats"].items():
            stats_rows.append({
                "metric": f"{name}_{k}", "value": v,
            })
    stats_df = pd.DataFrame(stats_rows)
    stats_out = TABLES_DIR / "table_2_model_stats.csv"
    stats_df.to_csv(stats_out, index=False)
    print(f"  Saved: {stats_out.name}")

    # Fix #8: dedicated ICC/VPC table for downstream reporting
    vpc = icc  # VPC = ICC for logistic random-intercept (latent scale)
    icc_df = pd.DataFrame([{
        "model": "Null (country random intercept)",
        "icc": icc,
        "vpc": vpc,
        "note": "ICC = tau^2 / (tau^2 + pi^2/3); empirical approximation (not a formal GLMM variance component); country-level clustering",
    }])
    icc_out = TABLES_DIR / "table_1b_icc_vpc.csv"
    icc_df.to_csv(icc_out, index=False)
    print(f"  Saved: {icc_out.name}")

    del df
    gc.collect()


if __name__ == "__main__":
    main()
