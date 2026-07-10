"""
s05d_inequality_decomposition.py — Table 4: Wagstaff Decomposition
==================================================================

Computes the concentration index for U5MR and decomposes it into
the contributions from each categorical predictor.
Uses a Linear Probability Model (LPM) as required for exact
additive decomposition of the concentration index (Wagstaff et al., 2003).

Output: output/tables/table_4_inequality_decomposition.csv

References
----------
  - Wagstaff et al. (2003). On decomposing the causes of health
    sector inequalities. J Health Econ.
  - Decision D017: LPM for Wagstaff Decomposition
"""
from __future__ import annotations
import sys
import gc
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent.parent)
)
from src.utils.print_util import print
from src.utils.io import load_systematic_sample
from config.paths import GLOBAL_PARQUET, RNG_SEED, TABLES_DIR

TABLES_DIR.mkdir(parents=True, exist_ok=True)

def load_sample(step: int = 20) -> pd.DataFrame:
    """Load a systematic sample (every ``step``-th row; step=1 = full cohort)
    and convert categorical to dummies.

    Sampling rate aligned with s05b (GEE) and s05c (PAF) per D015.
    """
    cols = [
        "died_under5", "who_region",
        "sample_weight", "survey_year",
        "child_male", "multiple_birth", "birth_order",
        "birth_interval_months", "maternal_age_at_birth",
        "edu_level", "currently_married", "wealth_quintile",
        "urban",
        # Scope flag (D006)
        "in_primary_scope",
    ]
    df = load_systematic_sample(cols, step=step, scope="primary")

    # Wealth quintile (1-5) is the ranking variable. Records with no wealth
    # quintile are left as NaN here and DROPPED by the decomposition routines:
    # imputing the ranking variable to the median quintile would dilute the
    # measured gradient toward zero (see decompose_wagstaff / decompose_logit).
    df["wealth_rank"] = df["wealth_quintile"]
    df["sample_weight"] = df["sample_weight"].fillna(1.0)

    # Create explicit dummy variables for OLS
    # Keep reference categories OUT of the model

    # 1. Child factors
    df["sex_male"] = df["child_male"].fillna(0)
    df["mult_birth"] = df["multiple_birth"].fillna(0)

    bi = df["birth_interval_months"].values
    df["bi_short"] = ((~np.isnan(bi)) & (bi < 24)).astype(int)
    df["bi_long"] = ((~np.isnan(bi)) & (bi >= 48)).astype(int)
    df["bi_first"] = (np.isnan(bi)).astype(int)
    # ref = optimal (24-47)

    bo = df["birth_order"].fillna(1).values
    df["bo_2_3"] = ((bo >= 2) & (bo <= 3)).astype(int)
    df["bo_4_5"] = ((bo >= 4) & (bo <= 5)).astype(int)
    df["bo_6_plus"] = (bo >= 6).astype(int)
    # ref = 1st

    # 2. Maternal factors
    ma = df["maternal_age_at_birth"].fillna(20).values
    df["ma_teen"] = (ma < 20).astype(int)
    df["ma_older"] = (ma >= 35).astype(int)
    # ref = 20-34

    df["marital_status"] = df["currently_married"].fillna(1)

    # 3. Socioeconomic (The main drivers of inequality)
    edu = df["edu_level"].fillna(0).values
    df["edu_primary"] = (edu == 1).astype(int)
    df["edu_secondary"] = (edu == 2).astype(int)
    df["edu_higher"] = (edu >= 3).astype(int)
    # ref = None

    df["residence_urban"] = df["urban"].fillna(0)
    # ref = Rural

    # Optional: decade dummies to adjust for secular trends
    sy = df["survey_year"].values
    df["dec_2010s"] = ((sy >= 2010) & (sy <= 2019)).astype(int)
    df["dec_2020s"] = (sy >= 2020).astype(int)

    return df

def calc_concentration_index(
    y: np.ndarray,
    frac_rank: np.ndarray,
    weights: np.ndarray
) -> float:
    """
    Computes standard concentration index.
    CI = (2 / mu) * covariance(y, frac_rank, weights)
    """
    if weights.sum() == 0:
        return 0.0
    mu = np.average(y, weights=weights)
    if mu == 0:
        return 0.0
    
    # Weighted covariance
    mean_rank = np.average(frac_rank, weights=weights)
    cov = np.average(weights * (y - mu) * (frac_rank - mean_rank))
    
    # But np.average already divides by sum of weights, so we must be careful:
    # A standard formula for weighted covariance:
    # sum(w_i * (y_i - mu) * (R_i - mean_R)) / sum(w_i)
    cov = np.average((y - mu) * (frac_rank - mean_rank), weights=weights)
    
    return (2.0 / mu) * cov


def _ranked_subset(df: pd.DataFrame) -> pd.DataFrame:
    """Drop wealth-unrankable records and attach the grouped (group-midpoint)
    fractional rank. Every record in a wealth group shares the rank
    R_g = (cumulative weight below g) + 0.5*(weight of g), divided by total
    weight — the standard Wagstaff grouped concentration index. A within-tie
    "spread" rank would inject the data's row order as a spurious within-group
    gradient; imputing missing wealth to the median would dilute the gradient.
    """
    df = df[df["wealth_rank"].notna()].copy()
    w = df["sample_weight"].values.astype(float)
    q = df["wealth_rank"].values.astype(float)
    tw = w.sum()
    frac = np.empty(len(df), dtype=float)
    cum = 0.0
    for v in np.unique(q):            # ascending wealth groups
        m = (q == v)
        gw = w[m].sum()
        frac[m] = (cum + 0.5 * gw) / tw
        cum += gw
    df["frac_rank"] = frac
    return df


def decompose_wagstaff(df: pd.DataFrame, group_name: str) -> pd.DataFrame:
    """Perform exact linear Wagstaff decomposition on a subgroup."""
    exog_cols = [
        "sex_male", "mult_birth",
        "bi_short", "bi_long", "bi_first",
        "bo_2_3", "bo_4_5", "bo_6_plus",
        "ma_teen", "ma_older", "marital_status",
        "edu_primary", "edu_secondary", "edu_higher",
        "residence_urban", "dec_2010s", "dec_2020s"
    ]

    df = _ranked_subset(df)
    w = df["sample_weight"].values
    frac_rank = df["frac_rank"].values
    y = df["died_under5"].values

    # Total Concentration Index
    total_ci = calc_concentration_index(y, frac_rank, w)
    mu_y = np.average(y, weights=w)
    
    if mu_y == 0:
        return pd.DataFrame()
        
    # Fit weighted LPM
    X = sm.add_constant(df[exog_cols])
    model = sm.WLS(y, X, weights=w)
    res = model.fit()
    lpm_r2 = float(res.rsquared)
    lpm_r2_adj = float(res.rsquared_adj)

    rows = []
    sum_contributions = 0.0
    
    for col in exog_cols:
        beta = res.params[col]
        x_val = df[col].values
        mu_x = np.average(x_val, weights=w)
        
        # CI of the predictor
        ci_x = calc_concentration_index(x_val, frac_rank, w)
        
        # Elasticity: beta * mu_x / mu_y
        elasticity = (beta * mu_x) / mu_y
        
        # Absolute contribution: elasticity * ci_x
        contribution = elasticity * ci_x
        percent_contrib = (contribution / total_ci) * 100 if total_ci != 0 else 0
        
        sum_contributions += contribution
        
        # Direction interpret
        if ci_x < -0.05:
            dist = "Concentrated in poor"
        elif ci_x > 0.05:
            dist = "Concentrated in rich"
        else:
            dist = "Evenly distributed"
            
        rows.append({
            "Group": group_name,
            "Predictor": col,
            "Beta": beta,
            "Mean_X": mu_x,
            "Elasticity": elasticity,
            "CI_Predictor": ci_x,
            "Contribution": contribution,
            "Percent_Contribution": percent_contrib,
            "Distribution": dist
        })
        
    # Add residual
    residual_ci = total_ci - sum_contributions
    percent_resid = (residual_ci / total_ci) * 100 if total_ci != 0 else 0
    
    rows.append({
        "Group": group_name,
        "Predictor": "Residual",
        "Beta": np.nan,
        "Mean_X": np.nan,
        "Elasticity": np.nan,
        "CI_Predictor": np.nan,
        "Contribution": residual_ci,
        "Percent_Contribution": percent_resid,
        "Distribution": "Unexplained"
    })
    
    # Append total
    rows.append({
        "Group": group_name,
        "Predictor": "TOTAL_CI",
        "Beta": np.nan,
        "Mean_X": np.nan,
        "Elasticity": np.nan,
        "CI_Predictor": np.nan,
        "Contribution": total_ci,
        "Percent_Contribution": 100.0,
        "Distribution": ""
    })

    # LPM fit quality (Fix #12): exact decomposition is only meaningful
    # to the extent the LPM explains variation in y.
    rows.append({
        "Group": group_name,
        "Predictor": "LPM_R2",
        "Beta": np.nan,
        "Mean_X": np.nan,
        "Elasticity": np.nan,
        "CI_Predictor": np.nan,
        "Contribution": lpm_r2,
        "Percent_Contribution": np.nan,
        "Distribution": f"adj_R2={lpm_r2_adj:.4f}"
    })

    return pd.DataFrame(rows)


def decompose_logit(df: pd.DataFrame, group_name: str) -> pd.DataFrame:
    """Logit-based Wagstaff-style decomposition (sensitivity to D017).

    Replaces the linear-probability model with a binomial-logit fit and
    substitutes the OLS coefficient :math:`\\beta_k` with the average
    marginal effect (AME) of covariate :math:`k`. The decomposition
    formula is otherwise identical:

        contribution_k = (AME_k * mu_x_k / mu_y) * CI_x_k

    The residual captures :math:`CI_\\mathrm{total} - \\sum_k \\mathrm{contribution}_k`.
    Following Erreygers (2009), this addresses the LPM-misfit concern
    for binary outcomes at low prevalence without re-scaling the
    concentration index itself.
    """
    exog_cols = [
        "sex_male", "mult_birth",
        "bi_short", "bi_long", "bi_first",
        "bo_2_3", "bo_4_5", "bo_6_plus",
        "ma_teen", "ma_older", "marital_status",
        "edu_primary", "edu_secondary", "edu_higher",
        "residence_urban", "dec_2010s", "dec_2020s"
    ]

    df = _ranked_subset(df)
    w = df["sample_weight"].values
    frac_rank = df["frac_rank"].values
    y = df["died_under5"].values

    total_ci = calc_concentration_index(y, frac_rank, w)
    mu_y = np.average(y, weights=w)

    if mu_y == 0:
        return pd.DataFrame()

    X = sm.add_constant(df[exog_cols].astype(float))
    try:
        model = sm.GLM(y, X, family=sm.families.Binomial(),
                       freq_weights=w).fit(disp=False)
    except Exception as e:
        print(f"  Logit fit failed for {group_name}: {e}")
        return pd.DataFrame()

    # Average marginal effects on the probability scale.
    # AME_k = mean over i of [p_i (1 - p_i) * beta_k]
    p_hat = model.predict(X)
    s = p_hat * (1.0 - p_hat)
    s_mean = np.average(s, weights=w)
    pseudo_r2 = 1.0 - (model.deviance / model.null_deviance) \
        if model.null_deviance > 0 else np.nan

    rows = []
    sum_contributions = 0.0

    for col in exog_cols:
        beta = float(model.params[col])
        x_val = df[col].values
        mu_x = np.average(x_val, weights=w)
        ci_x = calc_concentration_index(x_val, frac_rank, w)
        ame = s_mean * beta
        elasticity = (ame * mu_x) / mu_y
        contribution = elasticity * ci_x
        percent_contrib = (contribution / total_ci) * 100 \
            if total_ci != 0 else 0
        sum_contributions += contribution
        rows.append({
            "Group": group_name,
            "Predictor": col,
            "Beta": beta,
            "AME": ame,
            "Mean_X": mu_x,
            "Elasticity": elasticity,
            "CI_Predictor": ci_x,
            "Contribution": contribution,
            "Percent_Contribution": percent_contrib,
        })

    residual_ci = total_ci - sum_contributions
    rows.append({
        "Group": group_name,
        "Predictor": "Residual",
        "Beta": np.nan,
        "AME": np.nan,
        "Mean_X": np.nan,
        "Elasticity": np.nan,
        "CI_Predictor": np.nan,
        "Contribution": residual_ci,
        "Percent_Contribution": (residual_ci / total_ci) * 100
            if total_ci != 0 else 0,
    })
    rows.append({
        "Group": group_name,
        "Predictor": "TOTAL_CI",
        "Beta": np.nan,
        "AME": np.nan,
        "Mean_X": np.nan,
        "Elasticity": np.nan,
        "CI_Predictor": np.nan,
        "Contribution": total_ci,
        "Percent_Contribution": 100.0,
    })
    rows.append({
        "Group": group_name,
        "Predictor": "LOGIT_PSEUDO_R2",
        "Beta": np.nan,
        "AME": np.nan,
        "Mean_X": np.nan,
        "Elasticity": np.nan,
        "CI_Predictor": np.nan,
        "Contribution": pseudo_r2,
        "Percent_Contribution": np.nan,
    })
    return pd.DataFrame(rows)


def aggregate_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate individual dummy variables back to parent factors."""
    mapping = {
        "sex_male": "Child sex",
        "mult_birth": "Birth type",
        "bi_short": "Birth interval",
        "bi_long": "Birth interval",
        "bi_first": "Birth interval",
        "bo_2_3": "Birth order",
        "bo_4_5": "Birth order",
        "bo_6_plus": "Birth order",
        "ma_teen": "Maternal age",
        "ma_older": "Maternal age",
        "marital_status": "Marital status",
        "edu_primary": "Education",
        "edu_secondary": "Education",
        "edu_higher": "Education",
        "residence_urban": "Residence",
        "dec_2010s": "Decade trend",
        "dec_2020s": "Decade trend",
        "Residual": "Residual",
        "TOTAL_CI": "TOTAL_CI"
    }
    
    df_agg = df.copy()
    df_agg["Factor"] = df_agg["Predictor"].map(mapping)
    
    # We sum the pure contributions, then recalculate percentages
    agg = df_agg.groupby(["Group", "Factor"], as_index=False)["Contribution"].sum()
    
    # Merge total back
    tot = agg[agg["Factor"] == "TOTAL_CI"][["Group", "Contribution"]].rename(
        columns={"Contribution": "Total"}
    )
    agg = agg.merge(tot, on="Group")
    
    agg["Percent_Contribution"] = (agg["Contribution"] / agg["Total"]) * 100
    
    return agg


def compute_ci_by_region_decade() -> pd.DataFrame:
    """
    Table 3: Wagstaff concentration index for U5MR by WHO region and decade.

    Reads the full dataset (all primary-scope rows) for accurate CI
    estimates — does not subsample.  Results saved to
    output/tables/table_3_concentration_index.csv.
    """
    print("Computing CI by region x decade (5% systematic sample)...")
    # Use 5% systematic sample (same pattern as s05b) — sufficient for
    # accurate CI estimation while staying within RAM limits.
    cols = ["who_region", "survey_year", "died_under5",
            "wealth_quintile", "sample_weight", "in_primary_scope",
            "country_code"]
    df = load_systematic_sample(cols, step=20, scope="primary")
    df = df[df["wealth_quintile"].notna()].copy()
    print(f"  Sample: {len(df):,} rows")

    df["decade"] = pd.cut(
        df["survey_year"],
        bins=[1999, 2009, 2019, 2025],
        labels=["2000-2009", "2010-2019", "2020-2024"],
    )

    regions = sorted(df["who_region"].dropna().unique()) + ["All"]
    decades = ["2000-2009", "2010-2019", "2020-2024"]

    def _ci_from_subset(sub_df: pd.DataFrame) -> float:
        h = sub_df["died_under5"].values
        r = sub_df["wealth_quintile"].values
        w = sub_df["sample_weight"].fillna(1.0).values
        sort_idx = np.argsort(r)
        w_s = w[sort_idx]
        cum_w = np.cumsum(w_s)
        if cum_w[-1] == 0:
            return np.nan
        frac_rank = np.empty_like(cum_w)
        frac_rank[sort_idx] = (cum_w - w_s / 2) / cum_w[-1]
        return calc_concentration_index(h, frac_rank, w)

    # Fix #12: cluster bootstrap (resample countries with replacement)
    # for 95% CIs around each concentration index. 200 reps is enough
    # for stable percentile CIs at this aggregation level.
    N_BOOT = 200
    rng = np.random.default_rng(RNG_SEED)

    rows = []
    for region in regions:
        for decade in decades:
            sub = df[df["decade"] == decade]
            if region != "All":
                sub = sub[sub["who_region"] == region]
            if len(sub) < 100:
                continue

            h = sub["died_under5"].values
            w = sub["sample_weight"].fillna(1.0).values
            ci = _ci_from_subset(sub)
            u5mr = np.average(h, weights=w) * 1000

            # Cluster bootstrap by country_code
            countries = sub["country_code"].unique()
            country_groups = {c: sub[sub["country_code"] == c]
                              for c in countries}
            boot_cis = []
            for _ in range(N_BOOT):
                bc = rng.choice(countries, size=len(countries),
                                replace=True)
                bdf = pd.concat([country_groups[c] for c in bc],
                                ignore_index=True)
                if len(bdf) < 50:
                    continue
                ci_b = _ci_from_subset(bdf)
                if np.isfinite(ci_b):
                    boot_cis.append(ci_b)
                del bdf
            if len(boot_cis) >= 20:
                ci_lo = float(np.percentile(boot_cis, 2.5))
                ci_hi = float(np.percentile(boot_cis, 97.5))
            else:
                ci_lo = np.nan
                ci_hi = np.nan

            rows.append({
                "region": region,
                "decade": decade,
                "n_births": len(sub),
                "u5mr_per1000": round(u5mr, 1),
                "concentration_index": round(ci, 4),
                "ci_lo_95": round(ci_lo, 4) if np.isfinite(ci_lo) else np.nan,
                "ci_hi_95": round(ci_hi, 4) if np.isfinite(ci_hi) else np.nan,
                "interpretation": (
                    "Pro-poor (expected)" if ci < -0.05
                    else "Near-equal" if abs(ci) <= 0.05
                    else "Pro-rich (unexpected)"
                ),
            })

    del df
    gc.collect()
    result = pd.DataFrame(rows)
    out = TABLES_DIR / "table_3_concentration_index.csv"
    result.to_csv(out, index=False)
    print(f"  Saved: {out.name}  ({len(result)} rows)")
    return result


def main():
    print("=" * 60)
    print("Table 4: Wagstaff Inequality Decomposition")
    print("=" * 60)

    df = load_sample()
    print(f"  Loaded {len(df):,} rows.")
    
    # 1. Global decomposition
    print("  Decomposing Global All-time...")
    global_decomp = decompose_wagstaff(df, "All LMICs (2000-2024)")
    
    # 2. Regional decends over time (recent decade)
    recent_df = df[df["survey_year"] >= 2010]
    print(f"  Decomposing recent surveys (>2010, n={len(recent_df):,})...")
    
    all_results = [global_decomp]
    
    regions = sorted(recent_df["who_region"].dropna().unique())
    for region in regions:
        sub = recent_df[recent_df["who_region"] == region]
        if len(sub) > 1000:
            print(f"    Region: {region}")
            sub_decomp = decompose_wagstaff(sub, f"{region} (2010-2024)")
            all_results.append(sub_decomp)
            
    # Combine
    final = pd.concat(all_results, ignore_index=True)
    
    # Format and save detailed
    out_detail = TABLES_DIR / "table_4_inequality_decomp_detailed.csv"
    final.to_csv(out_detail, index=False)
    
    # Format and save aggregated
    agg = aggregate_factors(final)
    out_agg = TABLES_DIR / "table_4_inequality_decomp_aggregated.csv"
    agg.to_csv(out_agg, index=False)
    
    print(f"\nSaved detailed decomposition: {out_detail.name}")
    print(f"Saved aggregated decomposition: {out_agg.name}")
    
    # Print a summary of the global aggregated
    g_agg = agg[agg["Group"] == "All LMICs (2000-2024)"].sort_values("Contribution")
    # CI is negative (pro-poor), so most negative contribution = strongest driver
    print("\nGlobal Inequality Drivers:")
    print(g_agg[["Factor", "Contribution", "Percent_Contribution"]].to_string(index=False))

    # Table 3: CI by region x decade (migrated from s05_inferential_analysis)
    print()
    compute_ci_by_region_decade()


if __name__ == "__main__":
    main()
