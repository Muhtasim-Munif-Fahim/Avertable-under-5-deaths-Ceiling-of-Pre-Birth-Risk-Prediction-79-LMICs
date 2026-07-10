"""
s05L_fulldata_glm_regression.py — FULL-DATA Model 3 associations
================================================================

Country-clustered GEE cannot be fit on all 10.7M records (statsmodels' IRLS
solver fails with a LAPACK `init_gelsd` workspace error, and small regions go
singular). This script delivers the **full-cohort** Model 3 associations the
proper way for data at this scale:

  * logistic MLE via scikit-learn (lbfgs, no penalty) — handles 10.7M rows;
  * Huber-White **country cluster-robust** sandwich covariance computed in
    NumPy (79 clusters) — the full-data analogue of GEE with an independence
    working correlation;
  * degenerate dummies (zero variance within a stratum, e.g. no 2020-24 LAC
    surveys) are dropped automatically, so Latin America no longer goes
    singular.

Covariates = Model 3 (child + maternal + socioeconomic + temporal; no
healthcare, no country fixed effects — country is the cluster).

Output: output/tables/table_s8_regional_gee.csv  (now includes a `Pooled`
stratum + all 5 WHO regions; schema unchanged so journal_tables reads it
directly), Method column = "GLM-CR".

Run: python -m src.inferential.s05L_fulldata_glm_regression
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.io import load_primary
from config.paths import TABLES_DIR

OUT = TABLES_DIR / "table_s_fulldata_glm.csv"  # sensitivity; GEE table_s8 stays primary

# Dummy order = the codes journal_tables.build_T1_stratified expects.
DUMMIES = ["male", "is_multiple", "bo_2_3", "bo_4_5", "bo_6plus",
           "bi_firstmissing", "bi_short", "bi_long", "age_under20", "age_over35",
           "edu_primary", "edu_secondary", "edu_higher", "wq_2", "wq_3", "wq_4",
           "wq_5", "married", "is_urban", "decade_2010", "decade_2020"]
REGIONS = ["SSA", "SSEA", "LAC", "MENA", "ECA"]


def make_design(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["male"] = (df["child_male"].fillna(0) > 0).astype(float)
    X["is_multiple"] = (df["multiple_birth"].fillna(0) > 0).astype(float)
    bo = df["birth_order"].fillna(1).values
    X["bo_2_3"] = ((bo >= 2) & (bo <= 3)).astype(float)
    X["bo_4_5"] = ((bo >= 4) & (bo <= 5)).astype(float)
    X["bo_6plus"] = (bo >= 6).astype(float)
    bi = df["birth_interval_months"].values
    X["bi_firstmissing"] = np.isnan(bi).astype(float)
    X["bi_short"] = ((~np.isnan(bi)) & (bi < 24)).astype(float)
    X["bi_long"] = ((~np.isnan(bi)) & (bi >= 48)).astype(float)
    ma = df["maternal_age_at_birth"].fillna(20).values
    X["age_under20"] = (ma < 20).astype(float)
    X["age_over35"] = (ma >= 35).astype(float)
    edu = df["edu_level"].fillna(0).values
    X["edu_primary"] = (edu == 1).astype(float)
    X["edu_secondary"] = (edu == 2).astype(float)
    X["edu_higher"] = (edu >= 3).astype(float)
    wq = df["wealth_quintile"].values
    X["wq_2"] = (wq == 2).astype(float)
    X["wq_3"] = (wq == 3).astype(float)
    X["wq_4"] = (wq == 4).astype(float)
    X["wq_5"] = (wq == 5).astype(float)
    X["married"] = (df["currently_married"].fillna(1) > 0).astype(float)
    X["is_urban"] = (df["urban"].fillna(0) > 0).astype(float)
    sy = df["survey_year"].values
    X["decade_2010"] = ((sy >= 2010) & (sy <= 2019)).astype(float)
    X["decade_2020"] = (sy >= 2020).astype(float)
    return X


def cluster_robust_logit(X: np.ndarray, y: np.ndarray, groups: np.ndarray):
    """Logistic MLE (sklearn) + country cluster-robust sandwich covariance.
    X includes an intercept column at position 0. Returns (beta, se)."""
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000,
                            tol=1e-9, fit_intercept=False)
    lr.fit(X, y)
    beta = lr.coef_.ravel()
    eta = X @ beta
    p = 1.0 / (1.0 + np.exp(-eta))
    w = p * (1.0 - p)
    bread = np.linalg.inv((X * w[:, None]).T @ X)        # inv(X' W X)
    u = (y - p)                                          # working residual
    score = X * u[:, None]                               # (n, p)
    uniq = np.unique(groups)
    Sg = np.empty((len(uniq), X.shape[1]))
    for i, g in enumerate(uniq):
        Sg[i] = score[groups == g].sum(0)
    meat = Sg.T @ Sg                                     # sum_g s_g s_g'
    cov = bread @ meat @ bread
    return beta, np.sqrt(np.diag(cov))


def fit_stratum(df: pd.DataFrame, label: str) -> list[dict]:
    X = make_design(df)
    keep = [c for c in DUMMIES if X[c].nunique() > 1]    # drop degenerate dummies
    Xm = np.column_stack([np.ones(len(df)), X[keep].values])
    y = df["died_under5"].astype(float).values
    g = df["country_code"].values
    beta, se = cluster_robust_logit(Xm, y, g)
    rows = []
    for j, name in enumerate(keep, start=1):
        z = beta[j] / se[j]
        rows.append({
            "Region": label, "Variable": name, "N": int(len(df)),
            "OR": round(float(np.exp(beta[j])), 3),
            "CI_lower": round(float(np.exp(beta[j] - 1.96 * se[j])), 3),
            "CI_upper": round(float(np.exp(beta[j] + 1.96 * se[j])), 3),
            "p_value": round(float(2 * stats.norm.sf(abs(z))), 4),
            "Significance": "***" if 2 * stats.norm.sf(abs(z)) < 1e-3 else "",
            "Method": "GLM-CR",
        })
    return rows


def main() -> None:
    df = load_primary(
        columns=["country_code", "who_region", "survey_year", "died_under5",
                 "child_male", "multiple_birth", "birth_order",
                 "birth_interval_months", "maternal_age_at_birth", "edu_level",
                 "wealth_quintile", "currently_married", "urban"],
        scope="primary",
    )
    df = df[df["died_under5"].notna()].copy()  # outcome must be observed
    print(f"Full cohort: {len(df):,} rows, {df['country_code'].nunique()} countries")
    rows: list[dict] = []
    rows += fit_stratum(df, "Pooled")
    print("  Pooled fit OK")
    for R in REGIONS:
        sub = df[df["who_region"] == R]
        rows += fit_stratum(sub, R)
        print(f"  {R} fit OK (n={len(sub):,})")
    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"\nWrote {OUT} ({len(out)} rows; strata: {sorted(out['Region'].unique())})")
    # quick check: pooled key ORs
    chk = out[(out.Region == "Pooled") & (out.Variable.isin(
        ["is_multiple", "edu_secondary", "edu_higher", "age_under20"]))]
    print(chk[["Variable", "OR", "CI_lower", "CI_upper"]].to_string(index=False))


if __name__ == "__main__":
    main()
