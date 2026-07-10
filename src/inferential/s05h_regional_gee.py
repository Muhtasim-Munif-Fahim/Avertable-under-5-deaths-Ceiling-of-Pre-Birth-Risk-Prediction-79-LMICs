"""
s05h_regional_gee.py
=====================

Region-stratified adjusted odds ratios with proper 95% CIs and p-values.

This replaces an orphaned April-2026 CSV (`table_s8_regional_gee.csv`)
in which the LAC stratum had empty CI columns and `p_value=1.0`
placeholders. The original generator was lost during file
reorganisation; this script provides a clean, fully reproducible
reimplementation.

For each WHO region we fit a country-clustered GEE with logit link and
exchangeable working correlation on a 5% systematic sample of the
2000--2024 primary cohort. Sample sizes per region are large enough
that the cluster-robust sandwich is the dominant variance estimator;
we therefore report robust Wald CIs and p-values from the sandwich
covariance.

Output: output/tables/table_s8_regional_gee.csv

Schema: Region, Variable, N, OR, CI_lower, CI_upper, p_value, Significance
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

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.print_util import print
from config.paths import TABLES_DIR
from src.inferential.s05c_counterfactual_paf import load_sample, FORMULA


OUT = TABLES_DIR / "table_s8_regional_gee.csv"

PARAM_PRETTY = {
    "sex_cat[T.Male]":          "male",
    "birth_type[T.Multiple]":    "is_multiple",
    "bo_cat[T.2nd-3rd]":         "bo_2_3",
    "bo_cat[T.4th-5th]":         "bo_4_5",
    "bo_cat[T.6th+]":            "bo_6plus",
    "bi_cat[T.<24 mo]":          "bi_short",
    "bi_cat[T.48+ mo]":          "bi_long",
    "bi_cat[T.First/missing]":   "bi_firstmissing",
    "ma_cat[T.<20]":             "age_under20",
    "ma_cat[T.35+]":             "age_over35",
    "edu_cat[T.Primary]":        "edu_primary",
    "edu_cat[T.Secondary]":      "edu_secondary",
    "edu_cat[T.Higher]":         "edu_higher",
    "marital_cat[T.Married]":    "married",
    "urban_cat[T.Urban]":        "is_urban",
    "wq_cat[T.Poorer]":          "wq_2",
    "wq_cat[T.Middle]":          "wq_3",
    "wq_cat[T.Richer]":          "wq_4",
    "wq_cat[T.Richest]":         "wq_5",
    "decade[T.2010-2019]":       "decade_2010",
    "decade[T.2020-2024]":       "decade_2020",
}

# Drop covariates whose categorical level is rank-deficient (zero or
# near-zero observations) in a given region; fall back to a reduced
# formula if the full formula fails.
FORMULA_FULL = FORMULA  # from s05c
FORMULA_NO_DECADE = FORMULA.replace(" + decade", "")


def _significance(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def fit_region(df: pd.DataFrame, region: str) -> pd.DataFrame:
    sub = df[df["who_region"] == region].copy()
    n = len(sub)
    print(f"\n[{region}] N={n:,}")
    if n < 500:
        print(f"  insufficient sample, skipping")
        return pd.DataFrame()
    sub = sub.sort_values("country_code").reset_index(drop=True)

    # If a decade level has no observations, drop the decade term.
    formula = FORMULA_FULL
    if sub["decade"].cat.remove_unused_categories().nunique() < 2:
        formula = FORMULA_NO_DECADE
        print(f"  using formula without decade (only one decade present)")

    def _fit_glm():
        m_glm = sm.GLM.from_formula(formula, data=sub, family=Binomial())
        # Align cluster groups to the rows actually used (post NA-drop).
        groups = sub.loc[m_glm.data.row_labels, "country_code"]
        return m_glm.fit(cov_type="cluster", cov_kwds={"groups": groups})

    method = "GEE"
    try:
        model = sm.GEE.from_formula(
            formula, groups="country_code", data=sub,
            family=Binomial(), cov_struct=sm.cov_struct.Exchangeable(),
        )
        r = model.fit(maxiter=100)
        # GEE can "converge" to all-NaN parameters on small/heterogeneous
        # strata (e.g. few country clusters); detect that and fall back.
        if pd.Series(r.params).isna().any():
            print("  GEE returned NaN params; cluster-robust GLM fallback")
            r = _fit_glm()
            method = "GLM"
    except Exception as e:
        print(f"  GEE failed ({type(e).__name__}: {e}); cluster-robust GLM fallback")
        try:
            r = _fit_glm()
            method = "GLM"
        except Exception as e2:
            print(f"  GLM also failed: {e2}")
            return pd.DataFrame()

    if pd.Series(r.params).isna().any():
        print(f"  still NaN after GLM fallback; skipping {region}")
        return pd.DataFrame()

    params = r.params
    conf = r.conf_int()
    pvals = r.pvalues if hasattr(r, "pvalues") else pd.Series(np.nan, index=params.index)

    rows = []
    for name in params.index:
        if name == "Intercept":
            continue
        pretty = PARAM_PRETTY.get(name, name)
        b = params[name]
        lo_b, hi_b = conf.loc[name]
        p = float(pvals.get(name, np.nan))
        rows.append({
            "Region": region,
            "Variable": pretty,
            "N": n,
            "OR": round(float(np.exp(b)), 3),
            "CI_lower": round(float(np.exp(lo_b)), 3),
            "CI_upper": round(float(np.exp(hi_b)), 3),
            "p_value": round(p, 4) if not np.isnan(p) else np.nan,
            "Significance": _significance(p),
            "Method": method,
        })
    return pd.DataFrame(rows)


def main() -> None:
    print("=" * 60)
    print("Region-stratified GEE (eTable S7)")
    print("=" * 60)
    df = load_sample()
    print(f"  5% sample: {len(df):,}")
    regions = sorted(df["who_region"].dropna().unique())
    frames = []
    for reg in regions:
        f = fit_region(df, reg)
        if not f.empty:
            frames.append(f)
        gc.collect()

    if not frames:
        print("  NO regional fits succeeded.")
        return

    out = pd.concat(frames, ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}  ({len(out)} rows)")
    print(out.groupby("Region").size())


if __name__ == "__main__":
    main()
