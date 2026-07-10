"""
s05p_wealth_missingness.py — Wealth-missingness sensitivity for the equity leg
==============================================================================
Reviewer (biostatistics/equity) asked what fraction of the concentration-index /
Wagstaff sample is lost to wealth missingness (the pooled concentration index and
decomposition are complete-case on the household wealth index), how the excluded
records differ in mortality, and whether that exclusion could bias the pro-poor
gradient.

The pooled concentration index (−0·128) is already SURVEY-WEIGHTED (it uses
sample_weight; s05k_per_country_equity). The remaining question is missingness, so
this script:

  1. quantifies the wealth-missing fraction overall, by WHO region, and flags the
     surveys/regions where it concentrates (expected: LAC/MENA);
  2. contrasts survey-weighted recorded under-five mortality between wealth-observed
     and wealth-missing records (does complete-case drop a higher- or lower-mortality
     group?);
  3. brackets the pooled concentration index with a missing-indicator bounds
     analysis: within the wealth-index countries, add every wealth-missing record
     back assigned to the POOREST quintile (Q1) and, separately, to the RICHEST (Q5),
     and recompute the pooled grouped concentration index under each extreme. The two
     bounds show the maximum the pooled index could move if missingness were maximally
     informative in either direction. Country-cluster bootstrap CIs match s05k.

Outputs (output/tables/):
  table_s_wealth_missingness.csv          (overall + by-region missingness & U5M)
  table_s_wealth_missingness_ci_bounds.csv (complete-case vs missing->Q1 / ->Q5)

Run (SOLO — full-data load; never concurrent with another full-data job):
  python -m src.inferential.s05p_wealth_missingness
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from src.utils.io import load_primary
from config.paths import TABLES_DIR, RNG_SEED

OUT_SUMMARY = TABLES_DIR / "table_s_wealth_missingness.csv"
OUT_BOUNDS = TABLES_DIR / "table_s_wealth_missingness_ci_bounds.csv"
N_BOOT = 200
# who_region is stored as short codes in the parquet; map to readable labels for
# the output table (unknown codes pass through unchanged).
REGION_NAMES = {
    "SSA": "Sub-Saharan Africa", "SSEA": "South and Southeast Asia",
    "MENA": "Middle East and North Africa",
    "LAC": "Latin America and the Caribbean", "ECA": "Europe and Central Asia",
}


def _ci_from_arrays(h: np.ndarray, q: np.ndarray, w: np.ndarray) -> float:
    """Grouped (group-midpoint) Wagstaff concentration index (matches s05k)."""
    if w.sum() == 0:
        return np.nan
    mu = np.average(h, weights=w)
    if mu == 0:
        return np.nan
    tw = w.sum()
    frac_rank = np.empty_like(w, dtype=float)
    cum = 0.0
    for v in np.unique(q):
        m = (q == v)
        gw = w[m].sum()
        frac_rank[m] = (cum + 0.5 * gw) / tw
        cum += gw
    mean_rank = np.average(frac_rank, weights=w)
    cov = np.average((h - mu) * (frac_rank - mean_rank), weights=w)
    return (2.0 / mu) * cov


def _wrate(h, w):
    return float(np.average(h, weights=w) * 1000.0) if w.sum() > 0 else np.nan


def main() -> None:
    print("=" * 60)
    print("Wealth-missingness sensitivity for the equity leg")
    print("=" * 60)
    df = load_primary(
        columns=["country_code", "who_region", "survey_year", "died_under5",
                 "wealth_quintile", "sample_weight"],
        scope="primary",
    )
    df = df[df["died_under5"].notna()].copy()
    df["sample_weight"] = df["sample_weight"].fillna(1.0)
    n_total = len(df)

    miss = df["wealth_quintile"].isna().values
    h = df["died_under5"].values.astype(float)
    w = df["sample_weight"].values.astype(float)
    reg = df["who_region"].values

    # ---- 1 + 2: missingness fraction and mortality contrast, overall + by region
    rows = []

    def _summ(label, mask_rows):
        hh, ww, mm = h[mask_rows], w[mask_rows], miss[mask_rows]
        n = int(mask_rows.sum())
        pct_miss = 100.0 * float(ww[mm].sum() / ww.sum()) if ww.sum() > 0 else np.nan
        pct_miss_unwt = 100.0 * float(mm.mean()) if n else np.nan
        u5m_obs = _wrate(hh[~mm], ww[~mm]) if (~mm).any() else np.nan
        u5m_miss = _wrate(hh[mm], ww[mm]) if mm.any() else np.nan
        rows.append({
            "stratum": label, "n_records": n,
            "pct_wealth_missing_wtd": round(pct_miss, 2) if np.isfinite(pct_miss) else np.nan,
            "pct_wealth_missing_unwt": round(pct_miss_unwt, 2) if np.isfinite(pct_miss_unwt) else np.nan,
            "u5m_wealth_observed": round(u5m_obs, 1) if np.isfinite(u5m_obs) else np.nan,
            "u5m_wealth_missing": round(u5m_miss, 1) if np.isfinite(u5m_miss) else np.nan,
        })

    _summ("Overall", np.ones(n_total, dtype=bool))
    # Discover the region codes actually present; order by weighted missingness
    # (descending) so the LAC/MENA concentration surfaces at the top.
    uniq_reg = [r for r in pd.unique(reg) if pd.notna(r)]
    miss_by_reg = {}
    for r in uniq_reg:
        m = (reg == r)
        miss_by_reg[r] = float(w[m & miss].sum() / w[m].sum()) if w[m].sum() > 0 else 0.0
    for r in sorted(uniq_reg, key=lambda x: -miss_by_reg[x]):
        _summ(REGION_NAMES.get(str(r), str(r)), (reg == r))
    summ_df = pd.DataFrame(rows)
    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    summ_df.to_csv(OUT_SUMMARY, index=False)
    print(summ_df.to_string(index=False))

    # Surveys where missingness concentrates (top 10 by weighted missing fraction).
    dfx = df.assign(_miss=miss, _w=w)
    g = dfx.groupby(["country_code", "survey_year"]).apply(
        lambda s: pd.Series({
            "pct_missing": 100.0 * (s["_w"][s["_miss"]].sum() / s["_w"].sum()),
            "n": len(s),
        }), include_groups=False).reset_index()
    top = g[g["n"] >= 5000].sort_values("pct_missing", ascending=False).head(10)
    print("\nSurveys with the highest wealth-missingness (>=5k records):")
    print(top.to_string(index=False))

    # ---- 3: missing-indicator bounds on the pooled concentration index ----------
    # Restrict to countries that HAVE a wealth index (>=1 non-missing wealth record);
    # the 2 fully-missing countries stay excluded (as in the canonical CI).
    cc = df["country_code"].values
    has_wealth = df.groupby("country_code")["wealth_quintile"].transform(
        lambda s: s.notna().any()).values
    keep = has_wealth
    hk, wk, mk, cck = h[keep], w[keep], miss[keep], cc[keep]
    qk = df["wealth_quintile"].values[keep]

    # complete-case (observed wealth only) -> reproduces canonical -0.128
    obs = ~mk
    q_cc = qk[obs].astype(float)
    variants = {
        "complete_case": (hk[obs], q_cc, wk[obs], cck[obs]),
        "missing_to_Q1": (hk, np.where(mk, 1.0, qk).astype(float), wk, cck),
        "missing_to_Q5": (hk, np.where(mk, 5.0, qk).astype(float), wk, cck),
    }
    rng = np.random.default_rng(RNG_SEED)
    brows = []
    for name, (hv, qv, wv, cv) in variants.items():
        point = _ci_from_arrays(hv, qv, wv)
        countries = np.unique(cv)
        idx_by_c = {c: np.where(cv == c)[0] for c in countries}
        boots = []
        for _ in range(N_BOOT):
            bc = rng.choice(countries, size=len(countries), replace=True)
            bidx = np.concatenate([idx_by_c[c] for c in bc])
            cb = _ci_from_arrays(hv[bidx], qv[bidx], wv[bidx])
            if np.isfinite(cb):
                boots.append(cb)
        lo, hi = np.percentile(boots, [2.5, 97.5]) if len(boots) >= 30 else (np.nan, np.nan)
        brows.append({
            "variant": name, "n_records": int(len(hv)),
            "conc_index": round(float(point), 4),
            "ci_lo_95": round(float(lo), 4), "ci_hi_95": round(float(hi), 4),
            "conc_index_fmt": f"{point:.3f} ({lo:.3f} to {hi:.3f})",
        })
        print(f"  {name}: CI = {point:.4f} ({lo:.4f}, {hi:.4f}); n={len(hv):,}")
    pd.DataFrame(brows).to_csv(OUT_BOUNDS, index=False)
    print(f"\nWrote {OUT_SUMMARY.name} and {OUT_BOUNDS.name}")


if __name__ == "__main__":
    main()
