"""
s05k_per_country_equity.py — Per-country wealth-related inequality in U5M
========================================================================

Builds the per-country inputs for main Table 3 (and the supplement equity
tables). For each of the 79 countries we compute, on the full primary
birth-history cohort (load_primary, scope="primary"; all analyses migrated
off the retired 5% systematic sample):

  - weighted recorded under-five mortality per 1,000 (all wealth groups)
  - recorded U5M/1,000 in the poorest wealth quintile (Q1) and richest (Q5)
  - the Q1:Q5 ratio (how many times higher mortality is among the poorest)
  - the Wagstaff concentration index (CI) with a 95% nonparametric
    bootstrap interval (resampling birth records within the country)

Concentration-index machinery is identical to
s05d_inequality_decomposition.calc_concentration_index / _ci_from_subset
(rank fractionally by within-country wealth quintile).

Output: output/tables/table_s_country_equity.csv
Run: python -m src.inferential.s05k_per_country_equity
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.io import load_primary
from config.paths import TABLES_DIR, RNG_SEED

OUT = TABLES_DIR / "table_s_country_equity.csv"
N_BOOT = 300
MIN_DEATHS = 30   # suppress CI when a country has too few recorded deaths
# NB: the concentration-index POINT and its bootstrap CI are both computed on the
# FULL country (and full pooled) cohort -- no record/cluster cap. An earlier
# capped-base variant biased the grouped index (e.g. pooled -0.106 vs the
# canonical full-data -0.128) because subsampling shifts the weighted joint
# (death x wealth-rank) distribution; do NOT reintroduce a cap on the point.


def _ci_from_arrays(h: np.ndarray, q: np.ndarray, w: np.ndarray) -> float:
    """Concentration index from death indicator h, wealth quintile q, weights w."""
    if w.sum() == 0:
        return np.nan
    mu = np.average(h, weights=w)
    if mu == 0:
        return np.nan
    # Grouped-tie fractional rank: every record in a wealth group shares the
    # group-midpoint rank R_g = (cumulative weight below g) + 0.5*(weight of g),
    # divided by total weight (standard Wagstaff grouped concentration index).
    # A within-tie "spread" rank would inject the data's original row order as a
    # spurious within-quintile gradient into the point estimate, which the
    # record bootstrap then washes out — leaving the point outside its own CI.
    tw = w.sum()
    frac_rank = np.empty_like(w, dtype=float)
    cum = 0.0
    for v in np.unique(q):            # ascending; ~5 wealth groups
        m = (q == v)
        gw = w[m].sum()
        frac_rank[m] = (cum + 0.5 * gw) / tw
        cum += gw
    mean_rank = np.average(frac_rank, weights=w)
    cov = np.average((h - mu) * (frac_rank - mean_rank), weights=w)
    return (2.0 / mu) * cov


def _wrate(h: np.ndarray, w: np.ndarray) -> float:
    if w.sum() == 0:
        return np.nan
    return float(np.average(h, weights=w) * 1000.0)


def build() -> pd.DataFrame:
    df = load_primary(
        columns=["country_code", "who_region", "died_under5",
                 "wealth_quintile", "sample_weight"],
        scope="primary",
    )
    df = df[df["died_under5"].notna() & df["wealth_quintile"].notna()].copy()
    df["sample_weight"] = df["sample_weight"].fillna(1.0)
    inv = pd.read_csv(TABLES_DIR / "table_s1_survey_inventory.csv")
    names = inv.groupby("country_code")["country_name"].first()

    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for cc, sub in df.groupby("country_code"):
        h = sub["died_under5"].values.astype(float)
        q = sub["wealth_quintile"].values.astype(float)
        w = sub["sample_weight"].values.astype(float)
        n_deaths = float(np.nansum(h * w))

        u5m_all = _wrate(h, w)
        m1, m5 = q == 1, q == 5
        u5m_q1 = _wrate(h[m1], w[m1]) if m1.any() else np.nan
        u5m_q5 = _wrate(h[m5], w[m5]) if m5.any() else np.nan
        # Ratio from the displayed (1-dp) rates so it is order-independent: the
        # raw weighted-mean summation order can flip the 2nd decimal of the ratio
        # for borderline countries, which would otherwise make reruns non-exact.
        ratio = (round(u5m_q1, 1) / round(u5m_q5, 1)) if (u5m_q5 and u5m_q5 > 0 and not np.isnan(u5m_q1)) else np.nan
        # Point on the FULL country cohort; 95% CI = percentile of N_BOOT
        # record bootstraps resampled over the same full cohort (no cap).
        ci = _ci_from_arrays(h, q, w)

        ci_lo = ci_hi = np.nan
        if n_deaths >= MIN_DEATHS and np.isfinite(ci):
            n = len(h)
            boots = []
            for _ in range(N_BOOT):
                idx = rng.integers(0, n, n)
                cb = _ci_from_arrays(h[idx], q[idx], w[idx])
                if np.isfinite(cb):
                    boots.append(cb)
            if len(boots) >= 30:
                ci_lo = float(np.percentile(boots, 2.5))
                ci_hi = float(np.percentile(boots, 97.5))

        rows.append({
            "country_code": cc,
            "country_name": names.get(cc, cc),
            "who_region": sub["who_region"].iloc[0],
            "n_records": int(len(sub)),
            "n_deaths_wtd": round(n_deaths, 0),
            "u5m_all": round(u5m_all, 1) if np.isfinite(u5m_all) else np.nan,
            "u5m_q1_poorest": round(u5m_q1, 1) if np.isfinite(u5m_q1) else np.nan,
            "u5m_q5_richest": round(u5m_q5, 1) if np.isfinite(u5m_q5) else np.nan,
            "ratio_q1_q5": float(np.round(ratio, 2)) if np.isfinite(ratio) else np.nan,
            "conc_index": round(ci, 4) if np.isfinite(ci) else np.nan,
            "ci_lo_95": round(ci_lo, 4) if np.isfinite(ci_lo) else np.nan,
            "ci_hi_95": round(ci_hi, 4) if np.isfinite(ci_hi) else np.nan,
        })

    out = pd.DataFrame(rows).sort_values(["who_region", "u5m_all"],
                                         ascending=[True, False]).reset_index(drop=True)

    # Pooled "All LMICs": point on the FULL cohort; 95% CI = country-cluster
    # bootstrap (resample the countries with replacement, each contributing its
    # full set of records -- no per-country cap, so the resample point is unbiased
    # and the interval brackets the full-data point).
    H = df["died_under5"].values.astype(float)
    Q = df["wealth_quintile"].values.astype(float)
    W = df["sample_weight"].values.astype(float)
    CC = df["country_code"].values
    pm1, pm5 = Q == 1, Q == 5
    pq1, pq5 = _wrate(H[pm1], W[pm1]), _wrate(H[pm5], W[pm5])
    countries = np.unique(CC)
    idx_by_c = {c: np.where(CC == c)[0] for c in countries}
    pooled_ci = _ci_from_arrays(H, Q, W)  # point on the full pooled cohort
    boots = []
    for _ in range(N_BOOT):
        bc = rng.choice(countries, size=len(countries), replace=True)
        bidx = np.concatenate([idx_by_c[c] for c in bc])
        cb = _ci_from_arrays(H[bidx], Q[bidx], W[bidx])
        if np.isfinite(cb):
            boots.append(cb)
    plo, phi = np.percentile(boots, [2.5, 97.5])
    pooled = {
        "country_code": "ALL", "country_name": "All LMICs", "who_region": "ALL",
        "n_records": int(len(df)), "n_deaths_wtd": round(float(np.nansum(H * W)), 0),
        "u5m_all": round(_wrate(H, W), 1),
        "u5m_q1_poorest": round(pq1, 1), "u5m_q5_richest": round(pq5, 1),
        "ratio_q1_q5": float(np.round(round(pq1, 1) / round(pq5, 1), 2)) if pq5 else np.nan,
        "conc_index": round(pooled_ci, 4),
        "ci_lo_95": round(float(plo), 4), "ci_hi_95": round(float(phi), 4),
    }
    out = pd.concat([pd.DataFrame([pooled]), out], ignore_index=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    n_ci = int(out.iloc[1:]["ci_lo_95"].notna().sum())
    print(f"Wrote {OUT}  ({len(out) - 1} countries + pooled; {n_ci} with bootstrap CI)")
    print(f"  pooled concentration index: {pooled_ci:.4f} ({plo:.4f}, {phi:.4f})")
    return out


if __name__ == "__main__":
    build()
