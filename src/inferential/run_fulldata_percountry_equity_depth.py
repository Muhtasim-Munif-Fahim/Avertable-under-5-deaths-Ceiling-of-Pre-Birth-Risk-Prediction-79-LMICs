"""
run_fulldata_percountry_equity_depth.py — per-country equity depth on FULL data (P1 C1.7/C2.5)
==============================================================================================

LGH country-resolution + equity-spine layer (R1/R2). For every country, on the FULL cohort:
  - the wealth-related concentration index (CI) within each survey decade (2000-09/10-19/20-24)
  - the slope index of inequality (SII): the absolute mortality gap per 1,000 between the
    bottom and top of the wealth-rank distribution (weighted OLS of the death indicator on the
    grouped fractional wealth rank).

Pure weighted rank statistics (no model fit, no bootstrap) → cheap at full-data scale. Writes a
NEW file; does NOT touch table_s_country_equity.csv (that canonical, full-data file is produced
elsewhere and must not be overwritten by a 5%-sample path).

Output: output/tables/table_s_percountry_equity_depth.csv
Schema: country_code, country_name, who_region, n_records,
        ci_2000s, ci_2010s, ci_2020s, sii_per1000   ("--"/NaN where not estimable)
"""
from __future__ import annotations
import os, sys
from pathlib import Path
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.paths import TABLES_DIR
from src.inferential import _fulldata as FD

OUT = TABLES_DIR / "table_s_percountry_equity_depth.csv"
MIN_DEATHS = 30  # suppress a decade cell / SII with too few weighted deaths


def _frac_rank(q: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Grouped (group-midpoint) fractional wealth rank — same convention as s05k/s05d."""
    tw = w.sum()
    fr = np.empty_like(w, dtype=float)
    cum = 0.0
    for v in np.unique(q):
        m = (q == v)
        gw = w[m].sum()
        fr[m] = (cum + 0.5 * gw) / tw
        cum += gw
    return fr


def _conc_index(h: np.ndarray, q: np.ndarray, w: np.ndarray) -> float:
    if w.sum() == 0:
        return np.nan
    mu = np.average(h, weights=w)
    if mu == 0:
        return np.nan
    fr = _frac_rank(q, w)
    mr = np.average(fr, weights=w)
    cov = np.average((h - mu) * (fr - mr), weights=w)
    return (2.0 / mu) * cov


def _sii_per1000(h: np.ndarray, q: np.ndarray, w: np.ndarray) -> float:
    """Slope index of inequality: weighted OLS slope of death-indicator on fractional rank,
    expressed per 1,000 (predicted gap from rank 0 to rank 1)."""
    if w.sum() == 0:
        return np.nan
    fr = _frac_rank(q, w)
    mr = np.average(fr, weights=w)
    var = np.average((fr - mr) ** 2, weights=w)
    if var <= 0:
        return np.nan
    mu = np.average(h, weights=w)
    cov = np.average((h - mu) * (fr - mr), weights=w)
    return float((cov / var) * 1000.0)


def _decade(yr: np.ndarray) -> np.ndarray:
    d = np.full(yr.shape, "", dtype=object)
    d[(yr >= 2000) & (yr <= 2009)] = "2000s"
    d[(yr >= 2010) & (yr <= 2019)] = "2010s"
    d[(yr >= 2020) & (yr <= 2024)] = "2020s"
    return d


def main():
    df = FD.load_full(
        ["country_code", "who_region", "survey_year", "died_under5",
         "wealth_quintile", "sample_weight"], scope="primary")
    df = df[df["wealth_quintile"].notna()].copy()
    df["sample_weight"] = df["sample_weight"].fillna(1.0)
    df["decade"] = _decade(df["survey_year"].values.astype(float))
    try:
        inv = pd.read_csv(TABLES_DIR / "table_s1_survey_inventory.csv")
        names = inv.groupby("country_code")["country_name"].first()
    except Exception:
        names = pd.Series(dtype=object)

    rows = []
    for cc, sub in df.groupby("country_code", sort=True):
        h = sub["died_under5"].values.astype(float)
        q = sub["wealth_quintile"].values.astype(float)
        w = sub["sample_weight"].values.astype(float)
        dec = sub["decade"].values
        row = {
            "country_code": cc,
            "country_name": names.get(cc, cc),
            "who_region": sub["who_region"].iloc[0],
            "n_records": int(len(sub)),
        }
        for tag, col in [("2000s", "ci_2000s"), ("2010s", "ci_2010s"), ("2020s", "ci_2020s")]:
            m = dec == tag
            if m.sum() and float(np.nansum(h[m] * w[m])) >= MIN_DEATHS:
                v = _conc_index(h[m], q[m], w[m])
                row[col] = round(float(v), 4) if np.isfinite(v) else np.nan
            else:
                row[col] = np.nan
        if float(np.nansum(h * w)) >= MIN_DEATHS:
            s = _sii_per1000(h, q, w)
            row["sii_per1000"] = round(float(s), 1) if np.isfinite(s) else np.nan
        else:
            row["sii_per1000"] = np.nan
        rows.append(row)

    out = pd.DataFrame(rows).sort_values(
        ["who_region", "n_records"], ascending=[True, False]).reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    n_sii = int(out["sii_per1000"].notna().sum())
    print(f"Wrote {OUT} ({len(out)} countries; {n_sii} with SII)")
    print(out.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
