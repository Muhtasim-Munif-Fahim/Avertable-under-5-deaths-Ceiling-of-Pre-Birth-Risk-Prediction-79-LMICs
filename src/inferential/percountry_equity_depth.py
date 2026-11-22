"""percountry_equity_depth.py — per-country equity depth on the v2 exact cohort.

Replaces the v1 `run_fulldata_percountry_equity_depth.py` (archived 2026-09-24), which computed
the same quantities on the frozen v1 cohort with the old imputed-follow-up outcome. For each
country, on the exact >=60-month follow-up cohort with complete wealth (the cohort, weights and
rank convention of `percountry_equity`):

  * the wealth concentration index within each survey decade (2000-09, 2010-19, 2020-24);
  * the slope index of inequality (SII): the weighted least-squares slope of the death
    indicator on the grouped fractional wealth rank, per 1,000 (rank 0 -> rank 1).

Pure weighted rank statistics, no model fit and no bootstrap. A decade cell (or the SII) with
fewer than MIN_DEATHS recorded deaths is left empty, matching `percountry_equity`.

Output: output/phase4/manuscript/table_percountry_equity_depth.csv
Run:    python -m src.inferential.percountry_equity_depth
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from config.paths import HARMONIZED_DIR, OUTPUT_DIR
from src.inferential.phase2_analysis import _positive_normalized_weights, _read_pilot
from src.inferential.s05d_inequality import _conc_index, _grouped_fractional_rank

OUT = OUTPUT_DIR / "phase4" / "manuscript"
MIN_DEATHS = 30
DECADES = [("ci_2000s", 2000, 2009), ("ci_2010s", 2010, 2019), ("ci_2020s", 2020, 2024)]


def _sii_per1000(y: np.ndarray, wq: np.ndarray, w: np.ndarray) -> float:
    rank = _grouped_fractional_rank(wq, w)
    mr = np.average(rank, weights=w)
    var = np.average((rank - mr) ** 2, weights=w)
    if var <= 0:
        return np.nan
    cov = np.average((y - np.average(y, weights=w)) * (rank - mr), weights=w)
    return float(cov / var * 1000.0)


def main() -> None:
    frame = _read_pilot(HARMONIZED_DIR / "global_analysis_v2_candidate.parquet")
    w_all = _positive_normalized_weights(frame)
    keep = (w_all.notna()
            & frame["wealth_quintile"].notna()
            & pd.to_numeric(frame["died_under5_complete"], errors="coerce").notna())
    d = frame.loc[keep, ["country_code", "survey_year", "wealth_quintile",
                         "died_under5_complete"]].copy()
    d["w"] = w_all.loc[keep].to_numpy(dtype=float)
    d["y"] = pd.to_numeric(d["died_under5_complete"], errors="coerce").to_numpy(dtype=float)
    d["wq"] = pd.to_numeric(d["wealth_quintile"], errors="coerce").to_numpy(dtype=float)
    d["yr"] = pd.to_numeric(d["survey_year"], errors="coerce").to_numpy(dtype=float)

    rows = []
    for cc, g in d.groupby("country_code", sort=True):
        y, w, wq, yr = (g[c].to_numpy() for c in ("y", "w", "wq", "yr"))
        row = {"country_code": cc}
        for col, lo, hi in DECADES:
            m = (yr >= lo) & (yr <= hi) & (w > 0)
            if (y[m]).sum() >= MIN_DEATHS and len(np.unique(wq[m])) > 1:
                v = _conc_index(y[m], _grouped_fractional_rank(wq[m], w[m]), w[m])
                row[col] = round(float(v), 4) if np.isfinite(v) else np.nan
            else:
                row[col] = np.nan
        m = w > 0
        if (y[m]).sum() >= MIN_DEATHS and len(np.unique(wq[m])) > 1:
            s = _sii_per1000(y[m], wq[m], w[m])
            row["sii_per1000"] = round(s, 1) if np.isfinite(s) else np.nan
        else:
            row["sii_per1000"] = np.nan
        rows.append(row)

    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "table_percountry_equity_depth.csv", index=False)
    print(f"wrote table_percountry_equity_depth.csv ({len(out)} countries; "
          f"{int(out['sii_per1000'].notna().sum())} with SII)")


if __name__ == "__main__":
    main()
