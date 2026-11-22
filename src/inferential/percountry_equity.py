"""percountry_equity.py — per-country wealth equity on the v2 exact cohort.

Closes the display-item gap left by the v2 rebuild: the legacy per-country equity
table (poorest/richest under-five mortality, their ratio, and the concentration
index with a bootstrap CI) had no v2 counterpart, because the v2 inequality port
regenerated only the pooled Wagstaff decomposition.

For each country, on the exact >=60-month follow-up cohort with complete wealth:
  * survey-weighted under-five risk per 1,000 in the poorest (Q1) and richest (Q5)
    within-survey wealth quintile, and their ratio;
  * the wealth concentration index using the same grouped (midpoint) fractional
    rank as the pooled decomposition, so country and pooled values are comparable;
  * a within-country nonparametric percentile CI from resampling records.

Countries with fewer than MIN_DEATHS recorded deaths report ".." for the unstable
quantities rather than a spuriously precise number.

Output: output/phase4/manuscript/table_percountry_equity.csv
Run:    python -m src.inferential.percountry_equity
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from config.paths import HARMONIZED_DIR, OUTPUT_DIR, RNG_SEED
from src.inferential.phase2_analysis import _positive_normalized_weights, _read_pilot
from src.inferential.s05d_inequality import _conc_index, _grouped_fractional_rank

OUT = OUTPUT_DIR / "phase4" / "manuscript"
MIN_DEATHS = 30          # below this the quintile contrasts are unstable
BOOT_REPS = 300          # matches the legacy per-country bootstrap depth


def main(reps: int = BOOT_REPS, seed: int = RNG_SEED) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = _read_pilot(HARMONIZED_DIR / "global_analysis_v2_candidate.parquet")

    w_all = _positive_normalized_weights(frame)
    keep = (w_all.notna()
            & frame["wealth_quintile"].notna()
            & pd.to_numeric(frame["died_under5_complete"], errors="coerce").notna())
    d = frame.loc[keep, ["country_code", "who_region", "wealth_quintile",
                         "died_under5_complete"]].copy()
    d["w"] = w_all.loc[keep].to_numpy(dtype=float)
    d["y"] = pd.to_numeric(d["died_under5_complete"], errors="coerce").to_numpy(dtype=float)
    d["wq"] = pd.to_numeric(d["wealth_quintile"], errors="coerce").to_numpy(dtype=float)

    inv = pd.read_csv(OUT / "table_s1_survey_inventory.csv")
    names = inv.groupby("country_code")["country_name"].first().to_dict()

    rng = np.random.default_rng(seed)
    rows = []
    for cc, g in d.groupby("country_code", sort=True):
        y = g["y"].to_numpy(); w = g["w"].to_numpy(); wq = g["wq"].to_numpy()
        n_deaths = float((y * (w > 0)).sum())

        def _q_rate(q):
            m = wq == q
            return 1000.0 * np.average(y[m], weights=w[m]) if m.any() and w[m].sum() > 0 else np.nan

        q1, q5 = _q_rate(1), _q_rate(5)
        ratio = q1 / q5 if (np.isfinite(q1) and np.isfinite(q5) and q5 > 0) else np.nan
        ci_point = _conc_index(y, _grouped_fractional_rank(wq, w), w)

        lo = hi = np.nan
        if n_deaths >= MIN_DEATHS and len(y) > 1:
            draws = []
            n = len(y)
            for _ in range(reps):
                idx = rng.integers(0, n, size=n)
                yb, wb, qb = y[idx], w[idx], wq[idx]
                if wb.sum() <= 0:
                    continue
                v = _conc_index(yb, _grouped_fractional_rank(qb, wb), wb)
                if np.isfinite(v):
                    draws.append(v)
            if len(draws) >= 20:
                lo, hi = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))

        unstable = n_deaths < MIN_DEATHS
        rows.append({
            "country_code": cc,
            "country_name": names.get(cc, cc),
            "who_region": g["who_region"].astype(str).iloc[0],
            "n_births": int(len(g)),
            "n_deaths": int(round(n_deaths)),
            "u5m_q1_poorest": np.nan if unstable else round(q1, 1),
            "u5m_q5_richest": np.nan if unstable else round(q5, 1),
            "ratio_q1_q5": np.nan if unstable else round(ratio, 2),
            "concentration_index": round(ci_point, 4),
            "ci_lo": round(lo, 4) if np.isfinite(lo) else np.nan,
            "ci_hi": round(hi, 4) if np.isfinite(hi) else np.nan,
            "unstable_lt30_deaths": bool(unstable),
        })

    out = pd.DataFrame(rows).sort_values(
        ["who_region", "u5m_q1_poorest"], ascending=[True, False])
    out.to_csv(OUT / "table_percountry_equity.csv", index=False)

    n_ci = int(out["ci_lo"].notna().sum())
    n_uns = int(out["unstable_lt30_deaths"].sum())
    pro_poor = int((out["concentration_index"] < 0).sum())
    print(f"[equity-v2] {len(out)} countries; {n_ci} with bootstrap CIs; "
          f"{n_uns} flagged unstable (<{MIN_DEATHS} deaths)")
    print(f"[equity-v2] pro-poor (negative index): {pro_poor}/{len(out)}")
    print(f"[equity-v2] wrote {OUT / 'table_percountry_equity.csv'}")


if __name__ == "__main__":
    main()
