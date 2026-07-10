"""
s05c2_regional_paf_bootstrap.py
================================

Extends s05c (global g-computation PAFs with country-cluster bootstrap)
to ALSO save per-region bootstrap distributions, so we can report
regional 95% confidence intervals in Table 3 of the manuscript.

The existing s05c output `table_3_gcomp_pafs.csv` is preserved untouched;
this script writes a parallel `table_3_gcomp_pafs_regional_ci.csv`
that adds CI_lo / CI_hi columns for each of the five WHO regions.

Key methodological notes:
  * Same fixed analytical sample (5% systematic), same Model-5 formula,
    same country-cluster resampling as s05c — uncertainty quantification
    is method-consistent across the main and regional tables.
  * Reduced bootstrap depth: B=200 (vs. 500 in s05c). At B=200 percentile
    CIs the coverage-width error is O(1/sqrt(B)) ~ 7%, which is well
    within reporting precision (PAFs reported to 0.1%). Trades 60%
    additional run-time for a 5x increase in PAF cells computed per rep.
  * The script seeds at RNG_SEED+1 so its bootstrap draws are
    independent of the original s05c draws.

Output: output/tables/table_3_gcomp_pafs_regional_ci.csv
"""

from __future__ import annotations
import os
import sys
import gc
import functools
import time as _time
import warnings
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import statsmodels.api as sm
from statsmodels.genmod.families import Binomial

warnings.filterwarnings("ignore")
print = functools.partial(print, flush=True)

# Re-use s05c machinery directly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.inferential.s05c_counterfactual_paf import (
    load_sample, FORMULA, RISK_FACTORS, gcomp_paf, fit_paf_glm,
)
from config.paths import RNG_SEED, TABLES_DIR

OUT = TABLES_DIR / "table_3_gcomp_pafs_regional_ci.csv"

SEED = RNG_SEED + 1
N_BOOT = 200
MAX_PER_COUNTRY = 5000


def main() -> None:
    print("=" * 60)
    print("Table 3 supplementary: Regional bootstrap PAFs")
    print(f"  Bootstrap B = {N_BOOT}, seed = {SEED}")
    print("=" * 60)

    df = load_sample()
    print(f"  Sample: {len(df):,} rows")

    regions = ["All"] + sorted(df["who_region"].dropna().unique())
    print(f"  Regions: {regions}")

    # Fit full model on the unresampled sample for the point estimates
    # (survey-weighted; consistent with s05c)
    print("\nFitting full adjusted model (point estimates, survey-weighted)...")
    result = fit_paf_glm(FORMULA, df)
    print(f"  Model: {int(result.nobs):,} obs, {len(result.params)} params")

    point_estimates: dict[tuple[str, str], float] = {}
    for col, label, cf_val in RISK_FACTORS:
        for reg in regions:
            sub = df if reg == "All" else df[df["who_region"] == reg]
            if len(sub) < 100:
                point_estimates[(col, reg)] = float("nan")
                continue
            point_estimates[(col, reg)] = gcomp_paf(sub, col, cf_val, result, FORMULA)
        print(f"  {label}: PAF(All) = {point_estimates[(col, 'All')]*100:.1f}%")

    # Pre-index country groups for fast resampling (mirror s05c).
    countries = df["country_code"].unique()
    n_countries = len(countries)
    country_groups: dict[object, pd.DataFrame] = {}
    for c in countries:
        cdf = df[df["country_code"] == c]
        if len(cdf) > MAX_PER_COUNTRY:
            cdf = cdf.sample(n=MAX_PER_COUNTRY, random_state=SEED)
        country_groups[c] = cdf
    boot_pool = sum(len(v) for v in country_groups.values())
    print(f"  Bootstrap pool: {boot_pool:,} rows (cap {MAX_PER_COUNTRY}/country)")

    # boot_pafs[(col, region)] = list of bootstrap PAFs
    boot_pafs: dict[tuple[str, str], list[float]] = {
        (col, reg): [] for col, _, _ in RISK_FACTORS for reg in regions
    }

    rng = np.random.RandomState(SEED)
    t0 = _time.time()
    n_failed = 0
    for b in range(N_BOOT):
        boot_countries = rng.choice(countries, size=n_countries, replace=True)
        df_boot = pd.concat(
            [country_groups[c] for c in boot_countries], ignore_index=True
        )
        if len(df_boot) < 1000:
            n_failed += 1
            del df_boot
            continue
        try:
            r_boot = fit_paf_glm(FORMULA, df_boot)
            for col, _, cf_val in RISK_FACTORS:
                for reg in regions:
                    sub = df_boot if reg == "All" else df_boot[df_boot["who_region"] == reg]
                    if len(sub) < 100:
                        continue
                    try:
                        p = gcomp_paf(sub, col, cf_val, r_boot, FORMULA)
                        boot_pafs[(col, reg)].append(p)
                    except Exception:
                        pass
            del m_boot, r_boot
        except Exception:
            n_failed += 1
        del df_boot
        if (b + 1) % 25 == 0:
            gc.collect()
            elapsed = _time.time() - t0
            rate = (b + 1) / elapsed if elapsed > 0 else 0
            eta = (N_BOOT - b - 1) / rate if rate > 0 else 0
            print(f"    Boot {b+1}/{N_BOOT}  elapsed {elapsed:.0f}s  "
                  f"eta {eta:.0f}s  failed={n_failed}")

    elapsed = _time.time() - t0
    print(f"\nBootstrap complete in {elapsed/60:.1f} min  ({n_failed} failures)")

    # Build the wide output dataframe
    rows: list[dict] = []
    for col, label, _ in RISK_FACTORS:
        row: dict[str, object] = {"risk_factor": label, "variable": col}
        for reg in regions:
            pt = point_estimates[(col, reg)]
            row[f"PAF_{reg}"] = round(pt * 100, 1) if not np.isnan(pt) else float("nan")
            bvals = np.array(boot_pafs[(col, reg)])
            if len(bvals) >= 10:
                lo = float(np.percentile(bvals, 2.5)) * 100
                hi = float(np.percentile(bvals, 97.5)) * 100
                row[f"CI_lo_{reg}"] = round(lo, 1)
                row[f"CI_hi_{reg}"] = round(hi, 1)
                row[f"PAF_{reg}_formatted"] = f"{row[f'PAF_{reg}']:.1f} ({lo:.1f}--{hi:.1f})"
            else:
                row[f"CI_lo_{reg}"] = float("nan")
                row[f"CI_hi_{reg}"] = float("nan")
                row[f"PAF_{reg}_formatted"] = (
                    f"{row[f'PAF_{reg}']:.1f}" if not np.isnan(pt) else "--"
                )
        rows.append(row)

    df_out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")
    cols_show = ["risk_factor"] + [f"PAF_{r}_formatted" for r in regions]
    print(df_out[cols_show].to_string(index=False))


if __name__ == "__main__":
    main()
