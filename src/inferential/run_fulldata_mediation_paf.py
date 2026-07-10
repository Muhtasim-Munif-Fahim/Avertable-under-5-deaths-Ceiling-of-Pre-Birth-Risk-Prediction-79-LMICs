"""run_fulldata_mediation_paf.py  (P1-11, 2026-06-23)
================================================================
Total-effect education PAF, to report ALONGSIDE the primary mediator-adjusted
PAF (37.5%). The primary Model-3 g-computation conditions on wealth quintile,
urban residence, preceding birth interval, and birth order -- all plausible
MEDIATORS of maternal education -> child survival. Conditioning on mediators
estimates a controlled-direct-effect-like quantity that UNDER-states the total
effect. This script re-estimates the education PAF adjusting for CONFOUNDERS /
precision covariates only (child sex, multiple birth, maternal age, marital
status, survey decade), dropping the four mediators, so the two PAFs bracket
the education contribution.

Reuses the validated full-data g-computation engine in run_fulldata_all
(unweighted GLM + whole-population shift, per the s05c provenance decision).
Point estimate uses the full cohort; the cluster bootstrap caps each country's
resample pool at 20,000 rows for tractability (same convention as the joint
counterfactual), affecting only the CI width, not the point estimate.
"""
from __future__ import annotations
import sys, time, gc
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.paths import RNG_SEED
import src.inferential._fulldata as FD
from src.inferential.run_fulldata_all import (
    build_dummies, design_from, _paf_one, OUTDIR, log,
)

# Confounder / precision set for the TOTAL effect of education
# (drops the mediators wq_*, is_urban, bi_*, bo_*).
M_EDU_TOTAL = ["male", "is_multiple", "age_under20", "age_over35", "married",
               "decade_2010", "decade_2020",
               "edu_primary", "edu_secondary", "edu_higher"]
EDU_SHIFT = {"edu_primary": 0, "edu_secondary": 1, "edu_higher": 0}
N_BOOT = 500
BOOT_CAP = 20000


def main():
    log("P1-11 total-effect education PAF (confounder-only adjustment)")
    cols = ["country_code", "who_region", "survey_year", "died_under5",
            "child_male", "multiple_birth", "birth_order",
            "birth_interval_months", "maternal_age_at_birth", "edu_level",
            "wealth_quintile", "currently_married", "urban", "sample_weight"]
    df = FD.load_full(cols, scope="primary")
    df = df.sort_values("country_code").reset_index(drop=True)
    n = len(df)
    log(f"Loaded {n:,} rows, {df['country_code'].nunique()} countries")
    dummies = build_dummies(df)
    y = df["died_under5"].astype(np.float64).values

    keep = [k for k in M_EDU_TOTAL if dummies[k].std() > 0]
    X = design_from(dummies, keep, n)
    colidx = {nm: i + 1 for i, nm in enumerate(keep)}
    beta = FD.logit_coefs(X, y)
    shift = {k: v for k, v in EDU_SHIFT.items() if k in colidx}
    paf_total = _paf_one(X, beta, colidx, shift) * 100
    log(f"  Total-effect education PAF (point) = {paf_total:.2f}%")

    # capped cluster bootstrap
    cc = df["country_code"].values
    uniq, groups = FD.cluster_row_index(cc)
    rng = np.random.default_rng(RNG_SEED)
    groups_cap = {}
    for k, pos in groups.items():
        if len(pos) > BOOT_CAP:
            groups_cap[k] = pos[np.linspace(0, len(pos) - 1, BOOT_CAP).astype(np.intp)]
        else:
            groups_cap[k] = pos
    boot = []
    t0 = time.time()
    for b, idx in FD.iter_cluster_bootstrap(uniq, groups_cap, rng, N_BOOT):
        Xb, yb = X[idx], y[idx]
        try:
            bb = FD.logit_coefs(Xb, yb)
            boot.append(_paf_one(Xb, bb, colidx, shift) * 100)
        except Exception:
            pass
        del Xb, yb
        if (b + 1) % 50 == 0:
            log(f"    boot {b+1}/{N_BOOT} ({time.time()-t0:.0f}s)")
            gc.collect()
    bv = np.array(boot)
    lo, hi = (float(np.percentile(bv, 2.5)), float(np.percentile(bv, 97.5))
              ) if len(bv) >= 20 else (float("nan"), float("nan"))
    out = pd.DataFrame([{
        "estimand": "Education total-effect PAF (confounder-only)",
        "adjustment": "+".join(keep),
        "PAF_pct": round(paf_total, 2),
        "CI_lo": round(lo, 2), "CI_hi": round(hi, 2),
        "n_boot": len(bv),
        "primary_mediator_adjusted_PAF": 37.5,
    }])
    op = OUTDIR / "table_education_total_effect_paf_fulldata.csv"
    out.to_csv(op, index=False)
    log(f"  Wrote {op}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
