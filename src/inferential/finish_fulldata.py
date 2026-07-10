"""
finish_fulldata.py — sequential completion of the full-data run
===============================================================

Runs the remaining work in ONE process (no concurrency -> no OOM):
  1. Fast PAF cluster bootstrap (500 reps) with warm-started IRLS refits
     (3 Newton iters from the point estimate ~= 5x faster than cold lbfgs,
     identical MLE). Fills CI columns into table_3_gcomp_pafs_fulldata.csv.
  2. The four phase-2 functions that OOM'd under concurrency:
     s05g, s05k_parity, s05m, s05n (s05j + s05k_equity already done).

Run (background, ALONE — no other heavy job): python -m src.inferential.finish_fulldata
"""
from __future__ import annotations
import os, sys, gc, time, traceback
from datetime import datetime
from pathlib import Path
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.paths import RNG_SEED, TABLES_DIR
from src.inferential import _fulldata as FD
from src.inferential.run_fulldata_all import (
    build_dummies, design_from, M3, PAF_FACTORS, _paf_one, table4)
import src.inferential.run_fulldata_phase2 as P2

OUTDIR = TABLES_DIR.parent / "fulldata"
LOG = OUTDIR / "finish.log"
N_BOOT_PAF = 500


def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def paf_bootstrap(df, dummies):
    log("PAF cluster bootstrap (warm-started IRLS, 500 reps)")
    n = len(df)
    y = df["died_under5"].astype(np.float64).values
    keep = [k for k in M3 if dummies[k].std() > 0]
    X = design_from(dummies, keep, n)
    colidx = {nm: i + 1 for i, nm in enumerate(keep)}
    beta_pt = FD.irls_logit(X, y)            # point-estimate coefficients (warm-start seed)

    cc = df["country_code"].values
    uniq, groups = FD.cluster_row_index(cc)
    rng = np.random.default_rng(RNG_SEED)
    boot = {lab: [] for lab, _ in PAF_FACTORS}
    t0 = time.time()
    for b, idx in FD.iter_cluster_bootstrap(uniq, groups, rng, N_BOOT_PAF):
        Xb, yb = X[idx], y[idx]
        try:
            bb = FD.irls_logit(Xb, yb, start=beta_pt, max_iter=8)
            for lab, shift in PAF_FACTORS:
                sh = {k: v for k, v in shift.items() if k in colidx}
                boot[lab].append(_paf_one(Xb, bb, colidx, sh))
        except Exception:
            pass
        del Xb, yb
        if (b + 1) % 25 == 0:
            el = time.time() - t0
            eta = (N_BOOT_PAF - b - 1) / ((b + 1) / el)
            log(f"  boot {b+1}/{N_BOOT_PAF} ({el:.0f}s, ~{eta:.0f}s left)")
            gc.collect()

    # merge CIs into the existing point-estimate table
    path = OUTDIR / "table_3_gcomp_pafs_fulldata.csv"
    tbl = pd.read_csv(path)
    for lab, _ in PAF_FACTORS:
        bv = np.array(boot[lab]) * 100
        if len(bv) >= 20:
            lo, hi = np.percentile(bv, [2.5, 97.5])
            m = tbl["risk_factor"] == lab
            tbl.loc[m, "CI_lo"] = round(float(lo), 2)
            tbl.loc[m, "CI_hi"] = round(float(hi), 2)
            tbl.loc[m, "PAF_All_fmt"] = (
                f"{float(tbl.loc[m, 'PAF_All'].iloc[0])} "
                f"({round(float(lo),2)}–{round(float(hi),2)})")
    tbl.to_csv(path, index=False)
    log("  PAF bootstrap CIs written")


def main():
    log("=" * 70)
    log("FINISH BATCH START (sequential)")
    cols = ["country_code", "who_region", "survey_year", "died_under5",
            "child_male", "multiple_birth", "birth_order",
            "birth_interval_months", "maternal_age_at_birth", "edu_level",
            "wealth_quintile", "currently_married", "urban", "sample_weight",
            "_source_file", "years_before_survey"]
    df = FD.load_full(cols, scope="primary")
    df = df.sort_values("country_code").reset_index(drop=True)
    log(f"Loaded full cohort: {len(df):,} rows")
    dummies = build_dummies(df)

    for name, fn in [
        ("TABLE4_wagstaff", lambda: table4(df, dummies)),   # main batch died before this
        ("PAF_bootstrap", lambda: paf_bootstrap(df, dummies)),
        ("s05g", lambda: P2.s05g_fully_exposed(df, dummies)),
        ("s05k_parity", lambda: P2.s05k_parity(df, dummies)),
        ("s05m", lambda: P2.s05m_paf_sens(df, dummies)),
        ("s05n", lambda: P2.s05n_gee_sens(df, dummies)),
    ]:
        try:
            t0 = time.time()
            fn()
            log(f"{name} OK ({time.time()-t0:.0f}s)")
        except Exception:
            log(f"{name} CRASHED:\n{traceback.format_exc()}")
        gc.collect()
    log("FINISH BATCH COMPLETE")


if __name__ == "__main__":
    main()
