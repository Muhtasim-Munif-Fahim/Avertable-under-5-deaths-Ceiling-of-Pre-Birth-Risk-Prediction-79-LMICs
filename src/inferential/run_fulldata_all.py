"""
run_fulldata_all.py — FULL-DATASET batch for Tables 2, 3, 4 (no subsampling)
============================================================================

Runs the primary inferential analyses on ALL 10,669,363 in-scope births using
the validated, memory/overflow-safe engine in `_fulldata.py`. Designed to run
unattended in the background:

  * each table is wrapped in try/except and its result saved as soon as it is
    computed, so one failure never discards completed work;
  * a heartbeat log (output/fulldata/run.log) records every step with
    timestamps for crash recovery;
  * point estimates are computed and saved BEFORE the (long) bootstraps, so even
    an interrupted run leaves usable estimates.

Outputs (output/fulldata/):
  table_2_sequential_models_fulldata.csv   (GEE-Exch, robust sandwich CIs)
  table_3_gcomp_pafs_fulldata.csv          (g-computation PAFs + cluster boot CI)
  table_4_wagstaff_fulldata.csv            (decomposition) + table_4_ci_fulldata.csv

Run (background): python -m src.inferential.run_fulldata_all
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
from scipy import stats
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.paths import RNG_SEED, TABLES_DIR, COUNTRY_COVARIATES
from src.inferential import _fulldata as FD
import pyarrow.parquet as pq

OUTDIR = TABLES_DIR.parent / "fulldata"
OUTDIR.mkdir(parents=True, exist_ok=True)
LOG = OUTDIR / "run.log"

N_BOOT_PAF = 500      # D016 cluster bootstrap reps for PAF CIs
N_BOOT_CI = 500       # cluster bootstrap reps for concentration-index CIs


def log(msg: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


# --------------------------------------------------------------------------- #
# Shared dummy construction (matches s05L / s05b coding exactly)              #
# --------------------------------------------------------------------------- #
DUMMY_NAMES = ["male", "is_multiple", "bo_2_3", "bo_4_5", "bo_6plus",
               "bi_firstmissing", "bi_short", "bi_long", "age_under20",
               "age_over35", "edu_primary", "edu_secondary", "edu_higher",
               "wq_2", "wq_3", "wq_4", "wq_5", "married", "is_urban",
               "decade_2010", "decade_2020"]

M1 = ["male", "is_multiple", "bo_2_3", "bo_4_5", "bo_6plus",
      "bi_firstmissing", "bi_short", "bi_long"]
M2 = M1 + ["age_under20", "age_over35", "edu_primary", "edu_secondary",
           "edu_higher", "wq_2", "wq_3", "wq_4", "wq_5", "married", "is_urban"]
M3 = M2 + ["decade_2010", "decade_2020"]


def build_dummies(df) -> dict:
    """All Model-3 dummy columns as float64 arrays (keyed by name)."""
    d = {}
    d["male"] = (df["child_male"].fillna(0).values > 0).astype(np.float64)
    d["is_multiple"] = (df["multiple_birth"].fillna(0).values > 0).astype(np.float64)
    bo = df["birth_order"].fillna(1).values
    d["bo_2_3"] = ((bo >= 2) & (bo <= 3)).astype(np.float64)
    d["bo_4_5"] = ((bo >= 4) & (bo <= 5)).astype(np.float64)
    d["bo_6plus"] = (bo >= 6).astype(np.float64)
    bi = df["birth_interval_months"].values
    nb = np.isnan(bi)
    d["bi_firstmissing"] = nb.astype(np.float64)
    d["bi_short"] = ((~nb) & (bi < 24)).astype(np.float64)
    d["bi_long"] = ((~nb) & (bi >= 48)).astype(np.float64)
    ma = df["maternal_age_at_birth"].fillna(20).values
    d["age_under20"] = (ma < 20).astype(np.float64)
    d["age_over35"] = (ma >= 35).astype(np.float64)
    edu = df["edu_level"].fillna(0).values
    d["edu_primary"] = (edu == 1).astype(np.float64)
    d["edu_secondary"] = (edu == 2).astype(np.float64)
    d["edu_higher"] = (edu >= 3).astype(np.float64)
    wq = df["wealth_quintile"].values
    d["wq_2"] = (wq == 2).astype(np.float64)
    d["wq_3"] = (wq == 3).astype(np.float64)
    d["wq_4"] = (wq == 4).astype(np.float64)
    d["wq_5"] = (wq == 5).astype(np.float64)
    d["married"] = (df["currently_married"].fillna(1).values > 0).astype(np.float64)
    d["is_urban"] = (df["urban"].fillna(0).values > 0).astype(np.float64)
    sy = df["survey_year"].values
    d["decade_2010"] = ((sy >= 2010) & (sy <= 2019)).astype(np.float64)
    d["decade_2020"] = (sy >= 2020).astype(np.float64)
    return d


def design_from(dummies: dict, names: list[str], n: int) -> np.ndarray:
    cols = [np.ones(n, dtype=np.float64)] + [dummies[k] for k in names]
    return np.column_stack(cols)


def _or_rows(beta, se, names, method, n, extra=None):
    rows = []
    for j, nm in enumerate(names, start=1):   # skip intercept at 0
        z = beta[j] / se[j]
        rows.append({
            "Variable": nm, "N": int(n),
            "OR": round(float(np.exp(beta[j])), 4),
            "CI_lower": round(float(np.exp(beta[j] - 1.96 * se[j])), 4),
            "CI_upper": round(float(np.exp(beta[j] + 1.96 * se[j])), 4),
            "p_value": round(float(2 * stats.norm.sf(abs(z))), 5),
            "Method": method,
        })
        if extra:
            rows[-1].update(extra)
    return rows


# --------------------------------------------------------------------------- #
# TABLE 2 — sequential GEE models on full data                                #
# --------------------------------------------------------------------------- #
def table2(df, groups_sorted, dummies):
    log("TABLE 2: sequential exchangeable GEE (Models 1-5), full data")
    n = len(df)
    y = df["died_under5"].astype(np.float64).values
    all_rows = []

    for label, names in [("Model1", M1), ("Model2", M2), ("Model3", M3)]:
        keep = [k for k in names if dummies[k].std() > 0]
        X = design_from(dummies, keep, n)
        t0 = time.time()
        beta, se, alpha, nit, conv = FD.fit_exchangeable_gee(
            X, y, groups_sorted, verbose=False)
        log(f"  {label}: conv={conv} alpha={alpha:.4f} iters={nit} "
            f"({time.time()-t0:.0f}s)")
        all_rows += _or_rows(beta, se, keep, "GEE-Exch-FullData", n,
                             extra={"Model": label})
        pd.DataFrame(all_rows).to_csv(
            OUTDIR / "table_2_sequential_models_fulldata.csv", index=False)
        del X; gc.collect()

    # Model 4: + WDI country covariates (rows with all three present)
    try:
        wdi = pq.read_table(str(COUNTRY_COVARIATES)).to_pandas()
        wdi["survey_year"] = wdi["survey_year"].astype(int)
        wdi = wdi[["country_code", "survey_year", "gdp_per_capita_ppp",
                   "health_exp_pct_gdp", "physicians_per_1000"]]
        m = df[["country_code", "survey_year"]].copy()
        m["survey_year"] = m["survey_year"].astype(int)
        m = m.merge(wdi, on=["country_code", "survey_year"], how="left")
        log_gdp = np.log(np.clip(m["gdp_per_capita_ppp"].values, 1, None))
        hexp = m["health_exp_pct_gdp"].values
        phys = m["physicians_per_1000"].values
        ok = np.isfinite(log_gdp) & np.isfinite(hexp) & np.isfinite(phys)
        log(f"  Model4 WDI coverage: {ok.sum():,}/{n:,}")
        keep = [k for k in M3 if dummies[k][ok].std() > 0]
        X3 = design_from(dummies, keep, n)[ok]
        Xc = np.column_stack([X3, log_gdp[ok], hexp[ok], phys[ok]])
        g4 = groups_sorted[ok]
        # groups must stay contiguous-sorted; ok-mask preserves order so re-pack
        order = np.argsort(g4, kind="stable")
        Xc, y4, g4 = Xc[order], y[ok][order], g4[order]
        names4 = keep + ["log_gdp_ppp", "health_exp_pct_gdp", "physicians_per_1000"]
        t0 = time.time()
        beta, se, alpha, nit, conv = FD.fit_exchangeable_gee(Xc, y4, g4)
        log(f"  Model4: conv={conv} alpha={alpha:.4f} iters={nit} "
            f"({time.time()-t0:.0f}s)")
        all_rows += _or_rows(beta, se, names4, "GEE-Exch-FullData", int(ok.sum()),
                             extra={"Model": "Model4"})
        pd.DataFrame(all_rows).to_csv(
            OUTDIR / "table_2_sequential_models_fulldata.csv", index=False)
        del X3, Xc; gc.collect()
    except Exception as e:
        log(f"  Model4 FAILED: {e}")

    # Model 5: + healthcare (recent births only)
    try:
        rb = (df.get("is_recent_birth", pd.Series(np.zeros(n))).fillna(0).values == 1)
        log(f"  Model5 recent-birth rows: {rb.sum():,}")
        hc = {}
        anc = df["anc_visits"].values
        hc["anc_none"] = ((~np.isnan(anc)) & (anc == 0)).astype(np.float64)
        hc["anc_1_3"] = ((~np.isnan(anc)) & (anc >= 1) & (anc <= 3)).astype(np.float64)
        hc["anc_missing"] = np.isnan(anc).astype(np.float64)
        hc["delivery_home"] = (df["facility_delivery"].fillna(-1).values != 1).astype(np.float64)
        hc["csection_yes"] = (df["caesarean"].fillna(0).values == 1).astype(np.float64)
        hc["tetanus_not"] = (df["tetanus_protected"].fillna(0).values != 1).astype(np.float64)
        keep3 = [k for k in M3 if dummies[k][rb].std() > 0]
        hckeep = [k for k in hc if hc[k][rb].std() > 0]
        cols = [np.ones(int(rb.sum()))]
        cols += [dummies[k][rb] for k in keep3]
        cols += [hc[k][rb] for k in hckeep]
        X5 = np.column_stack(cols)
        g5 = groups_sorted[rb]
        order = np.argsort(g5, kind="stable")
        X5, y5, g5 = X5[order], y[rb][order], g5[order]
        names5 = keep3 + hckeep
        t0 = time.time()
        beta, se, alpha, nit, conv = FD.fit_exchangeable_gee(X5, y5, g5)
        log(f"  Model5: conv={conv} alpha={alpha:.4f} iters={nit} "
            f"({time.time()-t0:.0f}s)")
        all_rows += _or_rows(beta, se, names5, "GEE-Exch-FullData", int(rb.sum()),
                             extra={"Model": "Model5"})
        pd.DataFrame(all_rows).to_csv(
            OUTDIR / "table_2_sequential_models_fulldata.csv", index=False)
        del X5; gc.collect()
    except Exception as e:
        log(f"  Model5 FAILED: {e}")

    log("TABLE 2 done")


# --------------------------------------------------------------------------- #
# TABLE 3 — g-computation PAFs on full data + cluster bootstrap               #
# --------------------------------------------------------------------------- #
# Whole-population shift: set every record to the target level (UNWEIGHTED GLM,
# per the s05c provenance decision — do NOT weight, do NOT use worse-only).
PAF_FACTORS = [
    ("Education < Secondary -> Secondary+", {"edu_primary": 0, "edu_secondary": 1, "edu_higher": 0}),
    ("Wealth < Richest -> Richest", {"wq_2": 0, "wq_3": 0, "wq_4": 0, "wq_5": 1}),
    ("Rural -> Urban", {"is_urban": 1}),
    ("Maternal age != 20-34 -> 20-34", {"age_under20": 0, "age_over35": 0}),
    ("Birth interval != 24-47mo -> 24-47mo", {"bi_firstmissing": 0, "bi_short": 0, "bi_long": 0}),
    ("Unmarried -> Married", {"married": 1}),
    ("Multiple -> Singleton", {"is_multiple": 0}),
    ("Male -> Female", {"male": 0}),
]


def _paf_one(X, beta, colidx, shift):
    # memory-frugal: edit only the shifted columns in place, then restore
    # (avoids a full ~1.75 GB X.copy() per call).
    pf = expit(X @ beta)
    rf = pf.mean()
    saved = {}
    for cname, val in shift.items():
        ci = colidx[cname]
        saved[ci] = X[:, ci].copy()
        X[:, ci] = val
    pcf = expit(X @ beta)
    for ci, col in saved.items():
        X[:, ci] = col
    return float((rf - pcf.mean()) / rf) if rf > 0 else 0.0


def table3(df, dummies):
    log("TABLE 3: g-computation PAFs (Model-3 surface), full data")
    n = len(df)
    y = df["died_under5"].astype(np.float64).values
    keep = [k for k in M3 if dummies[k].std() > 0]
    X = design_from(dummies, keep, n)
    colidx = {nm: i + 1 for i, nm in enumerate(keep)}   # +1 for intercept
    beta = FD.logit_coefs(X, y)

    region = df["who_region"].values
    regions = ["All"] + sorted(pd.unique(region[pd.notna(region)]))
    rows = []
    for label, shift in PAF_FACTORS:
        shift = {k: v for k, v in shift.items() if k in colidx}
        rec = {"risk_factor": label}
        for reg in regions:
            mask = slice(None) if reg == "All" else (region == reg)
            paf = _paf_one(X[mask], beta, colidx, shift)
            rec[f"PAF_{reg}"] = round(paf * 100, 2)
        rows.append(rec)
        log(f"  {label}: PAF(All)={rec['PAF_All']}%")
    pd.DataFrame(rows).to_csv(OUTDIR / "table_3_gcomp_pafs_fulldata.csv", index=False)
    log("  PAF point estimates saved; starting cluster bootstrap...")

    # Cluster bootstrap (All region), FULL rows per country (no cap).
    cc = df["country_code"].values
    uniq, groups = FD.cluster_row_index(cc)
    rng = np.random.default_rng(RNG_SEED)
    boot = {lab: [] for lab, _ in PAF_FACTORS}
    t0 = time.time()
    for b, idx in FD.iter_cluster_bootstrap(uniq, groups, rng, N_BOOT_PAF):
        Xb, yb = X[idx], y[idx]
        try:
            bb = FD.logit_coefs(Xb, yb)
            for lab, shift in PAF_FACTORS:
                sh = {k: v for k, v in shift.items() if k in colidx}
                boot[lab].append(_paf_one(Xb, bb, colidx, sh))
        except Exception:
            pass
        del Xb, yb
        if (b + 1) % 25 == 0:
            el = time.time() - t0
            eta = (N_BOOT_PAF - b - 1) / ((b + 1) / el)
            log(f"    PAF boot {b+1}/{N_BOOT_PAF} ({el:.0f}s, ~{eta:.0f}s left)")
            gc.collect()
    for rec, (lab, _) in zip(rows, PAF_FACTORS):
        bv = np.array(boot[lab]) * 100
        if len(bv) >= 20:
            rec["CI_lo"] = round(float(np.percentile(bv, 2.5)), 2)
            rec["CI_hi"] = round(float(np.percentile(bv, 97.5)), 2)
            rec["PAF_All_fmt"] = f"{rec['PAF_All']} ({rec['CI_lo']}–{rec['CI_hi']})"
    pd.DataFrame(rows).to_csv(OUTDIR / "table_3_gcomp_pafs_fulldata.csv", index=False)
    log("TABLE 3 done")


# --------------------------------------------------------------------------- #
# TABLE 4 — Wagstaff decomposition + concentration index, full data          #
# --------------------------------------------------------------------------- #
WAG_COLS = ["male", "is_multiple", "bi_short", "bi_long", "bi_firstmissing",
            "bo_2_3", "bo_4_5", "bo_6plus", "age_under20", "age_over35",
            "married", "edu_primary", "edu_secondary", "edu_higher",
            "is_urban", "decade_2010", "decade_2020"]


def _conc_index(y, frac_rank, w):
    mu = np.average(y, weights=w)
    if mu == 0:
        return 0.0
    mr = np.average(frac_rank, weights=w)
    cov = np.average((y - mu) * (frac_rank - mr), weights=w)
    return (2.0 / mu) * cov


def _wls_beta(X, y, w):
    """Weighted least squares via normal equations (no SVD; full-N safe)."""
    Xw = X * w[:, None]
    XtWX = X.T @ Xw
    XtWy = Xw.T @ y
    return np.linalg.solve(XtWX, XtWy)


def table4(df, dummies):
    log("TABLE 4: Wagstaff decomposition (weighted LPM), full data")
    # Restrict to wealth-rankable records (drop missing wealth) — imputing the
    # ranking variable to the median quintile dilutes the gradient toward zero
    # and is inconsistent with the directly-measured concentration index (s05k
    # / Table 3). Use the grouped (group-midpoint) wealth rank for the same
    # reason it is used there (see _ci_from_arrays in run_fulldata_phase2).
    wmask = df["wealth_quintile"].notna().values
    n = int(wmask.sum())
    y = df["died_under5"].astype(np.float64).values[wmask]
    w = df["sample_weight"].fillna(1.0).astype(np.float64).values[wmask]
    q = df["wealth_quintile"].astype(np.float64).values[wmask]
    dummies = {k: np.asarray(dummies[k])[wmask] for k in dummies}
    log(f"  wealth-rankable rows: {n:,} (dropped {int((~wmask).sum()):,} missing wealth)")

    tw = w.sum()
    frac = np.empty(n)
    cum = 0.0
    for v in np.unique(q):            # grouped midpoint rank; ascending quintiles
        m = (q == v)
        gw = w[m].sum()
        frac[m] = (cum + 0.5 * gw) / tw
        cum += gw

    keep = [k for k in WAG_COLS if dummies[k].std() > 0]
    X = np.column_stack([np.ones(n)] + [dummies[k] for k in keep])
    beta = _wls_beta(X, y, w)
    total_ci = _conc_index(y, frac, w)
    mu_y = np.average(y, weights=w)

    rows = []
    ssum = 0.0
    for j, col in enumerate(keep, start=1):
        xv = dummies[col]
        mu_x = np.average(xv, weights=w)
        ci_x = _conc_index(xv, frac, w)
        elast = (beta[j] * mu_x) / mu_y
        contrib = elast * ci_x
        ssum += contrib
        rows.append({"Predictor": col, "Beta": round(float(beta[j]), 6),
                     "Mean_X": round(float(mu_x), 5),
                     "CI_Predictor": round(float(ci_x), 5),
                     "Elasticity": round(float(elast), 5),
                     "Contribution": round(float(contrib), 6),
                     "Pct_Contribution": round(float(contrib / total_ci * 100), 2)})
    rows.append({"Predictor": "Residual", "Contribution": round(float(total_ci - ssum), 6),
                 "Pct_Contribution": round(float((total_ci - ssum) / total_ci * 100), 2)})
    rows.append({"Predictor": "TOTAL_CI", "Contribution": round(float(total_ci), 6),
                 "Pct_Contribution": 100.0})
    pd.DataFrame(rows).to_csv(OUTDIR / "table_4_wagstaff_fulldata.csv", index=False)
    log(f"  total CI={total_ci:.5f}; decomposition saved")
    log("TABLE 4 done")


# --------------------------------------------------------------------------- #
def main():
    log("=" * 70)
    log("FULL-DATASET BATCH START")
    cols = ["country_code", "who_region", "survey_year", "died_under5",
            "child_male", "multiple_birth", "birth_order",
            "birth_interval_months", "maternal_age_at_birth", "edu_level",
            "wealth_quintile", "currently_married", "urban", "sample_weight",
            "anc_visits", "facility_delivery", "caesarean", "tetanus_protected",
            "is_recent_birth"]
    df = FD.load_full(cols, scope="primary")
    df = df.sort_values("country_code").reset_index(drop=True)
    n = len(df)
    log(f"Loaded full cohort: {n:,} rows, {df['country_code'].nunique()} countries")
    groups_sorted = pd.factorize(df["country_code"].values)[0].astype(np.int64)
    dummies = build_dummies(df)

    for name, fn in [("TABLE2", lambda: table2(df, groups_sorted, dummies)),
                     ("TABLE3", lambda: table3(df, dummies)),
                     ("TABLE4", lambda: table4(df, dummies))]:
        try:
            fn()
        except Exception:
            log(f"{name} CRASHED:\n{traceback.format_exc()}")
        gc.collect()

    log("FULL-DATASET BATCH COMPLETE")


if __name__ == "__main__":
    main()
