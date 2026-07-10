"""
run_fulldata_phase2.py — FULL-DATASET sensitivity analyses (phase 2)
====================================================================

Ports the secondary/sensitivity analyses to the full 10.7M-row cohort using the
validated `_fulldata.py` engine. Mirrors the originals exactly except: (a) full
data instead of the 5% systematic sample, and (b) the per-country / cluster
bootstrap CAPS in the originals (s05c MAX_PER_COUNTRY, s05k BOOT_CAP/CLUSTER_CAP)
are REMOVED — full cluster rows are used.

Covers:
  s05g  fully-exposed cohort (GEE M3 all vs fully-exposed; PAFs)        -> s22*
  s05k  parity-truncation (GEE M3 fine vs truncated parity, DHS-only)   -> parity*
  s05k  per-country wealth equity (concentration index + bootstrap)     -> country_equity
  s05m  PAF sensitivities: survey-weighted (R2.2) + DHS-only (R1.3)      -> paf_sens
  s05n  GEE aOR sensitivities: weighted (R2.2) + DHS-only (R1.3)         -> gee_sens
  s05j  MICS-vs-DHS comparability (weighted means + SMD)                 -> comparability

Run (background, AFTER the main batch finishes to avoid CPU contention):
  python -m src.inferential.run_fulldata_phase2
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
from config.paths import RNG_SEED, TABLES_DIR
from src.inferential import _fulldata as FD
from src.inferential.run_fulldata_all import (
    build_dummies, design_from, _or_rows, M3, PAF_FACTORS, _paf_one)

OUTDIR = TABLES_DIR.parent / "fulldata"
OUTDIR.mkdir(parents=True, exist_ok=True)
LOG = OUTDIR / "phase2.log"

N_BOOT_EQUITY = 300


def log(msg: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


# Truncated-parity dummies (MICS-style: 1st / 2nd-3rd / 4th+)
def trunc_parity(df):
    bo = df["birth_order"].fillna(1).values
    return {"bo_2_3": ((bo >= 2) & (bo <= 3)).astype(np.float64),
            "bo_4plus": (bo >= 4).astype(np.float64)}


M3_NOBO = [c for c in M3 if not c.startswith("bo_")]   # M3 without birth-order


def _gee_or_table(df, dummies, names, label, weights=None):
    """Fit exchangeable GEE on full data and return OR rows (data pre-sorted)."""
    n = len(df)
    y = df["died_under5"].astype(np.float64).values
    keep = [k for k in names if dummies[k].std() > 0]
    X = design_from(dummies, keep, n)
    g = pd.factorize(df["country_code"].values)[0].astype(np.int64)
    w = None if weights is None else df[weights].fillna(1.0).clip(lower=1e-9).values
    t0 = time.time()
    beta, se, alpha, nit, conv = FD.fit_exchangeable_gee(X, y, g, weights=w)
    log(f"    {label}: conv={conv} alpha={alpha:.4f} iters={nit} ({time.time()-t0:.0f}s)")
    return _or_rows(beta, se, keep, "GEE-Exch-FullData", n, extra={"fit": label})


# --------------------------------------------------------------------------- #
def s05g_fully_exposed(df, dummies):
    log("s05g: fully-exposed cohort sensitivity (full data)")
    yb = df["years_before_survey"].values
    fe_mask = yb >= 5
    log(f"  fully-exposed rows: {int(fe_mask.sum()):,} / {len(df):,}")

    # GEE Model 3: all vs fully-exposed
    rows = []
    rows += _gee_or_table(df, dummies, M3, "all_records")
    fe = df[fe_mask].sort_values("country_code").reset_index(drop=True)
    fe_dummies = build_dummies(fe)
    rows += _gee_or_table(fe, fe_dummies, M3, "fully_exposed")
    pd.DataFrame(rows).to_csv(OUTDIR / "s05g_fully_exposed_gee_fulldata.csv", index=False)

    # PAFs: all vs fully-exposed (point estimates)
    paf_rows = []
    for label, sub, sub_dum in [("all_records", df, dummies),
                                ("fully_exposed", fe, fe_dummies)]:
        n = len(sub)
        y = sub["died_under5"].astype(np.float64).values
        keep = [k for k in M3 if sub_dum[k].std() > 0]
        X = design_from(sub_dum, keep, n)
        colidx = {nm: i + 1 for i, nm in enumerate(keep)}
        beta = FD.logit_coefs(X, y)
        for fac, shift in PAF_FACTORS:
            sh = {k: v for k, v in shift.items() if k in colidx}
            paf_rows.append({"cohort": label, "risk_factor": fac,
                             "PAF": round(_paf_one(X, beta, colidx, sh) * 100, 2)})
    pd.DataFrame(paf_rows).to_csv(OUTDIR / "s05g_fully_exposed_pafs_fulldata.csv", index=False)
    log("  s05g done")


def s05k_parity(df, dummies):
    log("s05k: parity-truncation sensitivity, DHS-only (full data)")
    dhs = df[~df["_source_file"].astype(str).str.contains("MICS", case=False, na=False)]
    dhs = dhs.sort_values("country_code").reset_index(drop=True)
    log(f"  DHS-only rows: {len(dhs):,} ({dhs['country_code'].nunique()} countries)")
    dd = build_dummies(dhs)
    tp = trunc_parity(dhs)

    rows = []
    # Spec A: fine parity (native M3)
    rows += _gee_or_table(dhs, dd, M3, "A_fine_parity")
    # Spec B: truncated parity (replace bo_4_5+bo_6plus with bo_4plus)
    dd_trunc = dict(dd); dd_trunc["bo_4plus"] = tp["bo_4plus"]
    names_b = M3_NOBO + ["bo_2_3", "bo_4plus"]
    rows += _gee_or_table(dhs, dd_trunc, names_b, "B_truncated_parity")
    pd.DataFrame(rows).to_csv(OUTDIR / "s05k_parity_truncation_fulldata.csv", index=False)
    log("  s05k-parity done")


def _ci_from_arrays(h, q, w):
    # Grouped-tie fractional rank: every record in a wealth group shares the
    # group-midpoint rank R_g = (cumulative weight below g) + 0.5*(weight of g),
    # divided by total weight. This is the standard Wagstaff grouped
    # concentration index. Unlike a within-tie "spread" rank, it makes the
    # record bootstrap re-centre on the point estimate when wealth is a
    # 5-category ordinal with heavy ties (otherwise the bootstrap biases toward
    # zero and the point falls outside its own CI on full data).
    tw = w.sum()
    if tw == 0:
        return np.nan
    mu = np.average(h, weights=w)
    if mu == 0:
        return np.nan
    fr = np.empty_like(w, dtype=float)
    cum = 0.0
    for v in np.unique(q):            # ascending; ~5 wealth groups
        m = (q == v)
        gw = w[m].sum()
        fr[m] = (cum + 0.5 * gw) / tw
        cum += gw
    mr = np.average(fr, weights=w)
    return (2.0 / mu) * np.average((h - mu) * (fr - mr), weights=w)


def _wrate(h, w):
    return float(np.average(h, weights=w) * 1000.0) if w.sum() > 0 else np.nan


def s05k_equity(df):
    log("s05k: per-country wealth equity (full data, NO caps)")
    sub0 = df[df["wealth_quintile"].notna()]
    H = sub0["died_under5"].values.astype(float)
    Q = sub0["wealth_quintile"].values.astype(float)
    W = sub0["sample_weight"].fillna(1.0).values.astype(float)
    CC = sub0["country_code"].values
    rng = np.random.default_rng(RNG_SEED)

    try:
        inv = pd.read_csv(TABLES_DIR / "table_s1_survey_inventory.csv")
        names = inv.groupby("country_code")["country_name"].first()
    except Exception:
        names = pd.Series(dtype=object)

    uniq, groups = FD.cluster_row_index(CC)
    rows = []
    for cc in uniq:
        idx = groups[cc]
        h, q, w = H[idx], Q[idx], W[idx]
        nd = float(np.nansum(h * w))
        ci = _ci_from_arrays(h, q, w)       # full data, no cap
        m1, m5 = q == 1, q == 5
        u1 = _wrate(h[m1], w[m1]) if m1.any() else np.nan
        u5 = _wrate(h[m5], w[m5]) if m5.any() else np.nan
        ci_lo = ci_hi = np.nan
        if nd >= 30 and np.isfinite(ci):
            nb = len(h); boots = []
            for _ in range(N_BOOT_EQUITY):
                bi = rng.integers(0, nb, nb)
                cb = _ci_from_arrays(h[bi], q[bi], w[bi])
                if np.isfinite(cb):
                    boots.append(cb)
            if len(boots) >= 30:
                ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
        rows.append({"country_code": cc, "country_name": names.get(cc, cc),
                     "n_records": len(idx), "n_deaths_wtd": round(nd, 0),
                     "u5m_all": round(_wrate(h, w), 1),
                     "u5m_q1_poorest": round(u1, 1) if np.isfinite(u1) else np.nan,
                     "u5m_q5_richest": round(u5, 1) if np.isfinite(u5) else np.nan,
                     "conc_index": round(ci, 4) if np.isfinite(ci) else np.nan,
                     "ci_lo_95": round(float(ci_lo), 4) if np.isfinite(ci_lo) else np.nan,
                     "ci_hi_95": round(float(ci_hi), 4) if np.isfinite(ci_hi) else np.nan})
    # pooled with country-cluster bootstrap (full rows, no cap)
    pooled_ci = _ci_from_arrays(H, Q, W)
    boots = []
    for b, idx in FD.iter_cluster_bootstrap(uniq, groups, rng, N_BOOT_EQUITY):
        cb = _ci_from_arrays(H[idx], Q[idx], W[idx])
        if np.isfinite(cb):
            boots.append(cb)
    plo, phi = np.percentile(boots, [2.5, 97.5])
    rows.insert(0, {"country_code": "ALL", "country_name": "All LMICs",
                    "n_records": len(sub0), "conc_index": round(pooled_ci, 4),
                    "ci_lo_95": round(float(plo), 4), "ci_hi_95": round(float(phi), 4)})
    pd.DataFrame(rows).to_csv(OUTDIR / "s05k_country_equity_fulldata.csv", index=False)
    log(f"  pooled CI={pooled_ci:.4f} ({plo:.4f},{phi:.4f}); s05k-equity done")


def s05m_paf_sens(df, dummies):
    log("s05m: PAF sensitivities — weighted (R2.2) + DHS-only (R1.3), full data")
    is_mics = df["_source_file"].astype(str).str.contains("MICS", case=False, na=False).values
    n = len(df)
    y = df["died_under5"].astype(np.float64).values
    keep = [k for k in M3 if dummies[k].std() > 0]
    X = design_from(dummies, keep, n)
    colidx = {nm: i + 1 for i, nm in enumerate(keep)}
    wt = df["sample_weight"].fillna(1.0).clip(lower=1e-9).values

    def scenario_pafs(mask, weights):
        Xs = X[mask]; ys = y[mask]
        ws = None if weights is None else weights[mask]
        beta = FD.logit_coefs(Xs, ys, weights=ws)
        out = {}
        pf = expit(Xs @ beta)
        for fac, shift in PAF_FACTORS:
            sh = {k: v for k, v in shift.items() if k in colidx}
            saved = {}
            for cn, val in sh.items():
                ci = colidx[cn]; saved[ci] = Xs[:, ci].copy(); Xs[:, ci] = val
            pcf = expit(Xs @ beta)
            for ci, col in saved.items():
                Xs[:, ci] = col
            if ws is None:
                rf, rc = pf.mean(), pcf.mean()
            else:
                rf, rc = np.average(pf, weights=ws), np.average(pcf, weights=ws)
            out[fac] = round((rf - rc) / rf * 100, 2) if rf > 0 else 0.0
        return out

    allm = np.ones(n, dtype=bool)
    canon = scenario_pafs(allm, None)
    weighted = scenario_pafs(allm, wt)
    dhs = scenario_pafs(~is_mics, None)
    rows = [{"risk_factor": fac, "PAF_canonical": canon[fac],
             "PAF_weighted": weighted[fac], "PAF_dhs_only": dhs[fac]}
            for fac, _ in PAF_FACTORS]
    pd.DataFrame(rows).to_csv(OUTDIR / "s05m_paf_sensitivity_fulldata.csv", index=False)
    log("  s05m done")


HEADLINE = ["edu_secondary", "edu_higher", "bi_short", "bi_long", "is_multiple",
            "age_under20", "bo_6plus", "is_urban", "wq_5"]


def s05n_gee_sens(df, dummies):
    log("s05n: GEE aOR sensitivities — weighted (R2.2) + DHS-only (R1.3), full data")
    is_mics = df["_source_file"].astype(str).str.contains("MICS", case=False, na=False).values
    scen = {}
    # canonical (unwt, all)
    scen["canonical"] = {r["Variable"]: r["OR"]
                         for r in _gee_or_table(df, dummies, M3, "canonical")}
    # weighted (R2.2)
    scen["weighted"] = {r["Variable"]: r["OR"]
                        for r in _gee_or_table(df, dummies, M3, "weighted",
                                               weights="sample_weight")}
    # DHS-only (R1.3)
    dhs = df[~is_mics].sort_values("country_code").reset_index(drop=True)
    dhs_dum = build_dummies(dhs)
    scen["dhs_only"] = {r["Variable"]: r["OR"]
                        for r in _gee_or_table(dhs, dhs_dum, M3, "dhs_only")}
    rows = [{"determinant": h, "aOR_canonical": scen["canonical"].get(h),
             "aOR_weighted": scen["weighted"].get(h),
             "aOR_dhs_only": scen["dhs_only"].get(h)} for h in HEADLINE]
    pd.DataFrame(rows).to_csv(OUTDIR / "s05n_gee_sensitivity_fulldata.csv", index=False)
    log("  s05n done")


def _wmean(x, w):
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    return np.average(x[m], weights=w[m]) if m.sum() else np.nan


def _wvar(x, w):
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if m.sum() < 2:
        return np.nan
    mu = np.average(x[m], weights=w[m])
    return float(np.average((x[m] - mu) ** 2, weights=w[m]))


def s05j_comparability(df):
    log("s05j: MICS-vs-DHS comparability (full data)")
    is_mics = df["_source_file"].astype(str).str.contains("MICS", case=False, na=False).values
    edu = df["edu_level"].values
    wq = df["wealth_quintile"].values
    cols = {
        "Recorded U5M (per 1,000)": (df["died_under5"].values.astype(float), "binary", 1000),
        "Male child (%)": (df["child_male"].values.astype(float), "binary", 100),
        "Multiple birth (%)": (df["multiple_birth"].values.astype(float), "binary", 100),
        "Birth order (mean)": (df["birth_order"].values.astype(float), "cont", 1),
        "Preceding interval, months (mean)": (df["birth_interval_months"].values.astype(float), "cont", 1),
        "Maternal age at birth (mean)": (df["maternal_age_at_birth"].values.astype(float), "cont", 1),
        "No maternal education (%)": (np.where(np.isnan(edu), np.nan, (edu == 0).astype(float)), "binary", 100),
        "Secondary+ education (%)": (np.where(np.isnan(edu), np.nan, (edu >= 2).astype(float)), "binary", 100),
        "Currently married (%)": (df["currently_married"].values.astype(float), "binary", 100),
        "Poorest quintile (%)": (np.where(np.isnan(wq), np.nan, (wq == 1).astype(float)), "binary", 100),
        "Urban residence (%)": (df["urban"].values.astype(float), "binary", 100),
    }
    w = df["sample_weight"].values.astype(float)
    rows = []
    for label, (x, kind, scale) in cols.items():
        m1, m2 = _wmean(x[~is_mics], w[~is_mics]), _wmean(x[is_mics], w[is_mics])
        if kind == "binary":
            denom = np.sqrt((m1 * (1 - m1) + m2 * (1 - m2)) / 2.0)
        else:
            denom = np.sqrt((_wvar(x[~is_mics], w[~is_mics]) + _wvar(x[is_mics], w[is_mics])) / 2.0)
        smd = (m1 - m2) / denom if (np.isfinite(denom) and denom > 0) else np.nan
        rows.append({"variable": label,
                     "dhs_value": round(m1 * scale, 2) if np.isfinite(m1) else np.nan,
                     "mics_value": round(m2 * scale, 2) if np.isfinite(m2) else np.nan,
                     "smd_dhs_minus_mics": round(smd, 3) if np.isfinite(smd) else np.nan})
    pd.DataFrame(rows).to_csv(OUTDIR / "s05j_mics_dhs_comparability_fulldata.csv", index=False)
    log("  s05j done")


def main():
    log("=" * 70)
    log("PHASE-2 SENSITIVITIES BATCH START (full data)")
    cols = ["country_code", "who_region", "survey_year", "died_under5",
            "child_male", "multiple_birth", "birth_order",
            "birth_interval_months", "maternal_age_at_birth", "edu_level",
            "wealth_quintile", "currently_married", "urban", "sample_weight",
            "_source_file", "years_before_survey"]
    df = FD.load_full(cols, scope="primary")
    df = df.sort_values("country_code").reset_index(drop=True)
    log(f"Loaded full cohort: {len(df):,} rows, {df['country_code'].nunique()} countries")
    dummies = build_dummies(df)

    for name, fn in [
        ("s05j", lambda: s05j_comparability(df)),
        ("s05k_equity", lambda: s05k_equity(df)),
        ("s05g", lambda: s05g_fully_exposed(df, dummies)),
        ("s05k_parity", lambda: s05k_parity(df, dummies)),
        ("s05m", lambda: s05m_paf_sens(df, dummies)),
        ("s05n", lambda: s05n_gee_sens(df, dummies)),
    ]:
        try:
            fn()
        except Exception:
            log(f"{name} CRASHED:\n{traceback.format_exc()}")
        gc.collect()
    log("PHASE-2 SENSITIVITIES BATCH COMPLETE")


if __name__ == "__main__":
    main()
