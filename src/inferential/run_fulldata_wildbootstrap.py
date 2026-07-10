"""run_fulldata_wildbootstrap.py  (P1-8, 2026-06-23)
================================================================
Wild-cluster bootstrap CIs for the Model-3 GEE adjusted odds ratios, to
accompany the asymptotic country-cluster sandwich CIs. With only G=79 clusters
the sandwich SE can be anticonservative; the score-based wild-cluster bootstrap
(Cameron, Gelbach & Miller 2008; Kline & Santos score bootstrap) gives a
small-sample-robust interval that does not rely on the normal approximation.

Implementation (no refits): the fitted exchangeable GEE exposes, at beta-hat,
the bread H (p x p) and the per-cluster score contributions U (G x p), with
sandwich cov = H^-1 (U'U) H^-1. The wild-cluster bootstrap draws Rademacher
signs s_g in {-1,+1} per cluster and forms the coefficient deviation
  delta*_b = H^-1 (sum_g s_g U_g),
so beta*_b = beta-hat + delta*_b. Percentiles of exp(beta*_b) give the WCB
odds-ratio CIs. Also reports a small-sample t(G-1) df correction on the
sandwich SE for reference.

Run SOLO (full-data GEE fit). Output: output/fulldata/table_m3_wildbootstrap_fulldata.csv
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.paths import RNG_SEED
import src.inferential._fulldata as FD
from src.inferential.run_fulldata_all import build_dummies, design_from, M3, OUTDIR, log

B_WCB = 2000

# Headline determinants the reviewers care about (label -> dummy name)
HEADLINE = {
    "edu_secondary": "Secondary education", "edu_higher": "Higher education",
    "bi_short": "Short birth interval (<24mo)", "is_multiple": "Multiple birth",
    "bo_6plus": "Birth order 6+", "age_under20": "Maternal age <20",
}


def main():
    log("P1-8 wild-cluster bootstrap for Model-3 GEE (G=79 country clusters)")
    cols = ["country_code", "who_region", "survey_year", "died_under5",
            "child_male", "multiple_birth", "birth_order",
            "birth_interval_months", "maternal_age_at_birth", "edu_level",
            "wealth_quintile", "currently_married", "urban", "sample_weight"]
    df = FD.load_full(cols, scope="primary")
    df = df.sort_values("country_code").reset_index(drop=True)
    n = len(df)
    G = df["country_code"].nunique()
    log(f"Loaded {n:,} rows, {G} country clusters")
    groups = pd.factorize(df["country_code"].values)[0].astype(np.int64)
    dummies = build_dummies(df)
    y = df["died_under5"].astype(np.float64).values

    keep = [k for k in M3 if dummies[k].std() > 0]
    X = design_from(dummies, keep, n)
    names = ["(intercept)"] + keep

    t0 = time.time()
    beta, se, alpha, nit, conv = FD.fit_exchangeable_gee(X, y, groups, verbose=False)
    log(f"  GEE fit: conv={conv} alpha={alpha:.4f} iters={nit} ({time.time()-t0:.0f}s)")

    # Recompute H, U at beta-hat (same path the engine uses internally)
    w = np.ones(n)
    counts = np.bincount(groups).astype(np.intp)
    starts = FD._seg_starts(counts)
    ni_eff = np.add.reduceat(w, starts)
    _, H, U, _ = FD._score_H_U(beta, alpha, X, y, w, starts, ni_eff)
    Hinv = np.linalg.inv(H)
    assert U.shape[0] == G, f"U has {U.shape[0]} rows, expected {G} clusters"

    # Wild-cluster bootstrap (Rademacher signs per cluster; no refit)
    rng = np.random.default_rng(RNG_SEED)
    p = len(beta)
    boot = np.empty((B_WCB, p), dtype=np.float64)
    for b in range(B_WCB):
        signs = rng.integers(0, 2, size=G).astype(np.float64) * 2.0 - 1.0  # {-1,+1}
        boot[b] = beta + Hinv @ (signs @ U)
    or_lo = np.exp(np.percentile(boot, 2.5, axis=0))
    or_hi = np.exp(np.percentile(boot, 97.5, axis=0))

    # df-corrected sandwich (t(G-1), CR1 scaling G/(G-1)) for reference
    tcrit = stats.t.ppf(0.975, G - 1)
    cr1 = np.sqrt(G / (G - 1.0))
    or_pt = np.exp(beta)
    sand_lo = np.exp(beta - 1.96 * se);  sand_hi = np.exp(beta + 1.96 * se)
    df_lo = np.exp(beta - tcrit * cr1 * se); df_hi = np.exp(beta + tcrit * cr1 * se)

    rows = []
    for j, nm in enumerate(names):
        rows.append({
            "term": nm, "headline": HEADLINE.get(nm, ""),
            "OR": round(float(or_pt[j]), 4),
            "sandwich_lo": round(float(sand_lo[j]), 4), "sandwich_hi": round(float(sand_hi[j]), 4),
            "wildcluster_lo": round(float(or_lo[j]), 4), "wildcluster_hi": round(float(or_hi[j]), 4),
            "dfcorr_lo": round(float(df_lo[j]), 4), "dfcorr_hi": round(float(df_hi[j]), 4),
        })
    out = pd.DataFrame(rows)
    op = OUTDIR / "table_m3_wildbootstrap_fulldata.csv"
    out.to_csv(op, index=False)
    log(f"  Wrote {op} (B={B_WCB} wild-cluster reps, G={G})")
    print(out[out["headline"] != ""].to_string(index=False))


if __name__ == "__main__":
    main()
