"""s05b_regional_aor.py — WHO-region-stratified adjusted odds ratios (v2).

Closes the second display-item gap left by the v2 rebuild: the legacy main table
carried pooled *and* region-stratified determinant columns, showing how each
association travels across settings, but the v2 port regenerated only the pooled
surface.

Refits the same primary surface (`exact_wave_fe_complete_wealth`, survey-weighted,
survey-wave fixed effects, explicit missing states) independently within each WHO
region, with the identical country-cluster-robust sandwich used for the pooled
table (`s05b_table2`), so pooled and regional columns are directly comparable.

Regions with too few country clusters for a credible cluster-robust variance are
still fitted, but flagged: with few clusters the sandwich is known to be
anti-conservative, so `n_clusters` is reported alongside every estimate.

Small-cluster correction (2026-08-03 external-review amendment)
---------------------------------------------------------------
Flagging "few clusters" in a footnote is not a correction, and three of the five
regions have only 7-8 country clusters. Each region is therefore reported with
three interval types:

  ci_lo/ci_hi            CR0 sandwich with normal critical values (as published)
  ci_lo_cr1/ci_hi_cr1    CR1 finite-sample scaling with t(G-1) critical values,
                         the standard small-cluster correction
  ci_lo_wild/ci_hi_wild  score wild cluster bootstrap (Rademacher), enumerated
                         exhaustively over all 2^G sign vectors when G is small

The CR1+t interval is the recommended one for regional claims; with G=7 the t
critical value alone is 2.447 against 1.96, so the published intervals understate
regional uncertainty by roughly a quarter before any variance correction.

Output: output/phase4/manuscript/table_2_regional_aor.csv
Run:    python -m src.inferential.s05b_regional_aor
"""
from __future__ import annotations

import itertools
import sys

import numpy as np
import pandas as pd
from scipy import sparse
from scipy import stats

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from config.paths import HARMONIZED_DIR, OUTPUT_DIR
from src.inferential.phase2_analysis import _fit_surface, _read_pilot
from src.inferential.s05b_table2 import _is_nuisance

OUT = OUTPUT_DIR / "phase4" / "manuscript"
REGION_ORDER = ["SSA", "SSEA", "LAC", "ECA", "MENA"]
MIN_CLUSTERS = 5   # below this a cluster-robust sandwich is not trustworthy


def _sandwich_aors(result) -> tuple[pd.DataFrame, int, int]:
    """Country-cluster-robust aORs for one fitted surface (same maths as s05b_table2)."""
    coef = np.r_[float(result.model.intercept_[0]), result.model.coef_[0]]
    names = ["(intercept)"] + [str(n) for n in result.feature_names]
    n = result.matrix.shape[0]
    Xa = sparse.hstack([sparse.csr_matrix(np.ones((n, 1))), result.matrix]).tocsr()
    w = result.weights.astype(float)
    y = result.outcome.astype(float)
    p = result.factual_predictions.astype(float)

    V = w * p * (1.0 - p)
    XVX = (Xa.T @ sparse.diags(V) @ Xa).toarray()
    try:
        bread = np.linalg.inv(XVX)
    except np.linalg.LinAlgError:
        bread = np.linalg.pinv(XVX)

    g = w * (y - p)
    country = result.frame["country_code"].astype(str).to_numpy()
    codes, inv = np.unique(country, return_inverse=True)
    S = np.zeros((len(codes), Xa.shape[1]))
    for ci in range(len(codes)):
        rows = np.flatnonzero(inv == ci)
        S[ci] = np.asarray(Xa[rows].T @ g[rows]).ravel()
    cov = bread @ (S.T @ S) @ bread
    se = np.sqrt(np.clip(np.diag(cov), 0, None))

    # --- small-cluster corrections -------------------------------------------
    G, p_dim = len(codes), Xa.shape[1]
    # CR1: the finite-sample scaling Stata applies by default to vce(cluster).
    cr1_scale = np.sqrt((G / max(G - 1, 1)) * ((n - 1) / max(n - p_dim, 1)))
    se_cr1 = se * cr1_scale
    t_crit = float(stats.t.ppf(0.975, df=max(G - 1, 1)))

    # Score wild cluster bootstrap: perturb each cluster's score contribution by a
    # Rademacher sign and read off the induced coefficient shift. With few clusters
    # the sign space is small enough to enumerate exhaustively, which removes
    # bootstrap simulation error entirely.
    signs = (np.array(list(itertools.product([-1.0, 1.0], repeat=G)))
             if G <= 16 else
             np.random.RandomState(20260505).choice([-1.0, 1.0], size=(9999, G)))
    deltas = signs @ S @ bread.T                    # (n_draws, p_dim)
    lo_d, hi_d = np.percentile(deltas, [2.5, 97.5], axis=0)

    rows = []
    for j, nm in enumerate(names):
        if _is_nuisance(nm):
            continue
        b = coef[j]
        rows.append({
            "feature": nm,
            "aOR": round(float(np.exp(b)), 3),
            "ci_lo": round(float(np.exp(b - 1.96 * se[j])), 3),
            "ci_hi": round(float(np.exp(b + 1.96 * se[j])), 3),
            "ci_lo_cr1": round(float(np.exp(b - t_crit * se_cr1[j])), 3),
            "ci_hi_cr1": round(float(np.exp(b + t_crit * se_cr1[j])), 3),
            # Basic bootstrap interval; the Rademacher distribution is symmetric, so
            # this coincides with the percentile interval.
            "ci_lo_wild": round(float(np.exp(b - hi_d[j])), 3),
            "ci_hi_wild": round(float(np.exp(b + hi_d[j])), 3),
        })
    meta = {
        "cr1_scale": round(float(cr1_scale), 4),
        "t_critical_value": round(t_crit, 4),
        "wild_draws": int(len(signs)),
        "wild_enumerated": bool(G <= 16),
    }
    return pd.DataFrame(rows), len(codes), n, meta


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = _read_pilot(HARMONIZED_DIR / "global_analysis_v2_candidate.parquet")
    regions = [r for r in REGION_ORDER if (frame["who_region"].astype(str) == r).any()]

    out = []
    for reg in regions:
        sub = frame.loc[frame["who_region"].astype(str) == reg].reset_index(drop=True)
        try:
            result = _fit_surface(
                surface="exact_wave_fe_complete_wealth", role=f"regional_{reg}",
                frame=sub, outcome_column="died_under5_complete", legacy_states=False,
                fixed_effects="survey", wealth_mode="complete", weighted=True)
        except Exception as exc:                      # keep the other regions
            print(f"[regional] {reg}: FIT FAILED ({type(exc).__name__}: {exc})")
            continue
        df, n_clusters, n_rows, meta = _sandwich_aors(result)
        df.insert(0, "region", reg)
        df["n_clusters"] = n_clusters
        df["n_births"] = n_rows
        df["clusters_below_min"] = n_clusters < MIN_CLUSTERS
        for k, v in meta.items():
            df[k] = v
        out.append(df)
        flag = "  [FEW CLUSTERS]" if n_clusters < MIN_CLUSTERS else ""
        print(f"[regional] {reg}: {n_rows:,} births, {n_clusters} country clusters"
              f" | CR1 scale {meta['cr1_scale']:.3f}, t crit {meta['t_critical_value']:.3f},"
              f" wild draws {meta['wild_draws']}"
              f"{' (enumerated)' if meta['wild_enumerated'] else ''}{flag}")

    if not out:
        raise SystemExit("no regional fit succeeded")
    res = pd.concat(out, ignore_index=True)
    res.to_csv(OUT / "table_2_regional_aor.csv", index=False)

    key = "edu_cat_Secondary"
    print("\nSecondary-education aOR by region (transportability check):")
    print(f"  {'':5} {'aOR':>6}  {'CR0 (published)':>22}  {'CR1 + t(G-1)':>22}"
          f"  {'wild cluster':>22}")
    for reg in regions:
        m = res[(res["region"] == reg) & (res["feature"] == key)]
        if len(m):
            r = m.iloc[0]
            print(f"  {reg:5} {r['aOR']:6.3f}  {r['ci_lo']:10.3f}-{r['ci_hi']:<11.3f}"
                  f"  {r['ci_lo_cr1']:10.3f}-{r['ci_hi_cr1']:<11.3f}"
                  f"  {r['ci_lo_wild']:10.3f}-{r['ci_hi_wild']:<11.3f}")
    print(f"\n[regional] wrote {OUT / 'table_2_regional_aor.csv'} ({len(res)} rows)")


if __name__ == "__main__":
    main()
