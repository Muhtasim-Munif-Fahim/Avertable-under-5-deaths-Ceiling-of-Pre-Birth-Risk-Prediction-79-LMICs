"""
_fulldata.py — shared full-dataset estimation engine (float64, memory/overflow-safe)
====================================================================================

Primitives for running the inferential analyses on ALL ~10.7M in-scope births
without subsampling. Used by the s05*_fulldata scripts and the orchestrator.

Why this exists: statsmodels' GEE/GLM cannot fit at full N on a 32 GB machine —
patsy duplicates the design matrix, the SVD starting-value step OOMs, and the
exchangeable correlation update overflows 32-bit pair counts (~9.1e9 for 135K-row
country clusters) returning NaN. These routines avoid all three.

The GEE estimator here was validated to 4 decimals against statsmodels' GEE on
the 5% sample (ORs and SEs); the weighted variant matches to within max |OR diff|
0.034. Validation results recorded in docs/FULLDATA_MIGRATION_2026-06-15.md.
"""
from __future__ import annotations
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

# --------------------------------------------------------------------------- #
# Data loading                                                                #
# --------------------------------------------------------------------------- #

def load_full(columns, scope: str = "primary", drop_outcome_na: bool = True):
    """Full primary-scope load with numeric downcast (no subsampling)."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.utils.io import load_primary
    df = load_primary(columns=list(columns), scope=scope)
    if drop_outcome_na and "died_under5" in df.columns:
        df = df[df["died_under5"].notna()].copy()
    # downcast floats/ints to shrink footprint (design matrices are rebuilt as
    # float64 where the math needs it).
    for c in df.columns:
        if df[c].dtype == "float64":
            df[c] = pd.to_numeric(df[c], downcast="float")
        elif df[c].dtype == "int64":
            df[c] = pd.to_numeric(df[c], downcast="integer")
    return df


# --------------------------------------------------------------------------- #
# Exchangeable GEE (validated)                                                #
# --------------------------------------------------------------------------- #

def _seg_starts(counts: np.ndarray) -> np.ndarray:
    return np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.intp)


def _estimate_alpha(e, w, starts, neff_pairs, n, p):
    # weighted Pearson moment estimator; w are observation (frequency) weights
    we = w * e
    sume = np.add.reduceat(we, starts)
    sumsq = np.add.reduceat(w * e * e, starts)
    cross = 0.5 * (sume * sume - sumsq)
    scale = float(np.sum(w * e * e) / (np.sum(w) - p))
    alpha = float(np.sum(cross) / scale / neff_pairs)
    return min(max(alpha, 0.0), 0.95)


def _score_H_U(beta, alpha, X, y, w, starts, ni_eff):
    """Weighted exchangeable-GEE estimating function (w=1 -> unweighted).

    Observation weights w enter every per-row sum (frequency-weight form): with
    alpha=0 this reduces exactly to weighted logistic regression. ni_eff is the
    weighted cluster size (sum of w within cluster).
    """
    eta = np.clip(X @ beta, -30.0, 30.0)
    mu = expit(eta)
    var = np.clip(mu * (1.0 - mu), 1e-10, None)
    s = np.sqrt(var)
    resid = y - mu
    e = resid / s
    wsume = np.add.reduceat(w * e, starts)
    c_i = alpha / (1.0 + (ni_eff - 1.0) * alpha)
    inv1ma = 1.0 / (1.0 - alpha)
    wsX = (w * s)[:, None] * X
    V = np.add.reduceat(wsX, starts, axis=0)
    Xtq = np.add.reduceat((w * resid)[:, None] * X, starts, axis=0)
    XtVX = (X * (w * var)[:, None]).T @ X
    H = inv1ma * (XtVX - (V * c_i[:, None]).T @ V)
    U = inv1ma * (Xtq - (c_i * wsume)[:, None] * V)
    return U.sum(0), H, U, e


def fit_exchangeable_gee(X, y, groups, weights=None, max_iter=100, tol=1e-7,
                         verbose=False, start_beta=None):
    """Exchangeable-working-correlation GEE (binomial/logit), float64.

    `groups` must be CONTIGUOUS & sorted (sort the data by cluster first).
    `weights` (optional) are observation/survey weights (frequency-weight form).
    Returns (beta, se, alpha, n_iter, converged).
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, p = X.shape
    if weights is None:
        w = np.ones(n, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
    counts = np.bincount(groups).astype(np.intp)
    starts = _seg_starts(counts)
    # weighted cluster sizes and effective pair count (float64 -> no overflow)
    ni_eff = np.add.reduceat(w, starts)
    neff_pairs = float(np.sum(ni_eff * (ni_eff - 1.0) / 2.0)) - p

    if start_beta is not None:
        beta = np.asarray(start_beta, dtype=np.float64).copy()
    else:
        lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=500,
                                tol=1e-9, fit_intercept=False)
        lr.fit(X, y, sample_weight=w)
        beta = lr.coef_.ravel().astype(np.float64)

    converged = False
    H = U = None
    alpha = 0.0
    for it in range(max_iter):
        eta = np.clip(X @ beta, -30.0, 30.0)
        mu = expit(eta)
        e0 = (y - mu) / np.sqrt(np.clip(mu * (1.0 - mu), 1e-10, None))
        alpha = _estimate_alpha(e0, w, starts, neff_pairs, n, p)

        score, H, U, _ = _score_H_U(beta, alpha, X, y, w, starts, ni_eff)
        score_norm = np.linalg.norm(score)
        step = np.linalg.solve(H, score)

        t = 1.0
        accepted = False
        for _ in range(40):
            cand = beta + t * step
            if np.all(np.isfinite(cand)):
                s_new, _, _, _ = _score_H_U(cand, alpha, X, y, w, starts, ni_eff)
                if np.linalg.norm(s_new) < score_norm:
                    accepted = True
                    break
            t *= 0.5
        if not accepted:
            converged = True
            break
        delta = np.max(np.abs(t * step))
        beta = beta + t * step
        if verbose:
            print(f"    iter {it:2d}: alpha={alpha:.5f} t={t:.4f} "
                  f"max|step|={delta:.3e} ||score||={score_norm:.3e}")
        if delta < tol:
            converged = True
            break

    _, H, U, _ = _score_H_U(beta, alpha, X, y, w, starts, ni_eff)
    Hinv = np.linalg.inv(H)
    cov = Hinv @ (U.T @ U) @ Hinv
    se = np.sqrt(np.diag(cov))
    return beta, se, alpha, it + 1, converged


# --------------------------------------------------------------------------- #
# Fast full-data logistic + cluster-robust sandwich (independence working corr)#
# --------------------------------------------------------------------------- #

def fast_logit_cluster_robust(X, y, groups):
    """Logistic MLE (sklearn lbfgs) + country cluster-robust sandwich SE.

    X includes an intercept column. Returns (beta, se). Mirrors s05L.
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000,
                            tol=1e-9, fit_intercept=False)
    lr.fit(X, y)
    beta = lr.coef_.ravel()
    p = expit(X @ beta)
    w = p * (1.0 - p)
    bread = np.linalg.inv((X * w[:, None]).T @ X)
    score = X * (y - p)[:, None]
    uniq = np.unique(groups)
    Sg = np.empty((len(uniq), X.shape[1]))
    for i, g in enumerate(uniq):
        Sg[i] = score[groups == g].sum(0)
    cov = bread @ (Sg.T @ Sg) @ bread
    return beta, np.sqrt(np.diag(cov))


def irls_logit(X, y, start=None, weights=None, max_iter=30, tol=1e-9):
    """Newton/IRLS logistic, optionally warm-started from `start`.

    Warm-starting from the full-data point estimate converges in ~3 iters,
    making bootstrap refits several times faster than cold sklearn lbfgs while
    giving the identical MLE.
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, p = X.shape
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64)
    beta = np.zeros(p) if start is None else np.asarray(start, dtype=np.float64).copy()
    for _ in range(max_iter):
        eta = np.clip(X @ beta, -30.0, 30.0)
        mu = expit(eta)
        wv = np.clip(mu * (1.0 - mu), 1e-10, None) * w
        grad = X.T @ (w * (y - mu))
        H = (X * wv[:, None]).T @ X
        step = np.linalg.solve(H, grad)
        beta += step
        if np.max(np.abs(step)) < tol:
            break
    return beta


def logit_coefs(X, y, weights=None):
    """Logistic coefficients (no SE) — for fast bootstrap / PAF surfaces.

    `weights` (optional) are observation/survey weights (sklearn sample_weight).
    """
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000,
                            tol=1e-9, fit_intercept=False)
    lr.fit(np.ascontiguousarray(X, dtype=np.float64), np.asarray(y, float),
           sample_weight=None if weights is None else np.asarray(weights, float))
    return lr.coef_.ravel()


# --------------------------------------------------------------------------- #
# Country cluster bootstrap (full rows, no cap)                               #
# --------------------------------------------------------------------------- #

def cluster_row_index(country_codes: np.ndarray):
    """Map each unique country -> array of its row positions (for resampling)."""
    order = np.argsort(country_codes, kind="stable")
    sorted_c = country_codes[order]
    uniq, starts = np.unique(sorted_c, return_index=True)
    groups = {}
    for i, c in enumerate(uniq):
        lo = starts[i]
        hi = starts[i + 1] if i + 1 < len(starts) else len(sorted_c)
        groups[c] = order[lo:hi]
    return uniq, groups


def iter_cluster_bootstrap(uniq, groups, rng, n_boot):
    """Yield (rep_index, row_positions) for each cluster-bootstrap replicate.

    Resamples len(uniq) countries with replacement and concatenates ALL rows of
    the chosen countries (no per-country cap). Duplicated countries appear
    multiple times, as the cluster bootstrap requires.
    """
    k = len(uniq)
    for b in range(n_boot):
        pick = rng.integers(0, k, size=k)
        idx = np.concatenate([groups[uniq[j]] for j in pick])
        yield b, idx
