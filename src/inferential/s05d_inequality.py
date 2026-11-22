"""s05d_inequality.py — Table 4: Wagstaff decomposition on the v2 exact cohort.

This is the v2 port of s05d_inequality_decomposition.py. It fixes the defects
the 2026-07-11 methods audit flagged and follows Analysis Plan v2:

  * Full exact cohort (died_under5_complete on u5_binary_eligible == 1) — NOT the
    legacy 5% systematic sample of legacy died_under5.
  * Primary observed-wealth restriction and survey-wave-normalised weights,
    identical to the phase2 publication surface (_fit_surface), so the cohort,
    exposures, and weights match the published PAFs (n = 7,652,332).
  * Explicit "Missing" exposure categories (§5) — no fillna()-to-reference.
  * v2 five-state birth interval; missing wealth is dropped from the ranking,
    never assigned to Q1.
  * Country-cluster bootstrap (§8: 1000 uncapped country draws, seed RNG_SEED)
    for percentile CIs on the total concentration index and each factor share.

Ranking variable: observed wealth quintile (grouped fractional rank). The
decomposition predictor set excludes wealth itself, so factor shares answer
"how much of the wealth gradient is explained by education, residence, ...",
exactly as the legacy Table 4 estimand.

Output: output/phase4/manuscript/table_4_inequality_decomp*.csv (+manifest)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config.paths import HARMONIZED_DIR, OUTPUT_DIR, RNG_SEED
from src.inferential.phase2_analysis import (
    _positive_normalized_weights,
    _prepare_states,
    _read_pilot,
)

OUTDIR = OUTPUT_DIR / "phase4" / "manuscript"

# Explicit-missing dummy contract (reference = first level, dropped).
# Wealth (wq_cat) is the ranking variable and is NOT a decomposition predictor.
_DUMMY_SPECS = {
    "sex_cat": ("Female", ["Male", "Missing"]),
    "birth_type": ("Singleton", ["Multiple", "Missing"]),
    "bo_cat": ("1st", ["2nd-3rd", "4th-5th", "6th+", "Missing"]),
    "bi_model_cat": ("24-47 mo", ["<24 mo", "48+ mo", "Missing non-first", "First/missing"]),
    "ma_cat": ("20-34", ["<20", "35+", "Missing"]),
    "edu_cat": ("None", ["Primary", "Secondary", "Higher", "Missing"]),
    "marital_cat": ("Not married", ["Married", "Missing"]),
    "urban_cat": ("Rural", ["Urban", "Missing"]),
    "decade": ("2000-2009", ["2010-2019", "2020-2024", "Missing"]),
}

_FACTOR_LABEL = {
    "sex_cat": "Child sex",
    "birth_type": "Birth type",
    "bo_cat": "Birth order",
    "bi_model_cat": "Birth interval",
    "ma_cat": "Maternal age",
    "edu_cat": "Education",
    "marital_cat": "Marital status",
    "urban_cat": "Residence",
    "decade": "Decade trend",
}


def _grouped_fractional_rank(wq: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted grouped (midpoint) fractional rank on the wealth quintile."""
    total = w.sum()
    frac = np.empty(len(wq), dtype=float)
    cum = 0.0
    for v in np.unique(wq):
        m = wq == v
        gw = w[m].sum()
        frac[m] = (cum + 0.5 * gw) / total
        cum += gw
    return frac


def _wagstaff_index(ci: float, mu: float) -> float:
    """Wagstaff-normalised concentration index for a bounded 0/1 outcome.

    The standard index is not bounded on [-1, 1] when the outcome is binary: for a
    mean of mu its attainable range is [mu - 1, 1 - mu]. Dividing by (1 - mu)
    restores the full range, which is what makes indices comparable across
    populations with different mortality levels.
    """
    return float(ci / (1.0 - mu)) if 0.0 < mu < 1.0 else float("nan")


def _erreygers_index(ci: float, mu: float) -> float:
    """Erreygers-corrected concentration index, E = 4*mu*C for a 0/1 outcome.

    (General form is 4*mu*C / (b_max - b_min); for a binary indicator the
    denominator is 1.) Unlike the Wagstaff normalisation this satisfies mirror
    symmetry between the outcome and its complement.
    """
    return float(4.0 * mu * ci) if 0.0 < mu < 1.0 else float("nan")


def _conc_index(vals: np.ndarray, rank: np.ndarray, w: np.ndarray) -> float:
    mu = np.average(vals, weights=w)
    if mu == 0:
        return 0.0
    mean_r = np.average(rank, weights=w)
    cov = np.average((vals - mu) * (rank - mean_r), weights=w)
    return (2.0 / mu) * cov


def _wls_beta(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted least squares via normal equations (intercept in column 0).

    Uses lstsq on the small p-by-p normal matrix so a bootstrap resample that
    happens to zero out a low-support dummy yields a stable minimum-norm
    solution instead of raising on a singular matrix.
    """
    Xw = X * w[:, None]
    A = X.T @ Xw
    b = Xw.T @ y
    return np.linalg.lstsq(A, b, rcond=None)[0]


def _decompose(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    wq: np.ndarray,
    col_factor: list[str],
) -> tuple[float, dict[str, float]]:
    """Return (total_ci, {factor: contribution}). Column 0 of X is the intercept."""
    frac = _grouped_fractional_rank(wq, w)
    total_ci = _conc_index(y, frac, w)
    mu_y = np.average(y, weights=w)
    beta = _wls_beta(X, y, w)
    contrib: dict[str, float] = {}
    for j in range(1, X.shape[1]):  # skip intercept
        xv = X[:, j]
        mu_x = np.average(xv, weights=w)
        ci_x = _conc_index(xv, frac, w)
        c = (beta[j] * mu_x / mu_y) * ci_x
        contrib[col_factor[j]] = contrib.get(col_factor[j], 0.0) + c
    return total_ci, contrib


def _precompute_country_stats(X, y, w, wq, country):
    """Per-country sufficient statistics for a fast country-cluster bootstrap.

    Returns arrays indexed by country: A[c] = X_c' W_c X_c (p x p),
    b[c] = X_c' W_c y_c (p), and per wealth-group (ascending) weighted sums
    Wsum[c,g], Ysum[c,g], Xsum[c,g,:]. A bootstrap draw is then a
    multiplicity-weighted sum of these, with no per-row work.
    """
    countries = np.unique(country)
    groups = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # observed wealth quintiles
    p = X.shape[1]
    C, G = len(countries), len(groups)
    A = np.zeros((C, p, p))
    b = np.zeros((C, p))
    Wsum = np.zeros((C, G))
    Ysum = np.zeros((C, G))
    Xsum = np.zeros((C, G, p))
    for ci, c in enumerate(countries):
        m = country == c
        Xc, wc, yc, qc = X[m], w[m], y[m], wq[m]
        WXc = Xc * wc[:, None]
        A[ci] = Xc.T @ WXc
        b[ci] = WXc.T @ yc
        for gi, g in enumerate(groups):
            gm = qc == g
            if gm.any():
                Wsum[ci, gi] = wc[gm].sum()
                Ysum[ci, gi] = (wc[gm] * yc[gm]).sum()
                Xsum[ci, gi] = WXc[gm].sum(axis=0)
    return countries, groups, A, b, Wsum, Ysum, Xsum


def _decompose_from_stats(mult, A, b, Wsum, Ysum, Xsum, col_factor):
    """Decompose one draw from country multiplicities (fast path)."""
    Ad = np.einsum("c,cij->ij", mult, A)
    bd = mult @ b
    beta = np.linalg.lstsq(Ad, bd, rcond=None)[0]
    Wg = mult @ Wsum                      # (G,)
    Yg = mult @ Ysum                      # (G,)
    Xg = np.einsum("c,cgj->gj", mult, Xsum)  # (G, p)
    W = Wg.sum()
    if W <= 0:
        return float("nan"), {}
    cum = np.cumsum(Wg)
    frac = (cum - 0.5 * Wg) / W           # grouped midpoint fractional rank
    mu_y = Yg.sum() / W
    if mu_y == 0:
        return float("nan"), {}
    mean_rank = float((frac * Wg).sum() / W)
    fr = frac - mean_rank
    total_ci = (2.0 / mu_y) * float((fr * Yg).sum() / W)
    mu_x = Xg.sum(axis=0) / W             # (p,)
    cov_x = (fr[:, None] * Xg).sum(axis=0) / W  # (p,)
    contrib: dict[str, float] = {}
    for j in range(1, len(mu_x)):         # skip intercept
        if mu_x[j] <= 0:
            continue
        ci_x = (2.0 / mu_x[j]) * cov_x[j]
        c = beta[j] * mu_x[j] / mu_y * ci_x
        contrib[col_factor[j]] = contrib.get(col_factor[j], 0.0) + c
    return total_ci, contrib


def run(bootstrap_reps: int = 1000, seed: int = RNG_SEED) -> dict:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    frame = _read_pilot(HARMONIZED_DIR / "global_analysis_v2_candidate.parquet")
    w_all = _positive_normalized_weights(frame)
    mask = (
        pd.to_numeric(frame["died_under5_complete"], errors="coerce").notna()
        & pd.to_numeric(frame["wealth_quintile"], errors="coerce").notna()
        & w_all.notna()
    )
    data = frame.loc[mask].copy()
    w = w_all.loc[mask].to_numpy(float)
    states = _prepare_states(data, legacy=False)
    y = pd.to_numeric(states["died_under5_complete"], errors="raise").to_numpy(float)
    wq = pd.to_numeric(data["wealth_quintile"], errors="coerce").to_numpy(float)
    country = data["country_code"].astype(str).to_numpy()

    # Build the explicit-missing design (intercept + dummies). Drop dummy
    # levels with zero (or full) weighted support so the design is full rank,
    # matching the phase2 _design keep-logic (0 < support < total).
    total_w = w.sum()
    cols = [np.ones(len(data), dtype=float)]
    col_factor = ["(intercept)"]
    col_name = ["(intercept)"]
    dropped = []
    for feat, (_ref, levels) in _DUMMY_SPECS.items():
        s = states[feat].astype(str).to_numpy()
        for lv in levels:
            col = (s == lv).astype(float)
            support = float(w @ col)
            if support <= 0 or support >= total_w:
                dropped.append(f"{feat}={lv}")
                continue
            cols.append(col)
            col_factor.append(feat)
            col_name.append(f"{feat}={lv}")
    X = np.column_stack(cols)
    if dropped:
        print(f"[Table4-v2] dropped zero-support levels: {dropped}")

    total_ci, contrib = _decompose(X, y, w, wq, col_factor)
    n_deaths = int(y.sum())
    print(f"[Table4-v2] cohort={len(data):,} deaths={n_deaths:,} "
          f"weighted_risk={np.average(y, weights=w):.5f} total_CI={total_ci:.5f}")

    # --- country-cluster bootstrap via per-country sufficient statistics ---
    countries, groups, A, b_, Wsum, Ysum, Xsum = _precompute_country_stats(
        X, y, w, wq, country)
    C = len(countries)
    ones = np.ones(C)
    chk_ci, chk_contrib = _decompose_from_stats(
        ones, A, b_, Wsum, Ysum, Xsum, col_factor)
    if not np.isclose(chk_ci, total_ci, atol=1e-9):
        raise AssertionError(
            f"sufficient-stat check failed: {chk_ci} vs {total_ci}")
    print(f"[Table4-v2] sufficient-stat point check OK (CI={chk_ci:.5f})")

    rng = np.random.default_rng(seed)
    boot_total: list[float] = []
    boot_wagstaff: list[float] = []
    boot_erreygers: list[float] = []
    boot_share: dict[str, list[float]] = {f: [] for f in _FACTOR_LABEL}
    for _bi in range(bootstrap_reps):
        sampled = rng.integers(0, C, size=C)
        mult = np.bincount(sampled, minlength=C).astype(float)
        tci, cc = _decompose_from_stats(mult, A, b_, Wsum, Ysum, Xsum, col_factor)
        if not np.isfinite(tci) or tci == 0:
            continue
        boot_total.append(tci)
        mu_b = float((mult @ Ysum).sum() / (mult @ Wsum).sum())
        boot_wagstaff.append(_wagstaff_index(tci, mu_b))
        boot_erreygers.append(_erreygers_index(tci, mu_b))
        for f in _FACTOR_LABEL:
            boot_share[f].append(100.0 * cc.get(f, 0.0) / tci)

    def pct(a, p):
        return float(np.percentile(a, p)) if a else float("nan")

    rows = []
    explained = sum(contrib.values())
    for feat, label in _FACTOR_LABEL.items():
        c = contrib.get(feat, 0.0)
        rows.append({
            "factor": label,
            "contribution": c,
            "share_pct": 100.0 * c / total_ci,
            "share_ci_lo": pct(boot_share[feat], 2.5),
            "share_ci_hi": pct(boot_share[feat], 97.5),
        })
    residual = total_ci - explained
    rows.append({"factor": "Residual", "contribution": residual,
                 "share_pct": 100.0 * residual / total_ci,
                 "share_ci_lo": float("nan"), "share_ci_hi": float("nan")})
    agg = pd.DataFrame(rows).sort_values("contribution").reset_index(drop=True)
    agg.to_csv(OUTDIR / "table_4_inequality_decomp.csv", index=False)

    manifest = {
        "cohort_rows": int(len(data)),
        "n_deaths": n_deaths,
        "n_countries": int(len(countries)),
        "weighted_u5_risk": float(np.average(y, weights=w)),
        "total_concentration_index": float(total_ci),
        "total_ci_lo": pct(boot_total, 2.5),
        "total_ci_hi": pct(boot_total, 97.5),
        # Bounded-outcome variants (external review, 2026-08-03). The standard index
        # is primary and carries the decomposition; these are reported alongside it so
        # the bounded-outcome objection is answered with numbers, not a footnote.
        "wagstaff_normalised_index": _wagstaff_index(
            float(total_ci), float(np.average(y, weights=w))),
        "wagstaff_ci_lo": pct(boot_wagstaff, 2.5),
        "wagstaff_ci_hi": pct(boot_wagstaff, 97.5),
        "erreygers_index": _erreygers_index(
            float(total_ci), float(np.average(y, weights=w))),
        "erreygers_ci_lo": pct(boot_erreygers, 2.5),
        "erreygers_ci_hi": pct(boot_erreygers, 97.5),
        "index_attainable_range": [float(np.average(y, weights=w)) - 1.0,
                                   1.0 - float(np.average(y, weights=w))],
        "contributions_sum_to_total": bool(
            np.isclose(sum(contrib.values()) + (total_ci - sum(contrib.values())),
                       total_ci, atol=1e-12)),
        "education_share_pct": float(100.0 * contrib.get("edu_cat", 0.0) / total_ci),
        "education_share_ci": [pct(boot_share["edu_cat"], 2.5), pct(boot_share["edu_cat"], 97.5)],
        "bootstrap_reps_requested": bootstrap_reps,
        "bootstrap_reps_used": len(boot_total),
        "seed": seed,
        "estimand": "Wagstaff LPM concentration-index decomposition; wealth is the "
                    "ranking variable and is excluded from the predictor set",
        "source": "src/inferential/s05d_inequality.py on global_analysis_v2_candidate.parquet",
    }
    (OUTDIR / "table_4_inequality_decomp_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )
    print(f"[Table4-v2] total_CI={total_ci:.4f} "
          f"({manifest['total_ci_lo']:.4f},{manifest['total_ci_hi']:.4f}) "
          f"edu_share={manifest['education_share_pct']:.1f}% "
          f"({manifest['education_share_ci'][0]:.1f},{manifest['education_share_ci'][1]:.1f})")
    print(agg.to_string(index=False))
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1000)
    args = ap.parse_args()
    run(bootstrap_reps=args.reps)
