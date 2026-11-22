"""run_fulldata_recentdecade_paf_survival.py — recent-exposure PAF on the SURVIVAL surface.

External-review amendment (2026-08-03). The v2 analysis already re-standardises the
joint counterfactual to the exposure mix of the most recent birth cohort (2018-2024),
but only on the exact-binary surface (`run_fulldata_recentdecade_paf.py`, joint
32.20% -> 26.39%). Because the discrete-time survival model is the primary estimand,
and because the recent-exposure figure is now the paper's contemporary policy
headline, the same re-standardisation has to exist on the survival surface rather
than being carried across by a scaling ratio.

Estimand: identical in form to the binary port. The monthly-hazard model is fitted on
the full observed birth history (the published primary surface), each covariate
pattern's factual and counterfactual 60-month risks are read off that fit, and the
standardisation is then importance-weighted so the joint (education-state x
spacing-state) exposure cell distribution matches that of 2018-2024 births. Only
exposure prevalence changes; no association is refitted and nothing is extrapolated
outside the cells the cohort supports.

Exposure cells are built with `_prepare_states` on BOTH sides, so the cohort and
recent-birth cell definitions are identical by construction rather than by two
parallel reimplementations.

Uncertainty: uncapped country-cluster percentile bootstrap over the per-country x
per-cell sufficient statistics, holding the fitted surface fixed — the same estimator
the binary port uses, so the two are directly comparable.

Output: output/phase4/manuscript/table_recentdecade_paf_survival.csv
Run:    python -m src.inferential.run_fulldata_recentdecade_paf_survival
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from config.paths import HARMONIZED_DIR, OUTPUT_DIR, RNG_SEED
from src.inferential.phase2_analysis import _prepare_states, _read_pilot
from src.inferential.phase3_survival import (
    _fit_model, _prepare, _standardized_rows,
)

OUT = OUTPUT_DIR / "phase4" / "manuscript"
PILOT = HARMONIZED_DIR / "global_analysis_v2_candidate.parquet"
SCENARIOS = ("education", "spacing", "joint")
RECENT_LO, RECENT_HI = 2018, 2024

# Education states eligible for the counterfactual shift, and the short-interval state.
_ELIGIBLE_EDU = ("None", "Primary")
_SHORT_INTERVAL = "<24 mo"


def _cells(edu_cat, bi_cat) -> np.ndarray:
    """Joint education x spacing exposure cell label, as a plain object array."""
    e = pd.Series(edu_cat).astype(str).reset_index(drop=True)
    s = pd.Series(bi_cat).astype(str).reset_index(drop=True)
    return (e + "|" + s).to_numpy(dtype=object)


def _weighted_paf(factual: float, counterfactual: float) -> float:
    if factual <= 0:
        return float("nan")
    return (factual - counterfactual) / factual


def main(reps: int = 1000, seed: int = RNG_SEED) -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # --- primary survival surface (same fit as the published phase-3 run) ---
    print("[recent-survival] preparing survival cohort ...")
    prepared = _prepare(PILOT)
    print(f"[recent-survival] {prepared.n_births:,} births, "
          f"{prepared.n_countries} countries, {prepared.n_surveys} surveys, "
          f"{len(prepared.patterns):,} covariate patterns")
    fit = _fit_model(prepared.cells, prepared.patterns, compute_backend="auto")
    if not fit.converged:
        raise SystemExit(f"survival fit did not converge: {fit.warning_text}")

    patterns = prepared.patterns
    pattern_weight = patterns["birth_weight"].to_numpy(float)
    _rows, risks = _standardized_rows(fit, patterns, pattern_weight)

    factual = risks["factual"]
    cf = {s: risks[s] for s in SCENARIOS}
    pat_country = patterns["country_code"].astype(str).to_numpy()
    pat_cell = _cells(patterns["edu_cat"], patterns["bi_model_cat"])

    # --- recent (2018-2024) exposure distribution, same cell definition ---
    frame = _read_pilot(PILOT)
    by = pd.to_numeric(frame["birth_year"], errors="coerce")
    cohort_countries = set(np.unique(pat_country).tolist())
    in_cty = frame["country_code"].astype(str).isin(cohort_countries)
    sw = pd.to_numeric(frame["sample_weight"], errors="coerce")
    # Comparisons on nullable dtypes yield pd.NA, which is not a valid boolean mask;
    # an unknown birth year or weight is simply not a recent birth.
    rec_mask = (by.ge(RECENT_LO) & by.le(RECENT_HI) & in_cty
                & sw.notna() & sw.gt(0)).fillna(False).to_numpy(dtype=bool)
    rec = frame.loc[rec_mask].reset_index(drop=True)
    rec_states = _prepare_states(rec, legacy=False)
    rec_w = pd.to_numeric(rec["sample_weight"], errors="coerce").to_numpy(float)
    rec_country = rec["country_code"].astype(str).to_numpy()
    rec_cell = _cells(rec_states["edu_cat"], rec_states["bi_model_cat"])
    n_recent = len(rec)

    # --- per-country x per-cell sufficient statistics ---
    cells = sorted(set(pat_cell.tolist()) | set(rec_cell.tolist()))
    cix = {c: i for i, c in enumerate(cells)}
    countries = sorted(cohort_countries)
    coix = {c: i for i, c in enumerate(countries)}
    C, K = len(countries), len(cells)

    ci = np.array([coix[c] for c in pat_country])
    ki = np.array([cix[c] for c in pat_cell])
    S_w = np.zeros((C, K))
    S_wf = np.zeros((C, K))
    S_wcf = {s: np.zeros((C, K)) for s in SCENARIOS}
    np.add.at(S_w, (ci, ki), pattern_weight)
    np.add.at(S_wf, (ci, ki), pattern_weight * factual)
    for s in SCENARIOS:
        np.add.at(S_wcf[s], (ci, ki), pattern_weight * cf[s])

    rci = np.array([coix[c] for c in rec_country])
    rki = np.array([cix[c] for c in rec_cell])
    R_w = np.zeros((C, K))
    np.add.at(R_w, (rci, rki), rec_w)

    below = np.array([c.split("|")[0] in _ELIGIBLE_EDU for c in cells])
    short = np.array([c.split("|")[1] == _SHORT_INTERVAL for c in cells])
    ones = np.ones(C)

    def paf(mult, scen, reweight=True):
        """PAF on the fitted survival surface, optionally re-standardised.

        Importance weights are defined only on exposure cells the cohort supports
        (cw > 0); recent-birth weight in unsupported cells is dropped and the recent
        distribution renormalised over supported cells, so this stays a reweighting
        of the fitted surface rather than an extrapolation beyond it.
        """
        cw = mult @ S_w
        cwf = mult @ S_wf
        cwcf = mult @ S_wcf[scen]
        rw = mult @ R_w
        if cw.sum() <= 0:
            return np.nan
        if not reweight:
            return 100.0 * _weighted_paf(cwf.sum(), cwcf.sum())
        supported = cw > 0
        rw_sup = np.where(supported, rw, 0.0)
        if rw_sup.sum() <= 0:
            return np.nan
        ratio = np.zeros(K)
        ratio[supported] = ((rw_sup[supported] / rw_sup.sum())
                            / (cw[supported] / cw.sum()))
        return 100.0 * _weighted_paf((ratio * cwf).sum(), (ratio * cwcf).sum())

    def prev(mult, mask, source):
        m = mult @ source
        return 100.0 * m[mask].sum() / m.sum() if m.sum() > 0 else np.nan

    cw0, rw0 = ones @ S_w, ones @ R_w
    unsupported = 100.0 * rw0[cw0 <= 0].sum() / rw0.sum() if rw0.sum() > 0 else np.nan

    rng = np.random.default_rng(seed)
    boot_idx = rng.integers(0, C, size=(reps, C))

    rows = []
    for scen in SCENARIOS:
        pooled = paf(ones, scen, reweight=False)
        recent = paf(ones, scen, reweight=True)
        draws = []
        for b in range(reps):
            mult = np.bincount(boot_idx[b], minlength=C).astype(float)
            v = paf(mult, scen, reweight=True)
            if np.isfinite(v):
                draws.append(v)
        lo, hi = ((float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))
                  if draws else (np.nan, np.nan))
        rows.append({
            "scenario": scen,
            "estimand": "discrete-time survival (PRIMARY)",
            "paf_pooled_pct": round(pooled, 2),
            "paf_recent_prevalence_pct": round(recent, 2),
            "ci_lo": round(lo, 2), "ci_hi": round(hi, 2),
            "delta_recent_minus_pooled": round(recent - pooled, 2),
            "pct_below_secondary_pooled": round(prev(ones, below, S_w), 1),
            "pct_below_secondary_recent": round(prev(ones, below, R_w), 1),
            "pct_short_interval_pooled": round(prev(ones, short, S_w), 1),
            "pct_short_interval_recent": round(prev(ones, short, R_w), 1),
            "n_recent_2018_2024": n_recent,
            "n_analysis_births": int(prepared.n_births),
            "n_countries": C,
            "bootstrap_reps": reps,
            "pct_recent_weight_unsupported": round(float(unsupported), 3),
        })
        print(f"[recent-survival] {scen:10}: pooled {pooled:.2f}% -> "
              f"recent-prevalence {recent:.2f}% ({lo:.2f}-{hi:.2f})  "
              f"Δ={recent - pooled:+.2f}")

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "table_recentdecade_paf_survival.csv", index=False)
    print(f"[recent-survival] recent weight in cohort-unsupported cells: "
          f"{unsupported:.3f}%")
    print(f"[recent-survival] n_recent(2018-2024) = {n_recent:,}; "
          f"%<sec {rows[0]['pct_below_secondary_pooled']}->"
          f"{rows[0]['pct_below_secondary_recent']}, "
          f"%short {rows[0]['pct_short_interval_pooled']}->"
          f"{rows[0]['pct_short_interval_recent']}")
    print(f"[recent-survival] wrote "
          f"{OUT / 'table_recentdecade_paf_survival.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=1000)
    a = ap.parse_args()
    main(reps=a.reps)
