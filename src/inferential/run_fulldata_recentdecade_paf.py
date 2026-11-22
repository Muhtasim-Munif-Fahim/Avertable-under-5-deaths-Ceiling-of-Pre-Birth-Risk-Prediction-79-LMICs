"""run_fulldata_recentdecade_paf.py — recent-decade EXPOSURE-PREVALENCE PAF (v2 port).

Ports the legacy censoring-immune recent-decade PAF (`run_fulldata_recentdecade_paf.py`)
onto the v2 exact-follow-up wave-FE surface. A literal "2018-2024 births only" PAF is
invalid for under-five mortality — those births are right-censored to infancy — so this
isolates the EXPOSURE-PREVALENCE channel instead: the risk associations are estimated on
the exact >=60-month cohort (valid outcomes; the published primary surface), and the
g-computation PAF is then re-standardised to the joint education x birth-spacing EXPOSURE
distribution of the most recent birth cohort (2018-2024). Education and the preceding
interval are measured at/before birth, so they are complete for recent births regardless
of mortality censoring; only the exposure prevalence changes.

Estimand (v2): reweight the fitted exact-cohort surface by importance weights on the joint
(education-state x spacing-state) cell distribution, so the cohort's exposure mix matches
recent births'. This stays entirely within the fitted design (no out-of-cohort prediction,
so no survey fixed-effect extrapolation) and changes the PAF ONLY through exposure
prevalence. Uncertainty: uncapped country-cluster percentile bootstrap, recomputing both
the cohort and the recent exposure prevalences on each resample.

Output: output/phase4/manuscript/table_recentdecade_paf_reweighted.csv
Run:    python -m src.inferential.run_fulldata_recentdecade_paf
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy.special import expit

try:  # Windows console/file default cp1252 cannot encode the delta glyph
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from config.paths import HARMONIZED_DIR, OUTPUT_DIR, RNG_SEED
from src.inferential.phase2_analysis import (
    _counterfactual_eta_delta, _fit_surface, _read_pilot,
)
from src.inferential.derived_pafs import _reproduce_cohort_extras, _weighted_paf

OUT = OUTPUT_DIR / "phase4" / "manuscript"
SCENARIOS = ("education", "spacing", "joint")
RECENT_LO, RECENT_HI = 2018, 2024


def _edu_state(edu_level) -> np.ndarray:
    e = pd.to_numeric(pd.Series(edu_level).reset_index(drop=True), errors="coerce")
    out = np.full(len(e), "edu_higher", dtype=object)          # edu_level >= 3
    out[e.isin([0, 1]).to_numpy()] = "edu_below_sec"           # none / primary (eligible)
    out[(e == 2).to_numpy()] = "edu_secondary"
    out[e.isna().to_numpy()] = "edu_missing"
    return out


def _sp_state(birth_order, interval) -> np.ndarray:
    o = pd.to_numeric(pd.Series(birth_order).reset_index(drop=True), errors="coerce")
    i = pd.to_numeric(pd.Series(interval).reset_index(drop=True), errors="coerce")
    out = np.full(len(o), "sp_long", dtype=object)             # interval >= 48
    out[(i < 48).to_numpy(na_value=False)] = "sp_adequate"     # 24-47
    out[(i < 24).to_numpy(na_value=False)] = "sp_short"        # <24 (eligible, non-first)
    out[o.le(1).to_numpy(na_value=False)] = "sp_first"         # first birth: no interval
    non_first = o.gt(1).to_numpy(na_value=False)
    out[non_first & i.isna().to_numpy()] = "sp_missing"        # non-first, interval missing
    out[o.isna().to_numpy()] = "sp_missing"
    return out


def _cells(edu_level, birth_order, interval) -> np.ndarray:
    edu = _edu_state(edu_level).astype(str)
    sp = _sp_state(birth_order, interval).astype(str)
    return np.array([f"{e}|{s}" for e, s in zip(edu, sp)], dtype=object)


def main(reps: int = 1000, seed: int = RNG_SEED) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = _read_pilot(HARMONIZED_DIR / "global_analysis_v2_candidate.parquet")

    # --- primary surface (identical fit to derived_pafs / published PAFs) ---
    result = _fit_surface(
        surface="exact_wave_fe_complete_wealth", role="primary_observed_wealth",
        frame=frame, outcome_column="died_under5_complete", legacy_states=False,
        fixed_effects="survey", wealth_mode="complete", weighted=True)
    coef = result.model.coef_[0]
    eta = np.asarray(result.matrix @ coef).ravel() + float(result.model.intercept_[0])
    f = expit(eta)
    w = np.asarray(result.weights, dtype=float)
    country = result.frame["country_code"].astype(str).to_numpy()

    extras = _reproduce_cohort_extras(frame)
    assert len(extras) == len(result.frame), (len(extras), len(result.frame))
    assert (extras["country_code"].astype(str).to_numpy() == country).all(), "cohort misalignment"
    cf = {s: expit(eta + _counterfactual_eta_delta(result, s, "policy_eligible", coef)[0])
          for s in SCENARIOS}
    cohort_cell = _cells(extras["edu_level"], extras["birth_order"], extras["birth_interval_months"])

    # --- recent (2018-2024) exposure set: all recent births in cohort countries ---
    by = pd.to_numeric(frame["birth_year"], errors="coerce")
    cohort_countries = set(np.unique(country).tolist())
    in_cty = frame["country_code"].astype(str).isin(cohort_countries)
    rec = frame.loc[by.ge(RECENT_LO) & by.le(RECENT_HI) & in_cty].reset_index(drop=True)
    sw = pd.to_numeric(rec["sample_weight"], errors="coerce")
    keep = (sw > 0) & sw.notna()
    rec = rec.loc[keep.to_numpy()].reset_index(drop=True)
    rec_w = pd.to_numeric(rec["sample_weight"], errors="coerce").to_numpy()
    rec_country = rec["country_code"].astype(str).to_numpy()
    rec_cell = _cells(rec["edu_level"], rec["birth_order"], rec["birth_interval_months"])
    n_recent = len(rec)

    # --- per-country x per-cell sufficient statistics ---
    cells = sorted(set(cohort_cell.tolist()) | set(rec_cell.tolist()))
    cix = {c: i for i, c in enumerate(cells)}
    countries = sorted(cohort_countries)
    coix = {c: i for i, c in enumerate(countries)}
    C, K = len(countries), len(cells)

    ci = np.array([coix[c] for c in country]); ki = np.array([cix[c] for c in cohort_cell])
    S_w = np.zeros((C, K)); S_wf = np.zeros((C, K)); S_wcf = {s: np.zeros((C, K)) for s in SCENARIOS}
    np.add.at(S_w, (ci, ki), w)
    np.add.at(S_wf, (ci, ki), w * f)
    for s in SCENARIOS:
        np.add.at(S_wcf[s], (ci, ki), w * cf[s])
    rci = np.array([coix[c] for c in rec_country]); rki = np.array([cix[c] for c in rec_cell])
    R_w = np.zeros((C, K)); np.add.at(R_w, (rci, rki), rec_w)

    below = np.array([c.startswith("edu_below_sec") for c in cells])
    short = np.array([c.endswith("sp_short") for c in cells])

    def paf(mult, scen, reweight=True):
        """PAF on the fitted surface, optionally reweighted to the recent exposure mix.

        Importance weights are defined only on exposure cells the cohort actually
        supports (cw > 0). Recent-birth weight in unsupported cells is dropped and
        the recent distribution renormalised over supported cells, so the estimand
        stays a reweighting of the fitted surface (never an extrapolation).
        """
        cw = mult @ S_w; cwf = mult @ S_wf; cwcf = mult @ S_wcf[scen]; rw = mult @ R_w
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

    def unsupported_recent_share() -> float:
        """% of recent-birth weight falling in exposure cells the cohort lacks."""
        cw = ones @ S_w; rw = ones @ R_w
        return 100.0 * rw[cw <= 0].sum() / rw.sum() if rw.sum() > 0 else np.nan

    def prev(mult, mask, source):
        m = mult @ source
        return 100.0 * m[mask].sum() / m.sum() if m.sum() > 0 else np.nan

    ones = np.ones(C)
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
        lo, hi = (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))) if draws else (np.nan, np.nan)
        rows.append({
            "scenario": scen,
            "paf_pooled_pct": round(pooled, 2),
            "paf_recent_prevalence_pct": round(recent, 2),
            "ci_lo": round(lo, 2), "ci_hi": round(hi, 2),
            "delta_recent_minus_pooled": round(recent - pooled, 2),
            "pct_below_secondary_pooled": round(prev(ones, below, S_w), 1),
            "pct_below_secondary_recent": round(prev(ones, below, R_w), 1),
            "pct_short_interval_pooled": round(prev(ones, short, S_w), 1),
            "pct_short_interval_recent": round(prev(ones, short, R_w), 1),
            "n_recent_2018_2024": n_recent,
            "pct_recent_weight_unsupported": round(unsupported_recent_share(), 3),
        })
        print(f"[recent-v2] {scen:10}: pooled {pooled:.2f}% -> recent-prevalence "
              f"{recent:.2f}% ({lo:.2f}-{hi:.2f})  Δ={recent-pooled:+.2f}")

    pd.DataFrame(rows).to_csv(OUT / "table_recentdecade_paf_reweighted.csv", index=False)
    print(f"[recent-v2] recent weight in cohort-unsupported exposure cells: "
          f"{unsupported_recent_share():.3f}%")
    print(f"[recent-v2] n_recent(2018-2024) = {n_recent:,}; "
          f"%<sec {rows[0]['pct_below_secondary_pooled']}->{rows[0]['pct_below_secondary_recent']}, "
          f"%short {rows[0]['pct_short_interval_pooled']}->{rows[0]['pct_short_interval_recent']}")
    print(f"[recent-v2] wrote {OUT / 'table_recentdecade_paf_reweighted.csv'}")


if __name__ == "__main__":
    main()
