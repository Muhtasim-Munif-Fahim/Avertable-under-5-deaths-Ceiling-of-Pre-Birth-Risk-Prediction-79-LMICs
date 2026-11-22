"""derived_pafs.py — regional, per-country, and recent-decade PAFs on v2.

Global-surface standardisation (§6): fit the primary observed-wealth surface once
(the same fit that produced the published global PAFs), form per-row factual and
counterfactual (education / spacing / joint) predictions, then aggregate the PAF
over any subpopulation as 1 - sum(w*cf)/sum(w*factual). Uncertainty is an uncapped
country-cluster percentile bootstrap (§8) computed from per-country weighted sums.

Outputs (output/phase4/manuscript/):
  table_regional_paf.csv, table_percountry_paf.csv, table_recentdecade_paf.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit

from config.paths import HARMONIZED_DIR, OUTPUT_DIR, RNG_SEED
from src.inferential.phase2_analysis import (
    _counterfactual_eta_delta, _fit_surface, _positive_normalized_weights, _read_pilot,
)

OUT = OUTPUT_DIR / "phase4" / "manuscript"
SCENARIOS = ("education", "spacing", "joint")


def _reproduce_cohort_extras(frame):
    """Reproduce the primary observed-wealth row selection to recover birth_year
    aligned (in order) with the fitted result.frame."""
    cols = ["died_under5_complete", "country_code", "_source_file", "sample_weight",
            "survey_year", "birth_year", "child_male", "multiple_birth", "birth_order",
            "birth_interval_months", "maternal_age_at_birth", "edu_level",
            "wealth_quintile", "currently_married", "urban"]
    data = frame[cols].copy()
    normalized = _positive_normalized_weights(data)
    data = data[data["wealth_quintile"].notna()]
    data = data.loc[normalized.loc[data.index].notna()]
    data = data[pd.to_numeric(data["died_under5_complete"], errors="coerce").notna()]
    return data.reset_index(drop=True)


def _weighted_paf(num_wf, num_wcf):
    """PAF from weighted factual/counterfactual risk sums."""
    return 1.0 - num_wcf / num_wf if num_wf > 0 else np.nan


def _country_sums(mask, country, w, f, cf):
    """Per-country weighted sums of w*f and w*cf within mask. Returns dict."""
    idx = np.flatnonzero(mask)
    cc = country[idx]
    order = np.argsort(cc, kind="stable")
    cc_s = cc[order]
    wf = (w * f)[idx][order]
    wcf = (w * cf)[idx][order]
    uniq, starts = np.unique(cc_s, return_index=True)
    ends = np.r_[starts[1:], len(cc_s)]
    sums = {c: (float(wf[s:e].sum()), float(wcf[s:e].sum()))
            for c, s, e in zip(uniq, starts, ends)}
    return sums


def _boot_paf(country_sums, reps, seed):
    countries = np.array(list(country_sums.keys()))
    wf = np.array([country_sums[c][0] for c in countries])
    wcf = np.array([country_sums[c][1] for c in countries])
    rng = np.random.default_rng(seed)
    C = len(countries)
    draws = []
    for _ in range(reps):
        s = rng.integers(0, C, size=C)
        m = np.bincount(s, minlength=C).astype(float)
        paf = _weighted_paf((m * wf).sum(), (m * wcf).sum())
        if np.isfinite(paf):
            draws.append(100.0 * paf)
    return draws


def main(reps: int = 1000, seed: int = RNG_SEED) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = _read_pilot(HARMONIZED_DIR / "global_analysis_v2_candidate.parquet")
    result = _fit_surface(
        surface="exact_wave_fe_complete_wealth", role="primary_observed_wealth",
        frame=frame, outcome_column="died_under5_complete", legacy_states=False,
        fixed_effects="survey", wealth_mode="complete", weighted=True)
    coef = result.model.coef_[0]
    factual_eta = np.asarray(result.matrix @ coef).ravel() + float(result.model.intercept_[0])
    f = expit(factual_eta)
    w = result.weights
    country = result.frame["country_code"].astype(str).to_numpy()

    extras = _reproduce_cohort_extras(frame)
    assert len(extras) == len(result.frame), (len(extras), len(result.frame))
    assert (extras["country_code"].astype(str).to_numpy() == country).all(), "cohort misalignment"
    region_map = (frame[["country_code", "who_region"]].drop_duplicates("country_code")
                  .set_index("country_code")["who_region"].astype(str).to_dict())
    region = np.array([region_map.get(c, "NA") for c in country])
    birth_year = pd.to_numeric(extras["birth_year"], errors="coerce").to_numpy()

    cf = {s: expit(factual_eta + _counterfactual_eta_delta(result, s, "policy_eligible", coef)[0])
          for s in SCENARIOS}

    def ci(draws):
        return (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))) if draws else (np.nan, np.nan)

    # ---- Regional PAFs ----
    rows = []
    for reg in sorted(set(region)):
        m = region == reg
        for s in SCENARIOS:
            cs = _country_sums(m, country, w, f, cf[s])
            wf = sum(v[0] for v in cs.values()); wcf = sum(v[1] for v in cs.values())
            paf = 100.0 * _weighted_paf(wf, wcf)
            lo, hi = ci(_boot_paf(cs, reps, seed))
            rows.append({"region": reg, "scenario": s, "n_countries": len(cs),
                         "paf_pct": round(paf, 2), "ci_lo": round(lo, 2), "ci_hi": round(hi, 2)})
    pd.DataFrame(rows).to_csv(OUT / "table_regional_paf.csv", index=False)
    print(f"[regional] {len(rows)} rows")

    # ---- Per-country PAFs (point; joint scenario the headline) ----
    prows = []
    for c in sorted(set(country)):
        m = country == c
        wf_c = float((w * f)[m].sum())
        entry = {"country_code": c, "who_region": region_map.get(c, "NA"),
                 "n_births": int(m.sum())}
        for s in SCENARIOS:
            wcf_c = float((w * cf[s])[m].sum())
            entry[f"paf_{s}"] = round(100.0 * _weighted_paf(wf_c, wcf_c), 2)
        prows.append(entry)
    pd.DataFrame(prows).sort_values("paf_joint", ascending=False).to_csv(
        OUT / "table_percountry_paf.csv", index=False)
    print(f"[per-country] {len(prows)} countries")

    # ---- Recent-decade PAF: restandardise to 2018-2024 births ----
    recent = birth_year >= 2018
    rrows = []
    for s in SCENARIOS:
        cs = _country_sums(recent, country, w, f, cf[s])
        wf = sum(v[0] for v in cs.values()); wcf = sum(v[1] for v in cs.values())
        paf = 100.0 * _weighted_paf(wf, wcf)
        lo, hi = ci(_boot_paf(cs, reps, seed))
        rrows.append({"scenario": s, "births_2018_2024": int(recent.sum()),
                      "paf_pct": round(paf, 2), "ci_lo": round(lo, 2), "ci_hi": round(hi, 2)})
    pd.DataFrame(rrows).to_csv(OUT / "table_recentdecade_paf.csv", index=False)
    print(f"[recent-decade] n_recent={int(recent.sum()):,}")
    for r in rrows:
        print(f"  recent {r['scenario']:10}: {r['paf_pct']} ({r['ci_lo']}-{r['ci_hi']})")


if __name__ == "__main__":
    main()
