"""derived_extras.py — v2 mediation PAF, UN-IGME annualisation, SDG scoreboard.

1. Mediation / total-effect education PAF: education PAF from the wave-FE surface
   WITHOUT wealth adjustment (wealth is a downstream mediator), on the same
   complete-wealth cohort, vs the wealth-adjusted primary edu PAF (21.37).
2. UN-IGME annualisation: per-country joint PAF x IGME 2024 under-five deaths, on the
   survival country basis (primary) and the exact-binary basis (sensitivity).
3. SDG 3.2 scoreboard: per-country IGME 2024 U5MR x (1 - joint PAF) vs the 25/1000
   SDG target; count countries reaching it factually and under the counterfactual.

Reuses the legacy IGME crosswalk and the v2 per-country PAFs / v2 survey inventory.
Outputs to output/phase4/manuscript/.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit

from config.paths import HARMONIZED_DIR, OUTPUT_DIR
from src.inferential.phase2_analysis import (
    _counterfactual_eta_delta, _fit_surface, _read_pilot,
)
from src.inferential import country_basis as cb
from src.inferential.igme_crosswalk import build_crosswalk

OUT = OUTPUT_DIR / "phase4" / "manuscript"
IGME_CSV = HARMONIZED_DIR.parents[1] / "data" / "raw" / "un_igme" / "UNICEF-CME_DF_2021_WQ-1.0-download.csv"


def _global_edu_paf(frame, wealth_mode: str) -> float:
    result = _fit_surface(
        surface="s", role="r", frame=frame, outcome_column="died_under5_complete",
        legacy_states=False, fixed_effects="survey", wealth_mode=wealth_mode, weighted=True)
    coef = result.model.coef_[0]
    eta = np.asarray(result.matrix @ coef).ravel() + float(result.model.intercept_[0])
    f = expit(eta)
    cf = expit(eta + _counterfactual_eta_delta(result, "education", "policy_eligible", coef)[0])
    w = result.weights
    return 100.0 * (1.0 - np.average(cf, weights=w) / np.average(f, weights=w))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = _read_pilot(HARMONIZED_DIR / "global_analysis_v2_candidate.parquet")

    # 1. Mediation / total-effect education PAF -----------------------------------
    paf_adj = _global_edu_paf(frame, "complete")        # wealth-adjusted (primary)
    paf_total = _global_edu_paf(frame, "none_complete")  # no wealth (total effect)
    med = pd.DataFrame([
        {"estimand": "education_PAF_wealth_adjusted (primary)", "paf_pct": round(paf_adj, 2)},
        {"estimand": "education_PAF_total_effect (no wealth mediator)", "paf_pct": round(paf_total, 2)},
        {"estimand": "mediated_by_wealth (difference)", "paf_pct": round(paf_total - paf_adj, 2)},
    ])
    med.to_csv(OUT / "table_mediation_education_paf.csv", index=False)
    print(f"[mediation] edu PAF adjusted={paf_adj:.2f}  total-effect={paf_total:.2f}")

    # 2 & 3. IGME annualisation + SDG scoreboard, on both country bases ---------
    igme_and_sdg(cb.PERCOUNTRY_PAF, cb.IGME_ANNUALISATION, cb.SDG_SCOREBOARD)
    igme_and_sdg(cb.BINARY_PERCOUNTRY_PAF, cb.BINARY_IGME_ANNUALISATION,
                 cb.BINARY_SDG_SCOREBOARD)


def igme_and_sdg(paf_file: str, igme_out: str, sdg_out: str) -> None:
    """UN-IGME 2024 annualisation and SDG 3.2 scoreboard from one per-country PAF file."""
    pc = pd.read_csv(OUT / paf_file)
    inv = pd.read_csv(OUT / "table_s1_survey_inventory.csv")
    igme = pd.read_csv(IGME_CSV, encoding="utf-8", engine="python")
    our_names = inv.groupby("country_code")["country_name"].first().reset_index()
    cw = build_crosswalk(our_names, igme[["REF_AREA", "Geographic area"]].drop_duplicates())

    def _igme(indicator):
        d = igme[(igme["INDICATOR"] == indicator) & (igme["Sex"] == "Total")
                 & (igme["Series Name"] == "UN IGME estimate")].copy()
        d["year"] = d["TIME_PERIOD"].astype(str).str[:4].astype(int)
        d["val"] = pd.to_numeric(d["OBS_VALUE"], errors="coerce")
        return d[d["year"] == 2024].groupby("REF_AREA")["val"].first().to_dict()

    deaths2024 = _igme("TMY0T4")   # under-five deaths (count)
    u5mr2024 = _igme("MRY0T4")     # under-five mortality rate per 1000

    rows, sdg = [], []
    tot_deaths = tot_avert = 0.0
    fact_reach = cf_reach = newly = 0
    n_sdg = 0
    for _, r in pc.iterrows():
        cc = r["country_code"]; iso = cw.get(cc); pj = r["paf_joint"]
        d = deaths2024.get(iso); u5 = u5mr2024.get(iso)
        if iso is not None and d is not None and np.isfinite(d):
            av = pj / 100 * d
            tot_deaths += d; tot_avert += av
            rows.append({"country_code": cc, "iso3": iso, "who_region": r["who_region"],
                         "paf_joint_pct": pj, "igme_u5_deaths_2024": int(round(d)),
                         "avertable_per_year": int(round(av))})
        if iso is not None and u5 is not None and np.isfinite(u5):
            n_sdg += 1
            u5_cf = u5 * (1 - pj / 100)
            f_ok = u5 <= 25; c_ok = u5_cf <= 25
            fact_reach += int(f_ok); cf_reach += int(c_ok); newly += int(c_ok and not f_ok)
            sdg.append({"country_code": cc, "iso3": iso, "u5mr_2024": round(u5, 1),
                        "u5mr_counterfactual": round(u5_cf, 1),
                        "reaches_sdg_factual": f_ok, "reaches_sdg_counterfactual": c_ok})

    pd.DataFrame(rows).sort_values("avertable_per_year", ascending=False).to_csv(
        OUT / igme_out, index=False)
    pd.DataFrame(sdg).to_csv(OUT / sdg_out, index=False)
    print(f"[IGME:{paf_file}] pooled 2024 U5 deaths (mapped)={int(tot_deaths):,}  "
          f"avertable/yr under joint counterfactual={int(tot_avert):,} "
          f"({100*tot_avert/tot_deaths:.1f}%)")
    print(f"[SDG:{paf_file}] of {n_sdg} countries: {fact_reach} reach <=25 factually, "
          f"{cf_reach} under joint counterfactual (+{newly} newly)")


if __name__ == "__main__":
    main()
