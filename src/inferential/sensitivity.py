"""sensitivity.py — reviewer-facing sensitivity suite on the v2 exact cohort.

The v2 rebuild regenerated the primary estimates but not the legacy sensitivity
programme, so the appendix had no sensitivity section. This re-runs the analyses a
Lancet Global Health reviewer is most likely to ask for, each as a genuine refit of
the SAME locked surface with one design choice changed, so every row is comparable
to the primary and to every other row.

Analyses (`--analysis`):
  primary          the published specification, refitted here as the comparison anchor
  unweighted       survey weights off (is the estimand weight-driven?)
  country_fe       country fixed effects instead of survey-wave (does the finer wave
                   adjustment matter, or only country adjustment?)
  wealth_explicit  missing wealth retained as its own category instead of complete-case
  wealth_to_q1     missing wealth imputed to the poorest quintile   (bound)
  wealth_to_q5     missing wealth imputed to the richest quintile   (bound)
  dhs_only         MICS surveys excluded (how much rests on the 16 MICS waves?)
  pop_weighted     countries re-weighted to their share of UN-IGME 2024 live births
                   instead of their share of sampled births (see below)
  complete_case    all exposures complete-case instead of explicit missing categories
  evalues          E-values for the primary adjusted odds ratios (analytic, no refit)

The `pop_weighted` scenario answers the reviewer question "what population does the
pooled PAF represent?". DHS/MICS weights are normalised within survey, so a country
enters the pooled estimand in proportion to its *sampled* births — India, Nigeria and
Bangladesh dominate, and the 79 countries are not a probability sample of LMICs. This
scenario rescales each country's standardisation weight so that country totals are
proportional to UN-IGME 2024 implied live births (under-five deaths / U5MR) instead.
Only the standardisation population changes: the fitted associations are held at the
primary surface, which is the coherent comparison because survey-wave fixed effects
make those associations within-country contrasts already.

Each writes one CSV to output/phase4/manuscript/sensitivity/. `--analysis combine`
merges them into table_sensitivity.csv.

PAF uncertainty uses the uncapped country-cluster percentile bootstrap over
per-country weighted sufficient statistics (the same estimator as derived_pafs),
which is cheap once the fit exists.

Run one (each is a full-data fit; several can run concurrently on this box):
    python -m src.inferential.sensitivity --analysis unweighted
Then:
    python -m src.inferential.sensitivity --analysis combine
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from scipy.special import expit

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from config.paths import HARMONIZED_DIR, OUTPUT_DIR, RNG_SEED
from src.inferential.phase2_analysis import (
    _counterfactual_eta_delta, _fit_surface, _read_pilot,
)
from src.inferential.derived_pafs import _boot_paf, _country_sums, _weighted_paf

MV = OUTPUT_DIR / "phase4" / "manuscript"
OUT = MV / "sensitivity"
SCENARIOS = ("education", "spacing", "joint")

# Each entry overrides exactly one design choice of the primary specification.
SPECS: dict[str, dict] = {
    "primary": dict(label="Primary specification",
                    fixed_effects="survey", wealth_mode="complete", weighted=True),
    "unweighted": dict(label="Survey weights removed",
                       fixed_effects="survey", wealth_mode="complete", weighted=False),
    "country_fe": dict(label="Country fixed effects (not survey-wave)",
                       fixed_effects="country", wealth_mode="complete", weighted=True),
    "wealth_explicit": dict(label="Missing wealth as an explicit category",
                            fixed_effects="survey", wealth_mode="explicit", weighted=True),
    "wealth_to_q1": dict(label="Missing wealth imputed to poorest quintile",
                         fixed_effects="survey", wealth_mode="complete", weighted=True,
                         impute_wealth=1.0),
    "wealth_to_q5": dict(label="Missing wealth imputed to richest quintile",
                         fixed_effects="survey", wealth_mode="complete", weighted=True,
                         impute_wealth=5.0),
    "dhs_only": dict(label="DHS surveys only (MICS excluded)",
                     fixed_effects="survey", wealth_mode="complete", weighted=True,
                     dhs_only=True),
    "pop_weighted": dict(label="Countries weighted by UN-IGME live births",
                         fixed_effects="survey", wealth_mode="complete", weighted=True,
                         pop_weighted=True),
    "complete_case": dict(label="Complete-case on all exposures",
                          fixed_effects="survey", wealth_mode="complete", weighted=True,
                          complete_case=True),
}


def _igme_live_births() -> pd.Series:
    """Implied UN-IGME 2024 live births per country_code: U5 deaths / (U5MR/1000)."""
    deaths = pd.read_csv(MV / "table_igme_annualisation.csv")
    u5mr = pd.read_csv(MV / "table_sdg_scoreboard.csv")
    m = deaths.merge(u5mr[["country_code", "u5mr_2024"]], on="country_code", how="inner")
    births = m["igme_u5_deaths_2024"] / (m["u5mr_2024"] / 1000.0)
    out = pd.Series(births.to_numpy(), index=m["country_code"].astype(str))
    return out[np.isfinite(out) & (out > 0)]


def _apply_population_weights(w: np.ndarray, country: np.ndarray) -> np.ndarray:
    """Rescale weights so country totals track IGME live births, not sampled births.

    Countries without an IGME crosswalk keep their sampled-births weight; the count is
    printed so any coverage loss is visible rather than silent.
    """
    births = _igme_live_births()
    current = pd.Series(w).groupby(country).transform("sum").to_numpy()
    target = pd.Series(country).map(births).to_numpy(dtype=float)
    matched = np.isfinite(target) & (current > 0)
    n_unmatched = int(len(np.unique(country[~matched]))) if (~matched).any() else 0
    if n_unmatched:
        print(f"  pop_weighted: {n_unmatched} countries lack an IGME live-birth "
              f"estimate and keep their sampled-births weight")
    scaled = w.copy()
    scaled[matched] = w[matched] * (target[matched] / current[matched])
    # Renormalise to the original total so PAFs stay on the same scale.
    return scaled * (w.sum() / scaled.sum())


def _prepare_frame(spec: dict) -> pd.DataFrame:
    frame = _read_pilot(HARMONIZED_DIR / "global_analysis_v2_candidate.parquet")
    if spec.get("dhs_only"):
        # MICS survey ids do not follow the DHS ??BR* birth-recode convention.
        is_dhs = frame["_source_file"].astype(str).str.contains(r"BR", case=True, na=False)
        n_before = len(frame)
        frame = frame.loc[is_dhs].reset_index(drop=True)
        print(f"  DHS-only: {n_before:,} -> {len(frame):,} rows")
    if spec.get("impute_wealth") is not None:
        # Assign in place: _read_pilot already returns a fresh frame, and copying an
        # 8.2M-row frame here is what pushed these two jobs into an OOM when the
        # whole suite ran concurrently.
        wq = pd.to_numeric(frame["wealth_quintile"], errors="coerce")
        n_missing = int(wq.isna().sum())
        frame["wealth_quintile"] = wq.fillna(spec["impute_wealth"]).astype("float64")
        del wq
        print(f"  imputed {n_missing:,} missing wealth values to Q{int(spec['impute_wealth'])}")
    if spec.get("complete_case"):
        # Drop any birth with a missing value on an exposure the primary surface
        # would otherwise carry as an explicit "Missing" category.
        cols = ["edu_level", "birth_interval_months", "maternal_age_at_birth",
                "birth_order", "currently_married", "urban", "multiple_birth",
                "child_male"]
        present = [c for c in cols if c in frame.columns]
        keep = frame[present].notna().all(axis=1)
        # birth_interval_months is missing by design for first births, which are a
        # legitimate state rather than missing data; retain them.
        first_birth = pd.to_numeric(frame["birth_order"], errors="coerce").eq(1)
        keep = keep | (first_birth & frame[[c for c in present
                                            if c != "birth_interval_months"]]
                       .notna().all(axis=1))
        n_before = len(frame)
        frame = frame.loc[keep].reset_index(drop=True)
        print(f"  complete-case: {n_before:,} -> {len(frame):,} rows "
              f"({100 * (1 - len(frame) / n_before):.1f}% dropped)")
    return frame


def run_one(name: str, reps: int = 1000, seed: int = RNG_SEED) -> pd.DataFrame:
    spec = SPECS[name]
    print(f"[sensitivity] {name}: {spec['label']}")
    frame = _prepare_frame(spec)

    result = _fit_surface(
        surface="exact_wave_fe_complete_wealth", role=f"sensitivity_{name}",
        frame=frame, outcome_column="died_under5_complete", legacy_states=False,
        fixed_effects=spec["fixed_effects"], wealth_mode=spec["wealth_mode"],
        weighted=spec["weighted"])

    coef = result.model.coef_[0]
    eta = np.asarray(result.matrix @ coef).ravel() + float(result.model.intercept_[0])
    f = expit(eta)
    # _fit_surface returns weights=None for an unweighted fit; np.asarray(None) would
    # silently produce a 0-d NaN and propagate NaN through every PAF.
    w = (np.ones(len(f), dtype=float) if result.weights is None
         else np.asarray(result.weights, dtype=float))
    if not np.isfinite(w).all() or w.sum() <= 0:
        raise ValueError(f"invalid weights for {name}: sum={w.sum()}")
    country = result.frame["country_code"].astype(str).to_numpy()
    if spec.get("pop_weighted"):
        w = _apply_population_weights(w, country)
        if not np.isfinite(w).all() or w.sum() <= 0:
            raise ValueError(f"invalid population weights for {name}: sum={w.sum()}")
    mask = np.ones(len(f), dtype=bool)

    rows = []
    for scen in SCENARIOS:
        cf = expit(eta + _counterfactual_eta_delta(result, scen, "policy_eligible", coef)[0])
        cs = _country_sums(mask, country, w, f, cf)
        wf = sum(v[0] for v in cs.values())
        wcf = sum(v[1] for v in cs.values())
        paf = 100.0 * _weighted_paf(wf, wcf)
        draws = _boot_paf(cs, reps, seed)
        lo, hi = ((float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))
                  if draws else (np.nan, np.nan))
        rows.append({"analysis": name, "label": spec["label"], "scenario": scen,
                     "paf_pct": round(paf, 2), "ci_lo": round(lo, 2), "ci_hi": round(hi, 2),
                     "n_births": int(len(f)), "n_countries": len(cs)})
        print(f"    {scen:10} {paf:6.2f} ({lo:.2f}-{hi:.2f})")

    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / f"sensitivity_{name}.csv", index=False)
    print(f"[sensitivity] wrote {OUT / f'sensitivity_{name}.csv'}")
    return out


def run_evalues() -> pd.DataFrame:
    """E-values for the primary aORs: how strong must unmeasured confounding be to
    explain away the association? Uses the odds ratio as a risk-ratio approximation,
    which is conservative for an outcome this rare."""
    aor = pd.read_csv(MV / "table_2_primary_aor.csv")

    def ev(rr: float) -> float:
        if not np.isfinite(rr) or rr <= 0:
            return np.nan
        r = rr if rr > 1 else 1.0 / rr
        return r + np.sqrt(r * (r - 1.0))

    keep = ["edu_cat_Secondary", "edu_cat_Higher", "bi_model_cat_<24 mo",
            "bi_model_cat_48+ mo", "birth_type_Multiple", "ma_cat_<20",
            "urban_cat_Urban", "wq_cat_Q5"]
    d = aor[aor["feature"].isin(keep)].copy()
    # The CI limit closest to the null is the one an E-value must move.
    near_null = np.where(d["aOR"] > 1, d["ci_lo"], d["ci_hi"])
    d["e_value_point"] = [round(ev(x), 2) for x in d["aOR"]]
    d["e_value_ci"] = [round(ev(x), 2) for x in near_null]
    d = d[["feature", "aOR", "ci_lo", "ci_hi", "e_value_point", "e_value_ci"]]
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT / "sensitivity_evalues.csv", index=False)
    print("[sensitivity] E-values:")
    print(d.to_string(index=False))
    return d


def combine() -> None:
    files = sorted(OUT.glob("sensitivity_*.csv"))
    frames = [pd.read_csv(p) for p in files if "evalues" not in p.name]
    if not frames:
        raise SystemExit("no sensitivity results to combine")
    allr = pd.concat(frames, ignore_index=True)

    order = [k for k in SPECS if k in set(allr["analysis"])]
    allr["_o"] = allr["analysis"].map({k: i for i, k in enumerate(order)})
    allr = allr.sort_values(["_o", "scenario"]).drop(columns="_o")
    allr.to_csv(MV / "table_sensitivity.csv", index=False)

    joint = allr[allr["scenario"] == "joint"]
    base = joint[joint["analysis"] == "primary"]
    b = float(base["paf_pct"].iloc[0]) if len(base) else np.nan
    print(f"\nJoint PAF across specifications (primary = {b:.2f}%):")
    for _, r in joint.iterrows():
        delta = r["paf_pct"] - b
        print(f"  {r['label']:44} {r['paf_pct']:6.2f} ({r['ci_lo']:.2f}-{r['ci_hi']:.2f})"
              f"  {delta:+.2f}")
    print(f"\n[sensitivity] wrote {MV / 'table_sensitivity.csv'} ({len(allr)} rows)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", required=True,
                    choices=list(SPECS) + ["evalues", "combine"])
    ap.add_argument("--reps", type=int, default=1000)
    a = ap.parse_args()
    if a.analysis == "combine":
        combine()
    elif a.analysis == "evalues":
        run_evalues()
    else:
        run_one(a.analysis, reps=a.reps)


if __name__ == "__main__":
    main()
