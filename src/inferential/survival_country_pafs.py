"""Country- and region-level attributable fractions on the primary survival surface.

Why this exists
---------------
The supervisor's review (2026-10-10) asked that one estimand carry every main
display. Until now the pooled headline came from the discrete-time survival
surface, while every country and regional figure (main-text Figure 1, the country
rows of Table 2, eTable S21, the SDG scenario, eFigures S4 and S6) came from the
exact-binary surface. This module puts those quantities on the survival surface,
so the exact-binary model is left as a sensitivity analysis only.

What it estimates
-----------------
For each country, the policy-eligible education, spacing and joint fractions at
that country's OWN observed exposure distribution (its births, survey-weighted),
from the one pooled survival fit:

    PAF_c = 1 - sum_{i in c} w_i R_i(shifted) / sum_{i in c} w_i R_i(factual)

where R_i is the 60-month risk 1 - prod_t (1 - h_it). Country values are NOT
re-standardised to 2018-2024 exposures: 34 of the 77 countries have no
2018-2024 births, so a country-level contemporary profile does not exist for
them. The 2018-2024 re-standardisation stays the pooled policy headline.

Regional fractions pool the region's countries with their own weights.

Bootstrap
---------
The draws, seed, solver, warm start and convergence gate are those of
``sens2026_survival --model main``, so the pooled intervals reproduce that refit
run. In each replicate the model is refitted; the pooled and regional fractions
use the replicate's country multiplicities, and each country is scored on its own
rows with its own weights, only in replicates where it was drawn (its survey
fixed effects are unidentified otherwise -- the rule of ``s05r_percountry_paf_ci``).
Intervals are percentiles of the successful replicates.

Outputs (output/phase4/manuscript/)
    table_percountry_paf_survival.csv      point, same schema as table_percountry_paf.csv
    table_percountry_paf_ci_survival.csv   with intervals, schema of table_percountry_paf_ci.csv
    table_regional_paf_survival.csv        schema of table_regional_paf.csv
    survival_country_pafs/                 point, shard draws, diagnostics, manifest

Run:  python -m src.inferential.survival_country_pafs --shard 0 --shards 3
      python -m src.inferential.survival_country_pafs --summarise
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from config.paths import OUTPUT_DIR, RNG_SEED
from src.inferential import phase3_survival as p3
from src.inferential.phase2_analysis import _country_draws, _read_pilot
from src.inferential.sens2026_common import SCENARIOS, append_rows, git_head
from src.inferential.sens2026_survival import PILOT, build, check_published, risks

MV = OUTPUT_DIR / "phase4" / "manuscript"
RUN_DIR = MV / "survival_country_pafs"
REFIT_SUMMARY = OUTPUT_DIR / "phase4" / "sensitivity_2026-09" / "survival_main" / "summary.csv"


def country_regions(countries: list[str]) -> dict[str, str]:
    frame = _read_pilot(PILOT)[["country_code", "who_region"]]
    mapping = (frame.assign(country_code=frame["country_code"].astype(str))
               .groupby("country_code")["who_region"].first().astype(str).to_dict())
    missing = [c for c in countries if c not in mapping]
    if missing:
        raise SystemExit(f"no region for {missing}")
    return {c: mapping[c] for c in countries}


def evaluate(fit, model, designs, st, mult, region_of) -> list[dict]:
    """Pooled, regional and per-country PAF rows for one coefficient vector."""
    factual = risks(fit, model, designs["factual"])
    cf = {s: risks(fit, model, designs[s]) for s in SCENARIOS}
    rows = [{"level": "pooled", "unit": "all", "estimand": e, "scenario": s, "paf_pct": v}
            for (e, s), v in st.pafs(mult, factual, cf).items()]
    f_c = st._sum(factual).sum(axis=1)
    cf_c = {s: st._sum(cf[s]).sum(axis=1) for s in SCENARIOS}
    present = mult > 0
    for i, country in enumerate(st.countries):
        if not present[i] or f_c[i] <= 0:
            continue
        for s in SCENARIOS:
            rows.append({"level": "country", "unit": country, "estimand": "own_exposures",
                         "scenario": s, "paf_pct": 100.0 * (f_c[i] - cf_c[s][i]) / f_c[i]})
    regions = np.array([region_of[c] for c in st.countries])
    for region in sorted(set(regions)):
        m = mult * (regions == region)
        if m.sum() == 0:
            continue
        f_r = float(m @ f_c)
        for s in SCENARIOS:
            rows.append({"level": "region", "unit": region, "estimand": "own_exposures",
                         "scenario": s, "paf_pct": 100.0 * (f_r - float(m @ cf_c[s])) / f_r})
    return rows


def run(shard: int, shards: int, reps: int, limit: int | None) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    fit, patterns, designs, st, meta = build("main")
    region_of = country_regions(st.countries)
    ones = np.ones(st.C)
    point_rows = evaluate(fit, fit.model, designs, st, ones, region_of)
    pooled = {(r["estimand"], r["scenario"]): r["paf_pct"]
              for r in point_rows if r["level"] == "pooled"}
    check_published(pooled)
    births = patterns.groupby(patterns["country_code"].astype(str))["n_births"].sum()
    if shard == 0:
        pd.DataFrame(point_rows).to_csv(RUN_DIR / "point.csv", index=False)
        (RUN_DIR / "point_meta.json").write_text(json.dumps(
            {**meta, "region_of": region_of,
             "n_births_by_country": {k: int(v) for k, v in births.items()},
             "created_at_utc": datetime.now(timezone.utc).isoformat()},
            indent=2, default=str), encoding="utf-8")

    draws = _country_draws(patterns[["country_code"]], bootstrap_reps=reps, seed=RNG_SEED)
    boot_path = RUN_DIR / f"boot_shard{shard}.csv"
    diag_path = RUN_DIR / f"diag_shard{shard}.csv"
    done = set(pd.read_csv(diag_path)["replicate"].astype(int)) if diag_path.exists() else set()
    country_of_pattern = np.fromiter(
        (st.country_index[c] for c in patterns["country_code"].astype(str)),
        np.int64, len(patterns))
    age_keep = fit.keep_columns[fit.static_feature_count:]
    warm = copy.deepcopy(fit.model)
    new = 0
    for replicate, draw in enumerate(draws):
        if replicate % shards != shard or replicate in done:
            continue
        if limit is not None and new >= limit:
            break
        t0 = time.time()
        mult = st.multiplicity(draw)
        try:
            model = p3._solve_factorized(
                fit.static_matrix, fit.cells, age_keep, feature_scale=fit.feature_scale,
                likelihood_backend=fit.likelihood_backend,
                pattern_multiplicity=mult[country_of_pattern], initial_model=warm,
                max_iter=2000, tol=1e-6)
            n_iter = int(np.max(model.n_iter_))
            if (not model.success or n_iter >= model.max_iter
                    or model.max_abs_scaled_gradient > p3._BOOTSTRAP_MAX_ABS_SCALED_GRADIENT):
                raise RuntimeError(f"{model.message}; gradient {model.max_abs_scaled_gradient:.2e}")
            rows = evaluate(fit, model, designs, st, mult, region_of)
            append_rows(boot_path, [{"replicate": replicate, **r} for r in rows])
            append_rows(diag_path, [{"replicate": replicate, "success": True, "n_iter": n_iter,
                                     "seconds": round(time.time() - t0, 1), "error": ""}])
            warm = model
        except Exception as exc:  # recorded, not fatal: intervals use successes
            append_rows(diag_path, [{"replicate": replicate, "success": False, "n_iter": 0,
                                     "seconds": round(time.time() - t0, 1),
                                     "error": f"{type(exc).__name__}: {exc}"}])
            warm = copy.deepcopy(fit.model)
        new += 1
        print(f"[surv-country] shard {shard} replicate {replicate} {time.time() - t0:.0f}s",
              flush=True)


def _interval(values: pd.Series) -> tuple[float, float]:
    v = pd.to_numeric(values, errors="coerce").dropna()
    if not len(v):
        return np.nan, np.nan
    return round(float(v.quantile(0.025)), 2), round(float(v.quantile(0.975)), 2)


def summarise(reps: int) -> None:
    point = pd.read_csv(RUN_DIR / "point.csv")
    meta = json.loads((RUN_DIR / "point_meta.json").read_text(encoding="utf-8"))
    draws = pd.concat([pd.read_csv(p) for p in sorted(RUN_DIR.glob("boot_shard*.csv"))],
                      ignore_index=True)
    diag = pd.concat([pd.read_csv(p) for p in sorted(RUN_DIR.glob("diag_shard*.csv"))],
                     ignore_index=True).drop_duplicates("replicate", keep="last")
    draws = draws.drop_duplicates(["replicate", "level", "unit", "estimand", "scenario"])
    n_ok = int(diag["success"].astype(bool).sum())
    groups = {k: g["paf_pct"] for k, g in
              draws.groupby(["level", "unit", "estimand", "scenario"])}

    def ci(level, unit, estimand, scenario):
        return _interval(groups.get((level, unit, estimand, scenario), pd.Series(dtype=float)))

    region_of = meta["region_of"]
    births = meta["n_births_by_country"]
    pc_rows, ci_rows = [], []
    for country, g in point[point.level == "country"].groupby("unit"):
        p = g.set_index("scenario")["paf_pct"]
        pc_rows.append({"country_code": country, "who_region": region_of[country],
                        "n_births": births[country],
                        **{f"paf_{s}": round(float(p[s]), 2) for s in SCENARIOS}})
        n_draws = int(draws[(draws.level == "country") & (draws.unit == country)
                            & (draws.scenario == "joint")]["replicate"].nunique())
        entry = {"country_code": country, "n_draws": n_draws,
                 "pct_replicates_present": round(100.0 * n_draws / max(n_ok, 1), 1)}
        for s in SCENARIOS:
            lo, hi = ci("country", country, "own_exposures", s)
            entry.update({f"paf_{s}": round(float(p[s]), 2), f"paf_{s}_lo": lo,
                          f"paf_{s}_hi": hi})
        ci_rows.append(entry)
    pc = pd.DataFrame(pc_rows).sort_values("paf_joint", ascending=False)
    pc.to_csv(MV / "table_percountry_paf_survival.csv", index=False)
    pd.DataFrame(ci_rows).set_index("country_code").loc[pc["country_code"]].reset_index().to_csv(
        MV / "table_percountry_paf_ci_survival.csv", index=False)

    reg_rows = []
    n_countries = pd.Series(region_of).value_counts()
    for r in point[point.level == "region"].itertuples(index=False):
        lo, hi = ci("region", r.unit, r.estimand, r.scenario)
        reg_rows.append({"region": r.unit, "scenario": r.scenario,
                         "n_countries": int(n_countries[r.unit]),
                         "paf_pct": round(float(r.paf_pct), 2), "ci_lo": lo, "ci_hi": hi})
    pd.DataFrame(reg_rows).sort_values(["region", "scenario"]).to_csv(
        MV / "table_regional_paf_survival.csv", index=False)

    pooled = []
    for r in point[point.level == "pooled"].itertuples(index=False):
        lo, hi = ci("pooled", "all", r.estimand, r.scenario)
        pooled.append({"estimand": r.estimand, "scenario": r.scenario,
                       "paf_pct": round(float(r.paf_pct), 2), "ci_lo": lo, "ci_hi": hi})
    pooled = pd.DataFrame(pooled)
    pooled.to_csv(RUN_DIR / "summary_pooled.csv", index=False)

    # The pooled intervals must reproduce the September refit run (same draws).
    ref = pd.read_csv(REFIT_SUMMARY)
    cmp = pooled.merge(ref, on=["estimand", "scenario"], suffixes=("", "_ref"))
    worst = float(np.nanmax(np.abs(cmp[["paf_pct", "ci_lo", "ci_hi"]].to_numpy()
                                   - cmp[["paf_pct_ref", "ci_lo_ref", "ci_hi_ref"]].to_numpy())))
    manifest = {**{k: v for k, v in meta.items() if k not in ("region_of", "n_births_by_country")},
                "summarised_at_utc": datetime.now(timezone.utc).isoformat(),
                "git_head": git_head(), "reps": reps,
                "n_replicates_attempted": int(len(diag)), "n_replicates_failed": int(len(diag) - n_ok),
                "pooled_max_abs_diff_vs_sens2026_refit": round(worst, 3),
                "pooled": pooled.to_dict("records")}
    (RUN_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str),
                                           encoding="utf-8")
    print(pooled.to_string(index=False))
    print(f"\npooled max |diff| vs sens2026 refit summary: {worst:.3f} points; "
          f"{n_ok}/{len(diag)} replicates succeeded")
    print(pc.head(15).to_string(index=False))
    if worst > 0.1:
        raise SystemExit("pooled intervals do not reproduce the September refit run")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--summarise", action="store_true")
    args = ap.parse_args()
    if args.summarise:
        summarise(args.reps)
    else:
        run(args.shard, args.shards, args.reps, args.limit)


if __name__ == "__main__":
    main()
