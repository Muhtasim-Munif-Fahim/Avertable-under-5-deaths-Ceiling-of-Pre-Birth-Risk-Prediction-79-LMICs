"""Exact-binary sensitivity analyses for the 2026-09 reviewer round.

Two models on the published primary exact-binary surface
(``exact_wave_fe_complete_wealth``: survey-weighted, survey fixed effects,
complete-case wealth):

``main``
    The published specification, re-run so that the contemporary exact-binary
    estimand gets a bootstrap that refits the model on every replicate (the
    published contemporary interval held the surface fixed). Its long-window and
    contemporary points must reproduce the published values.

``interaction``
    Education x spacing entered as one combined categorical term.

The fit is the published grouped Bernoulli likelihood (identical design rows
collapsed to sufficient statistics), so a bootstrap replicate refits on pattern
weights multiplied by the country draw -- the estimator of
``phase2_analysis._bootstrap_grouped_surface``. Counterfactual designs are
re-encoded from shifted pattern frames, never built by coefficient-name lookup.

Outputs: output/phase4/sensitivity_2026-09/binary_<model>/
Run:     python -m src.inferential.sens2026_binary --model interaction --reps 1000 --shard 0 --shards 2
         python -m src.inferential.sens2026_binary --model interaction --summarise
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from config.paths import HARMONIZED_DIR, OUTPUT_DIR, RNG_SEED
from src.inferential import phase2_analysis as p2
from src.inferential.run_fulldata_recentdecade_paf import _cells as binary_cells
from src.inferential.sens2026_common import (
    OUT_ROOT, RECENT_HI, RECENT_LO, SCENARIOS, Standardiser, append_rows,
    completed_replicates, shard_paths, summarise,
)

PILOT = HARMONIZED_DIR / "global_analysis_v2_candidate.parquet"
BASE_FEATURES = list(p2._BASE_FEATURES)
ORIGINAL_PREPARE = p2._prepare_states
EXPOSURES = ("edu_cat", "bi_model_cat")
PUBLISHED_LONG = OUTPUT_DIR / "phase4" / "phase2_publication" / "phase2_pafs.csv"
# The published contemporary exact-binary estimate (26.39%). Not
# table_recentdecade_paf.csv, which holds an older capped-sample run.
PUBLISHED_NOW = OUTPUT_DIR / "phase4" / "manuscript" / "table_recentdecade_paf_reweighted.csv"


def _prepare_with_edu_bi(frame: pd.DataFrame, *, legacy: bool) -> pd.DataFrame:
    states = ORIGINAL_PREPARE(frame, legacy=legacy)
    states["edu_bi"] = states["edu_cat"].astype(str) + "|" + states["bi_model_cat"].astype(str)
    return states


def fit_surface(frame: pd.DataFrame, model_name: str):
    if model_name == "interaction":
        # _fit_surface reads both module globals at call time.
        p2._BASE_FEATURES = [f for f in BASE_FEATURES if f not in EXPOSURES] + ["edu_bi"]
        p2._prepare_states = _prepare_with_edu_bi
    try:
        return p2._fit_surface(
            surface=f"sens2026_{model_name}", role="sens2026", frame=frame,
            outcome_column="died_under5_complete", legacy_states=False,
            fixed_effects="survey", wealth_mode="complete", weighted=True)
    finally:
        p2._BASE_FEATURES = list(BASE_FEATURES)
        p2._prepare_states = ORIGINAL_PREPARE


def shift(frame: pd.DataFrame, scenario: str, model_name: str) -> pd.DataFrame:
    """Policy-eligible shift; the masks of ``phase2_analysis._counterfactual``."""
    cf = frame.copy()
    education = pd.to_numeric(cf["edu_level"], errors="coerce").isin([0, 1]).to_numpy(bool)
    order = pd.to_numeric(cf["birth_order"], errors="coerce")
    interval = pd.to_numeric(cf["birth_interval_months"], errors="coerce")
    spacing = (order.gt(1) & interval.ge(0) & interval.lt(24)).to_numpy(bool)
    edu_on = scenario in ("education", "joint")
    sp_on = scenario in ("spacing", "joint")
    if model_name == "main":
        for column in EXPOSURES:
            cf[column] = cf[column].astype(str)
        if edu_on:
            cf.loc[education, "edu_cat"] = "Secondary"
        if sp_on:
            cf.loc[spacing, "bi_model_cat"] = "24-47 mo"
    else:
        parts = cf["edu_bi"].astype(str).str.split("|", n=1, expand=True)
        edu = parts[0].to_numpy(object)
        spc = parts[1].to_numpy(object)
        if edu_on:
            edu = np.where(education, "Secondary", edu)
        if sp_on:
            spc = np.where(spacing, "24-47 mo", spc)
        cf["edu_bi"] = pd.Series(edu, index=cf.index).astype(str) + "|" + \
            pd.Series(spc, index=cf.index).astype(str)
    return cf


def design(result, frame: pd.DataFrame):
    matrix = result.encoder.transform(frame[result.features].astype(str)).tocsr()
    return matrix[:, result.keep_columns].tocsr()


def build(model_name: str):
    print(f"[sens-bin:{model_name}] loading cohort and fitting ...", flush=True)
    frame = _read = p2._read_pilot(PILOT)
    t0 = time.time()
    result = fit_surface(frame, model_name)
    if not result.converged or result.rank_deficient:
        raise SystemExit(f"point fit unusable: converged={result.converged}, "
                         f"rank_deficient={result.rank_deficient}")
    print(f"[sens-bin:{model_name}] point fit {time.time() - t0:.0f}s, "
          f"{len(result.grouped.frame):,} patterns", flush=True)
    patterns = result.grouped.frame
    likelihood = result.grouped.likelihood
    designs = {"factual": design(result, patterns)}
    if designs["factual"].shape != likelihood.pattern_matrix.shape or \
            abs(designs["factual"] - likelihood.pattern_matrix).max() > 0:
        raise SystemExit("re-encoded factual design differs from the fitted design")
    for scenario in SCENARIOS:
        designs[scenario] = design(result, shift(patterns, scenario, model_name))

    by = pd.to_numeric(frame["birth_year"], errors="coerce")
    sw = pd.to_numeric(frame["sample_weight"], errors="coerce")
    countries = set(patterns["country_code"].astype(str))
    recent_mask = (by.ge(RECENT_LO) & by.le(RECENT_HI)
                   & frame["country_code"].astype(str).isin(countries)
                   & sw.notna() & sw.gt(0)).fillna(False).to_numpy(bool)
    recent = frame.loc[recent_mask].reset_index(drop=True)
    del frame, _read
    standardiser = Standardiser(
        patterns["country_code"].astype(str).to_numpy(),
        binary_cells(patterns["edu_level"], patterns["birth_order"],
                     patterns["birth_interval_months"]),
        likelihood.pattern_weight,
        recent["country_code"].astype(str).to_numpy(),
        binary_cells(recent["edu_level"], recent["birth_order"],
                     recent["birth_interval_months"]),
        pd.to_numeric(recent["sample_weight"], errors="coerce").to_numpy(float),
    )
    meta = {"model": model_name, "features": result.features,
            "n_births": int(len(result.frame)), "n_patterns": int(len(patterns)),
            "n_params": int(result.model.coef_.shape[1] + 1), "seed": RNG_SEED}
    return result, patterns, designs, standardiser, meta


def evaluate(model, designs, standardiser, mult) -> dict:
    coef, intercept = model.coef_[0], float(model.intercept_[0])
    factual = expit(np.asarray(designs["factual"] @ coef).ravel() + intercept)
    cf = {s: expit(np.asarray(designs[s] @ coef).ravel() + intercept) for s in SCENARIOS}
    return standardiser.pafs(mult, factual, cf)


def check_published(point: dict) -> None:
    long = pd.read_csv(PUBLISHED_LONG)
    row = long[(long.surface == "exact_wave_fe_complete_wealth")
               & (long.contract == "policy_eligible") & (long.scenario == "joint")]
    pub_long = float(row["paf_pct"].iloc[0])
    now = pd.read_csv(PUBLISHED_NOW)
    pub_now = float(now.loc[now.scenario == "joint", "paf_recent_prevalence_pct"].iloc[0])
    got_long, got_now = point[("long_window", "joint")], point[("contemporary", "joint")]
    print(f"[sens-bin:main] reproduction: long-window {got_long:.3f} vs published "
          f"{pub_long:.3f}; contemporary {got_now:.3f} vs published {pub_now:.3f}", flush=True)
    if abs(got_long - pub_long) > 0.05 or abs(got_now - pub_now) > 0.05:
        raise SystemExit("main model does not reproduce the published PAFs; stopping")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=["main", "interaction"], required=True)
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--summarise", action="store_true")
    args = ap.parse_args()
    out_dir = OUT_ROOT / f"binary_{args.model}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.summarise:
        point = pd.read_csv(out_dir / "point.csv")
        meta = pd.read_json(out_dir / "point_meta.json", typ="series").to_dict()
        print(summarise(out_dir, point, meta).to_string(index=False))
        return

    result, patterns, designs, standardiser, meta = build(args.model)
    ones = np.ones(standardiser.C)
    point = evaluate(result.model, designs, standardiser, ones)
    if args.model == "main":
        check_published(point)
    pd.DataFrame([{"estimand": e, "scenario": s, "paf_pct": v}
                  for (e, s), v in point.items()]).to_csv(out_dir / "point.csv", index=False)
    pd.Series({**meta, "created_at_utc": datetime.now(timezone.utc).isoformat()}).to_json(
        out_dir / "point_meta.json", indent=2, default_handler=str)
    for (e, s), v in point.items():
        print(f"[sens-bin:{args.model}] point {e:12s} {s:9s} {v:6.2f}%", flush=True)

    likelihood = result.grouped.likelihood
    pattern_country = np.fromiter(
        (standardiser.country_index[c] for c in patterns["country_code"].astype(str)),
        np.int64, len(patterns))
    pseudo_country = pattern_country[likelihood.pseudo_pattern_ids]
    draws = p2._country_draws(patterns[["country_code"]], bootstrap_reps=args.reps,
                              seed=RNG_SEED)
    done = completed_replicates(out_dir, args.shard)
    boot_path, diag_path = shard_paths(out_dir, args.shard)

    def fresh():
        m = copy.deepcopy(result.model)
        m.warm_start, m.max_iter, m.tol = True, 2000, 1e-5
        return m

    model, new = fresh(), 0
    for replicate, draw in enumerate(draws):
        if replicate % args.shards != args.shard or replicate in done:
            continue
        if args.limit is not None and new >= args.limit:
            break
        t0 = time.time()
        mult = standardiser.multiplicity(draw)
        weight = likelihood.pseudo_weight * mult[pseudo_country]
        positive = weight > 0
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(likelihood.pseudo_matrix[positive],
                          likelihood.pseudo_outcome[positive], sample_weight=weight[positive])
            if any(issubclass(c.category, ConvergenceWarning) for c in caught) or \
                    int(np.max(model.n_iter_)) >= model.max_iter:
                raise RuntimeError("did not converge")
            values = evaluate(model, designs, standardiser, mult)
            append_rows(boot_path, [{"replicate": replicate, "estimand": e, "scenario": s,
                                     "paf_pct": v} for (e, s), v in values.items()])
            append_rows(diag_path, [{"replicate": replicate, "success": True,
                                     "n_iter": int(np.max(model.n_iter_)),
                                     "seconds": round(time.time() - t0, 1), "error": ""}])
        except Exception as exc:
            append_rows(diag_path, [{"replicate": replicate, "success": False, "n_iter": 0,
                                     "seconds": round(time.time() - t0, 1),
                                     "error": f"{type(exc).__name__}: {exc}"}])
            model = fresh()
        new += 1
        print(f"[sens-bin:{args.model}] shard {args.shard} replicate {replicate} "
              f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
