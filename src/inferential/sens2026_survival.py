"""Survival-surface sensitivity analyses for the 2026-09 reviewer round.

Two models on the published primary survival surface (``phase3_survival``):

``main``
    The published specification, re-run so that the CONTEMPORARY estimand (the
    2018-2024 re-standardised headline) gets a bootstrap that refits the model on
    every replicate. The published contemporary interval held the fitted surface
    fixed and resampled only exposure composition, so it omitted coefficient
    uncertainty. The long-window point and the contemporary point must reproduce
    the published values; the script refuses to continue otherwise.

``interaction``
    Education x spacing entered as one combined categorical term (a saturated
    interaction), so the joint counterfactual is no longer additive on the logit
    scale by construction.

Counterfactual risks are computed by re-encoding the shifted covariate patterns
through the fitted encoder -- never by looking coefficients up by name, which
silently returns zero for a term the model does not contain.

Bootstrap: whole countries resampled with replacement (the published draws, same
seed), the model refitted from a warm start on each replicate, and the
contemporary reweighting recomputed within the same replicate. Replicates can be
split across processes with ``--shard i --shards n``; each shard checkpoints after
every replicate and resumes where it stopped. ``--summarise`` merges the shards.

Outputs: output/phase4/sensitivity_2026-09/survival_<model>/
Run:     python -m src.inferential.sens2026_survival --model main --reps 1000 --shard 0 --shards 2
         python -m src.inferential.sens2026_survival --model main --summarise
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from config.paths import HARMONIZED_DIR, OUTPUT_DIR, RNG_SEED
from src.inferential import phase3_survival as p3
from src.inferential.phase2_analysis import _country_draws, _prepare_states, _read_pilot
from src.inferential.sens2026_common import (
    OUT_ROOT, RECENT_HI, RECENT_LO, SCENARIOS, Standardiser, append_rows,
    completed_replicates, shard_paths, summarise,
)

PILOT = HARMONIZED_DIR / "global_analysis_v2_candidate.parquet"
BASE_STATIC = list(p3._STATIC_FEATURES)
EXPOSURES = ("edu_cat", "bi_model_cat")
PUBLISHED_LONG = OUTPUT_DIR / "phase4" / "phase3_publication" / "phase3_bootstrap_summary.csv"
PUBLISHED_NOW = OUTPUT_DIR / "phase4" / "manuscript" / "table_recentdecade_paf_survival.csv"


def features_for(model: str) -> list[str]:
    if model == "main":
        return list(BASE_STATIC)
    if model == "interaction":
        return [f for f in BASE_STATIC if f not in EXPOSURES] + ["edu_bi"]
    raise ValueError(f"unknown model {model!r}")


def with_strings(patterns: pd.DataFrame) -> pd.DataFrame:
    out = patterns.copy()
    for column in EXPOSURES:
        out[column] = out[column].astype(str)
    out["edu_bi"] = out["edu_cat"] + "|" + out["bi_model_cat"]
    return out


def shift(patterns: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """Policy-eligible shift; the rule of ``phase3_survival._scenario_patterns``."""
    changed = patterns.copy()
    if scenario in ("education", "joint"):
        changed.loc[changed["edu_cat"].isin(["None", "Primary"]), "edu_cat"] = "Secondary"
    if scenario in ("spacing", "joint"):
        changed.loc[changed["bi_model_cat"].eq("<24 mo"), "bi_model_cat"] = "24-47 mo"
    changed["edu_bi"] = changed["edu_cat"] + "|" + changed["bi_model_cat"]
    return changed


def fit_surface(cells, patterns: pd.DataFrame, feats: list[str]):
    # _fit_model and _static_categories_for read the module global at call time.
    p3._STATIC_FEATURES = feats
    try:
        return p3._fit_model(cells, patterns, compute_backend="auto")
    finally:
        p3._STATIC_FEATURES = list(BASE_STATIC)


def static_design(fit, frame: pd.DataFrame, feats: list[str]):
    matrix = fit.static_encoder.transform(frame[feats].astype(str)).tocsr()
    return matrix[:, fit.keep_columns[: fit.static_feature_count]].tocsr()


def risks(fit, model, matrix) -> np.ndarray:
    n_static = matrix.shape[1]
    coef = model.coef_[0]
    eta = np.asarray(matrix @ coef[:n_static]).ravel() + float(model.intercept_[0])
    age = np.zeros(60)
    age_keep = fit.keep_columns[fit.static_feature_count:]
    age[np.flatnonzero(age_keep) + 1] = coef[n_static:]
    return fit.likelihood_backend.risks_from_eta(eta, age)


def build(model_name: str):
    feats = features_for(model_name)
    print(f"[sens-surv:{model_name}] preparing survival cohort ...", flush=True)
    prepared = p3._prepare(PILOT)
    patterns = with_strings(prepared.patterns)
    print(f"[sens-surv:{model_name}] {prepared.n_births:,} births, {prepared.n_countries} "
          f"countries, {len(patterns):,} patterns; fitting ...", flush=True)
    t0 = time.time()
    fit = fit_surface(prepared.cells, patterns, feats)
    if not fit.converged:
        raise SystemExit(f"point fit did not converge: {fit.warning_text}")
    print(f"[sens-surv:{model_name}] point fit {time.time() - t0:.0f}s, "
          f"{fit.n_iter} iterations, backend {fit.likelihood_backend.name}", flush=True)

    designs = {"factual": static_design(fit, patterns, feats)}
    if designs["factual"].shape != fit.factual_static_matrix.shape or \
            abs(designs["factual"] - fit.factual_static_matrix).max() > 0:
        raise SystemExit("re-encoded factual design differs from the fitted design")
    for scenario in SCENARIOS:
        designs[scenario] = static_design(fit, shift(patterns, scenario), feats)

    frame = _read_pilot(PILOT)
    by = pd.to_numeric(frame["birth_year"], errors="coerce")
    sw = pd.to_numeric(frame["sample_weight"], errors="coerce")
    countries = set(patterns["country_code"].astype(str))
    recent_mask = (by.ge(RECENT_LO) & by.le(RECENT_HI)
                   & frame["country_code"].astype(str).isin(countries)
                   & sw.notna() & sw.gt(0)).fillna(False).to_numpy(bool)
    recent = frame.loc[recent_mask].reset_index(drop=True)
    del frame
    rstates = _prepare_states(recent, legacy=False)
    standardiser = Standardiser(
        patterns["country_code"].astype(str).to_numpy(),
        (patterns["edu_cat"] + "|" + patterns["bi_model_cat"]).to_numpy(object),
        patterns["birth_weight"].to_numpy(float),
        recent["country_code"].astype(str).to_numpy(),
        (rstates["edu_cat"].astype(str) + "|" + rstates["bi_model_cat"].astype(str)).to_numpy(object),
        pd.to_numeric(recent["sample_weight"], errors="coerce").to_numpy(float),
    )
    loss = float(fit.model.objective) * float(prepared.cells.risk_weight.sum())
    meta = {"model": model_name, "features": feats, "n_births": prepared.n_births,
            "n_patterns": int(len(patterns)), "n_countries": prepared.n_countries,
            "n_params": int(len(fit.model.coef_[0]) + 1), "neg_log_likelihood": loss,
            "backend": fit.likelihood_backend.name, "seed": RNG_SEED}
    return fit, patterns, designs, standardiser, meta


# ---------------------------------------------------------------------------
# Age-band effect modification (point estimate). Education and spacing effects are
# allowed to differ in the neonatal month, months 1-11 and months 12-59. The
# factorised likelihood assumes one static linear predictor per pattern, so each
# pattern is split into three band-specific pseudo-patterns whose person-month
# weights are zero outside their band; the likelihood is unchanged and each band
# gets its own education and spacing terms.
# ---------------------------------------------------------------------------
BANDS = {0: np.arange(0, 1), 1: np.arange(1, 12), 2: np.arange(12, 60)}
EDU_REF, BI_REF = "None", "24-47 mo"
BAND_FEATURES = ["ex_b1", "ex_b2", "bx_b1", "bx_b2"]
BAND_CATEGORIES = {
    "ex_b1": ["ref", "Primary", "Secondary", "Higher", "Missing"],
    "ex_b2": ["ref", "Primary", "Secondary", "Higher", "Missing"],
    "bx_b1": ["ref", "<24 mo", "48+ mo", "Missing non-first", "First/missing"],
    "bx_b2": ["ref", "<24 mo", "48+ mo", "Missing non-first", "First/missing"],
}


def band_columns(frame: pd.DataFrame, band: int) -> pd.DataFrame:
    """Band-specific exposure terms; 'ref' outside the band or at the reference level."""
    out = frame.copy()
    edu = out["edu_cat"].to_numpy(object)
    bi = out["bi_model_cat"].to_numpy(object)
    for k in (1, 2):
        out[f"ex_b{k}"] = np.where((band == k) & (edu != EDU_REF), edu, "ref")
        out[f"bx_b{k}"] = np.where((band == k) & (bi != BI_REF), bi, "ref")
    return out


def band_risks(fit, model, frame: pd.DataFrame, feats: list[str],
               chunk: int = 200_000) -> np.ndarray:
    n_static = int(fit.keep_columns[: fit.static_feature_count].sum())
    coef = model.coef_[0]
    intercept = float(model.intercept_[0])
    age = np.zeros(60)
    age[np.flatnonzero(fit.keep_columns[fit.static_feature_count:]) + 1] = coef[n_static:]
    log_survival = np.zeros(len(frame))
    for band, months in BANDS.items():
        matrix = static_design(fit, band_columns(frame, band), feats)
        eta = np.asarray(matrix @ coef[:n_static]).ravel() + intercept
        for start in range(0, len(frame), chunk):
            block = eta[start:start + chunk, None] + age[None, months]
            log_survival[start:start + chunk] -= np.logaddexp(0.0, block).sum(axis=1)
    return -np.expm1(log_survival)


def run_ageband(out_dir) -> None:
    feats = list(BASE_STATIC) + BAND_FEATURES
    print("[sens-surv:ageband] preparing survival cohort ...", flush=True)
    prepared = p3._prepare(PILOT)
    patterns = with_strings(prepared.patterns)
    rw, ew = prepared.cells.risk_weight, prepared.cells.event_weight
    pseudo_rw, pseudo_ew, pseudo_frames = [], [], []
    for band, months in BANDS.items():
        mask = np.zeros(rw.shape[1], dtype=bool)
        mask[months] = True
        pseudo_rw.append(np.where(mask[None, :], rw, 0.0))
        pseudo_ew.append(np.where(mask[None, :], ew, 0.0))
        pseudo_frames.append(band_columns(patterns, band))
    pseudo_cells = p3._GroupedPersonMonths(risk_weight=np.vstack(pseudo_rw),
                                           event_weight=np.vstack(pseudo_ew))
    del pseudo_rw, pseudo_ew
    pseudo_patterns = pd.concat(pseudo_frames, ignore_index=True)
    added = [k for k in BAND_CATEGORIES if k not in p3._FIXED_CATEGORIES]
    p3._FIXED_CATEGORIES.update({k: BAND_CATEGORIES[k] for k in added})
    p3._STATIC_FEATURES = feats
    try:
        t0 = time.time()
        fit = p3._fit_model(pseudo_cells, pseudo_patterns, compute_backend="auto")
    finally:
        p3._STATIC_FEATURES = list(BASE_STATIC)
        for k in added:
            p3._FIXED_CATEGORIES.pop(k, None)
    if not fit.converged:
        raise SystemExit(f"age-band fit did not converge: {fit.warning_text}")
    print(f"[sens-surv:ageband] fit {time.time() - t0:.0f}s, {fit.n_iter} iterations, "
          f"backend {fit.likelihood_backend.name}", flush=True)

    frame = _read_pilot(PILOT)
    by = pd.to_numeric(frame["birth_year"], errors="coerce")
    sw = pd.to_numeric(frame["sample_weight"], errors="coerce")
    countries = set(patterns["country_code"].astype(str))
    recent = frame.loc[(by.ge(RECENT_LO) & by.le(RECENT_HI)
                        & frame["country_code"].astype(str).isin(countries)
                        & sw.notna() & sw.gt(0)).fillna(False).to_numpy(bool)].reset_index(drop=True)
    del frame
    rstates = _prepare_states(recent, legacy=False)
    standardiser = Standardiser(
        patterns["country_code"].astype(str).to_numpy(),
        (patterns["edu_cat"] + "|" + patterns["bi_model_cat"]).to_numpy(object),
        patterns["birth_weight"].to_numpy(float),
        recent["country_code"].astype(str).to_numpy(),
        (rstates["edu_cat"].astype(str) + "|" + rstates["bi_model_cat"].astype(str)).to_numpy(object),
        pd.to_numeric(recent["sample_weight"], errors="coerce").to_numpy(float),
    )
    factual = band_risks(fit, fit.model, patterns, feats)
    cf = {s: band_risks(fit, fit.model, shift(patterns, s), feats) for s in SCENARIOS}
    point = standardiser.pafs(np.ones(standardiser.C), factual, cf)
    pd.DataFrame([{"estimand": e, "scenario": s, "paf_pct": v}
                  for (e, s), v in point.items()]).to_csv(out_dir / "point.csv", index=False)
    loss = float(fit.model.objective) * float(pseudo_cells.risk_weight.sum())
    meta = {"model": "ageband", "features": feats, "n_patterns": int(len(patterns)),
            "n_params": int(len(fit.model.coef_[0]) + 1), "neg_log_likelihood": loss,
            "backend": fit.likelihood_backend.name,
            "created_at_utc": datetime.now(timezone.utc).isoformat()}
    pd.Series(meta).to_json(out_dir / "point_meta.json", indent=2, default_handler=str)
    for (e, s), v in point.items():
        print(f"[sens-surv:ageband] point {e:12s} {s:9s} {v:6.2f}%", flush=True)


def evaluate(fit, model, designs, standardiser, mult) -> dict:
    factual = risks(fit, model, designs["factual"])
    cf = {s: risks(fit, model, designs[s]) for s in SCENARIOS}
    return standardiser.pafs(mult, factual, cf)


def check_published(point: dict) -> None:
    long = pd.read_csv(PUBLISHED_LONG)
    pub_long = float(long[(long.scenario == "joint") & (long.metric == "paf_pct")]
                     ["point_estimate"].iloc[0])
    now = pd.read_csv(PUBLISHED_NOW)
    pub_now = float(now.loc[now.scenario == "joint", "paf_recent_prevalence_pct"].iloc[0])
    got_long, got_now = point[("long_window", "joint")], point[("contemporary", "joint")]
    print(f"[sens-surv:main] reproduction: long-window {got_long:.3f} vs published "
          f"{pub_long:.3f}; contemporary {got_now:.3f} vs published {pub_now:.3f}", flush=True)
    if abs(got_long - pub_long) > 0.05 or abs(got_now - pub_now) > 0.05:
        raise SystemExit("main model does not reproduce the published PAFs; stopping")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=["main", "interaction", "ageband"], required=True)
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after this many new replicates (timing runs)")
    ap.add_argument("--summarise", action="store_true")
    args = ap.parse_args()
    out_dir = OUT_ROOT / f"survival_{args.model}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.summarise:
        point = pd.read_csv(out_dir / "point.csv")
        meta = pd.read_json(out_dir / "point_meta.json", typ="series").to_dict()
        print(summarise(out_dir, point, meta).to_string(index=False))
        return
    if args.model == "ageband":
        run_ageband(out_dir)
        return

    fit, patterns, designs, standardiser, meta = build(args.model)
    ones = np.ones(standardiser.C)
    point = evaluate(fit, fit.model, designs, standardiser, ones)
    if args.model == "main":
        check_published(point)
    point_frame = pd.DataFrame([{"estimand": e, "scenario": s, "paf_pct": v}
                                for (e, s), v in point.items()])
    point_frame.to_csv(out_dir / "point.csv", index=False)
    pd.Series({**meta, "created_at_utc": datetime.now(timezone.utc).isoformat()}).to_json(
        out_dir / "point_meta.json", indent=2, default_handler=str)
    for (e, s), v in point.items():
        print(f"[sens-surv:{args.model}] point {e:12s} {s:9s} {v:6.2f}%", flush=True)

    draws = _country_draws(patterns[["country_code"]], bootstrap_reps=args.reps, seed=RNG_SEED)
    done = completed_replicates(out_dir, args.shard)
    boot_path, diag_path = shard_paths(out_dir, args.shard)
    country_of_pattern = np.fromiter(
        (standardiser.country_index[c] for c in patterns["country_code"].astype(str)),
        np.int64, len(patterns))
    age_keep = fit.keep_columns[fit.static_feature_count:]
    warm = copy.deepcopy(fit.model)
    new = 0
    for replicate, draw in enumerate(draws):
        if replicate % args.shards != args.shard or replicate in done:
            continue
        if args.limit is not None and new >= args.limit:
            break
        t0 = time.time()
        mult = standardiser.multiplicity(draw)
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
            values = evaluate(fit, model, designs, standardiser, mult)
            append_rows(boot_path, [{"replicate": replicate, "estimand": e, "scenario": s,
                                     "paf_pct": v} for (e, s), v in values.items()])
            append_rows(diag_path, [{"replicate": replicate, "success": True, "n_iter": n_iter,
                                     "seconds": round(time.time() - t0, 1), "error": ""}])
            warm = model
        except Exception as exc:  # recorded, not fatal: the interval uses successes
            append_rows(diag_path, [{"replicate": replicate, "success": False, "n_iter": 0,
                                     "seconds": round(time.time() - t0, 1),
                                     "error": f"{type(exc).__name__}: {exc}"}])
            warm = copy.deepcopy(fit.model)
        new += 1
        print(f"[sens-surv:{args.model}] shard {args.shard} replicate {replicate} "
              f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
