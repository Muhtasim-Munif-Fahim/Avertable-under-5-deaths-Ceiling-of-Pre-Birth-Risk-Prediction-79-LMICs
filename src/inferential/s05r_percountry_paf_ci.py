"""s05r_percountry_paf_ci.py -- per-country PAF confidence intervals (v2).

Adds uncertainty to the per-country population attributable fractions that
`derived_pafs.py` reports as point estimates only.

Why the existing bootstrap cannot do this
-----------------------------------------
`derived_pafs._boot_paf()` resamples countries over *per-country weighted
sums* with the fitted surface held fixed. For a single country that degenerates
to a point mass -- there is nothing to resample. Uncertainty in a one-country
PAF has to come from uncertainty in the fitted surface, which means refitting.

What this estimates
-------------------
A country-cluster **refit** bootstrap. In each replicate the 77 countries are
resampled with replacement, the primary surface is refitted, and every present
country's PAF is re-scored **on its own original rows with its own original
weights**. The interval therefore describes:

    uncertainty in the fitted global surface, evaluated at that country's
    fixed exposure profile

It is NOT an independent national estimate and must never be captioned as one.

Design notes
------------
1. *Resampling by reweighting.* Drawing country c m_c times is equivalent, for a
   weighted logistic MLE, to multiplying the weights of c's rows by m_c. That
   avoids materialising a duplicated 7.65M-row design.

2. *A country is scored only where present.* The surface carries 193 survey-wave
   fixed effects; when a country is absent its surveys' FE columns are
   unidentified. Under a cluster bootstrap each country is absent from ~37% of
   replicates, so each interval rests on ~63% of B. The realised count is
   reported per country in `n_draws`.

3. *The counterfactual delta is precomputed as an index lookup.*
   `phase2_analysis._counterfactual_eta_delta` rebuilds the delta from pandas
   columns on every call. It is really just
   `coef[target_level] - coef[current_level]` on a fixed row mask, so this module
   precomputes (mask, target index, per-row current index) once and evaluates the
   delta as a vectorised gather. `_selftest_delta_operator` asserts the operator
   reproduces the reference implementation exactly, at both the point estimate
   and a perturbed coefficient vector, before any replicate runs. This is what
   makes the replicate loop picklable -- and therefore parallelisable.

4. *Parallel across replicates.* Replicates are independent, and the per-replicate
   work is single-threaded scipy sparse arithmetic, so the job was previously
   using one of 32 logical cores. The parent fits once and writes a compact
   payload (~1.1 GB: the sparse design, outcome, weights, country index and the
   delta operators); workers load it once each and run a disjoint slice of
   replicate indices. Each replicate is seeded from `(RNG_SEED, b)` so the draws
   are reproducible and independent of how the work is partitioned.

Validation
----------
Point estimates recomputed here must reproduce `table_percountry_paf.csv`;
the run asserts this before bootstrapping.

Output
------
  output/phase4/manuscript/table_percountry_paf_ci.csv

Run:  python -m src.inferential.s05r_percountry_paf_ci --reps 1000 --workers 14
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.special import expit

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from config.paths import HARMONIZED_DIR, OUTPUT_DIR, RNG_SEED  # noqa: E402

OUT = OUTPUT_DIR / "phase4" / "manuscript"
CACHE = OUTPUT_DIR / "_paf_ci_payload"
SCENARIOS = ("education", "spacing", "joint")

# Worker-global state, populated by _init_worker exactly once per process.
_W: dict = {}


# --------------------------------------------------------------------------- #
# Counterfactual delta as a precomputed index operator                         #
# --------------------------------------------------------------------------- #

def _build_delta_operator(result) -> dict[str, dict]:
    """Precompute (mask, target index, per-row current index) per scenario.

    Mirrors `_counterfactual_eta_delta(..., contract="policy_eligible")`:
      education: rows with edu_level in {0,1} move to edu_cat "Secondary"
      spacing  : rows with birth_order>1 and 0 <= interval < 24 move to
                 bi_model_cat "24-47 mo"
    A level absent from `feature_names` is a reference level and contributes 0,
    which is encoded as index -1 and resolved through a zero-padded gather.
    """
    names = {str(n): i for i, n in enumerate(result.feature_names)}
    frame = result.frame
    n = len(frame)

    edu_level = pd.to_numeric(frame["edu_level"], errors="coerce")
    order = pd.to_numeric(frame["birth_order"], errors="coerce")
    interval = pd.to_numeric(frame["birth_interval_months"], errors="coerce")

    def col_index(feature: str, level: str) -> int:
        return names.get(f"{feature}_{level}", -1)

    def per_row(feature: str, series: pd.Series) -> np.ndarray:
        cur = series.astype(str).to_numpy()
        idx = np.full(n, -1, dtype=np.int64)
        for level in np.unique(cur):
            j = col_index(feature, str(level))
            if j >= 0:
                idx[cur == level] = j
        return idx

    ops: dict[str, dict] = {}
    ops["education"] = {
        "mask": edu_level.isin([0, 1]).to_numpy(dtype=bool),
        "target": col_index("edu_cat", "Secondary"),
        "current": per_row("edu_cat", frame["edu_cat"]),
    }
    ops["spacing"] = {
        "mask": (order.gt(1) & interval.ge(0) & interval.lt(24)).to_numpy(dtype=bool),
        "target": col_index("bi_model_cat", "24-47 mo"),
        "current": per_row("bi_model_cat", frame["bi_model_cat"]),
    }
    return ops


def _delta(ops: dict, scenario: str, coef: np.ndarray) -> np.ndarray:
    """Vectorised evaluation of the counterfactual logit shift."""
    padded = np.concatenate([coef, [0.0]])          # index -1 -> 0.0
    out = np.zeros(len(next(iter(ops.values()))["mask"]), dtype=float)
    parts = ("education", "spacing") if scenario == "joint" else (scenario,)
    for part in parts:
        op = ops[part]
        m = op["mask"]
        tgt = padded[op["target"]] if op["target"] >= 0 else 0.0
        out[m] += tgt - padded[op["current"][m]]
    return out


def _selftest_delta_operator(result, ops, beta0) -> None:
    """The operator must match the reference implementation bit-for-bit."""
    from src.inferential.phase2_analysis import _counterfactual_eta_delta
    rng = np.random.default_rng(0)
    perturbed = beta0[1:] + rng.normal(0, 0.05, size=len(beta0) - 1)
    for coef in (beta0[1:], perturbed):
        for s in SCENARIOS:
            ref = _counterfactual_eta_delta(result, s, "policy_eligible", coef)[0]
            mine = _delta(ops, s, coef)
            worst = float(np.max(np.abs(ref - mine)))
            assert worst < 1e-12, f"delta operator mismatch for {s}: {worst}"


# --------------------------------------------------------------------------- #
# Weighted logistic refit                                                      #
# --------------------------------------------------------------------------- #

def _sparse_irls(X, y, w, start, max_iter=3, eta_tol=1e-4, ridge=1e-6):
    """Warm-started Newton/IRLS for weighted logistic on a sparse design.

    Two departures from a textbook implementation, both forced by the design's
    conditioning (condition number ~2.8e8 with 193 survey fixed effects):

    * **Ridge-stabilised step.** Without a small ridge on the Hessian diagonal
      the Newton step acquires a component along a near-null direction and
      marches off at ~1.0 per iteration forever, never meeting a
      coefficient-based tolerance.
    * **Convergence on fitted values.** Those near-null directions barely move
      `eta`, and the counterfactual delta is a *difference* of coefficients
      within a block, so it is invariant to them.

    `max_iter=3` is empirically justified, not guessed:
      * the null replicate reproduces the published pooled exact-binary joint PAF
        of 32.20 -- sklearn gives 32.1988, this gives 32.1951 from iteration 3
        onward, unchanged through iteration 25;
      * on a real resampled replicate (50/77 countries present) the per-country
        PAFs are identical to three decimals from iteration 2 through 10.
    """
    beta = start.copy()
    eta = np.clip(X @ beta, -30.0, 30.0)
    for it in range(max_iter):
        mu = expit(eta)
        wv = np.clip(mu * (1.0 - mu), 1e-10, None) * w
        grad = X.T @ (w * (y - mu))
        H = (X.T @ sp.diags(wv) @ X).toarray()
        H[np.diag_indices_from(H)] += ridge * np.mean(np.diag(H))
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, grad, rcond=None)[0]
        beta = beta + step
        eta_new = np.clip(X @ beta, -30.0, 30.0)
        shift = np.max(np.abs(eta_new - eta))
        eta = eta_new
        if shift < eta_tol:
            break
    return beta, it + 1


def _percountry_paf(country_idx, n_countries, w, f, cf):
    wf = np.bincount(country_idx, weights=w * f, minlength=n_countries)
    wcf = np.bincount(country_idx, weights=w * cf, minlength=n_countries)
    with np.errstate(divide="ignore", invalid="ignore"):
        paf = 100.0 * (1.0 - wcf / wf)
    return np.where(wf > 0, paf, np.nan)


# --------------------------------------------------------------------------- #
# Parallel replicate execution                                                 #
# --------------------------------------------------------------------------- #

def _init_worker(cache_dir: str) -> None:
    """Load the shared payload once per worker process."""
    # scipy/numpy would otherwise each spawn threads inside every worker and
    # oversubscribe the box; the parallelism here is across replicates.
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = "1"
    c = Path(cache_dir)
    arrs = np.load(c / "arrays.npz", allow_pickle=False)
    _W["X"] = sp.load_npz(c / "design.npz").tocsr()
    _W["y"] = arrs["y"]
    _W["w"] = arrs["w"]
    _W["country_idx"] = arrs["country_idx"]
    _W["beta0"] = arrs["beta0"]
    _W["C"] = int(arrs["n_countries"])
    _W["ops"] = {
        "education": {"mask": arrs["edu_mask"], "target": int(arrs["edu_target"]),
                      "current": arrs["edu_current"]},
        "spacing": {"mask": arrs["spa_mask"], "target": int(arrs["spa_target"]),
                    "current": arrs["spa_current"]},
    }


def _run_replicate(b: int):
    """One bootstrap replicate. Returns (b, present mask, {scenario: paf vector})."""
    X, y, w = _W["X"], _W["y"], _W["w"]
    cidx, beta0, C, ops = _W["country_idx"], _W["beta0"], _W["C"], _W["ops"]

    # Seeded from (RNG_SEED, b) so draws do not depend on the partitioning.
    rng = np.random.default_rng([RNG_SEED, b])
    m = np.bincount(rng.integers(0, C, size=C), minlength=C).astype(float)
    present = m > 0

    beta_b, _ = _sparse_irls(X, y, w * m[cidx], beta0)
    eta_b = X @ beta_b
    f_b = expit(eta_b)

    out = {}
    for s in SCENARIOS:
        paf = _percountry_paf(cidx, C, w, f_b, expit(eta_b + _delta(ops, s, beta_b[1:])))
        out[s] = np.where(present, paf, np.nan)
    return b, present, out


def main(reps: int = 1000, workers: int = 14, seed: int = RNG_SEED) -> None:
    import multiprocessing as mp

    print("=" * 68)
    print(f"Per-country PAF confidence intervals (refit bootstrap, B={reps}, "
          f"{workers} workers)")
    print("=" * 68)
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    from src.inferential.phase2_analysis import _fit_surface, _read_pilot

    t0 = time.time()
    frame = _read_pilot(HARMONIZED_DIR / "global_analysis_v2_candidate.parquet")
    result = _fit_surface(
        surface="exact_wave_fe_complete_wealth", role="primary_observed_wealth",
        frame=frame, outcome_column="died_under5_complete", legacy_states=False,
        fixed_effects="survey", wealth_mode="complete", weighted=True)
    print(f"[fit] primary surface in {time.time() - t0:.1f}s")

    X = result.matrix.tocsr()
    n, p = X.shape
    Xf = sp.hstack([sp.csr_matrix(np.ones((n, 1))), X], format="csr")
    y = np.asarray(result.outcome, dtype=float)
    w = np.asarray(result.weights, dtype=float)
    beta0 = np.r_[float(result.model.intercept_[0]), result.model.coef_[0]]
    countries, country_idx = np.unique(
        result.frame["country_code"].astype(str).to_numpy(), return_inverse=True)
    C = len(countries)
    print(f"[data] {n:,} rows x {p} features, {C} countries")

    ops = _build_delta_operator(result)
    _selftest_delta_operator(result, ops, beta0)
    print("[check] delta operator reproduces the reference implementation exactly")

    # Point estimates must match the published per-country PAFs.
    eta0 = Xf @ beta0
    f0 = expit(eta0)
    point = {s: _percountry_paf(country_idx, C, w, f0, expit(eta0 + _delta(ops, s, beta0[1:])))
             for s in SCENARIOS}
    published = pd.read_csv(OUT / "table_percountry_paf.csv").set_index("country_code")
    for s in SCENARIOS:
        mine = pd.Series(point[s], index=countries).round(2)
        worst = (mine - published[f"paf_{s}"].reindex(mine.index)).abs().max()
        assert worst < 0.011, f"point estimates drifted for {s}: max |delta| = {worst}"
    print("[check] point estimates reproduce table_percountry_paf.csv")

    # Shared payload for the worker pool.
    t0 = time.time()
    sp.save_npz(CACHE / "design.npz", Xf)
    np.savez(CACHE / "arrays.npz", y=y, w=w, country_idx=country_idx, beta0=beta0,
             n_countries=np.array(C),
             edu_mask=ops["education"]["mask"], edu_target=np.array(ops["education"]["target"]),
             edu_current=ops["education"]["current"],
             spa_mask=ops["spacing"]["mask"], spa_target=np.array(ops["spacing"]["target"]),
             spa_current=ops["spacing"]["current"])
    payload_gb = sum(f.stat().st_size for f in CACHE.glob("*.npz")) / 1e9
    print(f"[payload] {payload_gb:.2f} GB written in {time.time() - t0:.1f}s")

    del frame, result, X
    import gc
    gc.collect()

    draws = {s: np.full((reps, C), np.nan) for s in SCENARIOS}
    present_count = np.zeros(C, dtype=int)

    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=workers, initializer=_init_worker,
                  initargs=(str(CACHE),)) as pool:
        done = 0
        for b, present, res in pool.imap_unordered(_run_replicate, range(reps),
                                                   chunksize=4):
            present_count += present.astype(int)
            for s in SCENARIOS:
                draws[s][b, :] = res[s]
            done += 1
            if done == 1 or done % 50 == 0 or done == reps:
                el = time.time() - t0
                rate = done / el
                print(f"  {done:>4}/{reps}  {rate:.2f} rep/s  elapsed {el/60:.1f}m  "
                      f"eta {(reps-done)/rate/60:.1f}m", flush=True)
    wall = time.time() - t0
    print(f"[boot] {reps} replicates in {wall/60:.1f}m "
          f"({reps/wall:.2f} rep/s across {workers} workers)")

    rows = []
    for i, c in enumerate(countries):
        entry = {"country_code": c,
                 "n_draws": int(np.isfinite(draws["joint"][:, i]).sum()),
                 "pct_replicates_present": round(100.0 * present_count[i] / reps, 1)}
        for s in SCENARIOS:
            d = draws[s][:, i]
            d = d[np.isfinite(d)]
            entry[f"paf_{s}"] = round(float(point[s][i]), 2)
            entry[f"paf_{s}_lo"] = round(float(np.percentile(d, 2.5)), 2) if len(d) else np.nan
            entry[f"paf_{s}_hi"] = round(float(np.percentile(d, 97.5)), 2) if len(d) else np.nan
        rows.append(entry)

    out = pd.DataFrame(rows).sort_values("paf_joint", ascending=False)
    dest = OUT / "table_percountry_paf_ci.csv"
    out.to_csv(dest, index=False)

    outside = out[(out["paf_joint"] < out["paf_joint_lo"])
                  | (out["paf_joint"] > out["paf_joint_hi"])]
    print(f"\nSaved: {dest}  ({len(out)} countries)")
    print(f"  draws per country : min {out['n_draws'].min()}, "
          f"median {int(out['n_draws'].median())}, max {out['n_draws'].max()}")
    print(f"  point outside its own CI: {len(outside)}"
          + ("" if outside.empty else f" -- {list(outside['country_code'])}"))
    print()
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--seed", type=int, default=RNG_SEED)
    main(**vars(ap.parse_args()))
