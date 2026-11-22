"""s05s_ablation_grid.py -- ablation grid for the joint counterfactual (v2).

Extends `sensitivity.py` along the two axes it does not cover. That suite varies
*fit-level* switches (weights, fixed effects, missing-wealth handling, DHS-only,
population weighting, complete-case). This one varies:

  A. **The adjustment set** -- refit without the child, maternal-age, marital,
     residence, wealth, or birth-period blocks in turn. Answers "how much of the
     joint fraction depends on what we controlled for?"

  B. **The counterfactual contract** -- vary how ambitious the shift is, and who is
     eligible for it, while holding the fitted surface fixed. Answers "how much
     depends on the policy we imagined?" This is arguably the more consequential
     arbitrary choice, and it is the one reviewers can most easily dispute.

Both are reported as delta percentage points against the primary joint fraction, so
the paper can make one bounded claim about specification sensitivity.

Why B needs no refit
--------------------
The counterfactual enters the model only as an additive logit shift
`coef[target_level] - coef[current_level]` on an eligible row set. Changing the
target or the eligibility rule therefore re-uses the primary fit: it changes the
shift, not the surface. Each contract variant costs milliseconds. Only the axis-A
ablations refit, and they parallelise across specifications.

What this deliberately does NOT do
----------------------------------
Re-binning the birth-interval threshold (e.g. an 18- or 36-month cut instead of 24)
is NOT included. That constant is hardcoded in three coordinated places --
`phase2_pilot._prepare_states` (the binning), `phase2_analysis._FIXED_CATEGORIES`
(the level list), and `_counterfactual_eta_delta` (eligibility and target) -- all of
which also produce the published primary. Changing it is a separate, properly
validated piece of work, not something to bolt onto an ablation script. The
`spacing_eligible_lt18` variant below gives a partial answer without re-binning: it
shifts only the very shortest intervals, using the raw month value rather than a
model category.

Outputs
-------
  output/phase4/manuscript/table_ablation_grid.csv

Worker count
------------
Keep it low. Unlike the PAF-CI bootstrap, which shares one compact payload, each
ablation worker calls `_read_pilot` and holds the full ~8 GB frame, so the ceiling
is memory rather than cores: 6 workers OOM'd this 64 GB box, 3 fits comfortably.

Run:  python -m src.inferential.s05s_ablation_grid --workers 3
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from config.paths import HARMONIZED_DIR, OUTPUT_DIR  # noqa: E402

MV = OUTPUT_DIR / "phase4" / "manuscript"
SCENARIOS = ("education", "spacing", "joint")

# ---------------------------------------------------------------- axis A specs
# Each drops one block of the adjustment set. The exposures themselves are never
# dropped (`_fit_surface` refuses); `wq_cat` is included because the
# wealth-unadjusted education fraction is already a headline quantity and this
# places it on the same grid as the other blocks.
COVARIATE_BLOCKS: dict[str, tuple[str, tuple[str, ...]]] = {
    "drop_child":        ("Child block dropped (sex, multiple birth, birth order)",
                          ("sex_cat", "birth_type", "bo_cat")),
    "drop_maternal_age": ("Maternal age block dropped", ("ma_cat",)),
    "drop_marital":      ("Marital status dropped", ("marital_cat",)),
    "drop_residence":    ("Urban/rural residence dropped", ("urban_cat",)),
    "drop_wealth":       ("Household wealth dropped (education fraction becomes total-effect)",
                          ("wq_cat",)),
    "drop_birth_period": ("Birth-period block dropped", ("birth_period",)),
}

# ------------------------------------------------------------- axis B contracts
# (label, education target level, education eligible levels,
#          spacing target level, spacing eligibility in raw months)
CONTRACTS: dict[str, dict] = {
    "primary": dict(
        label="Primary contract (to secondary; intervals <24 mo to 24-47 mo)",
        edu_target="Secondary", edu_eligible=(0, 1),
        spa_target="24-47 mo", spa_max_months=24),
    "edu_to_primary": dict(
        label="Education: no schooling raised to primary only",
        edu_target="Primary", edu_eligible=(0,),
        spa_target="24-47 mo", spa_max_months=24),
    "edu_to_higher": dict(
        label="Education: below higher raised to higher",
        edu_target="Higher", edu_eligible=(0, 1, 2),
        spa_target="24-47 mo", spa_max_months=24),
    "spacing_to_48": dict(
        label="Spacing: intervals <24 mo raised to 48+ mo",
        edu_target="Secondary", edu_eligible=(0, 1),
        spa_target="48+ mo", spa_max_months=24),
    "spacing_eligible_lt18": dict(
        label="Spacing: only intervals <18 mo shifted",
        edu_target="Secondary", edu_eligible=(0, 1),
        spa_target="24-47 mo", spa_max_months=18),
    "spacing_eligible_lt12": dict(
        label="Spacing: only intervals <12 mo shifted",
        edu_target="Secondary", edu_eligible=(0, 1),
        spa_target="24-47 mo", spa_max_months=12),
}

_W: dict = {}


# --------------------------------------------------------------------------- #

def _paf(w, f, cf) -> float:
    return 100.0 * (1.0 - float(np.sum(w * cf)) / float(np.sum(w * f)))


def _contract_delta(result, coef, contract: dict, scenario: str) -> np.ndarray:
    """Additive logit shift for one counterfactual contract.

    Generalises `phase2_analysis._counterfactual_eta_delta`, whose target levels
    and eligibility rule are fixed. Levels absent from the design are reference
    levels and contribute zero, matching that function's `effect()` helper.
    """
    names = {str(n): i for i, n in enumerate(result.feature_names)}
    frame = result.frame
    n = len(frame)
    delta = np.zeros(n, dtype=float)

    def eff(feature: str, level: str) -> float:
        j = names.get(f"{feature}_{level}")
        return 0.0 if j is None else float(coef[j])

    if scenario in ("education", "joint"):
        edu_level = pd.to_numeric(frame["edu_level"], errors="coerce")
        mask = edu_level.isin(list(contract["edu_eligible"])).to_numpy(bool)
        cur = frame["edu_cat"].astype(str).to_numpy()
        tgt = eff("edu_cat", contract["edu_target"])
        for level in np.unique(cur[mask]):
            rows = mask & (cur == level)
            delta[rows] += tgt - eff("edu_cat", str(level))

    if scenario in ("spacing", "joint"):
        order = pd.to_numeric(frame["birth_order"], errors="coerce")
        interval = pd.to_numeric(frame["birth_interval_months"], errors="coerce")
        mask = (order.gt(1) & interval.ge(0)
                & interval.lt(contract["spa_max_months"])).to_numpy(bool)
        cur = frame["bi_model_cat"].astype(str).to_numpy()
        tgt = eff("bi_model_cat", contract["spa_target"])
        for level in np.unique(cur[mask]):
            rows = mask & (cur == level)
            delta[rows] += tgt - eff("bi_model_cat", str(level))

    return delta


def _selftest_contract(result, coef) -> None:
    """The `primary` contract must reproduce the reference delta exactly."""
    from src.inferential.phase2_analysis import _counterfactual_eta_delta
    for s in SCENARIOS:
        ref = _counterfactual_eta_delta(result, s, "policy_eligible", coef)[0]
        mine = _contract_delta(result, coef, CONTRACTS["primary"], s)
        worst = float(np.max(np.abs(ref - mine)))
        assert worst < 1e-12, f"primary contract diverges from reference for {s}: {worst}"


# ------------------------------------------------------- axis A (parallel refits)

def _init_worker() -> None:
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = "1"


def _run_block_ablation(name: str) -> list[dict]:
    from src.inferential.phase2_analysis import (
        _counterfactual_eta_delta, _fit_surface, _read_pilot,
    )
    label, drop = COVARIATE_BLOCKS[name]
    frame = _read_pilot(HARMONIZED_DIR / "global_analysis_v2_candidate.parquet")
    result = _fit_surface(
        surface="exact_wave_fe_complete_wealth", role=f"ablation_{name}",
        frame=frame, outcome_column="died_under5_complete", legacy_states=False,
        fixed_effects="survey", wealth_mode="complete", weighted=True,
        drop_features=drop)
    coef = result.model.coef_[0]
    eta = np.asarray(result.matrix @ coef).ravel() + float(result.model.intercept_[0])
    f = expit(eta)
    w = (np.ones(len(f)) if result.weights is None
         else np.asarray(result.weights, dtype=float))
    rows = []
    for s in SCENARIOS:
        cf = expit(eta + _counterfactual_eta_delta(result, s, "policy_eligible", coef)[0])
        rows.append({"axis": "adjustment set", "ablation": name, "label": label,
                     "scenario": s, "paf_pct": round(_paf(w, f, cf), 2),
                     "n_births": int(len(f))})
    return rows


def main(workers: int = 6) -> None:
    import multiprocessing as mp
    from src.inferential.phase2_analysis import _fit_surface, _read_pilot

    print("=" * 68)
    print("Ablation grid: adjustment set and counterfactual contract")
    print("=" * 68)

    # ---- axis B: one fit, many contracts -----------------------------------
    t0 = time.time()
    frame = _read_pilot(HARMONIZED_DIR / "global_analysis_v2_candidate.parquet")
    result = _fit_surface(
        surface="exact_wave_fe_complete_wealth", role="ablation_primary",
        frame=frame, outcome_column="died_under5_complete", legacy_states=False,
        fixed_effects="survey", wealth_mode="complete", weighted=True)
    coef = result.model.coef_[0]
    eta = np.asarray(result.matrix @ coef).ravel() + float(result.model.intercept_[0])
    f = expit(eta)
    w = np.asarray(result.weights, dtype=float)
    print(f"[fit] primary surface in {time.time() - t0:.1f}s")

    _selftest_contract(result, coef)
    print("[check] primary contract reproduces the reference delta exactly")

    rows: list[dict] = []
    for name, contract in CONTRACTS.items():
        for s in SCENARIOS:
            cf = expit(eta + _contract_delta(result, coef, contract, s))
            rows.append({"axis": "counterfactual contract", "ablation": name,
                         "label": contract["label"], "scenario": s,
                         "paf_pct": round(_paf(w, f, cf), 2), "n_births": int(len(f))})
        j = [r for r in rows if r["ablation"] == name and r["scenario"] == "joint"][0]
        print(f"  contract {name:24} joint {j['paf_pct']:6.2f}")

    primary_joint = [r for r in rows
                     if r["ablation"] == "primary" and r["scenario"] == "joint"][0]["paf_pct"]
    published = 32.20
    assert abs(primary_joint - published) < 0.05, (
        f"primary contract gives {primary_joint}, expected the published {published}")
    print(f"[check] primary joint PAF {primary_joint} matches the published {published}")

    del frame, result
    import gc
    gc.collect()

    # ---- axis A: refit per dropped block, in parallel -----------------------
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=min(workers, len(COVARIATE_BLOCKS)),
                  initializer=_init_worker) as pool:
        for res in pool.imap_unordered(_run_block_ablation, list(COVARIATE_BLOCKS)):
            rows.extend(res)
            j = [r for r in res if r["scenario"] == "joint"][0]
            print(f"  ablation {j['ablation']:20} joint {j['paf_pct']:6.2f}  "
                  f"n={j['n_births']:,}")
    print(f"[refits] {len(COVARIATE_BLOCKS)} block ablations in {time.time() - t0:.1f}s")

    out = pd.DataFrame(rows)
    wide = out.pivot_table(index=["axis", "ablation", "label", "n_births"],
                           columns="scenario", values="paf_pct").reset_index()
    wide = wide.rename(columns={s: f"paf_{s}" for s in SCENARIOS})
    wide["delta_joint_vs_primary"] = (wide["paf_joint"] - primary_joint).round(2)
    wide = wide.sort_values(["axis", "delta_joint_vs_primary"])

    dest = MV / "table_ablation_grid.csv"
    wide.to_csv(dest, index=False)

    span = wide["delta_joint_vs_primary"].abs().max()
    print(f"\nSaved: {dest}  ({len(wide)} specifications)")
    print(f"  largest |delta| on the joint fraction: {span:.2f} percentage points")
    print()
    print(wide[["axis", "ablation", "paf_education", "paf_spacing", "paf_joint",
                "delta_joint_vs_primary"]].to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    main(**vars(ap.parse_args()))
