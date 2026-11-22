"""Order-robustness of the estimand-shift decomposition (2026-09 reviewer round).

The published waterfall (``decompose_paf_shift.py``, main-text Figure 2) corrects
the conventional pooled approach one choice at a time in a fixed order. For a
non-linear estimand that attribution can depend on the order. This script removes
the order by computing **Shapley values**: each choice's marginal contribution
averaged over all orderings.

Six choices, each switched from the conventional to the primary setting:

  contract  whole-population -> policy-eligible counterfactual   (no refit)
  cohort    legacy imputed follow-up -> exact >=60-month cohort
  states    collapsed -> explicit missing-exposure states
  weights   unweighted -> survey-weighted
  adjust    no fixed effects -> survey fixed effects (nests country)
  wealth    missing-wealth category -> complete-case wealth

The five refit choices define 2^5 = 32 surfaces; the contract is evaluated on each
without refitting, so all 64 coalition values come from 32 fits. The empty
coalition is the published conventional baseline and the full one the primary
exact-binary surface. The script also reports, for each choice, the smallest and
largest marginal contribution it takes over all orderings, and its contribution in
the published waterfall order.

Point estimates only: a bootstrap over 32 full-data surfaces is not feasible.

Outputs: output/phase4/sensitivity_2026-09/shapley/
Run:     python -m src.inferential.sens2026_shapley
"""

from __future__ import annotations

import itertools
import json
import math
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.special import expit

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from config.paths import HARMONIZED_DIR
from src.inferential import phase2_analysis as p2
from src.inferential.sens2026_common import OUT_ROOT, git_head

PILOT = HARMONIZED_DIR / "global_analysis_v2_candidate.parquet"
OUT = OUT_ROOT / "shapley"
REFIT = ("cohort", "states", "weights", "adjust", "wealth")
PLAYERS = ("contract",) + REFIT
WATERFALL_ORDER = ("contract", "cohort", "states", "weights", "adjust", "wealth")


def surface_kwargs(on: frozenset[str]) -> dict:
    return dict(
        outcome_column="died_under5_complete" if "cohort" in on else "died_under5_legacy_frozen",
        legacy_states="states" not in on,
        weighted="weights" in on,
        fixed_effects="survey" if "adjust" in on else "none",
        wealth_mode="complete" if "wealth" in on else "explicit",
    )


def joint_pafs(result) -> dict[str, float]:
    coef = result.model.coef_[0]
    eta = np.asarray(result.matrix @ coef).ravel() + float(result.model.intercept_[0])
    factual = p2._average(expit(eta), result.weights)
    out = {}
    for contract in ("legacy_whole_population", "policy_eligible"):
        delta, _ = p2._counterfactual_eta_delta(result, "joint", contract, coef)
        cf = p2._average(expit(eta + delta), result.weights)
        out[contract] = 100.0 * (factual - cf) / factual
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    values_path = OUT / "coalition_values.csv"
    done = pd.read_csv(values_path) if values_path.exists() else pd.DataFrame(
        columns=["refit_on", "contract", "joint_paf_pct", "n_births", "converged"])
    have = set(done["refit_on"].astype(str)) if len(done) else set()
    frame = p2._read_pilot(PILOT)
    for k in range(len(REFIT) + 1):
        for combo in itertools.combinations(REFIT, k):
            key = "+".join(combo) or "none"
            if key in have:
                continue
            t0 = time.time()
            result = p2._fit_surface(surface=f"shapley_{key}", role="sens2026", frame=frame,
                                     **surface_kwargs(frozenset(combo)))
            pafs = joint_pafs(result)
            rows = [{"refit_on": key, "contract": c, "joint_paf_pct": v,
                     "n_births": len(result.frame),
                     "converged": bool(result.converged),
                     "rank_deficient": bool(result.rank_deficient)}
                    for c, v in pafs.items()]
            pd.DataFrame(rows).to_csv(values_path, mode="a", header=not values_path.exists(),
                                      index=False)
            print(f"[shapley] {key:40s} whole-pop {pafs['legacy_whole_population']:6.2f}  "
                  f"policy {pafs['policy_eligible']:6.2f}  {time.time() - t0:.0f}s", flush=True)
            del result

    values = pd.read_csv(values_path)
    if "rank_deficient" not in values:
        values["rank_deficient"] = False
    values["rank_deficient"] = values["rank_deficient"].fillna(False).astype(bool)
    if not values["converged"].astype(bool).all():
        raise SystemExit("a coalition surface did not converge; see coalition_values.csv")
    # Survey fixed effects plus a missing-wealth category is rank-deficient by
    # construction: wealth is missing for whole survey rounds, so the missing-wealth
    # indicator is collinear with those surveys' own intercepts. Fitted
    # probabilities, and so the PAF, are still identified once the optimiser has
    # converged; only the redundant coefficient is not. That combination alone is
    # allowed to be rank-deficient; any other is refused.
    on = values["refit_on"].astype(str).str.split("+")
    expected = on.apply(lambda s: "adjust" in s and "wealth" not in s)
    unexpected = values["rank_deficient"] & ~expected
    if unexpected.any():
        raise SystemExit("unexpected rank deficiency: "
                         f"{sorted(values.loc[unexpected, 'refit_on'].unique())}")
    lookup = {}
    for r in values.itertuples(index=False):
        on = set() if r.refit_on == "none" else set(r.refit_on.split("+"))
        if r.contract == "policy_eligible":
            on.add("contract")
        lookup[frozenset(on)] = float(r.joint_paf_pct)

    n = len(PLAYERS)
    rows = []
    for player in PLAYERS:
        others = [p for p in PLAYERS if p != player]
        shapley, marginals = 0.0, []
        for k in range(n):
            for subset in itertools.combinations(others, k):
                s = frozenset(subset)
                marginal = lookup[s | {player}] - lookup[s]
                marginals.append(marginal)
                shapley += math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n) * marginal
        before = frozenset(WATERFALL_ORDER[: WATERFALL_ORDER.index(player)])
        rows.append({"choice": player, "shapley_pp": round(shapley, 2),
                     "min_marginal_pp": round(min(marginals), 2),
                     "max_marginal_pp": round(max(marginals), 2),
                     "waterfall_order_pp": round(lookup[before | {player}] - lookup[before], 2)})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "shapley_values.csv", index=False)
    total = lookup[frozenset(PLAYERS)] - lookup[frozenset()]
    manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "git_head": git_head(),
                "baseline_joint_paf": lookup[frozenset()],
                "primary_joint_paf": lookup[frozenset(PLAYERS)],
                "total_shift_pp": total,
                "shapley_sum_pp": float(table["shapley_pp"].sum()),
                "players": list(PLAYERS), "table": table.to_dict("records")}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(table.to_string(index=False))
    print(f"[shapley] baseline {lookup[frozenset()]:.2f} -> primary "
          f"{lookup[frozenset(PLAYERS)]:.2f}; total {total:+.2f}; "
          f"Shapley sum {table['shapley_pp'].sum():+.2f}")


if __name__ == "__main__":
    main()
