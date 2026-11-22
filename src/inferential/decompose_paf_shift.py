"""decompose_paf_shift.py — why the joint PAF fell from the legacy ~51% to ~32%.

Fits the phase2 bridge surfaces (legacy -> primary) at FULL data and reads the
joint / education / spacing PAF at each step, attributing the shift to each
correction: whole-population -> policy-eligible counterfactual; legacy -> exact
>=60-month outcome cohort; collapsed -> explicit-missing exposure states; survey
weighting; and wave fixed effects + complete-wealth adjustment.

The surface fits are independent, so they run in parallel across CPU cores (each
worker loads the candidate and fits one surface). Point estimates only.

Outputs (output/phase4/manuscript/):
  table_paf_decomposition_full.csv    (every surface x contract x scenario)
  table_paf_waterfall.csv             (curated joint-PAF waterfall with deltas)
"""
from __future__ import annotations

import os

# Cap per-worker BLAS threads so parallel workers do not oversubscribe the CPU.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import pandas as pd  # noqa: E402

from config.paths import HARMONIZED_DIR, OUTPUT_DIR  # noqa: E402

OUT = OUTPUT_DIR / "phase4" / "manuscript"
CANDIDATE = HARMONIZED_DIR / "global_analysis_v2_candidate.parquet"

WATERFALL = [
    ("legacy_binary_pooled", "legacy_whole_population", "Legacy baseline (whole-population shift)"),
    ("legacy_binary_pooled", "policy_eligible", "+ policy-eligible counterfactual"),
    ("exact_binary_pooled_legacy_states", "policy_eligible", "+ exact >=60-mo outcome cohort"),
    ("exact_binary_pooled", "policy_eligible", "+ explicit-missing exposure states"),
    ("exact_weighted_pooled", "policy_eligible", "+ survey weights"),
    ("exact_country_fe_explicit_missing", "policy_eligible", "+ country/survey fixed effects"),
    ("exact_wave_fe_complete_wealth", "policy_eligible", "+ complete-wealth & wave FE (primary)"),
]


def _fit_one(index: int):
    """Worker: load the candidate, fit surface `index`, return PAF rows as records."""
    from src.inferential.phase2_analysis import _fit_surface, _paf_rows, _read_pilot, _surface_specs
    frame = _read_pilot(CANDIDATE)
    spec = _surface_specs(frame)[index]
    try:
        result = _fit_surface(**spec)
        return spec["surface"], _paf_rows([result]).to_dict("records"), None
    except Exception as e:  # noqa: BLE001
        return spec["surface"], [], f"{type(e).__name__}: {e}"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    from src.inferential.phase2_analysis import _read_pilot, _surface_specs
    n_surfaces = len(_surface_specs(_read_pilot(CANDIDATE)))

    # 3 workers: each holds the full ~10.7M-row frame + a large design; more than
    # this spikes past RAM at peak (observed OOM at 6). Still parallel across cores.
    records, workers = [], min(3, n_surfaces)
    print(f"Fitting {n_surfaces} surfaces across {workers} worker processes...")
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for surface, recs, err in ex.map(_fit_one, range(n_surfaces)):
            if err:
                print(f"  {surface:38} FAILED: {err}")
            else:
                records.extend(recs)
                j = [r for r in recs if r["scenario"] == "joint"]
                for r in j:
                    print(f"  {surface:38} [{r['contract']:22}] joint PAF={r['paf_pct']:6.2f} "
                          f"n={int(r['n_births']):,}")

    full = pd.DataFrame(records)
    full.to_csv(OUT / "table_paf_decomposition_full.csv", index=False)

    def paf(surface, contract, scenario="joint"):
        m = full[(full.surface == surface) & (full.contract == contract) & (full.scenario == scenario)]
        return float(m["paf_pct"].iloc[0]) if len(m) else float("nan")

    rows, prev = [], None
    for surface, contract, label in WATERFALL:
        jp = paf(surface, contract, "joint")
        rows.append({"step": label, "joint_paf_pct": round(jp, 2),
                     "delta_from_prev": None if prev is None else round(jp - prev, 2),
                     "education_paf_pct": round(paf(surface, contract, "education"), 2),
                     "spacing_paf_pct": round(paf(surface, contract, "spacing"), 2)})
        prev = jp
    wf = pd.DataFrame(rows)
    wf.to_csv(OUT / "table_paf_waterfall.csv", index=False)
    print("\n=== JOINT PAF WATERFALL (legacy -> v2 primary) ===")
    print(wf.to_string(index=False))
    if pd.notna(wf.iloc[0]["joint_paf_pct"]) and pd.notna(wf.iloc[-1]["joint_paf_pct"]):
        print(f"\nTotal shift: {wf.iloc[0]['joint_paf_pct']} -> {wf.iloc[-1]['joint_paf_pct']} "
              f"({wf.iloc[-1]['joint_paf_pct'] - wf.iloc[0]['joint_paf_pct']:+.2f} pp)")


if __name__ == "__main__":
    main()
