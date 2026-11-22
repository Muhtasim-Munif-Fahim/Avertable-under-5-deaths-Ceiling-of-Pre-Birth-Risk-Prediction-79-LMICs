"""Primary full-data exact-binary publication analysis.

This module deliberately excludes the legacy bridge and pilot sensitivity
surfaces.  It fits the locked observed-wealth survey-wave model, evaluates the
three conditional standardisations, and obtains publication uncertainty from
1,000 common uncapped country draws using exact grouped Bernoulli likelihoods.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import sklearn

from config.paths import OUTPUT_DIR, RNG_SEED
from src.inferential.phase2_analysis import (
    AnalysisResult,
    _average,
    _bootstrap_models,
    _bootstrap_summary,
    _country_draws,
    _fit_surface,
    _model_outputs,
    _paf_rows,
    _read_pilot,
)


_OUTPUT_FILENAMES = {
    "model_diagnostics": "phase2_model_diagnostics.csv",
    "model_coefficients": "phase2_model_coefficients.csv",
    "pafs": "phase2_pafs.csv",
    "risk_comparison": "phase2_risk_comparison.csv",
    "bootstrap_draws": "phase2_bootstrap_draws.csv",
    "bootstrap_summary": "phase2_bootstrap_summary.csv",
    "bootstrap_diagnostics": "phase2_bootstrap_diagnostics.csv",
    "stop_rules": "phase2_stop_rules.csv",
    "report": "phase2_report.md",
    "run_manifest": "phase2_run_manifest.json",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_primary_binary_publication(
    candidate_path: str | Path,
    output_dir: str | Path = OUTPUT_DIR / "phase4" / "phase2_publication",
    *,
    bootstrap_reps: int = 1000,
    seed: int = RNG_SEED,
    require_publication_reps: bool = True,
) -> AnalysisResult:
    """Fit and bootstrap only the locked primary exact-binary surface."""

    candidate_path = Path(candidate_path)
    output_dir = Path(output_dir)
    if require_publication_reps and bootstrap_reps < 1000:
        raise ValueError("Publication binary analysis requires at least 1000 draws")
    if bootstrap_reps < 1:
        raise ValueError("bootstrap_reps must be positive")
    frame = _read_pilot(candidate_path)
    draws = _country_draws(frame, bootstrap_reps=bootstrap_reps, seed=seed)
    result = _fit_surface(
        surface="exact_wave_fe_complete_wealth",
        role="primary_observed_wealth",
        frame=frame,
        outcome_column="died_under5_complete",
        legacy_states=False,
        fixed_effects="survey",
        wealth_mode="complete",
        weighted=True,
    )
    del frame
    gc.collect()

    diagnostics, coefficients = _model_outputs([result])
    pafs = _paf_rows([result])
    risks = pd.DataFrame(
        [
            {
                "estimand": "model_factual_risk__exact_wave_fe_complete_wealth",
                "estimate": _average(result.factual_predictions, result.weights),
                "n_births": len(result.frame),
                "n_zero_or_invalid_weight": 0,
                "label": result.role,
            }
        ]
    )
    root = Path(__file__).resolve().parents[2]
    cache_dir = (
        root
        / "tmp"
        / "phase4_binary_jobs"
        / (
            f"{_sha256(candidate_path)[:12]}_"
            f"{_sha256(Path(__file__).resolve())[:12]}_"
            f"b{bootstrap_reps}_s{seed}"
        )
    )
    bootstrap_values, bootstrap_diagnostics = _bootstrap_models(
        [result],
        draws,
        cache_dir=cache_dir,
    )
    summary = _bootstrap_summary(
        bootstrap_values,
        pafs,
        risks,
        n_requested=bootstrap_reps,
    )
    success = int(
        bootstrap_diagnostics["success"].astype(str).str.lower().eq("true").sum()
    )
    required_success = math.ceil(0.95 * bootstrap_reps)
    primary = pafs.set_index("scenario")
    invalid_probabilities = int(
        (~pafs["factual_risk"].between(0, 1)).sum()
        + (~pafs["counterfactual_risk"].between(0, 1)).sum()
    )
    rules = pd.DataFrame(
        [
            {
                "rule_id": "point_model_valid",
                "decision_class": "METHOD_FAILURE",
                "value": (
                    f"converged={bool(result.converged)};"
                    f"rank_deficient={bool(result.rank_deficient)}"
                ),
                "threshold": "converged and full rank",
                "triggered": not result.converged or result.rank_deficient,
                "evaluable": True,
            },
            {
                "rule_id": "bootstrap_replicates_requested",
                "decision_class": "METHOD_FAILURE",
                "value": bootstrap_reps,
                "threshold": ">=1000 for publication",
                "triggered": require_publication_reps and bootstrap_reps < 1000,
                "evaluable": True,
            },
            {
                "rule_id": "paired_bootstrap_success",
                "decision_class": "METHOD_FAILURE",
                "value": success,
                "threshold": f">={required_success}",
                "triggered": success < required_success,
                "evaluable": True,
            },
            {
                "rule_id": "hazard_and_risk_validity",
                "decision_class": "METHOD_FAILURE",
                "value": invalid_probabilities,
                "threshold": "0 non-finite or out-of-bounds risks",
                "triggered": invalid_probabilities != 0,
                "evaluable": True,
            },
            {
                "rule_id": "counterfactual_monotonicity",
                "decision_class": "METHOD_FAILURE",
                "value": int(pafs["n_risk_increased"].sum()),
                "threshold": "0 eligible records with increased risk",
                "triggered": int(pafs["n_risk_increased"].sum()) != 0,
                "evaluable": True,
            },
            {
                "rule_id": "joint_same_surface_identity",
                "decision_class": "METHOD_FAILURE",
                "value": float(
                    primary.loc["joint", "same_surface_identity_max_abs_error"]
                ),
                "threshold": "<=1e-12",
                "triggered": float(
                    primary.loc["joint", "same_surface_identity_max_abs_error"]
                )
                > 1e-12,
                "evaluable": True,
            },
        ]
    )
    decision = "METHOD_FAILURE" if rules["triggered"].any() else "PASS"

    output_dir.mkdir(parents=True, exist_ok=True)
    staging = output_dir / ".phase4_binary_building"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    staged = {key: staging / name for key, name in _OUTPUT_FILENAMES.items()}
    diagnostics.to_csv(staged["model_diagnostics"], index=False)
    coefficients.to_csv(staged["model_coefficients"], index=False)
    pafs.to_csv(staged["pafs"], index=False)
    risks.to_csv(staged["risk_comparison"], index=False)
    bootstrap_values.to_csv(staged["bootstrap_draws"], index=False)
    summary.to_csv(staged["bootstrap_summary"], index=False)
    bootstrap_diagnostics.to_csv(staged["bootstrap_diagnostics"], index=False)
    rules.to_csv(staged["stop_rules"], index=False)
    staged["report"].write_text(
        "\n".join(
            [
                "# Phase 4 primary exact-binary publication analysis",
                "",
                f"**Decision class: {decision}.**",
                "",
                (
                    f"The primary model uses {len(result.frame):,} births from "
                    f"{result.frame['country_code'].nunique()} countries and "
                    f"{result.frame['survey_id'].nunique()} survey waves."
                ),
                "",
                *[
                    (
                        f"- {scenario}: PAF "
                        f"{float(primary.loc[scenario, 'paf_pct']):.2f}%; "
                        f"risk difference "
                        f"{float(primary.loc[scenario, 'risk_difference_per_1000']):.2f} "
                        "per 1,000"
                    )
                    for scenario in ("education", "spacing", "joint")
                ],
                "",
                (
                    f"Bootstrap: {bootstrap_reps} common uncapped country draws; "
                    f"{success} succeeded."
                ),
                "",
                "These are controlled associational standardisations, not identified causal effects.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    plan_path = root / "docs" / "ANALYSIS_PLAN_V2_2026-07-11.md"
    data_manifest = root / "output" / "phase4" / "phase4_data_manifest.json"
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "analysis_scope": "primary exact binary publication surface only",
        "candidate_path": str(candidate_path.resolve()),
        "candidate_sha256": _sha256(candidate_path),
        "analysis_code_path": str(Path(__file__).resolve()),
        "analysis_code_sha256": _sha256(Path(__file__).resolve()),
        "analysis_plan_path": str(plan_path.resolve()),
        "analysis_plan_sha256": _sha256(plan_path),
        "phase4_data_manifest_path": str(data_manifest.resolve()),
        "phase4_data_manifest_sha256":
            _sha256(data_manifest) if data_manifest.exists() else None,
        "bootstrap_reps": bootstrap_reps,
        "required_success": required_success,
        "seed": seed,
        "country_bootstrap_cap": None,
        "n_rows": int(len(result.frame)),
        "n_countries": int(result.frame["country_code"].nunique()),
        "n_surveys": int(result.frame["survey_id"].nunique()),
        "individual_level_output_written": False,
        "python_version": sys.version,
        "scikit_learn_version": sklearn.__version__,
    }
    manifest["aggregate_output_sha256"] = {
        key: _sha256(path)
        for key, path in staged.items()
        if key != "run_manifest"
    }
    staged["run_manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    outputs: dict[str, Path] = {}
    for key, name in _OUTPUT_FILENAMES.items():
        destination = output_dir / name
        os.replace(staged[key], destination)
        outputs[key] = destination
    shutil.rmtree(staging)
    return AnalysisResult(decision=decision, outputs=outputs)
