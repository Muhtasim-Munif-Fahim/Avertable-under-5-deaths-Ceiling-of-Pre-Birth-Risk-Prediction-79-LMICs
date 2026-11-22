"""Gated full-data rebuild and publication-analysis orchestration.

The Phase 4 interface keeps the rebuilt individual-level dataset at a
versioned candidate path.  The frozen legacy canonical parquet is read-only;
promotion is a separate operation that is unavailable until the complete
integrity audit passes.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from config.paths import HARMONIZED_DIR, OUTPUT_DIR, RNG_SEED
from src.inferential.counterfactuals import education_eligible, spacing_eligible
from src.inferential.phase2_pilot import (
    PilotBuildResult,
    PilotConfig,
    build_phase2_dataset,
)
from src.inferential.phase4_binary import (
    run_primary_binary_publication as run_phase2_analysis,
)
from src.inferential.phase3_survival import run_survival_validation


@dataclass(frozen=True)
class FullBuildConfig:
    """Paths and fixed settings for one complete v2 candidate build."""

    candidate_path: Path = HARMONIZED_DIR / "global_analysis_v2_candidate.parquet"
    manifest_path: Path = OUTPUT_DIR / "phase4" / "phase4_full_build_manifest.csv"
    legacy_canonical_path: Path = HARMONIZED_DIR / "global_analysis.parquet"
    bootstrap_reps: int = 1000
    seed: int = RNG_SEED
    rebuild_data: bool = False
    sample_step: int = 1
    survival_compute_backend: str = "auto"

    def __post_init__(self) -> None:
        candidate = Path(self.candidate_path).resolve()
        canonical = Path(self.legacy_canonical_path).resolve()
        if candidate == canonical:
            raise ValueError("Phase 4 candidate must not be the legacy canonical path")
        if self.sample_step != 1:
            raise ValueError("Phase 4 requires sample_step=1 for the complete rebuild")
        if self.bootstrap_reps < 1000:
            raise ValueError("Publication analysis requires at least 1000 bootstrap draws")
        if self.survival_compute_backend not in {"auto", "cpu", "cuda"}:
            raise ValueError(
                "survival_compute_backend must be one of: auto, cpu, cuda"
            )


def build_full_candidate(config: FullBuildConfig) -> PilotBuildResult:
    """Build or reuse the complete v2 candidate through the validated seam."""

    config.candidate_path.parent.mkdir(parents=True, exist_ok=True)
    config.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    builder_config = PilotConfig(
        sample_step=1,
        bootstrap_reps=config.bootstrap_reps,
        seed=config.seed,
        rebuild_data=config.rebuild_data,
        data_path_override=config.candidate_path,
        manifest_path_override=config.manifest_path,
    )
    return build_phase2_dataset(builder_config)


@dataclass(frozen=True)
class FullDataAuditResult:
    """Aggregate result of the locked full-candidate promotion checks."""

    passed: bool
    checks: pd.DataFrame
    audit_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class FullPublicationAnalysisResult:
    """Decisions returned by the two full-data publication analyses."""

    phase2_decision: str
    phase3_decision: str
    phase2_output_dir: Path
    phase3_output_dir: Path


_AUDIT_COLUMNS = [
    "country_code",
    "_source_file",
    "_source_row",
    "data_source",
    "birth_cmc",
    "interview_cmc",
    "followup_months",
    "age_at_death_months",
    "event_time_months",
    "u5_event",
    "u5_censored",
    "u5_binary_eligible",
    "died_under5_complete",
    "birth_order",
    "birth_interval_months",
    "edu_level",
    "wealth_quintile",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _csv_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def audit_full_candidate(
    config: FullBuildConfig,
    *,
    output_dir: str | Path = OUTPUT_DIR / "phase4",
    phase2_dir: str | Path = OUTPUT_DIR / "phase2",
    phase3_dir: str | Path = OUTPUT_DIR / "phase3",
) -> FullDataAuditResult:
    """Evaluate every locked promotion gate without modifying either dataset."""

    candidate = Path(config.candidate_path)
    canonical = Path(config.legacy_canonical_path)
    build_manifest_path = Path(config.manifest_path)
    output_dir = Path(output_dir)
    phase2_dir = Path(phase2_dir)
    phase3_dir = Path(phase3_dir)
    for path in (candidate, canonical, build_manifest_path):
        if not path.exists():
            raise FileNotFoundError(path)

    schema = set(pq.read_schema(str(candidate)).names)
    missing_columns = sorted(set(_AUDIT_COLUMNS) - schema)
    if missing_columns:
        raise ValueError(f"Phase 4 candidate lacks audit fields: {missing_columns}")
    frame = pd.read_parquet(candidate, columns=_AUDIT_COLUMNS)
    build_manifest = pd.read_csv(build_manifest_path)
    required_manifest = {
        "data_source",
        "survey_id",
        "sample_step",
        "n_source_births",
        "n_pilot_births",
        "frozen_canonical_sha256",
    }
    missing_manifest = sorted(required_manifest - set(build_manifest.columns))
    if missing_manifest:
        raise ValueError(f"Phase 4 build manifest lacks fields: {missing_manifest}")

    rows: list[dict] = []

    def add(check_id: str, value: object, threshold: str, passed: bool, detail: str = "") -> None:
        rows.append(
            {
                "check_id": check_id,
                "value": value,
                "threshold": threshold,
                "all_passed": bool(passed),
                "detail": detail,
            }
        )

    frozen_hashes = (
        build_manifest["frozen_canonical_sha256"].dropna().astype(str).str.lower().unique()
    )
    current_canonical_hash = _sha256(canonical)
    add(
        "legacy_canonical_unchanged",
        current_canonical_hash,
        "equals the single frozen hash in the build manifest",
        len(frozen_hashes) == 1 and frozen_hashes[0] == current_canonical_hash.lower(),
    )

    steps = pd.to_numeric(build_manifest["sample_step"], errors="coerce")
    source_births = pd.to_numeric(build_manifest["n_source_births"], errors="coerce")
    built_births = pd.to_numeric(build_manifest["n_pilot_births"], errors="coerce")
    complete_build = (
        steps.eq(1).all()
        and source_births.eq(built_births).all()
        and int(built_births.sum()) == len(frame)
    )
    add(
        "complete_step1_build",
        len(frame),
        "step=1; source=built births per survey; manifest total=disk rows",
        complete_build,
    )

    source_rows = pd.to_numeric(frame["_source_row"], errors="coerce")
    source_key_valid = (
        frame["_source_file"].notna().all()
        and source_rows.notna().all()
        and source_rows.ge(0).all()
        and np.equal(source_rows, np.floor(source_rows)).all()
        and not frame.assign(_audit_source_row=source_rows).duplicated(
            ["_source_file", "_audit_source_row"]
        ).any()
    )
    add(
        "source_keys_complete_unique",
        int(frame.duplicated(["_source_file", "_source_row"]).sum()),
        "zero invalid or duplicate source keys",
        source_key_valid,
    )

    birth = pd.to_numeric(frame["birth_cmc"], errors="coerce")
    interview = pd.to_numeric(frame["interview_cmc"], errors="coerce")
    followup = pd.to_numeric(frame["followup_months"], errors="coerce")
    eligible = pd.to_numeric(frame["u5_binary_eligible"], errors="coerce")
    valid_pair = (
        birth.notna()
        & interview.notna()
        & birth.gt(0)
        & interview.gt(0)
        & birth.eq(np.floor(birth))
        & interview.eq(np.floor(interview))
        & interview.ge(birth)
    )
    expected_followup = interview - birth
    exact_time = (
        followup.loc[valid_pair].eq(expected_followup.loc[valid_pair]).all()
        and eligible.loc[valid_pair].eq(expected_followup.loc[valid_pair].ge(60).astype(int)).all()
    )
    add(
        "exact_followup_and_binary_eligibility",
        int(valid_pair.sum()),
        "follow-up=interview-birth and eligibility=1(follow-up>=60)",
        exact_time,
    )

    complete = pd.to_numeric(frame["died_under5_complete"], errors="coerce")
    complete_valid = complete.notna()
    add(
        "complete_outcome_only_when_eligible",
        int(complete_valid.sum()),
        "every complete outcome has eligibility=1",
        eligible.loc[complete_valid].eq(1).all(),
    )

    event = pd.to_numeric(frame["u5_event"], errors="coerce")
    censored = pd.to_numeric(frame["u5_censored"], errors="coerce")
    event_time = pd.to_numeric(frame["event_time_months"], errors="coerce")
    known = event.notna() | censored.notna()
    survival_state = (
        event.loc[known].notna().all()
        and censored.loc[known].notna().all()
        and event.loc[known].add(censored.loc[known]).eq(1).all()
    )
    add(
        "known_survival_state_complete",
        int(known.sum()),
        "u5_event + u5_censored = 1 for every known record",
        survival_state,
    )
    time_valid = event_time.loc[known].between(0, 60, inclusive="both").all()
    add(
        "event_time_in_range",
        int((~event_time.loc[known].between(0, 60, inclusive="both")).sum()),
        "zero known records outside [0,60] months",
        time_valid,
    )

    death_age = pd.to_numeric(frame["age_at_death_months"], errors="coerce")
    events = event.eq(1)
    event_valid = (
        death_age.loc[events].notna().all()
        and death_age.loc[events].ge(0).all()
        and death_age.loc[events].lt(60).all()
        and death_age.loc[events].le(followup.loc[events]).all()
        and event_time.loc[events].eq(death_age.loc[events]).all()
    )
    add(
        "event_age_and_chronology_valid",
        int(events.sum()),
        "events have age in [0,60), age<=follow-up, and event time=age",
        event_valid,
    )

    sources = set(frame["data_source"].dropna().astype(str).str.upper())
    manifest_sources = set(
        build_manifest["data_source"].dropna().astype(str).str.upper()
    )
    source_contract = {"DHS", "MICS"}.issubset(sources & manifest_sources)
    add(
        "dhs_mics_exact_schema_retained",
        ",".join(sorted(sources)),
        "DHS and MICS both present with every exact-follow-up field",
        source_contract and not missing_columns,
    )

    order = pd.to_numeric(frame["birth_order"], errors="coerce")
    interval = pd.to_numeric(frame["birth_interval_months"], errors="coerce")
    education = pd.to_numeric(frame["edu_level"], errors="coerce")
    wealth = pd.to_numeric(frame["wealth_quintile"], errors="coerce")
    state_counts = {
        "first": int(order.eq(1).sum()),
        "missing_nonfirst": int((order.gt(1) & interval.isna()).sum()),
        "short": int((order.gt(1) & interval.between(0, 23)).sum()),
        "long": int((order.gt(1) & interval.ge(48)).sum()),
        "education_missing": int(education.isna().sum()),
        "education_higher": int(education.eq(3).sum()),
        "wealth_missing": int(wealth.isna().sum()),
    }
    add(
        "exposure_missing_states_distinct",
        json.dumps(state_counts, sort_keys=True),
        "all locked first/missing/higher/long states remain represented",
        all(value > 0 for value in state_counts.values()),
    )

    expected_education = education.isin([0, 1])
    expected_spacing = order.gt(1) & interval.ge(0) & interval.lt(24)
    intervention_contract = (
        education_eligible(frame).equals(expected_education)
        and spacing_eligible(frame).equals(expected_spacing)
        and expected_education.any()
        and expected_spacing.any()
    )
    add(
        "policy_eligibility_contract",
        f"education={int(expected_education.sum())};spacing={int(expected_spacing.sum())}",
        "only observed none/primary education and observed short non-first intervals change",
        intervention_contract,
    )

    phase2_manifest = json.loads(
        (phase2_dir / "phase2_run_manifest.json").read_text(encoding="utf-8")
    )
    phase3_manifest = json.loads(
        (phase3_dir / "phase3_run_manifest.json").read_text(encoding="utf-8")
    )
    phase2_rules = pd.read_csv(phase2_dir / "phase2_stop_rules.csv")
    phase3_rules = pd.read_csv(phase3_dir / "phase3_stop_rules.csv")
    phase2_hard = phase2_rules["decision_class"].isin(
        ["DATA_INTEGRITY_FAILURE", "METHOD_FAILURE"]
    ) & _csv_bool(phase2_rules["triggered"])
    phase3_any = _csv_bool(phase3_rules["triggered"])
    upstream_pass = (
        phase2_manifest.get("decision") in {"PASS", "NARRATIVE_ESCALATION"}
        and not phase2_hard.any()
        and phase3_manifest.get("decision") == "PASS"
        and not phase3_any.any()
    )
    add(
        "upstream_method_gates_passed",
        f"phase2={phase2_manifest.get('decision')};phase3={phase3_manifest.get('decision')}",
        "no Phase 2 hard failure and Phase 3 PASS with no triggered rule",
        upstream_pass,
    )

    checks = pd.DataFrame(rows)
    passed = bool(checks["all_passed"].all())
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = output_dir / "phase4_data_audit.csv"
    audit_building = output_dir / ".phase4_data_audit.building.csv"
    manifest_path = output_dir / "phase4_data_manifest.json"
    manifest_building = output_dir / ".phase4_data_manifest.building.json"
    audit_building.unlink(missing_ok=True)
    manifest_building.unlink(missing_ok=True)
    checks.to_csv(audit_building, index=False)
    manifest_payload = {
        "decision": "PASS" if passed else "FAIL",
        "candidate_path": str(candidate.resolve()),
        "candidate_sha256": _sha256(candidate),
        "legacy_canonical_path": str(canonical.resolve()),
        "legacy_canonical_sha256": current_canonical_hash,
        "build_manifest_path": str(build_manifest_path.resolve()),
        "build_manifest_sha256": _sha256(build_manifest_path),
        "n_rows": int(len(frame)),
        "n_countries": int(frame["country_code"].nunique()),
        "n_surveys": int(frame["_source_file"].nunique()),
        "n_exact_binary": int(complete.notna().sum()),
        "n_exact_deaths": int(complete.sum(skipna=True)),
        "n_survival_known": int(known.sum()),
        "audit_sha256": _sha256(audit_building),
    }
    manifest_building.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(audit_building, audit_path)
    os.replace(manifest_building, manifest_path)
    return FullDataAuditResult(
        passed=passed,
        checks=checks,
        audit_path=audit_path,
        manifest_path=manifest_path,
    )


def run_full_publication_analysis(
    config: FullBuildConfig,
    *,
    output_dir: str | Path = OUTPUT_DIR / "phase4",
    pilot_phase2_dir: str | Path = OUTPUT_DIR / "phase2",
    pilot_phase3_dir: str | Path = OUTPUT_DIR / "phase3",
) -> FullPublicationAnalysisResult:
    """Audit the candidate, then run binary and survival publication analyses."""

    output_dir = Path(output_dir)
    audit = audit_full_candidate(
        config,
        output_dir=output_dir,
        phase2_dir=pilot_phase2_dir,
        phase3_dir=pilot_phase3_dir,
    )
    if not audit.passed:
        failed = audit.checks.loc[~audit.checks["all_passed"], "check_id"].tolist()
        raise RuntimeError(f"Phase 4 candidate failed promotion audit: {failed}")

    phase2_output = output_dir / "phase2_publication"
    phase3_output = output_dir / "phase3_publication"
    phase2 = run_phase2_analysis(
        config.candidate_path,
        phase2_output,
        bootstrap_reps=config.bootstrap_reps,
        seed=config.seed,
        require_publication_reps=True,
    )
    if phase2.decision in {"DATA_INTEGRITY_FAILURE", "METHOD_FAILURE"}:
        raise RuntimeError(f"Full binary analysis failed: {phase2.decision}")
    phase3 = run_survival_validation(
        config.candidate_path,
        phase3_output,
        phase2_dir=phase2_output,
        require_decision_pilot_provenance=False,
        bootstrap_reps=config.bootstrap_reps,
        seed=config.seed,
        compute_backend=config.survival_compute_backend,
        checkpoint_bootstrap=True,
    )
    if phase3.decision in {"DATA_INTEGRITY_FAILURE", "METHOD_FAILURE"}:
        raise RuntimeError(f"Full survival analysis failed: {phase3.decision}")
    return FullPublicationAnalysisResult(
        phase2_decision=phase2.decision,
        phase3_decision=phase3.decision,
        phase2_output_dir=phase2_output,
        phase3_output_dir=phase3_output,
    )
