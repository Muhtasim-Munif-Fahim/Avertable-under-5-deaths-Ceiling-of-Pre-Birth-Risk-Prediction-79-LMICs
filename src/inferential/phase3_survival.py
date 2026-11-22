"""Phase 3 censoring-aware discrete-time survival validation.

The public interface, :func:`run_survival_validation`, accepts the persisted
Phase 2 pilot and emits aggregate artifacts only. Individual person-month rows
are never constructed; interval differences and weighted sufficient cells
represent the pooled-logistic likelihood exactly.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import numba
import pandas as pd
import scipy
import sklearn
from numba import njit, prange
from scipy import sparse
from scipy.special import expit
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

from config.paths import HARMONIZED_DIR, RNG_SEED
from src.inferential.phase2_analysis import (
    _BASE_FEATURES,
    _FIXED_CATEGORIES,
    _country_draws,
    _positive_normalized_weights,
    _prepare_states,
    _read_pilot,
    _validate_decision_pilot_provenance,
    _weighted_km_60,
)


_STATIC_FEATURES = _BASE_FEATURES + ["wq_cat", "survey_id"]
_MODEL_FEATURES = _STATIC_FEATURES + ["age_month"]
_SCENARIOS = ("education", "spacing", "joint")
_OUTPUT_FILENAMES = {
    "cohort_flow": "phase3_cohort_flow.csv",
    "person_month_audit": "phase3_person_month_audit.csv",
    "risk_comparison": "phase3_risk_comparison.csv",
    "model_diagnostics": "phase3_model_diagnostics.csv",
    "coefficients": "phase3_coefficients.csv",
    "standardized_risks": "phase3_standardized_risks.csv",
    "bootstrap_draws": "phase3_bootstrap_draws.csv",
    "bootstrap_summary": "phase3_bootstrap_summary.csv",
    "bootstrap_diagnostics": "phase3_bootstrap_diagnostics.csv",
    "stop_rules": "phase3_stop_rules.csv",
    "run_manifest": "phase3_run_manifest.json",
    "report": "phase3_report.md",
}

# Convergence acceptance tolerances for the pooled-logistic point fit and the
# per-draw bootstrap refits, expressed as max|gradient| / total risk weight.
#
# The likelihood objective and gradient are reduced in float64 on the selected
# compute backend. GPU float64 sum-reductions are non-associative, so the
# achievable residual-gradient floor over the ~5e5 grouped person-month cells is
# hardware-dependent (observed max|scaled gradient|: ~2.9e-8 on Ampere-class
# CUDA and ~3.6e-8 on CPU, but ~2.1e-7 on Blackwell-class CUDA) even though the
# fitted coefficients and standardized 60-month risks agree across backends to
# 1e-9 or better and every literal/factorized equivalence check passes at
# <=1e-12. The point tolerance is therefore set equal to the already-accepted
# per-draw bootstrap tolerance (1e-6) so the convergence gate is portable across
# GPU architectures while remaining an extremely strict optimum criterion.
# Rationale and reproduced diagnostics:
# docs/PHASE3_HARDWARE_PORTABILITY_AMENDMENT_2026-07-23.md.
_POINT_MAX_ABS_SCALED_GRADIENT = 1e-6
_BOOTSTRAP_MAX_ABS_SCALED_GRADIENT = 1e-6


@dataclass(frozen=True)
class SurvivalValidationResult:
    """Aggregate result contract for one Phase 3 validation run."""

    decision: str
    outputs: dict[str, Path]


@dataclass
class _PreparedSurvival:
    patterns: pd.DataFrame
    cells: _GroupedPersonMonths
    cohort_flow: pd.DataFrame
    audit: pd.DataFrame
    n_births: int
    n_events: int
    n_countries: int
    n_surveys: int
    km_point: float
    km_n: int


@dataclass
class _GroupedPersonMonths:
    """Dense sufficient statistics without a materialized model-row table."""

    risk_weight: np.ndarray
    event_weight: np.ndarray

    @property
    def n_patterns(self) -> int:
        return int(self.risk_weight.shape[0])

    @property
    def n_cells(self) -> int:
        return int(np.count_nonzero(self.risk_weight))

    @property
    def n_model_rows(self) -> int:
        non_event = self.risk_weight - self.event_weight
        return int(
            np.count_nonzero(self.event_weight)
            + np.count_nonzero(non_event > 1e-10)
        )


@dataclass
class _CpuLikelihoodBackend:
    cells: _GroupedPersonMonths
    name: str = "cpu-numba"

    def evaluate(
        self,
        static_eta: np.ndarray,
        age_coefficients: np.ndarray,
        pattern_multiplicity: np.ndarray,
        age_keep: np.ndarray,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        loss_by_pattern, residual = _parallel_likelihood_kernel(
            static_eta,
            age_coefficients,
            self.cells.risk_weight,
            self.cells.event_weight,
            pattern_multiplicity,
        )
        return (
            float(loss_by_pattern.sum()),
            residual.sum(axis=1),
            residual[:, 1:][:, age_keep].sum(axis=0),
        )

    def risks_from_eta(
        self,
        static_eta: np.ndarray,
        age_coefficients: np.ndarray,
        *,
        chunk_size: int = 20_000,
    ) -> np.ndarray:
        """Return 60-month risks without constructing person-month rows."""
        risks = np.empty(len(static_eta), dtype=np.float64)
        for start in range(0, len(static_eta), chunk_size):
            stop = min(start + chunk_size, len(static_eta))
            eta = static_eta[start:stop, None] + age_coefficients[None, :]
            log_survival = -np.logaddexp(0.0, eta).sum(axis=1)
            risks[start:stop] = -np.expm1(log_survival)
        return risks


class _CudaLikelihoodBackend:
    """Float64 Torch adapter retaining sufficient statistics on one GPU."""

    name = "cuda-torch-float64"

    def __init__(self, cells: _GroupedPersonMonths):
        import torch

        self.torch = torch
        self.risk_weight = torch.as_tensor(
            cells.risk_weight, device="cuda", dtype=torch.float64
        )
        self.event_weight = torch.as_tensor(
            cells.event_weight, device="cuda", dtype=torch.float64
        )

    def evaluate(
        self,
        static_eta: np.ndarray,
        age_coefficients: np.ndarray,
        pattern_multiplicity: np.ndarray,
        age_keep: np.ndarray,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        torch = self.torch
        with torch.no_grad():
            static = torch.as_tensor(static_eta, device="cuda", dtype=torch.float64)
            age = torch.as_tensor(
                age_coefficients, device="cuda", dtype=torch.float64
            )
            multiplicity = torch.as_tensor(
                pattern_multiplicity, device="cuda", dtype=torch.float64
            )
            eta = static[:, None] + age[None, :]
            loss_terms = torch.nn.functional.softplus(eta)
            loss_terms.mul_(self.risk_weight)
            loss_terms.addcmul_(self.event_weight, eta, value=-1.0)
            loss_terms.mul_(multiplicity[:, None])
            loss = float(loss_terms.sum().item())
            del loss_terms
            residual = torch.sigmoid(eta)
            del eta
            residual.mul_(self.risk_weight)
            residual.sub_(self.event_weight)
            residual.mul_(multiplicity[:, None])
            pattern_residual = residual.sum(dim=1).cpu().numpy()
            retained_months = torch.as_tensor(
                np.flatnonzero(age_keep) + 1,
                device="cuda",
                dtype=torch.long,
            )
            age_gradient = residual.index_select(1, retained_months).sum(dim=0)
            age_gradient = age_gradient.cpu().numpy()
        return loss, pattern_residual, age_gradient

    def risks_from_eta(
        self,
        static_eta: np.ndarray,
        age_coefficients: np.ndarray,
        *,
        chunk_size: int = 250_000,
    ) -> np.ndarray:
        """Return 60-month risks on the same float64 CUDA backend."""
        torch = self.torch
        risks = np.empty(len(static_eta), dtype=np.float64)
        with torch.no_grad():
            age = torch.as_tensor(
                age_coefficients, device="cuda", dtype=torch.float64
            )
            for start in range(0, len(static_eta), chunk_size):
                stop = min(start + chunk_size, len(static_eta))
                static = torch.as_tensor(
                    static_eta[start:stop], device="cuda", dtype=torch.float64
                )
                eta = static[:, None] + age[None, :]
                log_survival = -torch.nn.functional.softplus(eta).sum(dim=1)
                risks[start:stop] = (-torch.expm1(log_survival)).cpu().numpy()
        return risks


def _likelihood_backend(
    cells: _GroupedPersonMonths,
    requested: str,
) -> _CpuLikelihoodBackend | _CudaLikelihoodBackend:
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("compute_backend must be one of: auto, cpu, cuda")
    if requested == "cpu" or (requested == "auto" and cells.n_cells < 1_000_000):
        return _CpuLikelihoodBackend(cells)
    try:
        import torch
    except ImportError as exc:
        if requested == "cuda":
            raise RuntimeError("CUDA backend requires PyTorch") from exc
        return _CpuLikelihoodBackend(cells)
    if not torch.cuda.is_available():
        if requested == "cuda":
            raise RuntimeError("CUDA backend requested but no CUDA device is available")
        return _CpuLikelihoodBackend(cells)
    free_memory, _ = torch.cuda.mem_get_info()
    estimated_peak = cells.risk_weight.nbytes * 7 + 512 * 1024**2
    if free_memory < estimated_peak:
        if requested == "cuda":
            raise MemoryError(
                "CUDA backend does not have enough free memory for float64 likelihood"
            )
        return _CpuLikelihoodBackend(cells)
    return _CudaLikelihoodBackend(cells)


@dataclass
class _FactorizedLogisticModel:
    """Small sklearn-compatible result wrapper for the factorized solver."""

    coef_: np.ndarray
    intercept_: np.ndarray
    n_iter_: np.ndarray
    max_iter: int
    classes_: np.ndarray
    n_features_in_: int
    success: bool
    message: str
    objective: float
    max_abs_scaled_gradient: float

    def predict_proba(self, matrix: sparse.csr_matrix) -> np.ndarray:
        linear = np.asarray(matrix @ self.coef_[0]).ravel()
        probability = expit(linear + float(self.intercept_[0]))
        return np.column_stack([1.0 - probability, probability])


@dataclass
class _SurvivalFit:
    static_encoder: OneHotEncoder
    keep_columns: np.ndarray
    feature_names: np.ndarray
    static_matrix: sparse.csr_matrix
    factual_static_matrix: sparse.csr_matrix
    static_feature_count: int
    feature_scale: np.ndarray
    cells: _GroupedPersonMonths
    likelihood_backend: _CpuLikelihoodBackend | _CudaLikelihoodBackend
    model: _FactorizedLogisticModel
    converged: bool
    n_iter: int
    rank_deficient: bool
    condition_number: float
    warning_text: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _static_categories_for(frame: pd.DataFrame) -> list[list[str]]:
    categories: list[list[str]] = []
    for feature in _STATIC_FEATURES:
        if feature in _FIXED_CATEGORIES:
            categories.append(list(_FIXED_CATEGORIES[feature]))
        else:
            observed = sorted(frame[feature].astype(str).unique().tolist())
            if not observed:
                raise ValueError(f"No observed levels for {feature}")
            categories.append(observed)
    return categories


def _rank_diagnostics(matrix: sparse.csr_matrix) -> tuple[bool, float]:
    augmented = sparse.hstack(
        [sparse.csr_matrix(np.ones((matrix.shape[0], 1))), matrix],
        format="csr",
    )
    gram = (augmented.T @ augmented).toarray()
    rank = int(np.linalg.matrix_rank(gram))
    deficient = rank < gram.shape[0]
    condition = float(np.linalg.cond(gram)) if not deficient else float("inf")
    return deficient, condition


def _validate_survival_source(frame: pd.DataFrame) -> None:
    source_row = pd.to_numeric(frame["_source_row"], errors="coerce")
    source_key_invalid = (
        source_row.isna()
        | source_row.lt(0)
        | source_row.mod(1).ne(0)
        | frame["_source_file"].isna()
    )
    source_key_duplicate = pd.DataFrame(
        {
            "source_file": frame["_source_file"].astype(str),
            "source_row": source_row,
        }
    ).duplicated(["source_file", "source_row"])
    event = pd.to_numeric(frame["u5_event"], errors="coerce")
    censored = pd.to_numeric(frame["u5_censored"], errors="coerce")
    time = pd.to_numeric(frame["event_time_months"], errors="coerce")
    followup = pd.to_numeric(frame["followup_months"], errors="coerce")
    death_age = pd.to_numeric(frame["age_at_death_months"], errors="coerce")
    any_state = event.notna() | censored.notna() | time.notna()
    complete_state = (
        event.isin([0, 1])
        & censored.isin([0, 1])
        & (event + censored).eq(1)
        & time.between(0, 60)
        & time.mod(1).eq(0)
        & followup.notna()
        & followup.ge(0)
    )
    expected_censor_time = np.minimum(followup, 60)
    invariants = {
        "partial or invalid survival state": int((any_state & ~complete_state).sum()),
        "missing invalid or duplicate source key": int(
            source_key_invalid.sum() + source_key_duplicate.sum()
        ),
        "fractional observed follow-up": int(
            (complete_state & followup.mod(1).ne(0)).sum()
        ),
        "event death age/time mismatch": int(
            (
                complete_state
                & event.eq(1)
                & (
                    ~death_age.between(0, 59)
                    | death_age.mod(1).ne(0)
                    | ~time.eq(death_age)
                )
            ).sum()
        ),
        "event after observed follow-up": int(
            (complete_state & event.eq(1) & time.gt(followup)).sum()
        ),
        "censor time differs from observed follow-up": int(
            (
                complete_state
                & event.eq(0)
                & ~time.eq(expected_censor_time)
            ).sum()
        ),
    }
    failed = {name: count for name, count in invariants.items() if count}
    if failed:
        raise ValueError(f"Phase 3 survival source invariant failures: {failed}")


def _cohort_flow(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    _validate_survival_source(frame)
    event = pd.to_numeric(frame["u5_event"], errors="coerce")
    censored = pd.to_numeric(frame["u5_censored"], errors="coerce")
    time = pd.to_numeric(frame["event_time_months"], errors="coerce")
    known_mask = (
        event.isin([0, 1])
        & censored.isin([0, 1])
        & (event + censored).eq(1)
        & time.between(0, 60)
        & time.mod(1).eq(0)
    )
    known = frame.loc[known_mask].copy()
    normalized = _positive_normalized_weights(known)
    positive_mask = normalized.notna()
    wealth_mask = known["wealth_quintile"].notna()
    analysis_index = known.index[positive_mask & wealth_mask]
    analysis = known.loc[analysis_index].copy()
    analysis["_birth_weight"] = normalized.loc[analysis_index].to_numpy(float)
    rows = []
    for scope, selected in (
        ("pilot", frame),
        ("known_survival", known),
        ("known_survival_positive_weight", known.loc[positive_mask]),
        ("analysis_observed_wealth", analysis),
    ):
        rows.append(
            {
                "scope": scope,
                "n_births": int(len(selected)),
                "n_events": int(
                    pd.to_numeric(selected["u5_event"], errors="coerce")
                    .eq(1)
                    .sum()
                ),
                "n_countries": int(selected["country_code"].nunique()),
                "n_surveys": int(selected["_source_file"].nunique()),
            }
        )
    return analysis, pd.DataFrame(rows)


def _aggregate_person_months(
    births: pd.DataFrame,
    *,
    chunk_size: int = 500_000,
) -> tuple[_GroupedPersonMonths, pd.DataFrame]:
    event = pd.to_numeric(births["u5_event"], errors="raise").to_numpy(int)
    time = pd.to_numeric(
        births["event_time_months"], errors="raise"
    ).to_numpy(int)
    interval_count = np.where(event == 1, time + 1, time)
    if (interval_count < 0).any() or (interval_count > 60).any():
        raise ValueError("Invalid person-month interval count")
    pattern_id = births["_pattern_id"].to_numpy(np.int32)
    n_patterns = int(pattern_id.max()) + 1
    risk_difference = np.zeros(n_patterns * 61, dtype=np.float64)
    record_difference = np.zeros(n_patterns * 61, dtype=np.int32)
    event_weight = np.zeros(n_patterns * 60, dtype=np.float64)
    event_records = np.zeros(n_patterns * 60, dtype=np.uint32)
    birth_weight = births["_birth_weight"].to_numpy(float)
    for start in range(0, len(births), chunk_size):
        stop = min(start + chunk_size, len(births))
        counts = interval_count[start:stop]
        chunk_pattern = pattern_id[start:stop].astype(np.int64)
        starts = chunk_pattern * 61
        ends = starts + counts
        chunk_weight = birth_weight[start:stop]
        np.add.at(risk_difference, starts, chunk_weight)
        np.add.at(risk_difference, ends, -chunk_weight)
        np.add.at(record_difference, starts, 1)
        np.add.at(record_difference, ends, -1)
        event_row = event[start:stop].astype(bool)
        event_slot = (
            chunk_pattern[event_row] * 60 + time[start:stop][event_row]
        )
        np.add.at(event_weight, event_slot, chunk_weight[event_row])
        np.add.at(event_records, event_slot, 1)
    risk_grid = risk_difference.reshape(n_patterns, 61)
    record_grid = record_difference.reshape(n_patterns, 61)
    np.cumsum(risk_grid[:, :60], axis=1, out=risk_grid[:, :60])
    np.cumsum(record_grid[:, :60], axis=1, out=record_grid[:, :60])
    risk_weight = risk_grid[:, :60]
    risk_records = record_grid[:, :60]
    event_weight = event_weight.reshape(n_patterns, 60)
    event_records = event_records.reshape(n_patterns, 60)
    occupied = risk_records > 0
    if not occupied.any():
        raise ValueError("No observed person-months in survival cohort")
    risk_weight[~occupied] = 0.0
    non_event_weight = risk_weight - event_weight
    if (non_event_weight < -1e-10).any():
        raise ValueError("Grouped events exceed grouped risk-set weight")
    if (non_event_weight < 0).any():
        risk_weight[non_event_weight < 0] = event_weight[non_event_weight < 0]
    risk_weight = np.ascontiguousarray(risk_weight)
    weighted_event_source = float(
        (births["_birth_weight"] * births["u5_event"]).sum()
    )
    audit = pd.DataFrame(
        [
            {"metric": "source_births", "value": int(len(births))},
            {"metric": "source_events", "value": int(event.sum())},
            {
                "metric": "literal_person_months",
                "value": int(interval_count.sum()),
            },
            {
                "metric": "grouped_person_month_records",
                "value": int(risk_records.sum()),
            },
            {
                "metric": "grouped_event_records",
                "value": int(event_records.sum()),
            },
            {
                "metric": "weighted_event_source",
                "value": weighted_event_source,
            },
            {
                "metric": "weighted_event_grouped",
                "value": float(event_weight.sum()),
            },
            {
                "metric": "month_zero_events",
                "value": int(((event == 1) & (time == 0)).sum()),
            },
            {
                "metric": "same_month_nonevent_person_months",
                "value": int(interval_count[(event == 0) & (time == 0)].sum()),
            },
            {
                "metric": "age_months_with_risk_support",
                "value": int(np.count_nonzero(occupied.any(axis=0))),
            },
        ]
    )
    return _GroupedPersonMonths(risk_weight, event_weight), audit


def _synthetic_grouped_validation() -> dict[str, float]:
    """Validate construction, grouped coefficients, and standardized risks."""
    rows: list[dict] = []
    weights = [0.7, 1.1, 0.9, 1.3, 0.8, 1.2]
    for pattern_id in range(4):
        for replicate in range(18):
            event = int(replicate % 6 < pattern_id + 1)
            event_time = replicate % 4 if event else [1, 2, 4, 6][replicate % 4]
            rows.append(
                {
                    "_pattern_id": pattern_id,
                    "_birth_weight": weights[replicate % len(weights)],
                    "u5_event": event,
                    "event_time_months": event_time,
                }
            )
    births = pd.DataFrame(rows)
    cells, _ = _aggregate_person_months(births, chunk_size=10_000)

    literal_x: list[list[float]] = []
    literal_y: list[int] = []
    literal_weight: list[float] = []
    for pattern_id, birth_weight, event, event_time in births[
        ["_pattern_id", "_birth_weight", "u5_event", "event_time_months"]
    ].itertuples(index=False, name=None):
        last_month = (
            int(event_time)
            if event == 1
            else int(event_time) - 1
        )
        for month in range(last_month + 1):
            literal_x.append([float(pattern_id), float(month)])
            literal_y.append(
                int(event == 1 and month == event_time)
            )
            literal_weight.append(float(birth_weight))
    literal = LogisticRegression(
        penalty=None, solver="lbfgs", max_iter=5000, tol=1e-12
    ).fit(
        np.asarray(literal_x),
        np.asarray(literal_y),
        sample_weight=np.asarray(literal_weight),
    )
    event_pattern, event_age = np.nonzero(cells.event_weight > 0)
    non_event_weight = cells.risk_weight - cells.event_weight
    non_event_pattern, non_event_age = np.nonzero(non_event_weight > 1e-10)
    grouped_x = np.vstack(
        [
            np.column_stack([event_pattern, event_age]),
            np.column_stack([non_event_pattern, non_event_age]),
        ]
    ).astype(float)
    grouped_y = np.r_[
        np.ones(len(event_pattern), dtype=float),
        np.zeros(len(non_event_pattern), dtype=float),
    ]
    grouped_weight = np.r_[
        cells.event_weight[event_pattern, event_age],
        non_event_weight[non_event_pattern, non_event_age],
    ]
    grouped = LogisticRegression(
        penalty=None, solver="lbfgs", max_iter=5000, tol=1e-12
    ).fit(grouped_x, grouped_y, sample_weight=grouped_weight)
    literal_parameters = np.r_[literal.intercept_, literal.coef_.ravel()]
    grouped_parameters = np.r_[grouped.intercept_, grouped.coef_.ravel()]
    coefficient_error = float(
        np.max(np.abs(literal_parameters - grouped_parameters))
    )
    risk_error = 0.0
    for pattern_id in range(4):
        prediction_rows = np.column_stack(
            [np.full(6, pattern_id, dtype=float), np.arange(6, dtype=float)]
        )
        literal_hazard = literal.predict_proba(prediction_rows)[:, 1]
        grouped_hazard = grouped.predict_proba(prediction_rows)[:, 1]
        literal_risk = 1.0 - np.prod(1.0 - literal_hazard)
        grouped_risk = 1.0 - np.prod(1.0 - grouped_hazard)
        risk_error = max(risk_error, abs(literal_risk - grouped_risk))
    return {
        "grouped_likelihood_max_abs_coefficient_error": coefficient_error,
        "grouped_standardized_risk_max_abs_error": float(risk_error),
    }


def _synthetic_factorized_validation() -> dict[str, float]:
    """Compare the compiled sufficient-statistic kernel with literal rows."""
    static = np.asarray([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    static_coefficients = np.asarray([0.25, -0.4])
    intercept = -1.2
    age_coefficients = np.asarray([0.0, 0.1, -0.2, 0.3])
    static_eta = static @ static_coefficients + intercept
    risk = np.asarray(
        [[2.0, 1.7, 1.2, 0.5], [1.5, 1.2, 0.8, 0.2], [1.0, 0.9, 0.6, 0.4]]
    )
    event = np.asarray(
        [[0.2, 0.1, 0.0, 0.0], [0.0, 0.15, 0.1, 0.0], [0.1, 0.0, 0.05, 0.02]]
    )
    multiplicity = np.asarray([1.0, 2.0, 0.5])
    factorized_loss, residual = _parallel_likelihood_kernel(
        static_eta, age_coefficients, risk, event, multiplicity
    )
    factorized_gradient = np.r_[
        residual.sum(),
        static.T @ residual.sum(axis=1),
        residual[:, 1:].sum(axis=0),
    ]
    literal_loss = 0.0
    literal_gradient = np.zeros(1 + static.shape[1] + 3, dtype=float)
    for pattern in range(len(static)):
        design_static = static[pattern]
        for month in range(4):
            eta = static_eta[pattern] + age_coefficients[month]
            hazard = float(expit(eta))
            event_weight = multiplicity[pattern] * event[pattern, month]
            non_event_weight = multiplicity[pattern] * (
                risk[pattern, month] - event[pattern, month]
            )
            literal_loss -= event_weight * math.log(hazard)
            literal_loss -= non_event_weight * math.log1p(-hazard)
            row_residual = multiplicity[pattern] * (
                risk[pattern, month] * hazard - event[pattern, month]
            )
            literal_gradient[0] += row_residual
            literal_gradient[1:3] += design_static * row_residual
            if month > 0:
                literal_gradient[2 + month] += row_residual
    return {
        "factorized_loss_max_abs_error": abs(
            float(factorized_loss.sum()) - literal_loss
        ),
        "factorized_gradient_max_abs_error": float(
            np.max(np.abs(factorized_gradient - literal_gradient))
        ),
        "preconditioned_eta_max_abs_error": float(
            np.max(
                np.abs(
                    static @ static_coefficients
                    - (static / np.asarray([0.25, 1.7]))
                    @ (static_coefficients * np.asarray([0.25, 1.7]))
                )
            )
        ),
    }


def _prepare(pilot_path: Path) -> _PreparedSurvival:
    frame = _read_pilot(pilot_path)
    births, cohort_flow = _cohort_flow(frame)
    states = _prepare_states(births, legacy=False)
    states["_birth_weight"] = births["_birth_weight"].to_numpy(float)
    states["u5_event"] = pd.to_numeric(births["u5_event"], errors="raise").to_numpy()
    states["event_time_months"] = pd.to_numeric(
        births["event_time_months"], errors="raise"
    ).to_numpy()
    pattern_keys = _STATIC_FEATURES + ["country_code"]
    pattern_levels: dict[str, pd.Index] = {}
    encoded_patterns: dict[str, np.ndarray] = {}
    for key in pattern_keys:
        categorical = pd.Categorical(states[key])
        if (categorical.codes < 0).any():
            raise ValueError(f"Missing encoded survival pattern state: {key}")
        pattern_levels[key] = categorical.categories
        encoded_patterns[key] = categorical.codes
    multi_index = pd.MultiIndex.from_frame(pd.DataFrame(encoded_patterns))
    pattern_id, unique_patterns = pd.factorize(multi_index, sort=True)
    states["_pattern_id"] = pattern_id.astype(np.int32)
    unique_codes = unique_patterns.to_frame(index=False)
    unique_codes.columns = pattern_keys
    patterns = pd.DataFrame(
        {
            key: pd.Categorical.from_codes(
                unique_codes[key].to_numpy(np.int64),
                categories=pattern_levels[key],
            )
            for key in pattern_keys
        }
    )
    patterns.insert(0, "pattern_id", np.arange(len(patterns), dtype=np.int32))
    pattern_weight = np.bincount(
        pattern_id,
        weights=states["_birth_weight"].to_numpy(float),
        minlength=len(patterns),
    )
    pattern_n = np.bincount(pattern_id, minlength=len(patterns))
    patterns["birth_weight"] = pattern_weight
    patterns["n_births"] = pattern_n
    cells, audit = _aggregate_person_months(states)
    km_point, km_n = _weighted_km_60(
        states,
        pd.Series(states["_birth_weight"].to_numpy(float), index=states.index),
    )
    n_births = int(len(states))
    n_events = int(states["u5_event"].sum())
    n_countries = int(states["country_code"].nunique())
    n_surveys = int(states["survey_id"].nunique())
    validation = {
        **_synthetic_grouped_validation(),
        **_synthetic_factorized_validation(),
    }
    audit = pd.concat(
        [
            audit,
            pd.DataFrame(
                [
                    {"metric": metric, "value": value}
                    for metric, value in validation.items()
                ]
            ),
        ],
        ignore_index=True,
    )
    return _PreparedSurvival(
        patterns=patterns,
        cells=cells,
        cohort_flow=cohort_flow,
        audit=audit,
        n_births=n_births,
        n_events=n_events,
        n_countries=n_countries,
        n_surveys=n_surveys,
        km_point=km_point,
        km_n=km_n,
    )


def _factorized_rank_diagnostics(
    static_matrix: sparse.csr_matrix,
    cells: _GroupedPersonMonths,
    age_keep: np.ndarray,
) -> tuple[bool, float]:
    """Return weighted design rank without constructing pattern-month rows."""
    augmented = sparse.hstack(
        [sparse.csr_matrix(np.ones((static_matrix.shape[0], 1))), static_matrix],
        format="csr",
    )
    pattern_support = cells.risk_weight.sum(axis=1)
    weighted = augmented.multiply(np.sqrt(pattern_support)[:, None])
    static_gram = (weighted.T @ weighted).toarray()
    retained_months = np.flatnonzero(age_keep) + 1
    cross = np.column_stack(
        [
            np.asarray(augmented.T @ cells.risk_weight[:, month]).ravel()
            for month in retained_months
        ]
    )
    age_diagonal = np.diag(cells.risk_weight[:, retained_months].sum(axis=0))
    gram = np.block([[static_gram, cross], [cross.T, age_diagonal]])
    rank = int(np.linalg.matrix_rank(gram))
    deficient = rank < gram.shape[0]
    condition = float(np.linalg.cond(gram)) if not deficient else float("inf")
    return deficient, condition


def _factorized_loss_gradient(
    parameters: np.ndarray,
    static_matrix: sparse.csr_matrix,
    age_keep: np.ndarray,
    age_scale: np.ndarray,
    pattern_multiplicity: np.ndarray,
    likelihood_backend: _CpuLikelihoodBackend | _CudaLikelihoodBackend,
    objective_scale: float,
) -> tuple[float, np.ndarray]:
    """Evaluate the exact grouped likelihood without a model-row matrix."""
    n_static = static_matrix.shape[1]
    intercept = float(parameters[0])
    static_coefficients = parameters[1 : 1 + n_static]
    age_coefficients = np.zeros(60, dtype=float)
    retained_months = np.flatnonzero(age_keep) + 1
    age_coefficients[retained_months] = (
        parameters[1 + n_static :] / age_scale
    )
    static_eta = np.asarray(static_matrix @ static_coefficients).ravel() + intercept
    loss, pattern_residual, gradient_age = likelihood_backend.evaluate(
        static_eta,
        age_coefficients,
        pattern_multiplicity,
        age_keep,
    )
    gradient_intercept = float(pattern_residual.sum())
    gradient_static = np.asarray(static_matrix.T @ pattern_residual).ravel()
    gradient = np.r_[
        gradient_intercept,
        gradient_static,
        gradient_age / age_scale,
    ]
    return loss / objective_scale, gradient / objective_scale


@njit(parallel=True, cache=True)
def _parallel_likelihood_kernel(
    static_eta: np.ndarray,
    age_coefficients: np.ndarray,
    risk_weight: np.ndarray,
    event_weight: np.ndarray,
    pattern_multiplicity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compile the independent pattern likelihood loops across CPU threads."""
    n_patterns, n_months = risk_weight.shape
    loss_by_pattern = np.zeros(n_patterns, dtype=np.float64)
    residual = np.empty((n_patterns, n_months), dtype=np.float64)
    for pattern in prange(n_patterns):
        multiplicity = pattern_multiplicity[pattern]
        pattern_loss = 0.0
        for month in range(n_months):
            eta = static_eta[pattern] + age_coefficients[month]
            if eta >= 0.0:
                negative_exp = math.exp(-eta)
                log_normalizer = eta + math.log1p(negative_exp)
                hazard = 1.0 / (1.0 + negative_exp)
            else:
                positive_exp = math.exp(eta)
                log_normalizer = math.log1p(positive_exp)
                hazard = positive_exp / (1.0 + positive_exp)
            risk = risk_weight[pattern, month]
            event = event_weight[pattern, month]
            pattern_loss += multiplicity * (
                risk * log_normalizer - event * eta
            )
            residual[pattern, month] = multiplicity * (risk * hazard - event)
        loss_by_pattern[pattern] = pattern_loss
    return loss_by_pattern, residual


def _solve_factorized(
    static_matrix: sparse.csr_matrix,
    cells: _GroupedPersonMonths,
    age_keep: np.ndarray,
    *,
    feature_scale: np.ndarray,
    likelihood_backend: _CpuLikelihoodBackend | _CudaLikelihoodBackend,
    pattern_multiplicity: np.ndarray | None = None,
    initial_model: _FactorizedLogisticModel | None = None,
    max_iter: int,
    tol: float,
) -> _FactorizedLogisticModel:
    if pattern_multiplicity is None:
        pattern_multiplicity = np.ones(cells.n_patterns, dtype=float)
    else:
        pattern_multiplicity = np.asarray(pattern_multiplicity, dtype=float)
    n_features = static_matrix.shape[1] + int(age_keep.sum())
    if len(feature_scale) != n_features or (feature_scale <= 0).any():
        raise ValueError("Factorized feature preconditioner is invalid")
    age_scale = feature_scale[static_matrix.shape[1] :]
    objective_scale = float(
        np.asarray(cells.risk_weight.T @ pattern_multiplicity).sum()
    )
    if not np.isfinite(objective_scale) or objective_scale <= 0:
        raise ValueError("Factorized likelihood has no positive risk weight")
    if initial_model is None:
        initial = np.zeros(n_features + 1, dtype=float)
        risk_by_month = cells.risk_weight.T @ pattern_multiplicity
        event_by_month = cells.event_weight.T @ pattern_multiplicity
        monthly_hazard = np.divide(
            event_by_month,
            risk_by_month,
            out=np.full(60, 0.5, dtype=float),
            where=risk_by_month > 0,
        )
        monthly_logit = np.log(
            np.clip(monthly_hazard, 1e-8, 1 - 1e-8)
            / np.clip(1 - monthly_hazard, 1e-8, 1 - 1e-8)
        )
        initial[0] = monthly_logit[0]
        retained_months = np.flatnonzero(age_keep) + 1
        initial[1 + static_matrix.shape[1] :] = (
            monthly_logit[retained_months] - monthly_logit[0]
        ) * age_scale
    else:
        initial = np.r_[
            initial_model.intercept_[0],
            initial_model.coef_[0] * feature_scale,
        ]
        if len(initial) != n_features + 1:
            raise ValueError("Factorized warm start has an incompatible feature count")
    result = minimize(
        _factorized_loss_gradient,
        initial,
        args=(
            static_matrix,
            age_keep,
            age_scale,
            pattern_multiplicity,
            likelihood_backend,
            objective_scale,
        ),
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": max_iter,
            "ftol": 1e-12,
            "gtol": tol,
            "maxls": 50,
        },
    )
    return _FactorizedLogisticModel(
        coef_=np.asarray(result.x[1:] / feature_scale, dtype=float)[None, :],
        intercept_=np.asarray([result.x[0]], dtype=float),
        n_iter_=np.asarray([result.nit], dtype=int),
        max_iter=max_iter,
        classes_=np.asarray([0.0, 1.0]),
        n_features_in_=n_features,
        success=bool(result.success),
        message=str(result.message),
        objective=float(result.fun),
        max_abs_scaled_gradient=float(np.max(np.abs(result.jac))),
    )


def _fit_model(
    cells: _GroupedPersonMonths,
    patterns: pd.DataFrame,
    *,
    compute_backend: str,
) -> _SurvivalFit:
    static_encoder = OneHotEncoder(
        categories=_static_categories_for(patterns),
        drop="first",
        handle_unknown="ignore",
        sparse_output=True,
        dtype=np.float64,
    )
    static_patterns = static_encoder.fit_transform(
        patterns[_STATIC_FEATURES].astype(str)
    ).tocsr()
    pattern_support = cells.risk_weight.sum(axis=1)
    static_support = np.asarray(static_patterns.T @ pattern_support).ravel()
    age_support = cells.risk_weight[:, 1:].sum(axis=0)
    weighted_support = np.r_[static_support, age_support]
    total_weight = float(pattern_support.sum())
    keep = (weighted_support > 0) & (weighted_support < total_weight)
    static_feature_count = static_patterns.shape[1]
    static_keep = keep[:static_feature_count]
    age_keep = keep[static_feature_count:]
    unscaled_static = static_patterns[:, static_keep].tocsr()
    static_mean = static_support[static_keep] / total_weight
    age_mean = age_support[age_keep] / total_weight
    static_scale = np.sqrt(np.clip(static_mean * (1.0 - static_mean), 1e-12, None))
    age_scale = np.sqrt(np.clip(age_mean * (1.0 - age_mean), 1e-12, None))
    feature_scale = np.r_[static_scale, age_scale]
    static_matrix = unscaled_static.multiply(1.0 / static_scale).tocsr()
    full_feature_names = np.r_[
        static_encoder.get_feature_names_out(_STATIC_FEATURES),
        np.array([f"age_month_{month}" for month in range(1, 60)]),
    ]
    feature_names = full_feature_names[keep]
    rank_deficient, condition_number = _factorized_rank_diagnostics(
        unscaled_static, cells, age_keep
    )
    likelihood_backend = _likelihood_backend(cells, compute_backend)
    model = _solve_factorized(
        static_matrix,
        cells,
        age_keep,
        feature_scale=feature_scale,
        likelihood_backend=likelihood_backend,
        max_iter=3000,
        tol=1e-8,
    )
    n_iter = int(np.max(model.n_iter_))
    return _SurvivalFit(
        static_encoder=static_encoder,
        keep_columns=keep,
        feature_names=feature_names,
        static_matrix=static_matrix,
        factual_static_matrix=unscaled_static,
        static_feature_count=static_feature_count,
        feature_scale=feature_scale,
        cells=cells,
        likelihood_backend=likelihood_backend,
        model=model,
        converged=(
            model.success
            and n_iter < model.max_iter
            and model.max_abs_scaled_gradient <= _POINT_MAX_ABS_SCALED_GRADIENT
        ),
        n_iter=n_iter,
        rank_deficient=rank_deficient,
        condition_number=condition_number,
        warning_text=(
            ""
            if model.success
            and model.max_abs_scaled_gradient <= _POINT_MAX_ABS_SCALED_GRADIENT
            else model.message
        ),
    )


def _scenario_patterns(patterns: pd.DataFrame, scenario: str) -> tuple[pd.DataFrame, np.ndarray]:
    changed = patterns.copy()
    modified = np.zeros(len(changed), dtype=bool)
    if scenario in {"education", "joint"}:
        eligible = changed["edu_cat"].isin(["None", "Primary"]).to_numpy()
        changed.loc[eligible, "edu_cat"] = "Secondary"
        modified |= eligible
    if scenario in {"spacing", "joint"}:
        eligible = changed["bi_model_cat"].eq("<24 mo").to_numpy()
        changed.loc[eligible, "bi_model_cat"] = "24-47 mo"
        modified |= eligible
    return changed, modified


def _pattern_risk(
    fit: _SurvivalFit,
    patterns: pd.DataFrame,
    *,
    chunk_size: int = 20_000,
) -> np.ndarray:
    static_feature_count = len(
        fit.static_encoder.get_feature_names_out(_STATIC_FEATURES)
    )
    full_coefficients = np.zeros(static_feature_count + 59, dtype=float)
    full_coefficients[fit.keep_columns] = fit.model.coef_[0]
    static_coefficients = full_coefficients[:static_feature_count]
    age_coefficients = np.r_[0.0, full_coefficients[static_feature_count:]]
    risks = np.empty(len(patterns), dtype=float)
    for start in range(0, len(patterns), chunk_size):
        stop = min(start + chunk_size, len(patterns))
        chunk = patterns.iloc[start:stop]
        static = fit.static_encoder.transform(
            chunk[_STATIC_FEATURES].astype(str)
        ).tocsr()
        static_eta = (
            np.asarray(static @ static_coefficients).ravel()
            + float(fit.model.intercept_[0])
        )
        hazard = expit(static_eta[:, None] + age_coefficients[None, :])
        if not np.isfinite(hazard).all() or ((hazard < 0) | (hazard > 1)).any():
            raise ValueError("Non-finite or out-of-bounds monthly hazard")
        chunk_risk = 1.0 - np.prod(1.0 - hazard, axis=1)
        if not np.isfinite(chunk_risk).all() or (
            (chunk_risk < 0) | (chunk_risk > 1)
        ).any():
            raise ValueError("Non-finite or out-of-bounds 60-month risk")
        risks[start:stop] = chunk_risk
    return risks


def _factorized_risk_inputs(
    fit: _SurvivalFit,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Recover factual static eta, age effects, and retained coefficients."""
    n_static_kept = int(fit.keep_columns[: fit.static_feature_count].sum())
    coefficients = fit.model.coef_[0]
    static_eta = (
        np.asarray(
            fit.factual_static_matrix @ coefficients[:n_static_kept]
        ).ravel()
        + float(fit.model.intercept_[0])
    )
    age_coefficients = np.zeros(60, dtype=np.float64)
    age_keep = fit.keep_columns[fit.static_feature_count :]
    age_coefficients[np.flatnonzero(age_keep) + 1] = coefficients[n_static_kept:]
    coefficient_lookup = {
        str(feature): float(value)
        for feature, value in zip(fit.feature_names, coefficients, strict=True)
    }
    return static_eta, age_coefficients, coefficient_lookup


def _factorized_standardization_error(
    fit: _SurvivalFit,
    patterns: pd.DataFrame,
    *,
    n_patterns: int = 100,
) -> float:
    """Compare backend factorization with literal pattern prediction."""
    n_check = min(n_patterns, len(patterns))
    literal = _pattern_risk(fit, patterns.iloc[:n_check].copy())
    static_eta, age_coefficients, _ = _factorized_risk_inputs(fit)
    factorized = fit.likelihood_backend.risks_from_eta(
        static_eta[:n_check], age_coefficients
    )
    return float(np.max(np.abs(factorized - literal)))


def _algebraic_standardization_error(
    fit: _SurvivalFit,
    patterns: pd.DataFrame,
    *,
    n_patterns: int = 100,
) -> float:
    """Compare the fast additive calculation with literal expanded prediction."""
    check = patterns.iloc[: min(n_patterns, len(patterns))].copy()
    algebraic = _pattern_risk(fit, check)
    static = fit.static_encoder.transform(check[_STATIC_FEATURES].astype(str)).tocsr()
    static_rows = static[np.repeat(np.arange(len(check)), 60)]
    age = np.tile(np.arange(60), len(check))
    age_rows = sparse.csr_matrix(
        (
            np.ones(len(age), dtype=float),
            (np.arange(len(age)), age),
        ),
        shape=(len(age), 60),
    )[:, 1:]
    matrix = sparse.hstack([static_rows, age_rows], format="csr")
    matrix = matrix[:, fit.keep_columns]
    literal_hazard = fit.model.predict_proba(matrix)[:, 1].reshape(len(check), 60)
    literal = 1.0 - np.prod(1.0 - literal_hazard, axis=1)
    return float(np.max(np.abs(algebraic - literal)))


def _standardized_rows(
    fit: _SurvivalFit,
    patterns: pd.DataFrame,
    pattern_weight: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    static_eta, age_coefficients, coefficients = _factorized_risk_inputs(fit)
    factual = fit.likelihood_backend.risks_from_eta(
        static_eta, age_coefficients
    )
    factual_mean = float(np.average(factual, weights=pattern_weight))
    if not np.isfinite(factual_mean) or not 0 <= factual_mean <= 1:
        raise ValueError("Invalid standardized factual risk")
    risks = {"factual": factual}
    rows: list[dict] = []
    edu_none = patterns["edu_cat"].eq("None").to_numpy()
    edu_primary = patterns["edu_cat"].eq("Primary").to_numpy()
    edu_modified = edu_none | edu_primary
    spacing_modified = patterns["bi_model_cat"].eq("<24 mo").to_numpy()
    education_delta = np.zeros(len(patterns), dtype=np.float64)
    education_delta[edu_none] = coefficients.get("edu_cat_Secondary", 0.0)
    education_delta[edu_primary] = (
        coefficients.get("edu_cat_Secondary", 0.0)
        - coefficients.get("edu_cat_Primary", 0.0)
    )
    spacing_delta = np.zeros(len(patterns), dtype=np.float64)
    spacing_delta[spacing_modified] = -coefficients.get(
        "bi_model_cat_<24 mo", 0.0
    )
    scenario_inputs = {
        "education": (education_delta, edu_modified),
        "spacing": (spacing_delta, spacing_modified),
        "joint": (
            education_delta + spacing_delta,
            edu_modified | spacing_modified,
        ),
    }
    for scenario in _SCENARIOS:
        delta, modified = scenario_inputs[scenario]
        counterfactual = fit.likelihood_backend.risks_from_eta(
            static_eta + delta, age_coefficients
        )
        risks[scenario] = counterfactual
        counterfactual_mean = float(
            np.average(counterfactual, weights=pattern_weight)
        )
        if not np.isfinite(counterfactual_mean) or not 0 <= counterfactual_mean <= 1:
            raise ValueError("Invalid standardized counterfactual risk")
        increased = counterfactual > factual + 1e-12
        rows.append(
            {
                "scenario": scenario,
                "n_patterns": int(len(patterns)),
                "n_modified_patterns": int(modified.sum()),
                "factual_risk": factual_mean,
                "counterfactual_risk": counterfactual_mean,
                "risk_difference_per_1000":
                    (factual_mean - counterfactual_mean) * 1000,
                "risk_ratio": counterfactual_mean / factual_mean,
                "paf_pct":
                    (factual_mean - counterfactual_mean) / factual_mean * 100,
                "n_risk_increased_patterns": int((increased & modified).sum()),
                "max_risk_increase": float(
                    np.max(counterfactual[increased] - factual[increased])
                    if increased.any()
                    else 0.0
                ),
                "n_invalid_probabilities": 0,
            }
        )
    sequential_risk = fit.likelihood_backend.risks_from_eta(
        (static_eta + education_delta) + spacing_delta,
        age_coefficients,
    )
    joint_error = float(np.max(np.abs(sequential_risk - risks["joint"])))
    for row in rows:
        row["joint_same_surface_max_abs_error"] = joint_error
    return pd.DataFrame(rows), risks


def _bootstrap(
    fit: _SurvivalFit,
    patterns: pd.DataFrame,
    cells: _GroupedPersonMonths,
    *,
    bootstrap_reps: int,
    seed: int,
    cache_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    draw_frame = patterns[["country_code"]].copy()
    draws = _country_draws(draw_frame, bootstrap_reps=bootstrap_reps, seed=seed)
    values: list[dict] = []
    diagnostics: list[dict] = []
    base_pattern_weight = patterns["birth_weight"].to_numpy(float)
    countries = np.array(sorted(patterns["country_code"].astype(str).unique()))
    country_index = {country: index for index, country in enumerate(countries)}
    pattern_country_index = np.fromiter(
        (country_index[country] for country in patterns["country_code"].astype(str)),
        dtype=np.int16,
        count=len(patterns),
    )
    model = copy.deepcopy(fit.model)
    model.max_iter = 2000
    completed: set[int] = set()
    values_path = diagnostics_path = state_path = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        values_path = cache_dir / "bootstrap_values.csv"
        diagnostics_path = cache_dir / "bootstrap_diagnostics.csv"
        state_path = cache_dir / "warm_state.npz"
        if values_path.exists() and diagnostics_path.exists():
            cached_values = pd.read_csv(values_path)
            cached_diagnostics = pd.read_csv(diagnostics_path)
            values = cached_values.to_dict("records")
            diagnostics = cached_diagnostics.to_dict("records")
            completed = set(cached_diagnostics["replicate"].astype(int))
            if completed != set(range(len(completed))):
                raise ValueError("Phase 3 bootstrap cache is not contiguous")
            if completed and state_path.exists():
                with np.load(state_path) as state:
                    if int(state["replicate"]) != max(completed):
                        raise ValueError("Phase 3 bootstrap warm state is stale")
                    model.coef_ = state["coef"]
                    model.intercept_ = state["intercept"]
                    model.classes_ = state["classes"]
                    model.n_features_in_ = len(fit.feature_names)

    def checkpoint(replicate: int) -> None:
        if cache_dir is None:
            return
        assert values_path is not None
        assert diagnostics_path is not None
        assert state_path is not None
        values_temp = values_path.with_suffix(".building.csv")
        diagnostics_temp = diagnostics_path.with_suffix(".building.csv")
        state_temp = state_path.with_suffix(".building.npz")
        pd.DataFrame(
            values,
            columns=["replicate", "scenario", "metric", "value"],
        ).to_csv(values_temp, index=False)
        pd.DataFrame(
            diagnostics,
            columns=[
                "replicate",
                "success",
                "converged",
                "n_iter",
                "max_abs_scaled_gradient",
                "error",
            ],
        ).to_csv(diagnostics_temp, index=False)
        with state_temp.open("wb") as handle:
            np.savez(
                handle,
                replicate=np.array(replicate),
                coef=model.coef_,
                intercept=model.intercept_,
                classes=model.classes_,
            )
        values_temp.replace(values_path)
        diagnostics_temp.replace(diagnostics_path)
        state_temp.replace(state_path)

    for replicate, draw in enumerate(draws):
        if replicate in completed:
            continue
        country_multiplicity = np.zeros(len(countries), dtype=float)
        for country, count in draw.items():
            country_multiplicity[country_index[country]] = count
        pattern_multiplicity = country_multiplicity[pattern_country_index]
        pattern_weight = base_pattern_weight * pattern_multiplicity
        try:
            model = _solve_factorized(
                fit.static_matrix,
                cells,
                fit.keep_columns[fit.static_feature_count :],
                feature_scale=fit.feature_scale,
                likelihood_backend=fit.likelihood_backend,
                pattern_multiplicity=pattern_multiplicity,
                initial_model=model,
                max_iter=2000,
                tol=1e-6,
            )
            n_iter = int(np.max(model.n_iter_))
            if (
                not model.success
                or n_iter >= model.max_iter
                or model.max_abs_scaled_gradient > _BOOTSTRAP_MAX_ABS_SCALED_GRADIENT
            ):
                raise RuntimeError(
                    f"{model.message}; max_abs_scaled_gradient="
                    f"{model.max_abs_scaled_gradient:.3e}"
                )
            draw_fit = copy.copy(fit)
            draw_fit.model = model
            standardized, _ = _standardized_rows(
                draw_fit, patterns, pattern_weight
            )
            risk_weight_by_month = np.asarray(
                cells.risk_weight.T @ pattern_multiplicity
            ).ravel()
            event_weight_by_month = np.asarray(
                cells.event_weight.T @ pattern_multiplicity
            ).ravel()
            if (risk_weight_by_month <= 0).any():
                raise ValueError("bootstrap draw has an empty age-month risk set")
            km_risk = float(
                1.0
                - np.prod(
                    1.0 - event_weight_by_month / risk_weight_by_month
                )
            )
            values.append(
                {
                    "replicate": replicate,
                    "scenario": "nonparametric",
                    "metric": "km_60_risk",
                    "value": km_risk,
                }
            )
            for row in standardized.itertuples(index=False):
                for metric, value in (
                    ("paf_pct", row.paf_pct),
                    ("risk_difference_per_1000", row.risk_difference_per_1000),
                    ("risk_ratio", row.risk_ratio),
                    ("factual_risk", row.factual_risk),
                ):
                    if not np.isfinite(value):
                        raise ValueError("non-finite bootstrap metric")
                    values.append(
                        {
                            "replicate": replicate,
                            "scenario": row.scenario,
                            "metric": metric,
                            "value": value,
                        }
                    )
            diagnostics.append(
                {
                    "replicate": replicate,
                    "success": True,
                    "converged": True,
                    "n_iter": n_iter,
                    "max_abs_scaled_gradient": model.max_abs_scaled_gradient,
                    "error": "",
                }
            )
            checkpoint(replicate)
        except Exception as exc:
            values = [row for row in values if row["replicate"] != replicate]
            diagnostics.append(
                {
                    "replicate": replicate,
                    "success": False,
                    "converged": False,
                    "n_iter": 0,
                    "max_abs_scaled_gradient": np.nan,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            model = copy.deepcopy(fit.model)
            model.max_iter = 2000
            checkpoint(replicate)
        if (replicate + 1) % 10 == 0 or replicate + 1 == bootstrap_reps:
            print(
                f"[phase3] bootstrap {replicate + 1}/{bootstrap_reps}",
                flush=True,
            )
    return pd.DataFrame(values), pd.DataFrame(diagnostics)


def _bootstrap_summary(
    values: pd.DataFrame,
    point: pd.DataFrame,
    km_point: float,
    *,
    n_requested: int,
) -> pd.DataFrame:
    point_lookup: dict[tuple[str, str], float] = {}
    for row in point.itertuples(index=False):
        point_lookup[(row.scenario, "paf_pct")] = float(row.paf_pct)
        point_lookup[(row.scenario, "risk_difference_per_1000")] = float(
            row.risk_difference_per_1000
        )
        point_lookup[(row.scenario, "risk_ratio")] = float(row.risk_ratio)
        point_lookup[(row.scenario, "factual_risk")] = float(row.factual_risk)
    point_lookup[("nonparametric", "km_60_risk")] = float(km_point)
    rows: list[dict] = []
    for (scenario, metric), group in values.groupby(["scenario", "metric"]):
        finite = pd.to_numeric(group["value"], errors="coerce").dropna()
        rows.append(
            {
                "scenario": scenario,
                "metric": metric,
                "point_estimate": point_lookup[(scenario, metric)],
                "ci_lower": float(finite.quantile(0.025)),
                "ci_upper": float(finite.quantile(0.975)),
                "n_success": int(len(finite)),
                "n_requested": int(n_requested),
            }
        )
    return pd.DataFrame(rows)


def _phase2_reference(phase2_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    paf_path = phase2_dir / "phase2_pafs.csv"
    coefficient_path = phase2_dir / "phase2_model_coefficients.csv"
    manifest_path = phase2_dir / "phase2_run_manifest.json"
    for path in (paf_path, coefficient_path, manifest_path):
        if not path.exists():
            raise FileNotFoundError(f"Phase 2 reference artifact not found: {path}")
    pafs = pd.read_csv(paf_path)
    coefficients = pd.read_csv(coefficient_path)
    primary = pafs[
        pafs["surface"].eq("exact_wave_fe_complete_wealth")
        & pafs["contract"].eq("policy_eligible")
    ].copy()
    primary_coefficients = coefficients[
        coefficients["surface"].eq("exact_wave_fe_complete_wealth")
    ].copy()
    return primary, primary_coefficients


def _stop_rules(
    fit: _SurvivalFit,
    point: pd.DataFrame,
    bootstrap_diagnostics: pd.DataFrame,
    audit: pd.DataFrame,
    *,
    bootstrap_reps: int,
    phase2_dir: Path,
) -> tuple[pd.DataFrame, str]:
    audit_values = audit.set_index("metric")["value"]
    rules: list[dict] = []

    def add(rule_id, decision_class, value, threshold, triggered, detail=""):
        rules.append(
            {
                "rule_id": rule_id,
                "decision_class": decision_class,
                "value": value,
                "threshold": threshold,
                "triggered": bool(triggered),
                "evaluable": True,
                "detail": detail,
            }
        )

    arithmetic_error = max(
        abs(
            float(audit_values["source_events"])
            - float(audit_values["grouped_event_records"])
        ),
        abs(
            float(audit_values["literal_person_months"])
            - float(audit_values["grouped_person_month_records"])
        ),
        abs(
            float(audit_values["weighted_event_source"])
            - float(audit_values["weighted_event_grouped"])
        ),
    )
    add(
        "person_month_arithmetic",
        "DATA_INTEGRITY_FAILURE",
        arithmetic_error,
        "absolute discrepancy <=1e-8",
        arithmetic_error > 1e-8,
    )
    grouped_error = float(
        audit_values["grouped_likelihood_max_abs_coefficient_error"]
    )
    add(
        "grouped_likelihood_equivalence",
        "METHOD_FAILURE",
        grouped_error,
        "max coefficient difference <1e-8",
        grouped_error >= 1e-8,
    )
    grouped_risk_error = float(
        audit_values["grouped_standardized_risk_max_abs_error"]
    )
    add(
        "grouped_standardized_risk_equivalence",
        "METHOD_FAILURE",
        grouped_risk_error,
        "max standardized risk difference <1e-8",
        grouped_risk_error >= 1e-8,
    )
    factorized_loss_error = float(
        audit_values["factorized_loss_max_abs_error"]
    )
    factorized_gradient_error = float(
        audit_values["factorized_gradient_max_abs_error"]
    )
    add(
        "factorized_likelihood_equivalence",
        "METHOD_FAILURE",
        factorized_loss_error,
        "literal-versus-factorized loss difference <1e-10",
        factorized_loss_error >= 1e-10,
    )
    add(
        "factorized_gradient_equivalence",
        "METHOD_FAILURE",
        factorized_gradient_error,
        "literal-versus-factorized gradient difference <1e-10",
        factorized_gradient_error >= 1e-10,
    )
    preconditioned_eta_error = float(
        audit_values["preconditioned_eta_max_abs_error"]
    )
    add(
        "preconditioned_eta_equivalence",
        "METHOD_FAILURE",
        preconditioned_eta_error,
        "original-versus-preconditioned linear predictor difference <1e-12",
        preconditioned_eta_error >= 1e-12,
    )
    standardization_error = float(
        audit_values["algebraic_standardization_max_abs_error"]
    )
    add(
        "algebraic_standardization_equivalence",
        "METHOD_FAILURE",
        standardization_error,
        "max 60-month risk difference <=1e-12",
        standardization_error > 1e-12,
    )
    factorized_standardization_error = float(
        audit_values["factorized_standardization_max_abs_error"]
    )
    add(
        "factorized_standardization_equivalence",
        "METHOD_FAILURE",
        factorized_standardization_error,
        "literal-versus-factorized 60-month risk difference <=1e-12",
        factorized_standardization_error > 1e-12,
    )
    add(
        "point_model_valid",
        "METHOD_FAILURE",
        f"converged={fit.converged};rank_deficient={fit.rank_deficient}",
        "converged and full rank",
        not fit.converged or fit.rank_deficient,
    )
    add(
        "point_scaled_gradient",
        "METHOD_FAILURE",
        fit.model.max_abs_scaled_gradient,
        "maximum absolute gradient divided by total risk weight "
        f"<={_POINT_MAX_ABS_SCALED_GRADIENT:g}",
        fit.model.max_abs_scaled_gradient > _POINT_MAX_ABS_SCALED_GRADIENT,
    )
    age_support = int(audit_values["age_months_with_risk_support"])
    add(
        "all_age_months_supported",
        "METHOD_FAILURE",
        age_support,
        "60 months",
        age_support != 60,
    )
    increased = int(point["n_risk_increased_patterns"].sum())
    invalid_probabilities = int(point["n_invalid_probabilities"].sum())
    add(
        "hazard_and_risk_validity",
        "METHOD_FAILURE",
        invalid_probabilities,
        "0 non-finite or out-of-bounds probabilities",
        invalid_probabilities > 0,
    )
    add(
        "counterfactual_monotonicity",
        "METHOD_FAILURE",
        increased,
        "0 eligible patterns with increased risk",
        increased > 0,
    )
    identity = float(point["joint_same_surface_max_abs_error"].max())
    add(
        "joint_same_surface_identity",
        "METHOD_FAILURE",
        identity,
        "<=1e-12",
        identity > 1e-12,
    )
    add(
        "bootstrap_replicates_requested",
        "METHOD_FAILURE",
        bootstrap_reps,
        ">=200 for decision pilot",
        bootstrap_reps < 200,
    )
    successful = int(bootstrap_diagnostics["success"].sum())
    required = max(190, math.ceil(0.95 * bootstrap_reps))
    add(
        "paired_bootstrap_success",
        "METHOD_FAILURE",
        successful,
        f">={required}",
        successful < required,
    )
    reference, reference_coefficients = _phase2_reference(phase2_dir)
    for scenario in _SCENARIOS:
        survival = float(point.set_index("scenario").loc[scenario, "paf_pct"])
        binary = float(reference.set_index("scenario").loc[scenario, "paf_pct"])
        absolute = abs(survival - binary)
        relative = absolute / abs(binary) if abs(binary) >= 1 else 0.0
        add(
            f"binary_survival_paf_shift_{scenario}",
            "NARRATIVE_ESCALATION",
            absolute,
            ">=5 percentage points OR >=20% relative",
            absolute >= 5 or relative >= 0.20,
            f"survival={survival:.3f}; binary={binary:.3f}; relative={relative:.1%}",
        )
    indexed = point.set_index("scenario")["paf_pct"]
    add(
        "lever_order_reversal",
        "NARRATIVE_ESCALATION",
        bool(indexed["education"] > indexed["spacing"]),
        "education>spacing remains True",
        not bool(indexed["education"] > indexed["spacing"]),
    )
    coefficient_rows = pd.DataFrame(
        {"feature": fit.feature_names, "coefficient": fit.model.coef_[0]}
    ).set_index("feature")
    reference_coefficients = reference_coefficients.set_index("feature")
    nuisance_prefixes = ("survey_id_", "birth_period_", "age_month_")
    common_features = sorted(
        set(coefficient_rows.index).intersection(reference_coefficients.index)
    )
    substantive_features = [
        feature
        for feature in common_features
        if not feature.startswith(nuisance_prefixes)
        and "Missing" not in feature
        and "missing" not in feature
    ]
    flipped_features: list[str] = []
    for feature in substantive_features:
        current = float(coefficient_rows.loc[feature, "coefficient"])
        previous = float(reference_coefficients.loc[feature, "coefficient"])
        if abs(current) >= 0.02 and abs(previous) >= 0.02 and np.sign(current) != np.sign(previous):
            flipped_features.append(feature)
    add(
        "headline_coefficient_sign_flip",
        "NARRATIVE_ESCALATION",
        len(flipped_features),
        "0 common substantive flips outside abs(log OR)<0.02 tie band",
        bool(flipped_features),
        ";".join(flipped_features),
    )
    table = pd.DataFrame(rules)
    if table.loc[
        table["decision_class"].eq("DATA_INTEGRITY_FAILURE"), "triggered"
    ].any():
        decision = "DATA_INTEGRITY_FAILURE"
    elif table.loc[table["decision_class"].eq("METHOD_FAILURE"), "triggered"].any():
        decision = "METHOD_FAILURE"
    elif table.loc[
        table["decision_class"].eq("NARRATIVE_ESCALATION"), "triggered"
    ].any():
        decision = "NARRATIVE_ESCALATION"
    else:
        decision = "PASS"
    return table, decision


def _manifest(
    pilot_path: Path,
    prepared: _PreparedSurvival,
    phase2_dir: Path,
    *,
    bootstrap_reps: int,
    seed: int,
    likelihood_backend: str,
    checkpoint_bootstrap: bool,
) -> dict:
    root = Path(__file__).resolve().parents[2]
    protocol = root / "docs" / "PHASE3_IMPLEMENTATION_PROTOCOL_2026-07-12.md"
    amendment = root / "docs" / "PHASE3_METHOD_AMENDMENT_2026-07-12.md"
    compute_amendment = (
        root / "docs" / "PHASE3_COMPUTE_OPTIMIZATION_AMENDMENT_2026-07-13.md"
    )
    parent_plan = root / "docs" / "ANALYSIS_PLAN_V2_2026-07-11.md"
    canonical = HARMONIZED_DIR / "global_analysis.parquet"
    phase2_manifest = phase2_dir / "phase2_run_manifest.json"
    phase2_pafs = phase2_dir / "phase2_pafs.csv"
    phase2_coefficients = phase2_dir / "phase2_model_coefficients.csv"
    match = re.search(r"step(\d+)", pilot_path.name)
    backend_runtime: dict[str, str | int | None] = {
        "name": likelihood_backend,
    }
    if likelihood_backend == "cuda-torch-float64":
        import torch

        backend_runtime.update(
            {
                "torch_version": torch.__version__,
                "cuda_runtime_version": torch.version.cuda,
                "cuda_device_name": torch.cuda.get_device_name(),
                "cuda_device_total_memory_bytes": int(
                    torch.cuda.get_device_properties(0).total_memory
                ),
            }
        )
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pilot_path": str(pilot_path.resolve()),
        "pilot_sha256": _sha256(pilot_path),
        "sample_step": int(match.group(1)) if match else None,
        "analysis_code_path": str(Path(__file__).resolve()),
        "analysis_code_sha256": _sha256(Path(__file__).resolve()),
        "parent_plan_path": str(parent_plan.resolve()),
        "parent_plan_sha256": _sha256(parent_plan),
        "phase3_protocol_path": str(protocol.resolve()),
        "phase3_protocol_sha256": _sha256(protocol),
        "phase3_amendment_path": str(amendment.resolve()),
        "phase3_amendment_sha256": _sha256(amendment),
        "phase3_compute_amendment_path": str(compute_amendment.resolve()),
        "phase3_compute_amendment_sha256": _sha256(compute_amendment),
        "frozen_canonical_path": str(canonical.resolve()),
        "frozen_canonical_sha256": _sha256(canonical),
        "phase2_run_manifest_path": str(phase2_manifest.resolve()),
        "phase2_run_manifest_sha256": _sha256(phase2_manifest),
        "phase2_pafs_path": str(phase2_pafs.resolve()),
        "phase2_pafs_sha256": _sha256(phase2_pafs),
        "phase2_coefficients_path": str(phase2_coefficients.resolve()),
        "phase2_coefficients_sha256": _sha256(phase2_coefficients),
        "bootstrap_reps": int(bootstrap_reps),
        "bootstrap_checkpointing": checkpoint_bootstrap,
        "seed": int(seed),
        "age_intervals": 60,
        "age_baseline": "unrestricted categorical completed months 0-59",
        "link": "logit",
        "model": "unpenalized pooled logistic monthly hazard",
        "weight_normalization": (
            "positive weights normalized to mean one within wave on the full "
            "known-survival cohort before observed-wealth restriction"
        ),
        "n_analysis_births": prepared.n_births,
        "n_countries": prepared.n_countries,
        "n_surveys": prepared.n_surveys,
        "individual_person_months_written": False,
        "likelihood_representation": (
            "factorized grouped Bernoulli sufficient statistics"
        ),
        "materialized_pattern_month_design": False,
        "likelihood_backend": likelihood_backend,
        "backend_runtime": backend_runtime,
        "optimizer_objective_scaling": "divide loss and gradient by total risk weight",
        "point_max_abs_scaled_gradient_threshold": _POINT_MAX_ABS_SCALED_GRADIENT,
        "bootstrap_max_abs_scaled_gradient_threshold": (
            _BOOTSTRAP_MAX_ABS_SCALED_GRADIENT
        ),
        "publication_bootstrap_reps": 1000,
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "numba_version": numba.__version__,
        "scikit_learn_version": sklearn.__version__,
    }


def _bootstrap_cache_dir(
    pilot_path: Path,
    *,
    bootstrap_reps: int,
    seed: int,
    likelihood_backend: str,
) -> Path:
    code_path = Path(__file__).resolve()
    root = code_path.parents[2]
    payload = {
        "pilot_sha256": _sha256(pilot_path),
        "analysis_code_sha256": _sha256(code_path),
        "bootstrap_reps": int(bootstrap_reps),
        "seed": int(seed),
        "likelihood_backend": likelihood_backend,
    }
    cache_dir = (
        root
        / "tmp"
        / "phase3_bootstrap_jobs"
        / (
            f"{payload['pilot_sha256'][:12]}_"
            f"{payload['analysis_code_sha256'][:12]}_"
            f"{likelihood_backend.replace('-', '')}_b{bootstrap_reps}_s{seed}"
        )
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "job_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError("Phase 3 bootstrap cache manifest mismatch")
    else:
        manifest_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
    return cache_dir


def run_survival_validation(
    pilot_path: str | Path,
    output_dir: str | Path,
    *,
    phase2_dir: str | Path,
    require_decision_pilot_provenance: bool = True,
    bootstrap_reps: int = 200,
    seed: int = RNG_SEED,
    compute_backend: str = "auto",
    checkpoint_bootstrap: bool | None = None,
) -> SurvivalValidationResult:
    """Run the frozen Phase 3 survival gate and write aggregate artifacts."""
    if bootstrap_reps < 1:
        raise ValueError("bootstrap_reps must be at least 1")
    if checkpoint_bootstrap is None:
        checkpoint_bootstrap = (
            require_decision_pilot_provenance or bootstrap_reps >= 200
        )
    pilot_path = Path(pilot_path)
    output_dir = Path(output_dir)
    phase2_dir = Path(phase2_dir)
    if not pilot_path.exists():
        raise FileNotFoundError(pilot_path)
    _phase2_reference(phase2_dir)  # fail before expensive construction/fitting
    if require_decision_pilot_provenance:
        _validate_decision_pilot_provenance(pilot_path)
    prepared = _prepare(pilot_path)
    fit = _fit_model(
        prepared.cells,
        prepared.patterns,
        compute_backend=compute_backend,
    )
    bootstrap_cache = (
        _bootstrap_cache_dir(
            pilot_path,
            bootstrap_reps=bootstrap_reps,
            seed=seed,
            likelihood_backend=fit.likelihood_backend.name,
        )
        if checkpoint_bootstrap
        else None
    )
    prepared.audit = pd.concat(
        [
            prepared.audit,
            pd.DataFrame(
                [
                    {
                        "metric": "factorized_standardization_max_abs_error",
                        "value": _factorized_standardization_error(
                            fit, prepared.patterns
                        ),
                    },
                    {
                        "metric": "algebraic_standardization_max_abs_error",
                        "value": _algebraic_standardization_error(
                            fit, prepared.patterns
                        ),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    point, _ = _standardized_rows(
        fit, prepared.patterns, prepared.patterns["birth_weight"].to_numpy(float)
    )
    km_point, km_n = prepared.km_point, prepared.km_n
    risk_comparison = pd.DataFrame(
        [
            {
                "estimand": "weighted_km_60_month_risk_same_cohort",
                "estimate": km_point,
                "n_births": km_n,
                "label": "Nonparametric weighted risk on Phase 3 analytic cohort",
            },
            {
                "estimand": "model_standardized_60_month_risk",
                "estimate": float(point["factual_risk"].iloc[0]),
                "n_births": prepared.n_births,
                "label": "Birth-period-adjusted pooled-logistic standardisation",
            },
        ]
    )
    bootstrap_values, bootstrap_diagnostics = _bootstrap(
        fit,
        prepared.patterns,
        prepared.cells,
        bootstrap_reps=bootstrap_reps,
        seed=seed,
        cache_dir=bootstrap_cache,
    )
    summary = _bootstrap_summary(
        bootstrap_values,
        point,
        km_point,
        n_requested=bootstrap_reps,
    )
    rules, decision = _stop_rules(
        fit,
        point,
        bootstrap_diagnostics,
        prepared.audit,
        bootstrap_reps=bootstrap_reps,
        phase2_dir=phase2_dir,
    )
    diagnostics = pd.DataFrame(
        [
            {
                "surface": "full_followup_discrete_time_observed_wealth",
                "n_births": prepared.n_births,
                "n_events": prepared.n_events,
                "n_countries": prepared.n_countries,
                "n_surveys": prepared.n_surveys,
                "n_grouped_cells": prepared.cells.n_cells,
                "n_model_rows": prepared.cells.n_model_rows,
                "n_features": int(len(fit.feature_names)),
                "likelihood_backend": fit.likelihood_backend.name,
                "scaled_objective": fit.model.objective,
                "max_abs_scaled_gradient": fit.model.max_abs_scaled_gradient,
                "converged": fit.converged,
                "n_iter": fit.n_iter,
                "rank_deficient": fit.rank_deficient,
                "condition_number": fit.condition_number,
                "warning": fit.warning_text,
            }
        ]
    )
    coefficients = pd.DataFrame(
        {
            "surface": "full_followup_discrete_time_observed_wealth",
            "feature": fit.feature_names,
            "coefficient": fit.model.coef_[0],
            "odds_ratio": np.exp(np.clip(fit.model.coef_[0], -30, 30)),
        }
    )
    manifest = _manifest(
        pilot_path,
        prepared,
        phase2_dir,
        bootstrap_reps=bootstrap_reps,
        seed=seed,
        likelihood_backend=fit.likelihood_backend.name,
        checkpoint_bootstrap=checkpoint_bootstrap,
    )
    manifest["decision"] = decision
    output_dir.mkdir(parents=True, exist_ok=True)
    staging = output_dir / ".phase3_building"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    staged = {key: staging / name for key, name in _OUTPUT_FILENAMES.items()}
    prepared.cohort_flow.to_csv(staged["cohort_flow"], index=False)
    prepared.audit.to_csv(staged["person_month_audit"], index=False)
    risk_comparison.to_csv(staged["risk_comparison"], index=False)
    diagnostics.to_csv(staged["model_diagnostics"], index=False)
    coefficients.to_csv(staged["coefficients"], index=False)
    point.to_csv(staged["standardized_risks"], index=False)
    bootstrap_values.to_csv(staged["bootstrap_draws"], index=False)
    summary.to_csv(staged["bootstrap_summary"], index=False)
    bootstrap_diagnostics.to_csv(staged["bootstrap_diagnostics"], index=False)
    rules.to_csv(staged["stop_rules"], index=False)
    primary = point.set_index("scenario")
    staged["report"].write_text(
        "\n".join(
            [
                "# Phase 3 discrete-time survival validation",
                "",
                f"**Decision class: {decision}.**",
                "",
                (
                    f"The model uses {prepared.n_births:,} births from "
                    f"{prepared.n_countries} countries and "
                    f"{prepared.n_surveys} survey waves; "
                    f"converged={fit.converged}, rank_deficient={fit.rank_deficient}."
                ),
                "",
                "## Standardised 60-month counterfactuals",
                "",
                *[
                    (
                        f"- {scenario}: PAF {primary.loc[scenario, 'paf_pct']:.2f}%; "
                        f"risk difference "
                        f"{primary.loc[scenario, 'risk_difference_per_1000']:.2f} "
                        "per 1,000"
                    )
                    for scenario in _SCENARIOS
                ],
                "",
                (
                    f"Bootstrap: {bootstrap_reps} common uncapped country draws; "
                    f"{int(bootstrap_diagnostics['success'].sum())} succeeded."
                ),
                "",
                "These are controlled associational standardisations, not identified causal effects.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    manifest["aggregate_output_sha256"] = {
        key: _sha256(path)
        for key, path in staged.items()
        if key != "run_manifest"
    }
    staged["run_manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    outputs: dict[str, Path] = {}
    for key, filename in _OUTPUT_FILENAMES.items():
        destination = output_dir / filename
        staged[key].replace(destination)
        outputs[key] = destination
    shutil.rmtree(staging)
    return SurvivalValidationResult(decision=decision, outputs=outputs)
