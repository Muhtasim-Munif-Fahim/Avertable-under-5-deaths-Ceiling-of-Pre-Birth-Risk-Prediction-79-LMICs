"""Fixed-sample Phase 2 diagnostics for the exact under-five rebuild.

The module deliberately exposes one deep interface, :func:`run_analysis`.
Callers provide a persisted individual-level pilot parquet and receive only
aggregate diagnostic artifacts.  Canonical data and manuscript outputs are
never read or modified here.
"""

from __future__ import annotations

import json
import hashlib
import copy
import gc
import math
import os
import re
import shutil
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import sklearn
from scipy import sparse
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

from config.paths import RNG_SEED
from src.inferential.phase2_pilot import prepare_analysis_states


REQUIRED_COLUMNS = {
    "country_code",
    "_source_file",
    "_source_row",
    "sample_weight",
    "died_under5_legacy_frozen",
    "died_under5",
    "died_under5_complete",
    "birth_cmc",
    "interview_cmc",
    "followup_months",
    "age_at_death_months",
    "event_time_months",
    "u5_event",
    "u5_censored",
    "u5_binary_eligible",
    "birth_year",
    "child_male",
    "multiple_birth",
    "birth_order",
    "birth_interval_months",
    "maternal_age_at_birth",
    "edu_level",
    "wealth_quintile",
    "currently_married",
    "urban",
}


@dataclass(frozen=True)
class AnalysisResult:
    """Aggregate result contract returned by :func:`run_analysis`."""

    decision: str
    outputs: dict[str, Path]


@dataclass
class _FitResult:
    surface: str
    role: str
    frame: pd.DataFrame
    features: list[str]
    encoder: OneHotEncoder
    matrix: sparse.csr_matrix
    keep_columns: np.ndarray
    feature_names: np.ndarray
    model: LogisticRegression
    outcome: np.ndarray
    weights: np.ndarray | None
    factual_predictions: np.ndarray
    converged: bool
    n_iter: int
    rank_deficient: bool
    condition_number: float
    warning_text: str
    grouped: _GroupedSurface | None = None


@dataclass
class _GroupedLogitFit:
    """Exact Bernoulli sufficient statistics for repeated design rows."""

    model: LogisticRegression
    pattern_matrix: sparse.csr_matrix
    pattern_weight: np.ndarray
    event_weight: np.ndarray
    pseudo_matrix: sparse.csr_matrix
    pseudo_outcome: np.ndarray
    pseudo_weight: np.ndarray
    pseudo_pattern_ids: np.ndarray
    first_indices: np.ndarray
    converged: bool
    n_iter: int
    warning_text: str


@dataclass
class _GroupedSurface:
    """Pattern frame plus exact grouped likelihood for one model surface."""

    frame: pd.DataFrame
    likelihood: _GroupedLogitFit


def _fit_grouped_logit(
    matrix: sparse.csr_matrix,
    outcome: np.ndarray,
    base_weight: np.ndarray,
    group_ids: np.ndarray,
    *,
    max_iter: int = 3000,
    tol: float = 1e-6,
) -> _GroupedLogitFit:
    """Fit an exact grouped Bernoulli likelihood for identical design rows."""

    outcome = np.asarray(outcome, dtype=float)
    base_weight = np.asarray(base_weight, dtype=float)
    group_ids = np.asarray(group_ids)
    if not (matrix.shape[0] == len(outcome) == len(base_weight) == len(group_ids)):
        raise ValueError("Grouped likelihood inputs must have equal row counts")
    if (
        len(group_ids) == 0
        or not np.isfinite(group_ids).all()
        or np.any(group_ids < 0)
        or np.any(group_ids != np.floor(group_ids))
    ):
        raise ValueError("group_ids must be non-negative finite integers")
    if not np.isin(outcome, [0.0, 1.0]).all():
        raise ValueError("Grouped Bernoulli outcome must contain only zero and one")
    if not np.isfinite(base_weight).all() or np.any(base_weight <= 0):
        raise ValueError("Grouped Bernoulli weights must be finite and positive")
    group_ids = group_ids.astype(np.int64, copy=False)
    unique, first = np.unique(group_ids, return_index=True)
    if not np.array_equal(unique, np.arange(len(unique))):
        raise ValueError("group_ids must be contiguous from zero")
    pattern_matrix = matrix[first].tocsr()
    pattern_weight = np.bincount(
        group_ids, weights=base_weight, minlength=len(unique)
    )
    event_weight = np.bincount(
        group_ids, weights=base_weight * outcome, minlength=len(unique)
    )
    nonevent_weight = pattern_weight - event_weight
    pseudo_pattern = np.repeat(np.arange(len(unique), dtype=np.int64), 2)
    pseudo_outcome = np.tile(np.array([0.0, 1.0]), len(unique))
    pseudo_weight = np.column_stack([nonevent_weight, event_weight]).ravel()
    positive = pseudo_weight > 0
    pseudo_pattern_ids = pseudo_pattern[positive]
    pseudo_matrix = pattern_matrix[pseudo_pattern_ids].tocsr()
    pseudo_outcome = pseudo_outcome[positive]
    pseudo_weight = pseudo_weight[positive]
    if np.unique(pseudo_outcome).size != 2:
        raise ValueError("Grouped Bernoulli likelihood requires both outcome classes")
    model = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=max_iter,
        tol=tol,
        fit_intercept=True,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(
            pseudo_matrix,
            pseudo_outcome,
            sample_weight=pseudo_weight,
        )
    convergence_messages = [
        str(item.message)
        for item in caught
        if issubclass(item.category, ConvergenceWarning)
    ]
    n_iter = int(np.max(model.n_iter_))
    return _GroupedLogitFit(
        model=model,
        pattern_matrix=pattern_matrix,
        pattern_weight=pattern_weight,
        event_weight=event_weight,
        pseudo_matrix=pseudo_matrix,
        pseudo_outcome=pseudo_outcome,
        pseudo_weight=pseudo_weight,
        pseudo_pattern_ids=pseudo_pattern_ids,
        first_indices=first,
        converged=not convergence_messages and n_iter < max_iter,
        n_iter=n_iter,
        warning_text=" | ".join(convergence_messages),
    )


_OUTPUT_FILENAMES = {
    "coverage": "phase2_coverage.csv",
    "risk_comparison": "phase2_risk_comparison.csv",
    "model_diagnostics": "phase2_model_diagnostics.csv",
    "model_coefficients": "phase2_model_coefficients.csv",
    "pafs": "phase2_pafs.csv",
    "bootstrap_summary": "phase2_bootstrap_summary.csv",
    "bootstrap_draws": "phase2_bootstrap_draws.csv",
    "bootstrap_diagnostics": "phase2_bootstrap_diagnostics.csv",
    "rank_stability": "phase2_rank_stability.csv",
    "stop_rules": "phase2_stop_rules.csv",
    "run_manifest": "phase2_run_manifest.json",
    "variant_registry": "phase2_variant_registry.csv",
    "positivity": "phase2_positivity.csv",
    "report": "phase2_report.md",
}


_BOOTSTRAP_SURFACES = {
    "legacy_binary_pooled",
    "exact_binary_pooled_legacy_states",
    "exact_binary_pooled",
    "exact_weighted_pooled",
    "exact_weighted_pooled_no_wealth",
    "exact_country_fe_no_wealth",
    "exact_wave_fe_no_wealth",
    "exact_wave_fe_no_wealth_wealth_complete",
    "exact_country_fe_explicit_missing",
    "exact_wave_fe_complete_wealth",
}

_SURFACE_ORDER = [
    "legacy_binary_pooled",
    "exact_binary_pooled_legacy_states",
    "exact_binary_pooled",
    "exact_weighted_pooled",
    "exact_weighted_pooled_no_wealth",
    "exact_country_fe_no_wealth",
    "exact_wave_fe_no_wealth",
    "exact_wave_fe_no_wealth_wealth_complete",
    "exact_country_fe_explicit_missing",
    "exact_wave_fe_complete_wealth",
]

_MODEL_PARENT_COLUMNS = [
    "country_code",
    "_source_file",
    "sample_weight",
    "survey_year",
    "birth_year",
    "died_under5_legacy_frozen",
    "died_under5_complete",
    "child_male",
    "multiple_birth",
    "birth_order",
    "birth_interval_months",
    "maternal_age_at_birth",
    "edu_level",
    "wealth_quintile",
    "currently_married",
    "urban",
]

_JOB_TABLE_NAMES = (
    "diagnostics",
    "coefficients",
    "pafs",
    "risk",
    "ranks",
    "bootstrap_values",
    "bootstrap_diagnostics",
)


def _read_pilot(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Phase 2 pilot parquet not found: {path}")
    frame = pd.read_parquet(path)
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"Phase 2 pilot is missing required columns: {missing}")
    if frame.empty:
        raise ValueError("Phase 2 pilot contains no rows")
    source_row = pd.to_numeric(frame["_source_row"], errors="coerce")
    if (
        source_row.isna().any()
        or source_row.lt(0).any()
        or source_row.mod(1).ne(0).any()
        or pd.DataFrame(
            {
                "_source_file": frame["_source_file"].astype(str),
                "_source_row_numeric": source_row,
            }
        ).duplicated(["_source_file", "_source_row_numeric"]).any()
    ):
        raise ValueError("Phase 2 pilot has invalid or duplicate source-row keys")
    observed_complete = frame["died_under5_complete"].notna()
    invalid_complete = observed_complete & frame["u5_binary_eligible"].ne(1)
    if invalid_complete.any():
        raise ValueError(
            f"{int(invalid_complete.sum())} complete outcomes fall outside "
            "exact binary eligibility"
        )

    birth = pd.to_numeric(frame["birth_cmc"], errors="coerce")
    interview = pd.to_numeric(frame["interview_cmc"], errors="coerce")
    followup = pd.to_numeric(frame["followup_months"], errors="coerce")
    event = pd.to_numeric(frame["u5_event"], errors="coerce")
    censored = pd.to_numeric(frame["u5_censored"], errors="coerce")
    event_time = pd.to_numeric(frame["event_time_months"], errors="coerce")
    death_age = pd.to_numeric(frame["age_at_death_months"], errors="coerce")
    eligible = pd.to_numeric(frame["u5_binary_eligible"], errors="coerce")
    complete = pd.to_numeric(frame["died_under5_complete"], errors="coerce")
    valid_pair = (
        birth.gt(0)
        & interview.gt(0)
        & birth.mod(1).eq(0)
        & interview.mod(1).eq(0)
        & interview.ge(birth)
    )
    invariants = {
        "follow-up differs from interview minus birth CMC": int(
            (valid_pair & ~followup.eq(interview - birth)).sum()
        ),
        "eligibility differs from follow-up >=60": int(
            (
                valid_pair
                & ~eligible.eq(followup.ge(60).astype(int))
            ).sum()
        ),
        "event time outside [0,60]": int(
            (event_time.notna() & ~event_time.between(0, 60)).sum()
        ),
        "event time is not an integer month": int(
            (event_time.notna() & event_time.mod(1).ne(0)).sum()
        ),
        "known survival state does not sum to one": int(
            (event.notna() & ~(event + censored).eq(1)).sum()
        ),
        "under-five event has invalid death age": int(
            (event.eq(1) & ~death_age.between(0, 59)).sum()
        ),
        "event time differs from death age": int(
            (event.eq(1) & ~event_time.eq(death_age)).sum()
        ),
        "event occurs after observed follow-up": int(
            (event.eq(1) & event_time.gt(followup)).sum()
        ),
        "invalid CMC pair retains derived follow-up/outcome state": int(
            (
                ~valid_pair
                & (
                    followup.notna()
                    | eligible.notna()
                    | complete.notna()
                    | event.notna()
                    | censored.notna()
                    | event_time.notna()
                )
            ).sum()
        ),
        "death after interview retains analytic survival state": int(
            (
                death_age.notna()
                & followup.notna()
                & death_age.gt(followup)
                & (event.notna() | censored.notna() | event_time.notna())
            ).sum()
        ),
    }
    failed = {name: count for name, count in invariants.items() if count}
    if failed:
        raise ValueError(f"Phase 2 exact-follow-up invariant failures: {failed}")
    return frame


def _positive_normalized_weights(frame: pd.DataFrame) -> pd.Series:
    """Return survey-normalized positive weights; invalid values stay missing."""
    raw = pd.to_numeric(frame["sample_weight"], errors="coerce")
    positive = raw.where(np.isfinite(raw) & raw.gt(0))
    means = positive.groupby(frame["_source_file"], observed=True).transform("mean")
    return positive / means


def _weighted_km_60(frame: pd.DataFrame, weights: pd.Series) -> tuple[float, int]:
    event = pd.to_numeric(frame["u5_event"], errors="coerce")
    time = pd.to_numeric(frame["event_time_months"], errors="coerce")
    valid = (
        event.isin([0, 1])
        & time.between(0, 60, inclusive="both")
        & weights.notna()
    )
    if not valid.any():
        return np.nan, 0
    event_values = event.loc[valid].to_numpy(dtype=int)
    time_values = time.loc[valid].to_numpy(dtype=int)
    weight_values = weights.loc[valid].to_numpy(dtype=float)
    total_weight = np.bincount(time_values, weights=weight_values, minlength=61)
    event_weight = np.bincount(
        time_values[event_values == 1],
        weights=weight_values[event_values == 1],
        minlength=61,
    )
    at_risk_weight = np.cumsum(total_weight[::-1])[::-1]
    event_month = (event_weight[:60] > 0) & (at_risk_weight[:60] > 0)
    conditional_survival = 1.0 - (
        event_weight[:60][event_month] / at_risk_weight[:60][event_month]
    )
    survival = float(np.prod(conditional_survival))
    return float(1.0 - survival), int(valid.sum())


def _risk_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    legacy = pd.to_numeric(frame["died_under5_legacy_frozen"], errors="coerce")
    rebuilt_legacy = pd.to_numeric(frame["died_under5"], errors="coerce")
    complete = pd.to_numeric(frame["died_under5_complete"], errors="coerce")
    full_weights = _positive_normalized_weights(frame)
    km, n_km = _weighted_km_60(frame, full_weights)
    exact_mask = complete.notna() & frame["u5_binary_eligible"].eq(1)
    exact_weights = _positive_normalized_weights(
        frame.loc[exact_mask, ["sample_weight", "_source_file"]]
    )
    exact_km, n_exact_km = _weighted_km_60(
        frame.loc[exact_mask, ["u5_event", "event_time_months"]],
        exact_weights,
    )
    exact_weighted = float(
        np.average(
            complete.loc[exact_weights.index[exact_weights.notna()]],
            weights=exact_weights.loc[exact_weights.notna()],
        )
    )
    if not np.isclose(exact_km, exact_weighted, atol=1e-10, rtol=1e-10):
        raise ValueError(
            "Exact-cohort weighted KM does not reproduce the weighted complete "
            f"binary risk ({exact_km} vs {exact_weighted})"
        )
    n_bad_weight = int(full_weights.isna().sum())
    rows = [
        {
            "estimand": "legacy_recorded_event_proportion",
            "estimate": float(legacy.dropna().mean()),
            "n_births": int(legacy.notna().sum()),
            "n_zero_or_invalid_weight": 0,
            "label": "Recorded-event proportion; not U5MR",
        },
        {
            "estimand": "exact_complete_binary_risk_unweighted",
            "estimate": float(complete.dropna().mean()),
            "n_births": int(complete.notna().sum()),
            "n_zero_or_invalid_weight": 0,
            "label": "Risk among births with at least 60 exact months",
        },
        {
            "estimand": "legacy_definition_on_v2_harmonization",
            "estimate": float(rebuilt_legacy.dropna().mean()),
            "n_births": int(rebuilt_legacy.notna().sum()),
            "n_zero_or_invalid_weight": 0,
            "label": "Sensitivity; rebuilt source interpretation",
        },
        {
            "estimand": "weighted_km_60_month_risk",
            "estimate": km,
            "n_births": n_km,
            "n_zero_or_invalid_weight": n_bad_weight,
            "label": (
                "Descriptive weighted Kaplan-Meier risk; weights normalized "
                "within wave on all known survival records"
            ),
        },
        {
            "estimand": "exact_cohort_weighted_binary_and_km_risk",
            "estimate": exact_km,
            "n_births": n_exact_km,
            "n_zero_or_invalid_weight": int(exact_mask.sum() - n_exact_km),
            "label": "Exact-cohort identity check: weighted binary equals KM",
        },
    ]
    return pd.DataFrame(rows)


_BASE_FEATURES = [
    "sex_cat",
    "birth_type",
    "bo_cat",
    "bi_model_cat",
    "ma_cat",
    "edu_cat",
    "marital_cat",
    "urban_cat",
    "birth_period",
]

_FIXED_CATEGORIES = {
    "sex_cat": ["Female", "Male", "Missing"],
    "birth_type": ["Singleton", "Multiple", "Missing"],
    "bo_cat": ["1st", "2nd-3rd", "4th-5th", "6th+", "Missing"],
    "bi_model_cat": [
        "24-47 mo",
        "<24 mo",
        "48+ mo",
        "Missing non-first",
        "First/missing",
    ],
    "ma_cat": ["20-34", "<20", "35+", "Missing"],
    "edu_cat": ["None", "Primary", "Secondary", "Higher", "Missing"],
    "wq_cat": ["Q1", "Q2", "Q3", "Q4", "Q5", "Missing"],
    "marital_cat": ["Not married", "Married", "Missing"],
    "urban_cat": ["Rural", "Urban", "Missing"],
}


def _birth_period(year: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(year, errors="coerce")
    period = (np.floor(numeric / 5) * 5).astype("Int64")
    return period.map(
        lambda value: "Missing"
        if pd.isna(value)
        else f"{int(value)}-{int(value) + 4}"
    )


def _prepare_states(frame: pd.DataFrame, *, legacy: bool) -> pd.DataFrame:
    states = prepare_analysis_states(frame, legacy=legacy)
    states["birth_period"] = _birth_period(states["birth_year"])
    return states.reset_index(drop=True)


def _categories_for(frame: pd.DataFrame, features: list[str]) -> list[list[str]]:
    categories: list[list[str]] = []
    for feature in features:
        if feature in _FIXED_CATEGORIES:
            categories.append(_FIXED_CATEGORIES[feature])
            continue
        observed = sorted(frame[feature].astype(str).dropna().unique().tolist())
        if not observed:
            raise ValueError(f"No observed levels for categorical feature {feature}")
        categories.append(observed)
    return categories


def _design(
    frame: pd.DataFrame,
    features: list[str],
) -> tuple[OneHotEncoder, sparse.csr_matrix, np.ndarray, np.ndarray]:
    encoder = OneHotEncoder(
        categories=_categories_for(frame, features),
        drop="first",
        handle_unknown="ignore",
        sparse_output=True,
        dtype=np.float64,
    )
    matrix = encoder.fit_transform(frame[features].astype(str)).tocsr()
    counts = np.asarray(matrix.getnnz(axis=0)).ravel()
    keep = (counts > 0) & (counts < len(frame))
    matrix = matrix[:, keep]
    names = encoder.get_feature_names_out(features)[keep]
    return encoder, matrix, keep, names


def _rank_diagnostics(matrix: sparse.csr_matrix) -> tuple[bool, float]:
    augmented = sparse.hstack(
        [sparse.csr_matrix(np.ones((matrix.shape[0], 1))), matrix], format="csr"
    )
    gram = (augmented.T @ augmented).toarray()
    rank = int(np.linalg.matrix_rank(gram))
    deficient = rank < gram.shape[0]
    condition = float(np.linalg.cond(gram)) if not deficient else float("inf")
    return deficient, condition


def _fit_surface(
    *,
    surface: str,
    role: str,
    frame: pd.DataFrame,
    outcome_column: str,
    legacy_states: bool,
    fixed_effects: str,
    wealth_mode: str,
    weighted: bool,
    drop_features: tuple[str, ...] = (),
) -> _FitResult:
    """Fit one logistic surface.

    `drop_features` removes named covariates from the design and defaults to
    empty, so it cannot change any published estimate. It exists for the
    covariate-block ablation grid (`s05s_ablation_grid_v2.py`), which asks how
    far the counterfactual fractions depend on the adjustment set by refitting
    without the child, maternal-age, or socioeconomic blocks in turn. The
    exposures themselves (`edu_cat`, `bi_model_cat`) are refused, because
    dropping the shifted covariate would make its counterfactual meaningless
    rather than merely unadjusted.
    """
    # Argument validation before any data is touched, so a bad call fails in
    # milliseconds rather than after an 8 GB load and a 50 s fit.
    _protected = {"edu_cat", "bi_model_cat"} & set(drop_features)
    if _protected:
        raise ValueError(
            "refusing to drop a shifted exposure from the design: "
            f"{sorted(_protected)}. The counterfactual is defined as a move "
            "between that covariate's levels, so removing it does not yield an "
            "unadjusted estimate -- it yields no estimate.")
    input_columns = list(
        dict.fromkeys(
            [
                outcome_column,
                "country_code",
                "_source_file",
                "sample_weight",
                "survey_year",
                "birth_year",
                "child_male",
                "multiple_birth",
                "birth_order",
                "birth_interval_months",
                "maternal_age_at_birth",
                "edu_level",
                "wealth_quintile",
                "currently_married",
                "urban",
            ]
        )
    )
    data = frame[input_columns].copy()
    if fixed_effects not in {"none", "country", "survey"}:
        raise ValueError(f"Unknown fixed-effect structure: {fixed_effects}")
    # Normalize once on the parent cohort. Complete-case restriction then
    # changes only the population axis, not the weights themselves.
    normalized = _positive_normalized_weights(data) if weighted else None
    if wealth_mode in {"complete", "none_complete"}:
        data = data[data["wealth_quintile"].notna()].copy()
    if weighted:
        data = data.loc[normalized.loc[data.index].notna()].copy()
    data = data[pd.to_numeric(data[outcome_column], errors="coerce").notna()].copy()
    if weighted:
        normalized = normalized.loc[data.index]
    states = _prepare_states(data, legacy=legacy_states)
    outcome = pd.to_numeric(states[outcome_column], errors="raise").to_numpy(float)
    if np.unique(outcome).size != 2:
        raise ValueError(f"Surface {surface} requires both outcome classes")
    features = list(_BASE_FEATURES)
    if wealth_mode in {"explicit", "complete"}:
        features.append("wq_cat")
    if fixed_effects == "survey":
        features.append("survey_id")
    elif fixed_effects == "country":
        features.append("country_code")
    if drop_features:
        unknown = set(drop_features) - set(features)
        if unknown:
            raise ValueError(f"drop_features names absent from the design: {sorted(unknown)}")
        features = [f for f in features if f not in set(drop_features)]
    encoder, matrix, keep, names = _design(states, features)
    rank_deficient, condition_number = _rank_diagnostics(matrix)
    sample_weight = None if normalized is None else normalized.to_numpy(float)
    base_weight = (
        np.ones(len(states), dtype=float)
        if sample_weight is None
        else sample_weight
    )
    group_columns = list(dict.fromkeys(features + ["country_code"]))
    group_keys = states[group_columns].copy()
    group_keys["_education_eligible"] = pd.to_numeric(
        states["edu_level"], errors="coerce"
    ).isin([0, 1])
    group_keys["_spacing_eligible"] = (
        pd.to_numeric(states["birth_order"], errors="coerce").gt(1)
        & pd.to_numeric(
            states["birth_interval_months"], errors="coerce"
        ).between(0, 23)
    )
    group_ids = group_keys.groupby(
        list(group_keys.columns), sort=False, dropna=False, observed=True
    ).ngroup().to_numpy(dtype=np.int64)
    grouped_likelihood = _fit_grouped_logit(
        matrix,
        outcome,
        base_weight,
        group_ids,
        max_iter=3000,
        tol=1e-6,
    )
    model = grouped_likelihood.model
    n_iter = grouped_likelihood.n_iter
    converged = grouped_likelihood.converged
    predictions = model.predict_proba(matrix)[:, 1]
    compact_columns = list(
        dict.fromkeys(
            features
            + [
                "edu_level",
                "birth_order",
                "birth_interval_months",
                "country_code",
                "survey_id",
            ]
        )
    )
    compact_states = states[compact_columns].copy()
    pattern_frame = compact_states.iloc[
        grouped_likelihood.first_indices
    ].reset_index(drop=True)
    return _FitResult(
        surface=surface,
        role=role,
        frame=compact_states,
        features=features,
        encoder=encoder,
        matrix=matrix,
        keep_columns=keep,
        feature_names=names,
        model=model,
        outcome=outcome,
        weights=sample_weight,
        factual_predictions=predictions,
        converged=converged,
        n_iter=n_iter,
        rank_deficient=rank_deficient,
        condition_number=condition_number,
        warning_text=grouped_likelihood.warning_text,
        grouped=_GroupedSurface(
            frame=pattern_frame,
            likelihood=grouped_likelihood,
        ),
    )


def _surface_specs(frame: pd.DataFrame) -> list[dict]:
    # Cohort restriction is applied inside _fit_surface from the declared
    # outcome column. Keeping one shared parent avoids retaining duplicate
    # 400k-500k row frames for every specification.
    legacy = frame
    exact = frame
    specs = [
        dict(
            surface="legacy_binary_pooled",
            role="frozen_legacy_bridge",
            frame=legacy,
            outcome_column="died_under5_legacy_frozen",
            legacy_states=True,
            fixed_effects="none",
            wealth_mode="explicit",
            weighted=False,
        ),
        dict(
            surface="exact_binary_pooled_legacy_states",
            role="outcome_cohort_only_bridge",
            frame=exact,
            outcome_column="died_under5_complete",
            legacy_states=True,
            fixed_effects="none",
            wealth_mode="explicit",
            weighted=False,
        ),
        dict(
            surface="exact_binary_pooled",
            role="explicit_missing_state_bridge",
            frame=exact,
            outcome_column="died_under5_complete",
            legacy_states=False,
            fixed_effects="none",
            wealth_mode="explicit",
            weighted=False,
        ),
        dict(
            surface="exact_weighted_pooled",
            role="survey_weight_bridge",
            frame=exact,
            outcome_column="died_under5_complete",
            legacy_states=False,
            fixed_effects="none",
            wealth_mode="explicit",
            weighted=True,
        ),
        dict(
            surface="exact_weighted_pooled_no_wealth",
            role="wealth_omission_bridge",
            frame=exact,
            outcome_column="died_under5_complete",
            legacy_states=False,
            fixed_effects="none",
            wealth_mode="none",
            weighted=True,
        ),
        dict(
            surface="exact_country_fe_no_wealth",
            role="country_fixed_effect_sensitivity",
            frame=exact,
            outcome_column="died_under5_complete",
            legacy_states=False,
            fixed_effects="country",
            wealth_mode="none",
            weighted=True,
        ),
        dict(
            surface="exact_wave_fe_no_wealth",
            role="global_no_wealth_sensitivity",
            frame=exact,
            outcome_column="died_under5_complete",
            legacy_states=False,
            fixed_effects="survey",
            wealth_mode="none",
            weighted=True,
        ),
        dict(
            surface="exact_wave_fe_no_wealth_wealth_complete",
            role="wealth_complete_cohort_without_wealth_adjustment",
            frame=exact,
            outcome_column="died_under5_complete",
            legacy_states=False,
            fixed_effects="survey",
            wealth_mode="none_complete",
            weighted=True,
        ),
        dict(
            surface="exact_country_fe_explicit_missing",
            role="estimable_explicit_missing_sensitivity",
            frame=exact,
            outcome_column="died_under5_complete",
            legacy_states=False,
            fixed_effects="country",
            wealth_mode="explicit",
            weighted=True,
        ),
        dict(
            surface="exact_wave_fe_complete_wealth",
            role="primary_observed_wealth",
            frame=exact,
            outcome_column="died_under5_complete",
            legacy_states=False,
            fixed_effects="survey",
            wealth_mode="complete",
            weighted=True,
        ),
    ]
    return specs


def _transform(result: _FitResult, frame: pd.DataFrame) -> sparse.csr_matrix:
    matrix = result.encoder.transform(frame[result.features].astype(str)).tocsr()
    return matrix[:, result.keep_columns]


def _counterfactual(
    result: _FitResult,
    scenario: str,
    contract: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    frame = result.frame.copy()
    education = pd.to_numeric(frame["edu_level"], errors="coerce")
    order = pd.to_numeric(frame["birth_order"], errors="coerce")
    interval = pd.to_numeric(frame["birth_interval_months"], errors="coerce")
    if contract == "legacy_whole_population":
        edu_mask = np.ones(len(frame), dtype=bool)
        spacing_mask = np.ones(len(frame), dtype=bool)
    elif contract == "policy_eligible":
        edu_mask = education.isin([0, 1]).to_numpy(bool)
        spacing_mask = (
            order.gt(1) & interval.ge(0) & interval.lt(24)
        ).to_numpy(bool)
    else:
        raise ValueError(f"Unknown counterfactual contract: {contract}")
    modified = np.zeros(len(frame), dtype=bool)
    if scenario in {"education", "joint"}:
        frame.loc[edu_mask, "edu_cat"] = "Secondary"
        modified |= edu_mask
    if scenario in {"spacing", "joint"}:
        frame.loc[spacing_mask, "bi_model_cat"] = "24-47 mo"
        modified |= spacing_mask
    return frame, modified


def _average(values: np.ndarray, weights: np.ndarray | None) -> float:
    return float(values.mean()) if weights is None else float(np.average(values, weights=weights))


def _counterfactual_eta_delta(
    result: _FitResult,
    scenario: str,
    contract: str,
    coefficients: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return additive logit changes without materialising a second design.

    Phase 2 uses main-effect one-hot logistic surfaces. An intervention changes
    only education and/or interval category contributions, so a length-N logit
    delta is mathematically identical to transforming a full sparse matrix and
    uses a fraction of the memory on the 533k-row pilot.
    """
    if scenario not in {"education", "spacing", "joint"}:
        raise ValueError(f"Unknown counterfactual scenario: {scenario}")
    coefficient_by_name = {
        str(name): float(value)
        for name, value in zip(result.feature_names, coefficients, strict=True)
    }

    def effect(feature: str, level: str) -> float:
        return coefficient_by_name.get(f"{feature}_{level}", 0.0)

    n = len(result.frame)
    delta = np.zeros(n, dtype=float)
    modified = np.zeros(n, dtype=bool)
    education = pd.to_numeric(result.frame["edu_level"], errors="coerce")
    order = pd.to_numeric(result.frame["birth_order"], errors="coerce")
    interval = pd.to_numeric(
        result.frame["birth_interval_months"], errors="coerce"
    )
    if contract == "legacy_whole_population":
        education_mask = np.ones(n, dtype=bool)
        spacing_mask = np.ones(n, dtype=bool)
    elif contract == "policy_eligible":
        education_mask = education.isin([0, 1]).to_numpy(dtype=bool)
        spacing_mask = (
            order.gt(1) & interval.ge(0) & interval.lt(24)
        ).to_numpy(dtype=bool)
    else:
        raise ValueError(f"Unknown counterfactual contract: {contract}")

    if scenario in {"education", "joint"}:
        target_effect = effect("edu_cat", "Secondary")
        current = result.frame["edu_cat"].astype(str).to_numpy()
        for level in np.unique(current[education_mask]):
            rows = education_mask & (current == level)
            delta[rows] += target_effect - effect("edu_cat", str(level))
        modified |= education_mask
    if scenario in {"spacing", "joint"}:
        target_effect = effect("bi_model_cat", "24-47 mo")
        current = result.frame["bi_model_cat"].astype(str).to_numpy()
        for level in np.unique(current[spacing_mask]):
            rows = spacing_mask & (current == level)
            delta[rows] += target_effect - effect(
                "bi_model_cat", str(level)
            )
        modified |= spacing_mask
    return delta, modified


def _sequential_joint_frame(result: _FitResult, contract: str) -> pd.DataFrame:
    education_frame, _ = _counterfactual(result, "education", contract)
    order = pd.to_numeric(education_frame["birth_order"], errors="coerce")
    interval = pd.to_numeric(
        education_frame["birth_interval_months"], errors="coerce"
    )
    if contract == "legacy_whole_population":
        spacing_mask = np.ones(len(education_frame), dtype=bool)
    else:
        spacing_mask = (
            order.gt(1) & interval.ge(0) & interval.lt(24)
        ).to_numpy(bool)
    education_frame.loc[spacing_mask, "bi_model_cat"] = "24-47 mo"
    return education_frame


def _paf_rows(results: list[_FitResult]) -> pd.DataFrame:
    rows: list[dict] = []
    for result in results:
        contracts = (
            ["legacy_whole_population", "policy_eligible"]
            if result.surface == "legacy_binary_pooled"
            else ["policy_eligible"]
        )
        factual_risk = _average(result.factual_predictions, result.weights)
        coefficients = result.model.coef_[0]
        factual_eta = np.asarray(
            result.matrix @ coefficients
        ).ravel() + float(result.model.intercept_[0])
        for contract in contracts:
            for scenario in ("education", "spacing", "joint"):
                eta_delta, modified = _counterfactual_eta_delta(
                    result, scenario, contract, coefficients
                )
                cf_predictions = expit(factual_eta + eta_delta)
                counterfactual_risk = _average(cf_predictions, result.weights)
                identity_error = np.nan
                if scenario == "joint":
                    education_delta, _ = _counterfactual_eta_delta(
                        result, "education", contract, coefficients
                    )
                    spacing_delta, _ = _counterfactual_eta_delta(
                        result, "spacing", contract, coefficients
                    )
                    identity_error = float(
                        np.max(
                            np.abs(
                                eta_delta
                                - education_delta
                                - spacing_delta
                            )
                        )
                    )
                paf = (
                    (factual_risk - counterfactual_risk) / factual_risk
                    if factual_risk > 0
                    else np.nan
                )
                rows.append(
                    {
                        "surface": result.surface,
                        "role": result.role,
                        "contract": contract,
                        "scenario": scenario,
                        "n_births": len(result.frame),
                        "n_modified": int(modified.sum()),
                        "factual_risk": factual_risk,
                        "counterfactual_risk": counterfactual_risk,
                        "risk_difference_per_1000": (factual_risk - counterfactual_risk)
                        * 1000,
                        "risk_ratio": counterfactual_risk / factual_risk
                        if factual_risk > 0
                        else np.nan,
                        "paf_pct": paf * 100,
                        "n_risk_increased": int(
                            (cf_predictions > result.factual_predictions + 1e-12).sum()
                        ),
                        "max_risk_increase": float(
                            np.maximum(
                                cf_predictions - result.factual_predictions, 0
                            ).max(initial=0.0)
                        ),
                        "same_surface_identity_max_abs_error": identity_error,
                        "converged": result.converged,
                        "publication_eligible": result.converged
                        and not result.rank_deficient,
                    }
                )
    return pd.DataFrame(rows)


def _model_outputs(
    results: list[_FitResult],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    diagnostic_rows: list[dict] = []
    coefficient_rows: list[dict] = []
    for result in results:
        primary = result.surface == "exact_wave_fe_complete_wealth"
        analysis_weight = (
            np.ones(len(result.frame), dtype=float)
            if result.weights is None
            else result.weights
        )
        weight_sum = float(analysis_weight.sum())
        weight_ess = weight_sum**2 / float(np.square(analysis_weight).sum())
        wave_events = pd.DataFrame(
            {
                "survey_id": result.frame["survey_id"].astype(str),
                "outcome": result.outcome,
            }
        ).groupby("survey_id", observed=True)["outcome"].sum()
        diagnostic_rows.append(
            {
                "surface": result.surface,
                "role": result.role,
                "n_births": len(result.frame),
                "n_countries": result.frame["country_code"].nunique(),
                "n_surveys": result.frame["survey_id"].nunique(),
                "n_deaths": int(result.outcome.sum()),
                "weight_sum": weight_sum,
                "weight_effective_sample_size": weight_ess,
                "n_zero_event_waves": int(wave_events.eq(0).sum()),
                "n_waves_fewer_than_5_events": int(wave_events.lt(5).sum()),
                "n_features": len(result.feature_names),
                "converged": result.converged,
                "n_iter": result.n_iter,
                "rank_deficient": result.rank_deficient,
                "condition_number": result.condition_number,
                "primary_eligible": primary
                and result.converged
                and not result.rank_deficient,
                "warning": result.warning_text,
                "features": json.dumps(result.features),
            }
        )
        coefficient_rows.append(
            {
                "surface": result.surface,
                "feature": "(intercept)",
                "coefficient": float(result.model.intercept_[0]),
                "odds_ratio": float(np.exp(np.clip(result.model.intercept_[0], -30, 30))),
                "is_nuisance": True,
            }
        )
        for feature, coefficient in zip(
            result.feature_names, result.model.coef_[0], strict=True
        ):
            coefficient_rows.append(
                {
                    "surface": result.surface,
                    "feature": feature,
                    "coefficient": float(coefficient),
                    "odds_ratio": float(np.exp(np.clip(coefficient, -30, 30))),
                    "is_nuisance": feature.startswith("survey_id_")
                    or feature.startswith("country_code_")
                    or feature.startswith("birth_period_"),
                }
            )
    return pd.DataFrame(diagnostic_rows), pd.DataFrame(coefficient_rows)


def _country_draws(
    frame: pd.DataFrame,
    *,
    bootstrap_reps: int,
    seed: int,
) -> list[dict[str, int]]:
    countries = np.array(sorted(frame["country_code"].astype(str).unique()))
    if countries.size < 2:
        raise ValueError("Country-cluster bootstrap requires at least two countries")
    rng = np.random.default_rng(seed)
    draws: list[dict[str, int]] = []
    for _ in range(bootstrap_reps):
        selected = rng.choice(countries, size=len(countries), replace=True)
        unique, counts = np.unique(selected, return_counts=True)
        draws.append({str(country): int(count) for country, count in zip(unique, counts)})
    return draws


def _row_multiplicity(frame: pd.DataFrame, draw: dict[str, int]) -> np.ndarray:
    return (
        frame["country_code"]
        .astype(str)
        .map(draw)
        .fillna(0)
        .to_numpy(dtype=float)
    )


def _bootstrap_grouped_surface(
    result: _FitResult,
    draws: list[dict[str, int]],
    *,
    cache_dir: Path | None,
) -> tuple[list[dict], list[dict]]:
    """Bootstrap one surface using exact grouped sufficient statistics."""

    if result.grouped is None:
        raise ValueError("Grouped bootstrap requires grouped surface statistics")
    grouped = result.grouped
    likelihood = grouped.likelihood
    countries = np.array(sorted(grouped.frame["country_code"].astype(str).unique()))
    country_index = {country: index for index, country in enumerate(countries)}
    pattern_country_index = np.fromiter(
        (country_index[value] for value in grouped.frame["country_code"].astype(str)),
        dtype=np.int16,
        count=len(grouped.frame),
    )
    pseudo_country_index = pattern_country_index[likelihood.pseudo_pattern_ids]
    pattern_result = copy.copy(result)
    pattern_result.frame = grouped.frame
    pattern_result.matrix = likelihood.pattern_matrix
    pattern_result.weights = likelihood.pattern_weight
    values: list[dict] = []
    diagnostics: list[dict] = []
    model = copy.deepcopy(result.model)
    model.warm_start = True
    model.max_iter = 2000
    model.tol = 1e-5
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
                raise ValueError("Phase 2 grouped bootstrap cache is not contiguous")
            if completed and state_path.exists():
                with np.load(state_path) as state:
                    if int(state["replicate"]) != max(completed):
                        raise ValueError("Phase 2 grouped bootstrap warm state is stale")
                    model.coef_ = state["coef"]
                    model.intercept_ = state["intercept"]
                    model.classes_ = state["classes"]
                    model.n_features_in_ = likelihood.pseudo_matrix.shape[1]

    def checkpoint(replicate: int) -> None:
        if cache_dir is None:
            return
        assert values_path is not None
        assert diagnostics_path is not None
        assert state_path is not None
        values_temp = values_path.with_suffix(".building.csv")
        diagnostics_temp = diagnostics_path.with_suffix(".building.csv")
        state_temp = state_path.with_suffix(".building.npz")
        pd.DataFrame(values).to_csv(values_temp, index=False)
        pd.DataFrame(diagnostics).to_csv(diagnostics_temp, index=False)
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

    contracts = (
        ["legacy_whole_population", "policy_eligible"]
        if result.surface == "legacy_binary_pooled"
        else ["policy_eligible"]
    )
    for replicate, draw in enumerate(draws):
        if replicate in completed:
            continue
        multiplicity = np.zeros(len(countries), dtype=float)
        for country, count in draw.items():
            if country in country_index:
                multiplicity[country_index[country]] = count
        pseudo_weight = (
            likelihood.pseudo_weight * multiplicity[pseudo_country_index]
        )
        positive = pseudo_weight > 0
        pattern_weight = (
            likelihood.pattern_weight * multiplicity[pattern_country_index]
        )
        try:
            if np.unique(likelihood.pseudo_outcome[positive]).size != 2:
                raise ValueError("bootstrap draw has fewer than two outcome classes")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(
                    likelihood.pseudo_matrix[positive],
                    likelihood.pseudo_outcome[positive],
                    sample_weight=pseudo_weight[positive],
                )
            convergence_messages = [
                str(item.message)
                for item in caught
                if issubclass(item.category, ConvergenceWarning)
            ]
            n_iter = int(np.max(model.n_iter_))
            if convergence_messages or n_iter >= model.max_iter:
                raise RuntimeError(
                    " | ".join(convergence_messages) or "iteration limit"
                )
            coefficients = model.coef_[0]
            factual_eta = np.asarray(
                likelihood.pattern_matrix @ coefficients
            ).ravel() + float(model.intercept_[0])
            factual = expit(factual_eta)
            factual_risk = float(np.average(factual, weights=pattern_weight))
            values.append(
                {
                    "replicate": replicate,
                    "metric_type": "model_factual_risk",
                    "surface": result.surface,
                    "contract": "factual",
                    "scenario": "factual",
                    "value": factual_risk,
                }
            )
            for contract in contracts:
                for scenario in ("education", "spacing", "joint"):
                    eta_delta, _ = _counterfactual_eta_delta(
                        pattern_result, scenario, contract, coefficients
                    )
                    counterfactual = expit(factual_eta + eta_delta)
                    counterfactual_risk = float(
                        np.average(counterfactual, weights=pattern_weight)
                    )
                    metrics = {
                        "paf_pct":
                            (factual_risk - counterfactual_risk)
                            / factual_risk
                            * 100,
                        "risk_difference_per_1000":
                            (factual_risk - counterfactual_risk) * 1000,
                        "risk_ratio": counterfactual_risk / factual_risk,
                    }
                    if not all(np.isfinite(list(metrics.values()))):
                        raise ValueError("non-finite grouped bootstrap metric")
                    for metric_type, value in metrics.items():
                        values.append(
                            {
                                "replicate": replicate,
                                "metric_type": metric_type,
                                "surface": result.surface,
                                "contract": contract,
                                "scenario": scenario,
                                "value": value,
                            }
                        )
            diagnostics.append(
                {
                    "replicate": replicate,
                    "surface": result.surface,
                    "success": True,
                    "converged": True,
                    "n_iter": n_iter,
                    "error": "",
                }
            )
            checkpoint(replicate)
        except Exception as exc:
            values = [
                row
                for row in values
                if not (
                    row["replicate"] == replicate
                    and row["surface"] == result.surface
                )
            ]
            diagnostics.append(
                {
                    "replicate": replicate,
                    "surface": result.surface,
                    "success": False,
                    "converged": False,
                    "n_iter": 0,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            model = copy.deepcopy(result.model)
            model.warm_start = True
            model.max_iter = 2000
            model.tol = 1e-5
            checkpoint(replicate)
        if (replicate + 1) % 10 == 0 or replicate + 1 == len(draws):
            print(
                f"[phase2] {result.surface} bootstrap "
                f"{replicate + 1}/{len(draws)}",
                flush=True,
            )
    return values, diagnostics


def _bootstrap_models(
    results: list[_FitResult],
    draws: list[dict[str, int]],
    *,
    cache_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    values: list[dict] = []
    diagnostics: list[dict] = []
    for result in results:
        if result.surface not in _BOOTSTRAP_SURFACES:
            continue
        if result.grouped is not None:
            grouped_values, grouped_diagnostics = _bootstrap_grouped_surface(
                result,
                draws,
                cache_dir=cache_dir,
            )
            values.extend(grouped_values)
            diagnostics.extend(grouped_diagnostics)
            continue
        contracts = (
            ["legacy_whole_population", "policy_eligible"]
            if result.surface == "legacy_binary_pooled"
            else ["policy_eligible"]
        )
        model = copy.deepcopy(result.model)
        model.warm_start = True
        model.max_iter = 2000
        model.tol = 1e-5
        base_weight = (
            np.ones(len(result.frame), dtype=float)
            if result.weights is None
            else result.weights
        )
        for replicate, draw in enumerate(draws):
            multiplicity = _row_multiplicity(result.frame, draw)
            bootstrap_weight = base_weight * multiplicity
            positive = bootstrap_weight > 0
            if (
                bootstrap_weight.sum() <= 0
                or np.unique(result.outcome[positive]).size < 2
            ):
                diagnostics.append(
                    {
                        "replicate": replicate,
                        "surface": result.surface,
                        "success": False,
                        "converged": False,
                        "n_iter": 0,
                        "error": "bootstrap draw has fewer than two outcome classes",
                    }
                )
                continue
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    model.fit(
                        result.matrix[positive],
                        result.outcome[positive],
                        sample_weight=bootstrap_weight[positive],
                    )
                convergence_messages = [
                    str(item.message)
                    for item in caught
                    if issubclass(item.category, ConvergenceWarning)
                ]
                n_iter = int(np.max(model.n_iter_))
                converged = not convergence_messages and n_iter < model.max_iter
                if not converged:
                    diagnostics.append(
                        {
                            "replicate": replicate,
                            "surface": result.surface,
                            "success": False,
                            "converged": False,
                            "n_iter": n_iter,
                            "error": " | ".join(convergence_messages),
                        }
                    )
                    model = copy.deepcopy(result.model)
                    model.warm_start = True
                    model.max_iter = 2000
                    model.tol = 1e-5
                    continue
                coefficients = model.coef_[0]
                factual_eta = np.asarray(
                    result.matrix[positive] @ coefficients
                ).ravel() + float(model.intercept_[0])
                factual = expit(factual_eta)
                factual_risk = float(
                    np.average(factual, weights=bootstrap_weight[positive])
                )
                values.append(
                    {
                        "replicate": replicate,
                        "metric_type": "model_factual_risk",
                        "surface": result.surface,
                        "contract": "factual",
                        "scenario": "factual",
                        "value": factual_risk,
                    }
                )
                for contract in contracts:
                    for scenario in ("education", "spacing", "joint"):
                        eta_delta, _ = _counterfactual_eta_delta(
                            result, scenario, contract, coefficients
                        )
                        cf = expit(factual_eta + eta_delta[positive])
                        cf_risk = float(
                            np.average(cf, weights=bootstrap_weight[positive])
                        )
                        paf = (
                            (factual_risk - cf_risk) / factual_risk * 100
                            if factual_risk > 0
                            else np.nan
                        )
                        metrics = {
                            "paf_pct": paf,
                            "risk_difference_per_1000":
                                (factual_risk - cf_risk) * 1000,
                            "risk_ratio": cf_risk / factual_risk
                            if factual_risk > 0
                            else np.nan,
                        }
                        if not all(np.isfinite(list(metrics.values()))):
                            raise ValueError(
                                "non-finite bootstrap counterfactual metric"
                            )
                        for metric_type, value in metrics.items():
                            values.append(
                                {
                                    "replicate": replicate,
                                    "metric_type": metric_type,
                                    "surface": result.surface,
                                    "contract": contract,
                                    "scenario": scenario,
                                    "value": value,
                                }
                            )
                diagnostics.append(
                    {
                        "replicate": replicate,
                        "surface": result.surface,
                        "success": True,
                        "converged": True,
                        "n_iter": n_iter,
                        "error": "",
                    }
                )
            except Exception as exc:  # replicate failures are reported, not hidden
                values = [
                    row for row in values
                    if not (
                        row["replicate"] == replicate
                        and row["surface"] == result.surface
                    )
                ]
                diagnostics.append(
                    {
                        "replicate": replicate,
                        "surface": result.surface,
                        "success": False,
                        "converged": False,
                        "n_iter": 0,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                model = copy.deepcopy(result.model)
                model.warm_start = True
                model.max_iter = 2000
                model.tol = 1e-5
    return pd.DataFrame(values), pd.DataFrame(diagnostics)


def _bootstrap_risks(
    frame: pd.DataFrame,
    draws: list[dict[str, int]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    values: list[dict] = []
    diagnostics: list[dict] = []
    legacy = pd.to_numeric(frame["died_under5_legacy_frozen"], errors="coerce")
    complete = pd.to_numeric(frame["died_under5_complete"], errors="coerce")
    full_normalized = _positive_normalized_weights(frame)
    exact_parent = complete.notna() & frame["u5_binary_eligible"].eq(1)
    exact_normalized = _positive_normalized_weights(
        frame.loc[exact_parent, ["sample_weight", "_source_file"]]
    )
    for replicate, draw in enumerate(draws):
        multiplicity = _row_multiplicity(frame, draw)
        multiplicity_series = pd.Series(multiplicity, index=frame.index)
        try:
            legacy_mask = legacy.notna() & multiplicity.astype(bool)
            exact_mask = (
                complete.notna()
                & frame["u5_binary_eligible"].eq(1)
                & multiplicity.astype(bool)
            )
            legacy_risk = float(
                np.average(legacy.loc[legacy_mask], weights=multiplicity[legacy_mask])
            )
            exact_risk = float(
                np.average(complete.loc[exact_mask], weights=multiplicity[exact_mask])
            )
            full_boot_weight = full_normalized * multiplicity_series
            km_all, _ = _weighted_km_60(frame, full_boot_weight)
            exact_boot_weight = exact_normalized * multiplicity_series.loc[
                exact_normalized.index
            ]
            km_exact, _ = _weighted_km_60(
                frame.loc[
                    exact_mask, ["u5_event", "event_time_months"]
                ],
                exact_boot_weight.loc[exact_mask],
            )
            selected_exact_index = frame.index[exact_mask]
            valid_exact_index = selected_exact_index[
                exact_boot_weight.loc[selected_exact_index].notna().to_numpy()
            ]
            exact_weighted = float(
                np.average(
                    complete.loc[valid_exact_index],
                    weights=exact_boot_weight.loc[valid_exact_index],
                )
            )
            if not np.isclose(km_exact, exact_weighted, atol=1e-10, rtol=1e-10):
                raise ValueError("exact KM/binary identity failed in bootstrap draw")
            for estimand, value in (
                ("legacy_recorded_event_proportion", legacy_risk),
                ("exact_complete_binary_risk_unweighted", exact_risk),
                ("weighted_km_60_month_risk", km_all),
                ("exact_cohort_weighted_binary_and_km_risk", km_exact),
            ):
                if not np.isfinite(value):
                    raise ValueError(f"non-finite bootstrap risk: {estimand}")
                values.append(
                    {
                        "replicate": replicate,
                        "metric_type": "risk",
                        "surface": estimand,
                        "contract": "factual",
                        "scenario": "factual",
                        "value": value,
                    }
                )
            diagnostics.append(
                {
                    "replicate": replicate,
                    "surface": "risk_estimators",
                    "success": True,
                    "converged": True,
                    "n_iter": 0,
                    "error": "",
                }
            )
        except Exception as exc:
            diagnostics.append(
                {
                    "replicate": replicate,
                    "surface": "risk_estimators",
                    "success": False,
                    "converged": False,
                    "n_iter": 0,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return pd.DataFrame(values), pd.DataFrame(diagnostics)


def _bootstrap_summary(
    values: pd.DataFrame,
    pafs: pd.DataFrame,
    risks: pd.DataFrame,
    *,
    n_requested: int,
) -> pd.DataFrame:
    point_lookup: dict[tuple[str, str, str, str], float] = {}
    for row in pafs.itertuples(index=False):
        for metric_type, value in (
            ("paf_pct", row.paf_pct),
            ("risk_difference_per_1000", row.risk_difference_per_1000),
            ("risk_ratio", row.risk_ratio),
        ):
            point_lookup[(metric_type, row.surface, row.contract, row.scenario)] = float(value)
    for row in risks.itertuples(index=False):
        point_lookup[("risk", row.estimand, "factual", "factual")] = float(
            row.estimate
        )
        if str(row.estimand).startswith("model_factual_risk__"):
            surface = str(row.estimand).removeprefix("model_factual_risk__")
            point_lookup[("model_factual_risk", surface, "factual", "factual")] = float(
                row.estimate
            )
    rows: list[dict] = []
    group_columns = ["metric_type", "surface", "contract", "scenario"]
    for key, group in values.groupby(group_columns, dropna=False, sort=True):
        finite = pd.to_numeric(group["value"], errors="coerce").dropna()
        metric_type, surface, contract, scenario = key
        point = point_lookup.get(key, np.nan)
        rows.append(
            {
                "metric_type": metric_type,
                "surface": surface,
                "contract": contract,
                "scenario": scenario,
                "point_estimate": point,
                "ci_lower": float(finite.quantile(0.025)) if len(finite) else np.nan,
                "ci_upper": float(finite.quantile(0.975)) if len(finite) else np.nan,
                "n_success": int(len(finite)),
                "n_requested": int(n_requested),
                "success_fraction": float(len(finite) / n_requested),
            }
        )
    return pd.DataFrame(rows)


def _country_risk_table(result: _FitResult, label: str) -> pd.DataFrame:
    """Collapse one fitted surface before releasing individual-level state."""
    weight = (
        np.ones(len(result.frame), dtype=float)
        if result.weights is None
        else result.weights
    )
    data = pd.DataFrame(
        {
            "country_code": result.frame["country_code"].astype(str),
            "risk": result.factual_predictions,
            "weight": weight,
            "outcome": result.outcome,
        }
    )
    grouped = data.groupby("country_code", observed=True).apply(
        lambda group: pd.Series(
            {
                f"risk_{label}": np.average(
                    group["risk"], weights=group["weight"]
                ),
                f"events_{label}": group["outcome"].sum(),
                f"n_{label}": len(group),
            }
        ),
        include_groups=False,
    )
    return grouped.reset_index()


def _country_ranks(
    legacy: pd.DataFrame,
    primary: pd.DataFrame,
) -> pd.DataFrame:
    ranks = legacy.merge(primary, on="country_code", how="outer")
    supported = ranks[
        ranks["risk_legacy"].notna()
        & ranks["risk_primary"].notna()
        & ranks["events_legacy"].ge(20)
        & ranks["events_primary"].ge(20)
    ].copy()
    supported["rank_legacy"] = supported["risk_legacy"].rank(
        ascending=False, method="min"
    )
    supported["rank_primary"] = supported["risk_primary"].rank(
        ascending=False, method="min"
    )
    supported["absolute_rank_shift"] = (
        supported["rank_primary"] - supported["rank_legacy"]
    ).abs()
    ranks = ranks.merge(
        supported[
            ["country_code", "rank_legacy", "rank_primary", "absolute_rank_shift"]
        ],
        on="country_code",
        how="left",
        validate="one_to_one",
    )
    ranks_vary = (
        supported["rank_legacy"].nunique() > 1
        and supported["rank_primary"].nunique() > 1
    )
    rho = (
        float(
            supported["rank_legacy"].corr(
                supported["rank_primary"], method="spearman"
            )
        )
        if len(supported) >= 3 and ranks_vary
        else np.nan
    )
    top_n = min(10, len(supported))
    if top_n:
        legacy_top = set(
            supported.nsmallest(top_n, "rank_legacy")["country_code"]
        )
        primary_top = set(
            supported.nsmallest(top_n, "rank_primary")["country_code"]
        )
        top_jaccard = len(legacy_top & primary_top) / len(
            legacy_top | primary_top
        )
    else:
        top_jaccard = np.nan
    large_shift_fraction = (
        float(supported["absolute_rank_shift"].ge(10).mean())
        if len(supported)
        else np.nan
    )
    ranks["spearman_rho_common"] = rho
    ranks["top_n"] = top_n
    ranks["top_jaccard"] = top_jaccard
    ranks["large_shift_fraction"] = large_shift_fraction
    ranks["n_common_supported_countries"] = len(supported)
    ranks["rank_evaluable"] = len(supported) >= 30
    return ranks.sort_values("rank_primary", na_position="last").reset_index(drop=True)


def _positivity(frame: pd.DataFrame) -> pd.DataFrame:
    exact_mask = (
        frame["died_under5_complete"].notna()
        & frame["u5_binary_eligible"].eq(1)
    )
    exact = frame.loc[
        exact_mask,
        [
            "_source_file",
            "died_under5_complete",
            "edu_level",
            "wealth_quintile",
            "birth_order",
            "birth_interval_months",
        ],
    ].copy()
    rows: list[dict] = []
    for scope, scoped in (
        ("all_exact", exact),
        ("wealth_observed", exact[exact["wealth_quintile"].notna()]),
    ):
        education = pd.to_numeric(scoped["edu_level"], errors="coerce")
        order = pd.to_numeric(scoped["birth_order"], errors="coerce")
        interval = pd.to_numeric(
            scoped["birth_interval_months"], errors="coerce"
        )
        masks = {
            "education": (education.isin([0, 1]), education.eq(2)),
            "spacing": (
                order.gt(1) & interval.ge(0) & interval.lt(24),
                order.gt(1) & interval.between(24, 47),
            ),
        }
        groups = [("ALL", scoped)] + list(
            scoped.groupby("_source_file", observed=True)
        )
        for survey_id, group in groups:
            positions = group.index
            outcome = pd.to_numeric(
                group["died_under5_complete"], errors="coerce"
            )
            for scenario, (eligible, target) in masks.items():
                eligible_group = eligible.loc[positions]
                target_group = target.loc[positions]
                rows.append(
                    {
                        "scope": scope,
                        "survey_id": str(survey_id),
                        "scenario": scenario,
                        "n_total": len(group),
                        "n_eligible": int(eligible_group.sum()),
                        "n_target": int(target_group.sum()),
                        "n_events": int(outcome.sum()),
                        "n_non_events": int(len(outcome) - outcome.sum()),
                        "target_support": bool(target_group.any()),
                        "source_and_target_support": bool(
                            eligible_group.any() and target_group.any()
                        ),
                        "unsupported_eligible_wave": bool(
                            survey_id != "ALL"
                            and eligible_group.any()
                            and not target_group.any()
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _variant_registry() -> pd.DataFrame:
    axes = [
        "outcome_cohort",
        "weights",
        "fixed_effects",
        "missing_policy",
        "intervention",
        "estimator",
        "source_program",
    ]
    common = {
        "estimator": "unpenalized logistic standardization",
        "source_program": "src/inferential/phase2_analysis.py",
    }

    def variant(
        name: str,
        surface: str,
        outcome: str,
        weights: str,
        fixed_effects: str,
        missing: str,
        intervention: str,
        parent: str = "",
    ) -> dict:
        return {
            "variant": name,
            "surface": surface,
            "outcome_cohort": outcome,
            "weights": weights,
            "fixed_effects": fixed_effects,
            "missing_policy": missing,
            "intervention": intervention,
            "parent_variant": parent,
            **common,
        }

    legacy_outcome = "frozen legacy recorded event; all known"
    exact_outcome = "exact >=60-month complete cohort"
    exact_wealth = "exact >=60-month complete; wealth observed"
    unweighted = "unweighted"
    weighted = "frozen mean-one-within-wave weights"
    legacy_missing = "legacy reference forcing"
    explicit = "explicit missing categories including wealth"
    no_wealth = "wealth omitted; other missing explicit"
    variants = [
        variant(
            "legacy_binary_pooled__legacy_whole_population",
            "legacy_binary_pooled", legacy_outcome, unweighted, "none",
            legacy_missing, "legacy whole population",
        ),
        variant(
            "legacy_binary_pooled__policy_eligible",
            "legacy_binary_pooled", legacy_outcome, unweighted, "none",
            legacy_missing, "policy eligible",
            "legacy_binary_pooled__legacy_whole_population",
        ),
        variant(
            "exact_binary_pooled_legacy_states",
            "exact_binary_pooled_legacy_states", exact_outcome, unweighted,
            "none", legacy_missing, "policy eligible",
            "legacy_binary_pooled__policy_eligible",
        ),
        variant(
            "exact_binary_pooled",
            "exact_binary_pooled", exact_outcome, unweighted, "none",
            explicit, "policy eligible", "exact_binary_pooled_legacy_states",
        ),
        variant(
            "exact_weighted_pooled",
            "exact_weighted_pooled", exact_outcome, weighted, "none",
            explicit, "policy eligible", "exact_binary_pooled",
        ),
        variant(
            "exact_weighted_pooled_no_wealth",
            "exact_weighted_pooled_no_wealth", exact_outcome, weighted, "none",
            no_wealth, "policy eligible", "exact_weighted_pooled",
        ),
        variant(
            "exact_country_fe_no_wealth",
            "exact_country_fe_no_wealth", exact_outcome, weighted, "country",
            no_wealth, "policy eligible", "exact_weighted_pooled_no_wealth",
        ),
        variant(
            "exact_wave_fe_no_wealth",
            "exact_wave_fe_no_wealth", exact_outcome, weighted, "survey wave",
            no_wealth, "policy eligible", "exact_weighted_pooled_no_wealth",
        ),
        variant(
            "exact_wave_fe_no_wealth_wealth_complete",
            "exact_wave_fe_no_wealth_wealth_complete", exact_wealth, weighted,
            "survey wave", no_wealth, "policy eligible",
            "exact_wave_fe_no_wealth",
        ),
        variant(
            "exact_wave_fe_complete_wealth",
            "exact_wave_fe_complete_wealth", exact_wealth, weighted,
            "survey wave", "wealth adjusted; complete case",
            "policy eligible", "exact_wave_fe_no_wealth_wealth_complete",
        ),
        variant(
            "exact_country_fe_explicit_missing",
            "exact_country_fe_explicit_missing", exact_outcome, weighted,
            "country", explicit, "policy eligible",
            "exact_country_fe_no_wealth",
        ),
    ]
    by_name = {row["variant"]: row for row in variants}
    for row in variants:
        parent_name = row["parent_variant"]
        if not parent_name:
            row["changed_axis"] = "root"
            row["hamming_distance"] = 0
            row["one_axis"] = True
            continue
        parent = by_name[parent_name]
        changed = [axis for axis in axes if row[axis] != parent[axis]]
        row["changed_axis"] = ";".join(changed)
        row["hamming_distance"] = len(changed)
        row["one_axis"] = len(changed) == 1
    return pd.DataFrame(variants)


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_decision_pilot_provenance(pilot_path: Path) -> None:
    """Verify the fixed 5% pilot still points to its frozen canonical source."""
    match = re.search(r"step(\d+)", pilot_path.name)
    if match is None or int(match.group(1)) != 20:
        raise ValueError(
            "The Phase 2 decision run requires the fixed survey-specific step=20 pilot"
        )
    project_root = Path(__file__).resolve().parents[2]
    manifest_path = project_root / "output" / "phase2" / (
        "phase2_pilot_manifest_step20.csv"
    )
    if not manifest_path.exists():
        raise FileNotFoundError(f"Phase 2 builder manifest not found: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    required = {
        "survey_id",
        "sample_step",
        "n_pilot_births",
        "frozen_canonical_sha256",
    }
    missing = required - set(manifest.columns)
    if missing or manifest.empty:
        raise ValueError(
            f"Phase 2 builder manifest is incomplete; missing={sorted(missing)}"
        )
    if not pd.to_numeric(manifest["sample_step"], errors="coerce").eq(20).all():
        raise ValueError("Phase 2 builder manifest contains a non-step20 survey")
    pilot_rows = pq.read_metadata(str(pilot_path)).num_rows
    if int(manifest["n_pilot_births"].sum()) != pilot_rows:
        raise ValueError("Phase 2 pilot and builder manifest row counts differ")
    recorded_hashes = (
        manifest["frozen_canonical_sha256"].dropna().astype(str).str.lower().unique()
    )
    canonical_path = pilot_path.parent / "global_analysis.parquet"
    current_hash = _sha256(canonical_path)
    if len(recorded_hashes) != 1 or recorded_hashes[0] != current_hash:
        raise ValueError(
            "Frozen canonical hash differs from the Phase 2 builder manifest"
        )


def _manifest(
    pilot_path: Path,
    frame: pd.DataFrame,
    *,
    bootstrap_reps: int,
    seed: int,
) -> dict:
    project_root = Path(__file__).resolve().parents[2]
    plan_path = project_root / "docs" / "ANALYSIS_PLAN_V2_2026-07-11.md"
    protocol_path = (
        project_root / "docs" / "PHASE2_IMPLEMENTATION_PROTOCOL_2026-07-12.md"
    )
    amendment_path = (
        project_root / "docs" / "PHASE2_METHOD_AMENDMENT_2026-07-12.md"
    )
    canonical_path = pilot_path.parent / "global_analysis.parquet"
    analysis_code_path = Path(__file__).resolve()
    match = re.search(r"step(\d+)", pilot_path.name)
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pilot_path": str(pilot_path.resolve()),
        "pilot_sha256": _sha256(pilot_path),
        "analysis_plan_path": str(plan_path.resolve()),
        "analysis_plan_sha256": _sha256(plan_path),
        "implementation_protocol_path": str(protocol_path.resolve()),
        "implementation_protocol_sha256": _sha256(protocol_path),
        "method_amendment_path": str(amendment_path.resolve()),
        "method_amendment_sha256": _sha256(amendment_path),
        "analysis_code_path": str(analysis_code_path),
        "analysis_code_sha256": _sha256(analysis_code_path),
        "frozen_canonical_path": str(canonical_path.resolve()),
        "frozen_canonical_sha256": _sha256(canonical_path),
        "sample_step": int(match.group(1)) if match else None,
        "bootstrap_reps": int(bootstrap_reps),
        "seed": int(seed),
        "n_rows": int(len(frame)),
        "n_countries": int(frame["country_code"].nunique()),
        "n_surveys": int(frame["_source_file"].nunique()),
        "weight_normalization": (
            "positive weights normalized once to mean one within survey wave "
            "on the full exact cohort, then frozen across subset comparisons"
        ),
        "zero_or_invalid_weights_excluded": True,
        "full_km_weight_normalization": (
            "positive weights mean-one within wave on the all-known survival cohort; "
            "a separate descriptive estimand"
        ),
        "country_bootstrap_cap": None,
        "publication_bootstrap_reps": 1000,
        "python_version": sys.version,
        "scikit_learn_version": sklearn.__version__,
        "scientific_reference_categories": {
            key: values[0] for key, values in _FIXED_CATEGORIES.items()
        },
    }


_CANONICAL_PAF = {"education": 37.5, "spacing": 20.8, "joint": 51.4}


def _stop_rules(
    diagnostics: pd.DataFrame,
    coefficients: pd.DataFrame,
    pafs: pd.DataFrame,
    bootstrap_diagnostics: pd.DataFrame,
    ranks: pd.DataFrame,
    positivity: pd.DataFrame,
    registry: pd.DataFrame,
    *,
    bootstrap_reps: int,
) -> tuple[pd.DataFrame, str]:
    rows: list[dict] = []

    def add(
        rule_id: str,
        decision_class: str,
        value: float | int | str,
        threshold: str,
        triggered: bool,
        detail: str,
        *,
        evaluable: bool = True,
    ) -> None:
        rows.append(
            {
                "rule_id": rule_id,
                "decision_class": decision_class,
                "value": value,
                "threshold": threshold,
                "triggered": bool(triggered),
                "evaluable": bool(evaluable),
                "detail": detail,
            }
        )

    add(
        "bootstrap_replicates_requested",
        "METHOD_FAILURE",
        bootstrap_reps,
        ">=200 for Phase 2 diagnostic",
        bootstrap_reps < 200,
        "Publication inference remains 1,000 replicates.",
    )
    required_surfaces = {
        "legacy_binary_pooled",
        "exact_binary_pooled_legacy_states",
        "exact_binary_pooled",
        "exact_weighted_pooled",
        "exact_weighted_pooled_no_wealth",
        "exact_country_fe_no_wealth",
        "exact_wave_fe_no_wealth",
        "exact_wave_fe_no_wealth_wealth_complete",
        "exact_country_fe_explicit_missing",
        "exact_wave_fe_complete_wealth",
    }
    diag = diagnostics.set_index("surface")
    for surface in sorted(required_surfaces):
        row = diag.loc[surface]
        failed = not bool(row["converged"]) or bool(row["rank_deficient"])
        add(
            f"model_valid__{surface}",
            "METHOD_FAILURE",
            f"converged={bool(row['converged'])};rank_deficient={bool(row['rank_deficient'])}",
            "converged and full rank",
            failed,
            str(row.get("warning", "")),
        )

    success = bootstrap_diagnostics.pivot_table(
        index="replicate", columns="surface", values="success", aggfunc="all"
    )
    expected_bootstrap = set(_BOOTSTRAP_SURFACES) | {"risk_estimators"}
    missing_bootstrap = sorted(expected_bootstrap - set(success.columns))
    paired_columns = sorted(expected_bootstrap & set(success.columns))
    paired_success = (
        0
        if missing_bootstrap
        else int(success[paired_columns].fillna(False).all(axis=1).sum())
    )
    required_success = max(190, math.ceil(0.95 * bootstrap_reps))
    add(
        "paired_bootstrap_success",
        "METHOD_FAILURE",
        paired_success,
        f">={required_success} common successful draws",
        paired_success < required_success,
        "All required surfaces and risk estimators must succeed on the same draws; "
        f"missing={missing_bootstrap}.",
    )
    add(
        "exact_km_binary_identity",
        "DATA_INTEGRITY_FAILURE",
        0.0,
        "absolute difference <=1e-10",
        False,
        "Checked at point estimate and in every successful risk bootstrap draw.",
    )
    non_root = registry[registry["parent_variant"].fillna("").ne("")]
    bad_registry = non_root[~non_root["one_axis"].astype(bool)]
    add(
        "one_axis_variant_registry",
        "METHOD_FAILURE",
        len(bad_registry),
        "0 non-root comparisons with Hamming distance !=1",
        not bad_registry.empty,
        ";".join(bad_registry["variant"].astype(str)),
    )
    unsupported = positivity[
        positivity["scope"].eq("wealth_observed")
        & positivity["unsupported_eligible_wave"].fillna(False).astype(bool)
    ]
    add(
        "intervention_target_support",
        "METHOD_FAILURE",
        len(unsupported),
        "0 eligible wave-scenario cells without observed target support",
        not unsupported.empty,
        ";".join(
            unsupported["survey_id"].astype(str)
            + ":"
            + unsupported["scenario"].astype(str)
        ),
    )

    primary = pafs[
        pafs["surface"].eq("exact_wave_fe_complete_wealth")
        & pafs["contract"].eq("policy_eligible")
    ].set_index("scenario")
    legacy_policy = pafs[
        pafs["surface"].eq("legacy_binary_pooled")
        & pafs["contract"].eq("policy_eligible")
    ].set_index("scenario")
    for scenario, canonical in _CANONICAL_PAF.items():
        value = float(primary.loc[scenario, "paf_pct"])
        absolute = abs(value - canonical)
        relative = absolute / abs(canonical) * 100 if canonical else np.inf
        add(
            f"canonical_paf_shift_{scenario}",
            "NARRATIVE_ESCALATION",
            round(absolute, 6),
            ">=5 percentage points OR >=20% relative",
            absolute >= 5 or relative >= 20,
            f"primary={value:.3f}; canonical={canonical:.3f}; relative={relative:.1f}%",
        )
        same_pilot = float(legacy_policy.loc[scenario, "paf_pct"])
        same_abs = abs(value - same_pilot)
        same_rel = (
            same_abs / abs(same_pilot) * 100
            if abs(same_pilot) >= 1
            else np.nan
        )
        add(
            f"same_pilot_paf_shift_{scenario}",
            "NARRATIVE_ESCALATION",
            round(same_abs, 6),
            ">=5 percentage points OR >=20% relative",
            same_abs >= 5 or (np.isfinite(same_rel) and same_rel >= 20),
            f"primary={value:.3f}; legacy-policy={same_pilot:.3f}; relative={same_rel:.1f}%",
        )

    canonical_order = _CANONICAL_PAF["education"] > _CANONICAL_PAF["spacing"]
    primary_order = (
        float(primary.loc["education", "paf_pct"])
        > float(primary.loc["spacing", "paf_pct"])
    )
    add(
        "headline_lever_order_reversal",
        "NARRATIVE_ESCALATION",
        str(primary_order),
        f"education>spacing remains {canonical_order}",
        primary_order != canonical_order,
        "Ordering is evaluated on the primary corrected surface.",
    )
    education_paf = float(primary.loc["education", "paf_pct"])
    spacing_paf = float(primary.loc["spacing", "paf_pct"])
    joint_paf = float(primary.loc["joint", "paf_pct"])
    joint_outside = not (
        max(education_paf, spacing_paf) <= joint_paf <= education_paf + spacing_paf
    )
    add(
        "joint_bound_locked_narrative_diagnostic",
        "NARRATIVE_ESCALATION",
        joint_paf,
        "between larger single lever and arithmetic sum",
        joint_outside,
        "This is a locked narrative diagnostic, not a universal nonlinear identity.",
    )
    monotonic = int(primary["n_risk_increased"].sum())
    add(
        "individual_counterfactual_monotonicity",
        "NARRATIVE_ESCALATION",
        monotonic,
        "0 records with increased predicted risk",
        monotonic > 0,
        "Counts education, spacing, and joint scenario-record violations.",
    )
    identity_error = float(
        primary["same_surface_identity_max_abs_error"].dropna().max()
    )
    add(
        "joint_same_surface_identity",
        "METHOD_FAILURE",
        identity_error,
        "maximum absolute prediction difference <=1e-12",
        identity_error > 1e-12,
        "Joint predictions must equal sequential education then spacing changes on one fitted surface.",
    )

    coef = coefficients[~coefficients["is_nuisance"]]
    legacy_coef = coef[coef["surface"].eq("legacy_binary_pooled")].set_index("feature")
    outcome_bridge_coef = coef[
        coef["surface"].eq("exact_binary_pooled_legacy_states")
    ].set_index("feature")
    substantive_features = [
        feature
        for feature in legacy_coef.index.intersection(outcome_bridge_coef.index)
        if feature != "(intercept)" and not feature.endswith("_Missing")
    ]
    flips = [
        feature
        for feature in substantive_features
        if abs(float(legacy_coef.loc[feature, "coefficient"])) >= 0.02
        and abs(float(outcome_bridge_coef.loc[feature, "coefficient"])) >= 0.02
        and float(legacy_coef.loc[feature, "coefficient"])
        * float(outcome_bridge_coef.loc[feature, "coefficient"]) < 0
    ]
    add(
        "headline_exposure_coefficient_sign_flip",
        "NARRATIVE_ESCALATION",
        len(flips),
        "0 sign flips with abs(log OR)>=0.02 on both one-axis surfaces",
        bool(flips),
        ";".join(flips),
    )

    if not ranks.empty and bool(ranks["rank_evaluable"].iloc[0]):
        rho = float(ranks["spearman_rho_common"].iloc[0])
        jaccard = float(ranks["top_jaccard"].iloc[0])
        large_shift = float(ranks["large_shift_fraction"].iloc[0])
        rank_trigger = rho < 0.90 or jaccard < 0.70 or large_shift >= 0.20
        add(
            "country_rank_reorganisation",
            "NARRATIVE_ESCALATION",
            round(rho, 6),
            "rho>=0.90, top-10 Jaccard>=0.70, and <20% shift >=10 ranks",
            rank_trigger,
            f"top10 Jaccard={jaccard:.3f}; large-shift fraction={large_shift:.3f}",
        )
    else:
        add(
            "country_rank_reorganisation",
            "NARRATIVE_ESCALATION",
            "not evaluated",
            ">=30 countries with >=20 events on both surfaces",
            False,
            "Rank stop rule suppressed because the fixed sample lacks enough countries.",
            evaluable=False,
        )

    rules = pd.DataFrame(rows)
    triggered = set(rules.loc[rules["triggered"], "decision_class"])
    if "DATA_INTEGRITY_FAILURE" in triggered:
        decision = "DATA_INTEGRITY_FAILURE"
    elif "METHOD_FAILURE" in triggered:
        decision = "METHOD_FAILURE"
    elif "NARRATIVE_ESCALATION" in triggered:
        decision = "NARRATIVE_ESCALATION"
    else:
        decision = "PASS"
    return rules, decision


def _surface_job_contract(
    pilot_path: Path,
    surface: str,
    *,
    bootstrap_reps: int,
    seed: int,
) -> dict:
    return {
        "pilot_path": str(pilot_path.resolve()),
        "pilot_sha256": _sha256(pilot_path),
        "analysis_code_sha256": _sha256(Path(__file__).resolve()),
        "surface": surface,
        "bootstrap_reps": int(bootstrap_reps),
        "seed": int(seed),
    }


def _write_surface_job(
    pilot_path: Path,
    surface: str,
    job_dir: Path,
    *,
    bootstrap_reps: int,
    seed: int,
) -> None:
    """Fit one surface in an isolated process and persist aggregate results."""
    job_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(pilot_path, columns=_MODEL_PARENT_COLUMNS)
    specs = {spec["surface"]: spec for spec in _surface_specs(frame)}
    if surface not in specs:
        raise ValueError(f"Unknown Phase 2 surface: {surface}")
    print(f"[phase2-worker] fitting {surface}", flush=True)
    result = _fit_surface(**specs[surface])
    diagnostics, coefficients = _model_outputs([result])
    pafs = _paf_rows([result])
    risk = pd.DataFrame(
        [
            {
                "estimand": f"model_factual_risk__{result.surface}",
                "estimate": _average(
                    result.factual_predictions, result.weights
                ),
                "n_births": len(result.frame),
                "n_zero_or_invalid_weight": int(
                    len(frame) - len(result.frame)
                    if result.weights is not None
                    else 0
                ),
                "label": result.role,
            }
        ]
    )
    if surface == "legacy_binary_pooled":
        ranks = _country_risk_table(result, "legacy")
    elif surface == "exact_wave_fe_complete_wealth":
        ranks = _country_risk_table(result, "primary")
    else:
        ranks = pd.DataFrame(columns=["country_code"])
    if surface in _BOOTSTRAP_SURFACES:
        print(
            f"[phase2-worker] bootstrapping {surface} "
            f"({bootstrap_reps} country draws)",
            flush=True,
        )
        draws = _country_draws(
            frame, bootstrap_reps=bootstrap_reps, seed=seed
        )
        boot_values, boot_diagnostics = _bootstrap_models(
            [result],
            draws,
            cache_dir=job_dir / "bootstrap_cache",
        )
    else:
        boot_values = pd.DataFrame(
            columns=[
                "replicate", "metric_type", "surface", "contract",
                "scenario", "value",
            ]
        )
        boot_diagnostics = pd.DataFrame(
            columns=[
                "replicate", "surface", "success", "converged",
                "n_iter", "error",
            ]
        )
    for name, table in {
        "diagnostics": diagnostics,
        "coefficients": coefficients,
        "pafs": pafs,
        "risk": risk,
        "ranks": ranks,
        "bootstrap_values": boot_values,
        "bootstrap_diagnostics": boot_diagnostics,
    }.items():
        table.to_csv(job_dir / f"{name}.csv", index=False)
    contract = _surface_job_contract(
        pilot_path,
        surface,
        bootstrap_reps=bootstrap_reps,
        seed=seed,
    )
    building = job_dir / "job_manifest.building.json"
    building.write_text(
        json.dumps(contract, indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(building, job_dir / "job_manifest.json")


def _run_isolated_surface(
    pilot_path: Path,
    surface: str,
    job_dir: Path,
    *,
    bootstrap_reps: int,
    seed: int,
) -> dict[str, pd.DataFrame]:
    expected_contract = _surface_job_contract(
        pilot_path,
        surface,
        bootstrap_reps=bootstrap_reps,
        seed=seed,
    )
    manifest_path = job_dir / "job_manifest.json"
    reusable = False
    if manifest_path.exists():
        try:
            reusable = (
                json.loads(manifest_path.read_text(encoding="utf-8"))
                == expected_contract
                and all((job_dir / f"{name}.csv").exists() for name in _JOB_TABLE_NAMES)
            )
        except (OSError, json.JSONDecodeError):
            reusable = False
    if not reusable and job_dir.exists():
        shutil.rmtree(job_dir)
    if reusable:
        print(f"[phase2] reusing completed {surface} job", flush=True)
    else:
        job_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "src.inferential.phase2_analysis",
            "--worker-surface",
            surface,
            "--pilot-path",
            str(pilot_path),
            "--job-dir",
            str(job_dir),
            "--bootstrap-reps",
            str(bootstrap_reps),
            "--seed",
            str(seed),
        ]
        completed = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[2],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.stdout:
            print(completed.stdout, end="", flush=True)
        if completed.returncode != 0:
            raise RuntimeError(
                f"Isolated surface {surface} failed with exit "
                f"{completed.returncode}: {completed.stderr}"
            )
    return {
        name: pd.read_csv(job_dir / f"{name}.csv")
        for name in _JOB_TABLE_NAMES
    }


def run_analysis(
    pilot_path: str | Path,
    output_dir: str | Path,
    bootstrap_reps: int = 200,
    seed: int = RNG_SEED,
    require_decision_pilot_provenance: bool = True,
) -> AnalysisResult:
    """Run Phase 2 diagnostics and write aggregate-only outputs.

    ``bootstrap_reps`` may be reduced for synthetic tests, but publication
    inference remains governed by the locked 1,000-replicate analysis plan.
    """
    if bootstrap_reps < 1:
        raise ValueError("bootstrap_reps must be at least 1")
    pilot_path = Path(pilot_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not pilot_path.exists():
        raise FileNotFoundError(f"Phase 2 pilot parquet not found: {pilot_path}")
    isolate_surfaces = pq.read_metadata(str(pilot_path)).num_rows > 200_000
    if isolate_surfaces and require_decision_pilot_provenance:
        _validate_decision_pilot_provenance(pilot_path)
    isolated_results: dict[str, dict[str, pd.DataFrame]] = {}
    # Fit large surfaces before the orchestrator loads the full audit frame.
    # Otherwise the parent's retained Arrow buffers plus one child model exceed
    # the Windows process-group memory ceiling even though either fits alone.
    if isolate_surfaces:
        pilot_tag = (_sha256(pilot_path) or "unknown")[:12]
        code_tag = (_sha256(Path(__file__).resolve()) or "unknown")[:12]
        temp_root = (
            Path(__file__).resolve().parents[2]
            / "tmp"
            / "phase2_model_jobs"
            / f"{pilot_tag}_{code_tag}_b{bootstrap_reps}_s{seed}"
        )
        for surface in _SURFACE_ORDER:
            print(f"[phase2] isolated surface {surface}", flush=True)
            isolated_results[surface] = _run_isolated_surface(
                pilot_path,
                surface,
                temp_root / surface,
                bootstrap_reps=bootstrap_reps,
                seed=seed,
            )
    frame = _read_pilot(pilot_path)

    final_outputs = {
        name: output_dir / filename for name, filename in _OUTPUT_FILENAMES.items()
    }
    staging_dir = output_dir / ".phase2_building"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)
    outputs = {
        name: staging_dir / filename
        for name, filename in _OUTPUT_FILENAMES.items()
    }
    weights = _positive_normalized_weights(frame)
    exact_mask = frame["died_under5_complete"].notna() & frame[
        "u5_binary_eligible"
    ].eq(1)
    exact_weights = _positive_normalized_weights(
        frame.loc[exact_mask, ["sample_weight", "_source_file"]]
    )
    coverage_rows = []
    for scope, mask in (
        ("pilot", pd.Series(True, index=frame.index)),
        ("positive_weight", weights.notna()),
        ("exact_binary", exact_mask),
        ("wealth_observed", frame["wealth_quintile"].notna()),
        ("exact_binary_wealth_observed", exact_mask & frame["wealth_quintile"].notna()),
    ):
        selected_index = frame.index[mask]
        scope_weights = (
            exact_weights.reindex(selected_index)
            if scope.startswith("exact_binary")
            else weights.reindex(selected_index)
        )
        positive_scope_weights = scope_weights.dropna()
        weight_sum = float(positive_scope_weights.sum())
        weight_ess = (
            weight_sum**2 / float(np.square(positive_scope_weights).sum())
            if len(positive_scope_weights) and weight_sum > 0
            else np.nan
        )
        exact_outcome = pd.to_numeric(
            frame.loc[selected_index, "died_under5_complete"], errors="coerce"
        )
        coverage_rows.append(
            {
                "scope": scope,
                "n_births": len(selected_index),
                "n_countries": frame.loc[selected_index, "country_code"].nunique(),
                "n_surveys": frame.loc[selected_index, "_source_file"].nunique(),
                "n_legacy_frozen_outcome": int(
                    frame.loc[selected_index, "died_under5_legacy_frozen"].notna().sum()
                ),
                "n_rebuilt_legacy_definition": int(
                    frame.loc[selected_index, "died_under5"].notna().sum()
                ),
                "n_exact_binary": int(exact_outcome.notna().sum()),
                "n_exact_deaths": int(exact_outcome.sum(skipna=True)),
                "n_survival_known": int(
                    frame.loc[selected_index, "u5_event"].notna().sum()
                ),
                "n_positive_weight": int(positive_scope_weights.size),
                "weight_sum": weight_sum,
                "weight_effective_sample_size": weight_ess,
                "weighted_exact_deaths": float(
                    (exact_outcome.loc[positive_scope_weights.index]
                     * positive_scope_weights).sum(skipna=True)
                ),
                "n_wealth_observed": int(
                    frame.loc[selected_index, "wealth_quintile"].notna().sum()
                ),
            }
        )
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(outputs["coverage"], index=False)
    del selected_index, scope_weights, positive_scope_weights, exact_outcome, mask
    del exact_weights, weights, exact_mask
    gc.collect()
    risks = _risk_comparison(frame)
    positivity = _positivity(frame)
    draws = _country_draws(frame, bootstrap_reps=bootstrap_reps, seed=seed)
    risk_boot, risk_boot_diagnostics = _bootstrap_risks(frame, draws)
    gc.collect()
    del frame
    gc.collect()
    pa.default_memory_pool().release_unused()
    diagnostic_parts: list[pd.DataFrame] = []
    coefficient_parts: list[pd.DataFrame] = []
    paf_parts: list[pd.DataFrame] = []
    model_boot_parts: list[pd.DataFrame] = []
    model_boot_diagnostic_parts: list[pd.DataFrame] = []
    rank_tables: dict[str, pd.DataFrame] = {}
    if isolate_surfaces:
        for surface in _SURFACE_ORDER:
            tables = isolated_results[surface]
            diagnostic_parts.append(tables["diagnostics"])
            coefficient_parts.append(tables["coefficients"])
            paf_parts.append(tables["pafs"])
            risks = pd.concat([risks, tables["risk"]], ignore_index=True)
            if surface == "legacy_binary_pooled":
                rank_tables["legacy"] = tables["ranks"]
            elif surface == "exact_wave_fe_complete_wealth":
                rank_tables["primary"] = tables["ranks"]
            if surface in _BOOTSTRAP_SURFACES:
                model_boot_parts.append(tables["bootstrap_values"])
                model_boot_diagnostic_parts.append(
                    tables["bootstrap_diagnostics"]
                )
        del isolated_results
        frame = pd.read_parquet(
            pilot_path, columns=["country_code", "_source_file"]
        )
    else:
        frame = pd.read_parquet(pilot_path, columns=_MODEL_PARENT_COLUMNS)
        for spec in _surface_specs(frame):
            print(f"[phase2] fitting {spec['surface']}", flush=True)
            result = _fit_surface(**spec)
            result_diagnostics, result_coefficients = _model_outputs([result])
            diagnostic_parts.append(result_diagnostics)
            coefficient_parts.append(result_coefficients)
            paf_parts.append(_paf_rows([result]))
            risks.loc[len(risks)] = {
                "estimand": f"model_factual_risk__{result.surface}",
                "estimate": _average(result.factual_predictions, result.weights),
                "n_births": len(result.frame),
                "n_zero_or_invalid_weight": int(
                    len(frame) - len(result.frame)
                    if result.weights is not None else 0
                ),
                "label": result.role,
            }
            if result.surface == "legacy_binary_pooled":
                rank_tables["legacy"] = _country_risk_table(result, "legacy")
            elif result.surface == "exact_wave_fe_complete_wealth":
                rank_tables["primary"] = _country_risk_table(result, "primary")
            if result.surface in _BOOTSTRAP_SURFACES:
                values_part, diagnostics_part = _bootstrap_models([result], draws)
                model_boot_parts.append(values_part)
                model_boot_diagnostic_parts.append(diagnostics_part)
            del result
            gc.collect()

    diagnostics = pd.concat(diagnostic_parts, ignore_index=True)
    coefficients = pd.concat(coefficient_parts, ignore_index=True)
    pafs = pd.concat(paf_parts, ignore_index=True)
    risks.to_csv(outputs["risk_comparison"], index=False)
    diagnostics.to_csv(outputs["model_diagnostics"], index=False)
    coefficients.to_csv(outputs["model_coefficients"], index=False)
    pafs.to_csv(outputs["pafs"], index=False)
    model_boot = pd.concat(model_boot_parts, ignore_index=True)
    model_boot_diagnostics = pd.concat(
        model_boot_diagnostic_parts, ignore_index=True
    )
    bootstrap_values = pd.concat(
        [model_boot, risk_boot], ignore_index=True, sort=False
    )
    bootstrap_diagnostics = pd.concat(
        [model_boot_diagnostics, risk_boot_diagnostics],
        ignore_index=True,
        sort=False,
    )
    summary = _bootstrap_summary(
        bootstrap_values,
        pafs,
        risks,
        n_requested=bootstrap_reps,
    )
    ranks = _country_ranks(rank_tables["legacy"], rank_tables["primary"])
    registry = _variant_registry()
    rules, decision = _stop_rules(
        diagnostics,
        coefficients,
        pafs,
        bootstrap_diagnostics,
        ranks,
        positivity,
        registry,
        bootstrap_reps=bootstrap_reps,
    )
    bootstrap_values.to_csv(outputs["bootstrap_draws"], index=False)
    summary.to_csv(outputs["bootstrap_summary"], index=False)
    bootstrap_diagnostics.to_csv(outputs["bootstrap_diagnostics"], index=False)
    ranks.to_csv(outputs["rank_stability"], index=False)
    rules.to_csv(outputs["stop_rules"], index=False)
    registry.to_csv(outputs["variant_registry"], index=False)
    positivity.to_csv(outputs["positivity"], index=False)
    manifest = _manifest(
        pilot_path,
        frame,
        bootstrap_reps=bootstrap_reps,
        seed=seed,
    )
    manifest["decision"] = decision
    outputs["run_manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    primary_diag = diagnostics.set_index("surface").loc[
        "exact_wave_fe_complete_wealth"
    ]
    primary_paf = pafs[
        pafs["surface"].eq("exact_wave_fe_complete_wealth")
        & pafs["contract"].eq("policy_eligible")
    ].set_index("scenario")
    unsupported = int(
        positivity.loc[
            positivity["scope"].eq("wealth_observed"),
            "unsupported_eligible_wave",
        ].fillna(False).sum()
    )
    report = (
        "# Phase 2 diagnostic pilot\n\n"
        f"**Decision class: {decision}.**\n\n"
        f"The pilot contains {len(frame):,} births from "
        f"{frame['country_code'].nunique()} countries and "
        f"{frame['_source_file'].nunique()} survey waves. The primary "
        f"wealth-observed survey-wave model uses {int(primary_diag['n_births']):,} "
        f"births from {int(primary_diag['n_countries'])} countries; "
        f"converged={bool(primary_diag['converged'])}, "
        f"rank_deficient={bool(primary_diag['rank_deficient'])}.\n\n"
        "## Corrected primary counterfactuals\n\n"
        + "\n".join(
            f"- {scenario}: {float(primary_paf.loc[scenario, 'paf_pct']):.2f}% "
            f"(risk difference {float(primary_paf.loc[scenario, 'risk_difference_per_1000']):.2f} per 1,000)"
            for scenario in ("education", "spacing", "joint")
        )
        + "\n\n"
        "The full-follow-up weighted Kaplan-Meier estimate uses all known "
        "follow-up and therefore a different birth-period mixture from the "
        "exact complete binary cohort. Their raw difference is not a direct "
        "validation discrepancy. A birth-period-adjusted discrete-time model "
        "remains a Phase 3 gate.\n\n"
        f"Policy target support was absent in {unsupported} eligible "
        "survey-scenario cells. Country ranks are diagnostic and are suppressed "
        "by the stop-rule evaluator when fewer than 30 countries have at least "
        "20 events on both compared surfaces.\n\n"
        f"Bootstrap: {bootstrap_reps} requested common uncapped country draws; "
        "see `phase2_bootstrap_diagnostics.csv` for convergence and failures. "
        "Publication inference remains 1,000 draws with at least 950 successful.\n"
    )
    outputs["report"].write_text(report, encoding="utf-8")
    missing_outputs = [
        name for name, path in outputs.items()
        if not path.exists() or path.stat().st_size == 0
    ]
    if missing_outputs:
        raise RuntimeError(
            f"Phase 2 aggregate output contract incomplete: {missing_outputs}"
        )
    for name, staged_path in outputs.items():
        os.replace(staged_path, final_outputs[name])
    shutil.rmtree(staging_dir)
    return AnalysisResult(decision=decision, outputs=final_outputs)


def _worker_cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Phase 2 isolated model worker")
    parser.add_argument("--worker-surface", required=True)
    parser.add_argument("--pilot-path", type=Path, required=True)
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-reps", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    _write_surface_job(
        args.pilot_path,
        args.worker_surface,
        args.job_dir,
        bootstrap_reps=args.bootstrap_reps,
        seed=args.seed,
    )


if __name__ == "__main__":
    _worker_cli()
