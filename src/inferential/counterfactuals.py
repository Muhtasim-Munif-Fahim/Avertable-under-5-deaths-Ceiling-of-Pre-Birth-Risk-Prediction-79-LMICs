"""Policy-eligible counterfactuals shared by PAF analysis implementations.

The module defines the intervention contract once for both categorical model
frames and memory-efficient numeric design matrices. Protective, missing, and
inapplicable observations are preserved rather than overwritten.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Mapping

import numpy as np
import pandas as pd


EDUCATION_SECONDARY = "education_secondary"
SHORT_SPACING = "short_spacing_24_47"
JOINT_EDUCATION_SPACING = "joint_education_spacing"

_VALID_INTERVENTIONS = {
    EDUCATION_SECONDARY,
    SHORT_SPACING,
    JOINT_EDUCATION_SPACING,
}


def _validate(intervention: str) -> None:
    if intervention not in _VALID_INTERVENTIONS:
        raise ValueError(
            f"Unknown intervention {intervention!r}; expected one of "
            f"{sorted(_VALID_INTERVENTIONS)}"
        )


def _includes_education(intervention: str) -> bool:
    return intervention in {EDUCATION_SECONDARY, JOINT_EDUCATION_SPACING}


def _includes_spacing(intervention: str) -> bool:
    return intervention in {SHORT_SPACING, JOINT_EDUCATION_SPACING}


def education_eligible(frame: pd.DataFrame) -> pd.Series:
    """Rows whose observed education is none or primary."""
    education = pd.to_numeric(frame["edu_level"], errors="coerce")
    return education.isin([0, 1])


def spacing_eligible(frame: pd.DataFrame) -> pd.Series:
    """Non-first births with an observed, valid interval below 24 months."""
    order = pd.to_numeric(frame["birth_order"], errors="coerce")
    interval = pd.to_numeric(frame["birth_interval_months"], errors="coerce")
    return order.gt(1) & interval.ge(0) & interval.lt(24)


def apply_categorical_intervention(
    frame: pd.DataFrame,
    intervention: str,
) -> pd.DataFrame:
    """Return a counterfactual model frame under one defined policy shift.

    Education moves only observed none/primary values to secondary. Birth
    spacing moves only observed short intervals among non-first births to the
    24--47 month category. Higher education, long intervals, first births, and
    missing exposures remain factual.
    """
    _validate(intervention)
    result = frame.copy()

    if _includes_education(intervention):
        result.loc[education_eligible(frame), "edu_cat"] = "Secondary"
    if _includes_spacing(intervention):
        result.loc[spacing_eligible(frame), "bi_cat"] = "24-47 mo"

    return result


def _dummy_overrides(
    frame: pd.DataFrame,
    intervention: str,
) -> list[tuple[str, np.ndarray, float]]:
    overrides: list[tuple[str, np.ndarray, float]] = []
    if _includes_education(intervention):
        eligible = education_eligible(frame).to_numpy(dtype=bool)
        overrides.extend(
            [
                ("edu_primary", eligible, 0.0),
                ("edu_secondary", eligible, 1.0),
                ("edu_higher", eligible, 0.0),
            ]
        )
    if _includes_spacing(intervention):
        eligible = spacing_eligible(frame).to_numpy(dtype=bool)
        overrides.append(("bi_short", eligible, 0.0))
    return overrides


@contextmanager
def temporary_dummy_intervention(
    design: np.ndarray,
    column_index: Mapping[str, int],
    frame: pd.DataFrame,
    intervention: str,
) -> Iterator[np.ndarray]:
    """Temporarily apply an intervention to a numeric design matrix.

    Only eligible cells are copied and edited. The matrix is restored even if
    prediction raises, keeping full-data g-computation memory bounded and the
    factual design reusable across scenarios.
    """
    _validate(intervention)
    if len(design) != len(frame):
        raise ValueError("Design matrix and intervention frame must have equal rows")

    saved: list[tuple[int, np.ndarray, np.ndarray]] = []
    try:
        for name, mask, value in _dummy_overrides(frame, intervention):
            if name not in column_index or not mask.any():
                continue
            index = column_index[name]
            saved.append((index, mask, design[mask, index].copy()))
            design[mask, index] = value
        yield design
    finally:
        for index, mask, values in reversed(saved):
            design[mask, index] = values
