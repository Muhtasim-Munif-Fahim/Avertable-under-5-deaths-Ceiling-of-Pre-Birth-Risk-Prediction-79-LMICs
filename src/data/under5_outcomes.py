"""Exact follow-up construction for under-five mortality analyses."""

from __future__ import annotations

import numpy as np
import pandas as pd


DERIVED_UNDER5_COLUMNS = (
    "followup_months",
    "event_time_months",
    "u5_event",
    "u5_censored",
    "u5_binary_eligible",
    "died_under5_complete",
)

EXACT_FOLLOWUP_COLUMNS = (
    "birth_cmc",
    "interview_cmc",
    "age_at_death_months",
    *DERIVED_UNDER5_COLUMNS,
)


def derive_under5_outcomes(
    *,
    alive: pd.Series,
    age_at_death_months: pd.Series,
    birth_cmc: pd.Series,
    interview_cmc: pd.Series,
) -> pd.DataFrame:
    """Return exact 60-month binary eligibility and complete outcome.

    Binary-cohort eligibility is determined only by the elapsed months from
    birth to interview. It never depends on whether the child died.
    """
    birth = pd.to_numeric(birth_cmc, errors="coerce")
    interview = pd.to_numeric(interview_cmc, errors="coerce")
    valid_cmc = (
        birth.gt(0)
        & interview.gt(0)
        & birth.mod(1).eq(0)
        & interview.mod(1).eq(0)
    )
    followup = (interview - birth).where(valid_cmc)
    followup = followup.where(followup >= 0)

    eligible = pd.Series(np.nan, index=followup.index, dtype="float64")
    eligible.loc[followup.notna()] = (followup.loc[followup.notna()] >= 60).astype(float)

    alive_bool = alive.astype("boolean")
    death_age = pd.to_numeric(age_at_death_months, errors="coerce").where(
        lambda values: values >= 0
    )
    alive_observed = alive_bool.eq(True) & followup.notna()
    dead_observed = (
        alive_bool.eq(False)
        & death_age.notna()
        & followup.notna()
        & death_age.le(followup)
    )
    known_outcome = alive_observed | dead_observed
    event = dead_observed & death_age.lt(60)

    u5_event = pd.Series(np.nan, index=followup.index, dtype="float64")
    u5_event.loc[known_outcome.fillna(False)] = event.loc[
        known_outcome.fillna(False)
    ].astype(float)
    u5_censored = pd.Series(np.nan, index=followup.index, dtype="float64")
    u5_censored.loc[known_outcome.fillna(False)] = 1.0 - u5_event.loc[
        known_outcome.fillna(False)
    ]

    observed_time = pd.Series(np.nan, index=followup.index, dtype="float64")
    alive_known = alive_observed
    observed_time.loc[alive_known.fillna(False)] = np.minimum(
        followup.loc[alive_known.fillna(False)], 60.0
    )
    dead_known = dead_observed
    observed_time.loc[dead_known.fillna(False)] = np.minimum(
        death_age.loc[dead_known.fillna(False)], 60.0
    )

    complete = pd.Series(np.nan, index=followup.index, dtype="float64")
    complete_mask = eligible.eq(1) & known_outcome.fillna(False)
    complete.loc[complete_mask] = event.loc[complete_mask].astype(float)

    return pd.DataFrame(
        {
            "followup_months": followup.astype("float64"),
            "event_time_months": observed_time,
            "u5_event": u5_event,
            "u5_censored": u5_censored,
            "u5_binary_eligible": eligible,
            "died_under5_complete": complete,
        },
        index=followup.index,
    )
