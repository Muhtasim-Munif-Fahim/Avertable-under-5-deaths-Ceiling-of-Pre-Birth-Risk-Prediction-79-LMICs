"""Audit invariants — checks that pass on the canonical parquet."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.paths import GLOBAL_PARQUET


@pytest.fixture(scope="module")
def sample_df():
    """2% row sample for fast tests (every 50th row)."""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(str(GLOBAL_PARQUET))
    n = pf.metadata.num_rows
    idx = set(range(0, n, 50))
    frames = []
    offset = 0
    for batch in pf.iter_batches(batch_size=500_000,
                                  columns=["survey_year", "birth_year",
                                           "died_under5", "country_code",
                                           "in_primary_scope"]):
        df = batch.to_pandas()
        local = [i - offset for i in idx
                 if offset <= i < offset + len(df)]
        if local:
            frames.append(df.iloc[local])
        offset += len(df)
    return pd.concat(frames, ignore_index=True)


def test_in_primary_scope_range(sample_df):
    vals = sample_df["in_primary_scope"].dropna().unique()
    assert set(vals).issubset({0, 1}), f"Unexpected in_primary_scope values: {vals}"


def test_primary_scope_year_bounds(sample_df):
    primary = sample_df[sample_df["in_primary_scope"] == 1]
    assert primary["survey_year"].min() >= 2000
    assert primary["survey_year"].max() <= 2024


def test_died_under5_binary(sample_df):
    vals = sample_df["died_under5"].dropna().unique()
    assert set(vals).issubset({0, 1}), f"Unexpected died_under5 values: {vals}"


def test_birth_year_reasonable(sample_df):
    """Primary scope is survey-year based; birth histories can precede 1990."""
    primary = sample_df[sample_df["in_primary_scope"] == 1]
    byr = primary["birth_year"].dropna()
    assert (byr >= 1950).all(), "implausible birth_year < 1950 in primary scope"
    assert (byr <= 2025).all(), "birth_year > 2025 in primary scope"


def test_survey_year_ge_birth_year(sample_df):
    primary = sample_df[sample_df["in_primary_scope"] == 1].dropna(
        subset=["survey_year", "birth_year"])
    violation_frac = (primary["birth_year"] > primary["survey_year"] + 1).mean()
    # Allow up to 0.2% for known NP-1995/IA-2019 calendar artefacts
    assert violation_frac < 0.002, \
        f"birth_year > survey_year+1 in {violation_frac*100:.3f}% of rows (limit 0.2%)"


def test_country_code_not_null(sample_df):
    null_frac = sample_df["country_code"].isna().mean()
    assert null_frac == 0, f"{null_frac*100:.2f}% null country_code"
