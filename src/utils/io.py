"""
io.py — Shared I/O and statistical utilities.

All data loading goes through this module so that path management
and memory-safe patterns are centralized in one place.

The canonical analysis file is global_analysis.parquet, which contains
ALL years with an `in_primary_scope` flag (D006). This loader applies
the scope filter at read time so downstream code never has to think
about it.
"""
from __future__ import annotations
import gc
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# Add project root
sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent.parent)
)
from config.paths import COUNTRY_COVARIATES, GLOBAL_PARQUET


def load_primary(
    columns: Optional[list[str]] = None,
    scope: str = "primary",
) -> "pd.DataFrame":
    """
    Load the canonical analysis dataset with scope filtering.

    Parameters
    ----------
    columns : list[str], optional
        Subset of columns to load. If None, loads all columns.
        Always prefer specifying columns to reduce memory usage.
    scope : str
        "primary" (2000-2024, default), "supplement" (pre-2000),
        or "all" (no filter).

    Returns
    -------
    pd.DataFrame
        The analysis dataset filtered to the requested scope.
    """
    import pandas as pd

    # Ensure we can filter even if caller didn't request the flag column
    cols_to_read = columns
    need_filter = scope in ("primary", "supplement")
    if need_filter and cols_to_read is not None:
        if "in_primary_scope" not in cols_to_read:
            cols_to_read = list(cols_to_read) + ["in_primary_scope"]

    df = pq.read_table(
        str(GLOBAL_PARQUET), columns=cols_to_read
    ).to_pandas()

    # Apply scope filter (D006)
    if scope == "primary":
        df = df[df["in_primary_scope"] == 1]
    elif scope == "supplement":
        df = df[df["in_primary_scope"] == 0]

    # Drop the helper column if the caller didn't ask for it
    if (
        need_filter
        and columns is not None
        and "in_primary_scope" not in columns
        and "in_primary_scope" in df.columns
    ):
        df = df.drop(columns=["in_primary_scope"])

    return df.reset_index(drop=True)


def load_systematic_sample(
    columns: list[str],
    step: int = 20,
    scope: str = "primary",
) -> "pd.DataFrame":
    """
    Load a systematic sample (every step-th row) from the canonical parquet.

    Uses PyArrow batched streaming — never loads the full file into pandas.
    Implements decision D015 (5% systematic sample for inference).

    Parameters
    ----------
    columns : list[str]
        Columns to load. Must include "in_primary_scope" and "died_under5"
        for scope filtering and NA-dropping.
    step : int
        Sampling interval. step=20 gives a 5% systematic sample (~512K rows).
    scope : str
        "primary" (default), "supplement", or "all".

    Returns
    -------
    pd.DataFrame
        The systematic sample with scope filter applied, died_under5 NaN
        rows dropped, and dtype cast to int.
    """
    # Full-data override (decision D015 revisited): set INFERENCE_FULL_SAMPLE=1
    # to fit on every row (step=1) instead of the 5% systematic sample.
    import os as _os
    if _os.environ.get("INFERENCE_FULL_SAMPLE") == "1":
        step = 1
    pf = pq.ParquetFile(str(GLOBAL_PARQUET))

    frames = []
    row_offset = 0
    for batch in pf.iter_batches(batch_size=500_000, columns=columns):
        batch_df = batch.to_pandas()
        batch_len = len(batch_df)
        # O(1) per-row selection via modulo arithmetic (avoids building an index set)
        mask = np.arange(batch_len) + row_offset
        mask = mask % step == 0
        selected = batch_df.loc[mask]
        if len(selected) > 0:
            frames.append(selected)
        row_offset += batch_len
        del batch_df, batch

    df = pd.concat(frames, ignore_index=True)
    del frames
    gc.collect()

    # Scope filter
    if "in_primary_scope" in df.columns:
        if scope == "primary":
            df = df[df["in_primary_scope"] == 1]
        elif scope == "supplement":
            df = df[df["in_primary_scope"] == 0]
        df = df.drop(columns=["in_primary_scope"], errors="ignore")

    # Drop rows with missing outcome
    df = df.dropna(subset=["died_under5"]).copy()
    df["died_under5"] = df["died_under5"].astype(int)

    return df


def load_country_covariates() -> "pd.DataFrame":
    """
    Load country-level WDI covariates (output of s03d_country_covariates.py).

    Returns
    -------
    pd.DataFrame
        Columns: country_code, country_name, survey_year,
                 gdp_per_capita_ppp, health_exp_pct_gdp, physicians_per_1000
    """
    import pandas as pd

    if not COUNTRY_COVARIATES.exists():
        raise FileNotFoundError(
            f"Country covariates not found at {COUNTRY_COVARIATES}. "
            "Run src/data/s03d_country_covariates.py first."
        )
    return pq.read_table(str(COUNTRY_COVARIATES)).to_pandas()


def weighted_mean(
    x: "np.ndarray", w: "np.ndarray"
) -> float:
    """
    Compute weighted mean, handling NaN values.

    Parameters
    ----------
    x : array-like
        Values.
    w : array-like
        Weights (DHS sample_weight = v005/1e6).

    Returns
    -------
    float
        Weighted mean, or NaN if no valid observations.
    """
    mask = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if mask.sum() == 0:
        return np.nan
    return np.average(x[mask], weights=w[mask])


def weighted_proportion(
    x: "np.ndarray", w: "np.ndarray"
) -> float:
    """
    Compute weighted proportion (for binary 0/1 variables).

    Parameters
    ----------
    x : array-like
        Binary (0/1) values.
    w : array-like
        Weights.

    Returns
    -------
    float
        Weighted proportion, or NaN if no valid observations.
    """
    return weighted_mean(x, w)


def weighted_rate_per1000(
    x: "np.ndarray", w: "np.ndarray"
) -> float:
    """
    Compute weighted rate per 1,000 (e.g., U5MR).

    Parameters
    ----------
    x : array-like
        Binary outcome (0/1).
    w : array-like
        Weights.

    Returns
    -------
    float
        Rate per 1,000.
    """
    p = weighted_mean(x, w)
    return p * 1000 if np.isfinite(p) else np.nan


def weighted_mean_design(
    x: "np.ndarray",
    w: "np.ndarray",
    psu: "np.ndarray | None" = None,
    strata: "np.ndarray | None" = None,
) -> dict:
    """
    Design-based weighted mean with Taylor-linearized SE.

    Uses the influence-function (linearization) variance for a ratio
    estimator under stratified cluster sampling:

        Y_hat = sum(w*y) / sum(w)
        z_i   = w_i * (y_i - Y_hat) / sum(w)
        s_h_c = sum_{i in cluster c, stratum h} z_i
        Var(Y_hat) = sum_h n_h/(n_h-1) * sum_c (s_h_c - mean_h)^2

    If `psu` is not given, falls back to with-replacement SRS variance
    (still using the weights). `strata` is optional; without it the
    sample is treated as a single stratum.

    Returns dict with keys: estimate, se, ci_lo, ci_hi, n.
    """
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    mask = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if mask.sum() == 0:
        return {"estimate": np.nan, "se": np.nan,
                "ci_lo": np.nan, "ci_hi": np.nan, "n": 0}
    x = x[mask]
    w = w[mask]
    sw = w.sum()
    yhat = float(np.sum(w * x) / sw)
    z = w * (x - yhat) / sw  # influence values

    if psu is None:
        # SRS-with-replacement linearization: Var = sum(z^2)
        var = float(np.sum(z * z))
    else:
        psu = np.asarray(psu)[mask]
        if strata is None:
            strata = np.zeros_like(psu, dtype=object)
        else:
            strata = np.asarray(strata)[mask]
        var = 0.0
        # Aggregate z by (stratum, psu); within each stratum compute
        # the between-cluster variance with finite-sample correction.
        df = pd.DataFrame({"z": z, "psu": psu, "stratum": strata})
        cluster_sums = df.groupby(["stratum", "psu"], sort=False)["z"].sum()
        for stratum, s_c in cluster_sums.groupby(level="stratum"):
            n_c = len(s_c)
            if n_c < 2:
                continue
            mean_c = s_c.mean()
            var += (n_c / (n_c - 1.0)) * float(((s_c - mean_c) ** 2).sum())

    se = float(np.sqrt(max(var, 0.0)))
    ci_lo = yhat - 1.96 * se
    ci_hi = yhat + 1.96 * se
    return {"estimate": yhat, "se": se,
            "ci_lo": ci_lo, "ci_hi": ci_hi, "n": int(mask.sum())}
