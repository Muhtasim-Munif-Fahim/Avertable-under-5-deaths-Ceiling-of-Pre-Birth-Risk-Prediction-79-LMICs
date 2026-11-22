"""
s03_mics_pipeline.py - MICS Data Integration Pipeline
======================================================

Reads the MICS survey catalog, harmonizes each eligible survey via
s03_mics_harmonize.harmonize_mics_survey(), and appends the resulting
rows to the canonical global_analysis.parquet.

Geographic-gaps-only rule (enforced):
    No country_iso3 already present in the existing DHS parquet may be
    added from MICS. MICS fills the gap set (countries outside the 68
    DHS LMICs), not the DHS coverage set.

Design notes
------------
- Never loads full parquet into pandas (RAM limit: ~9 GB free).
- Writes in streaming batches via ParquetWriter (O(1) memory).
- Atomic rename: the canonical parquet is only replaced after a
  successful full write to a temp path.
- One bad survey must not abort the whole pipeline (try/except per row).
- Catalog missing -> clear error + sys.exit(1).
- RNG_SEED imported from config.paths (never hardcoded).

Output
------
- Expanded global_analysis.parquet (canonical; existing path overwritten
  atomically)
- output/mics_integration_manifest.csv (run log with per-survey stats)
"""

import argparse
import gc
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Project root on sys.path so config.* and src.* are importable
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from config.paths import (
    GLOBAL_PARQUET,
    LOG_DIR,
    OUTPUT_DIR,
    RAW_MICS_DIR,
    RNG_SEED,
)
from src.data.under5_outcomes import EXACT_FOLLOWUP_COLUMNS
from src.data.country_codes import normalize_country_iso3

# The harmonizer written by the parallel agent.
# Import is deferred to run time so the pipeline fails with a clear
# message if the module is not yet available.
def _import_harmonizer():
    try:
        from src.data.s03_mics_harmonize import harmonize_mics_survey
        return harmonize_mics_survey
    except ImportError as exc:
        _get_logger().error(
            "Cannot import harmonize_mics_survey from "
            "src.data.s03_mics_harmonize: %s", exc
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Required columns with expected dtypes (schema validation)
# Any column listed here must be present in the MICS batch before append.
# ---------------------------------------------------------------------------

# (column_name, accepted_pandas_dtype_kinds)
# dtype kinds: 'i'=signed int, 'u'=unsigned int, 'f'=float,
#              'b'=bool, 'O'=object/str, 'U'=unicode
_REQUIRED_COLS: list[tuple[str, tuple[str, ...]]] = [
    ("died_under5",              ("i", "u", "b", "f")),
    ("birth_cmc",                ("i", "u", "f")),
    ("interview_cmc",            ("i", "u", "f")),
    ("followup_months",          ("i", "u", "f")),
    ("age_at_death_months",      ("i", "u", "f")),
    ("event_time_months",        ("i", "u", "f")),
    ("u5_event",                 ("i", "u", "b", "f")),
    ("u5_censored",              ("i", "u", "b", "f")),
    ("u5_binary_eligible",       ("i", "u", "b", "f")),
    ("died_under5_complete",     ("i", "u", "b", "f")),
    ("child_male",               ("i", "u", "f")),
    ("birth_order",              ("i", "u", "f")),
    ("birth_interval_months",    ("i", "u", "f")),
    ("maternal_age_at_birth",    ("i", "u", "f")),
    ("edu_level",                ("i", "u", "f")),
    ("wealth_quintile",          ("i", "u", "f")),
    ("urban",                    ("i", "u", "f")),
    ("currently_married",        ("i", "u", "f")),
    ("sample_weight",            ("f",)),
    ("country_iso3",             ("O",)),
    ("survey_year",              ("i", "u", "f")),
    ("in_primary_scope",         ("i", "u", "b", "f")),
    ("data_source",              ("O",)),
    ("who_region",               ("O",)),
]

# Canonical column set (from existing parquet).  MICS columns outside
# this set are dropped; columns in this set absent from MICS are NaN-filled.
# Populated at runtime from the existing parquet schema.
_CANONICAL_COLUMNS: list[str] = []


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_logger: Optional[logging.Logger] = None


def _get_logger() -> logging.Logger:
    global _logger
    if _logger is not None:
        return _logger

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "s03_mics_pipeline.log"

    logger = logging.getLogger("mics_pipeline")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(str(log_path), encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    import io as _io
    safe_stdout = _io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )
    ch = logging.StreamHandler(safe_stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    _logger = logger
    return logger


# ---------------------------------------------------------------------------
# Catalog loading
# ---------------------------------------------------------------------------

def load_catalog(catalog_path: Path) -> pd.DataFrame:
    """
    Read mics_survey_catalog.csv and return rows where use_for_expansion == 1.

    Exits with code 1 if the file does not exist.
    """
    logger = _get_logger()
    if not catalog_path.exists():
        logger.error(
            "Catalog not found: %s  -- "
            "Run the catalog-building step first and try again.",
            catalog_path,
        )
        sys.exit(1)

    cat = pd.read_csv(catalog_path, dtype=str)
    logger.info("Catalog loaded: %d total rows from %s", len(cat), catalog_path)

    # Normalise use_for_expansion to int
    if "use_for_expansion" not in cat.columns:
        logger.error(
            "Catalog is missing required column 'use_for_expansion'."
        )
        sys.exit(1)

    cat["use_for_expansion"] = pd.to_numeric(
        cat["use_for_expansion"], errors="coerce"
    ).fillna(0).astype(int)

    eligible = cat[cat["use_for_expansion"] == 1].copy()
    logger.info(
        "%d surveys marked use_for_expansion=1 (out of %d total)",
        len(eligible), len(cat),
    )
    return eligible


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def validate_mics_schema(df: pd.DataFrame, country_iso3: str) -> list[str]:
    """
    Check that all required columns are present and have compatible dtypes.

    Returns a list of validation error strings (empty = pass).
    """
    errors: list[str] = []
    for col, accepted_kinds in _REQUIRED_COLS:
        if col not in df.columns:
            errors.append(f"[{country_iso3}] Missing required column: {col}")
            continue
        kind = df[col].dtype.kind
        if kind not in accepted_kinds:
            errors.append(
                f"[{country_iso3}] Column '{col}' has dtype kind '{kind}', "
                f"expected one of {accepted_kinds}"
            )
    return errors


def resolve_mics_zip_path(
    catalog_row,
    *,
    raw_mics_dir: Path = RAW_MICS_DIR,
) -> Path:
    """Resolve a MICS archive, tolerating stale absolute catalog paths."""
    catalog_value = str(catalog_row.get("zip_path", "")).strip()
    if catalog_value.lower() in {"", "nan", "none"}:
        catalog_value = ""
    catalog_path = Path(catalog_value) if catalog_value else None
    if catalog_path is not None and catalog_path.is_file():
        return catalog_path

    filename = str(catalog_row.get("zip_filename", "")).strip()
    if filename.lower() in {"", "nan", "none"}:
        raise ValueError("MICS catalog row has no usable zip filename")
    return Path(raw_mics_dir) / filename


# ---------------------------------------------------------------------------
# Column alignment: conform MICS DataFrame to canonical parquet schema
# ---------------------------------------------------------------------------

def _get_canonical_columns(parquet_path: Path) -> list[str]:
    """Read column names from the existing parquet schema (no data load)."""
    schema = pq.read_schema(str(parquet_path))
    return list(schema.names)


def align_to_canonical(df: pd.DataFrame, canonical_cols: list[str]) -> pd.DataFrame:
    """
    Align a MICS DataFrame to the canonical column set.

    - Columns in canonical_cols but absent in df -> added as NaN.
    - Columns in df but absent in canonical_cols -> dropped silently.
    - Column order matches canonical_cols exactly.
    """
    missing_exact = [
        column for column in EXACT_FOLLOWUP_COLUMNS if column not in canonical_cols
    ]
    if missing_exact:
        raise ValueError(
            "canonical schema lacks exact follow-up fields: "
            f"{missing_exact}. Rebuild DHS canonical output before MICS append."
        )

    for col in canonical_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Drop any MICS-only columns not in canonical set
    extra = [c for c in df.columns if c not in canonical_cols]
    if extra:
        _get_logger().info(
            "  Dropping %d MICS-only columns not in canonical schema: %s",
            len(extra), extra,
        )
    return df[canonical_cols].copy()


# ---------------------------------------------------------------------------
# Deduplication guard: enforce geographic-gaps-only rule
# ---------------------------------------------------------------------------

def get_existing_countries(parquet_path: Path) -> set[str]:
    """
    Return the set of country_iso3 values already in the existing parquet.

    Uses pyarrow column read (no full pandas load).
    """
    # The DHS parquet uses 'country_code' (2-letter), not 'country_iso3'.
    # We read whichever name is present in the schema.
    schema = pq.read_schema(str(parquet_path))
    col_names = schema.names

    # Prefer country_iso3 if present (future-proofed), fall back to country_code
    if "country_iso3" in col_names:
        tbl = pq.read_table(str(parquet_path), columns=["country_iso3"])
        col = tbl.column("country_iso3")
    elif "country_code" in col_names:
        tbl = pq.read_table(str(parquet_path), columns=["country_code"])
        col = tbl.column("country_code")
    else:
        _get_logger().warning(
            "Existing parquet has no country_iso3 / country_code column. "
            "Cannot enforce geographic-gaps-only rule."
        )
        return set()

    import pyarrow.compute as pc
    unique = pc.unique(col).to_pylist()
    normalized = {normalize_country_iso3(value) for value in unique}
    return {value for value in normalized if value is not None}


# ---------------------------------------------------------------------------
# PyArrow schema conformance helpers
# ---------------------------------------------------------------------------

def _normalize_pa_schema(schema: pa.Schema) -> pa.Schema:
    """
    Normalize PyArrow schema for cross-file unification.

    - dictionary<string> -> string
    - integer types -> float64 (avoids int64 vs double conflicts at union)
    - null type stays null (pa.unify_schemas promotes null -> other)
    """
    fields = []
    for field in schema:
        if pa.types.is_dictionary(field.type):
            fields.append(pa.field(field.name, pa.string()))
        elif pa.types.is_integer(field.type):
            fields.append(pa.field(field.name, pa.float64()))
        else:
            fields.append(field)
    return pa.schema(fields)


def _conform_pa_table(t: pa.Table, target: pa.Schema) -> pa.Table:
    """
    Conform a PyArrow Table to target schema.

    - Decode dictionary columns to plain string.
    - Cast null arrays to target type.
    - Cast mismatched types; on failure fill with nulls.
    - Add all-null columns for fields missing from t.
    """
    cols = {}
    for field in target:
        if field.name in t.schema.names:
            col = t.column(field.name)
            # Unwrap dictionary encoding
            if pa.types.is_dictionary(col.type):
                col = col.combine_chunks().dictionary_decode()
            # Handle null-typed arrays
            if col.type == pa.null():
                col = pa.nulls(len(t), type=field.type)
            elif col.type != field.type:
                try:
                    col = col.cast(field.type)
                except Exception:
                    col = pa.nulls(len(t), type=field.type)
        else:
            col = pa.nulls(len(t), type=field.type)
        cols[field.name] = col
    return pa.table(cols, schema=target)


# ---------------------------------------------------------------------------
# Core pipeline function
# ---------------------------------------------------------------------------

def run_mics_pipeline(
    catalog_path: str | Path,
    output_parquet_path: str | Path,
    dry_run: bool = False,
    only_country: Optional[str] = None,
) -> None:
    """
    Read MICS catalog, harmonize each eligible survey, append to canonical parquet.

    Parameters
    ----------
    catalog_path : str or Path
        Path to mics_survey_catalog.csv.
    output_parquet_path : str or Path
        Path to the canonical global_analysis.parquet (will be updated
        atomically on success).
    dry_run : bool
        If True, print what would be processed without writing anything.
    only_country : str, optional
        If given (ISO3), process only this one country. For testing.
    """
    logger = _get_logger()
    catalog_path = Path(catalog_path)
    output_parquet_path = Path(output_parquet_path)

    logger.info("=" * 60)
    logger.info("MICS Integration Pipeline")
    logger.info("  catalog : %s", catalog_path)
    logger.info("  output  : %s", output_parquet_path)
    logger.info("  dry_run : %s", dry_run)
    logger.info("  RNG_SEED: %d", RNG_SEED)
    logger.info("=" * 60)
    t0 = time.time()

    # ------------------------------------------------------------------
    # Step 1: Load catalog
    # ------------------------------------------------------------------
    eligible = load_catalog(catalog_path)

    if only_country is not None:
        iso3_upper = only_country.upper()
        eligible = eligible[eligible["country_iso3"].str.upper() == iso3_upper]
        if eligible.empty:
            logger.error(
                "--country %s not found in catalog or not eligible "
                "(use_for_expansion=1). Check the catalog.",
                only_country,
            )
            sys.exit(1)
        logger.info("Filtered to single country: %s (%d surveys)", iso3_upper, len(eligible))

    if eligible.empty:
        logger.warning("No eligible surveys found. Nothing to do.")
        return

    if dry_run:
        logger.info("DRY RUN — would process %d surveys:", len(eligible))
        for _, row in eligible.iterrows():
            logger.info(
                "  %s  year=%s  round=%s  zip=%s",
                row.get("country_iso3", "?"),
                row.get("survey_year", "?"),
                row.get("mics_round", "?"),
                row.get("zip_filename", "?"),
            )
        logger.info("Dry run complete. No files written.")
        return

    # ------------------------------------------------------------------
    # Step 2: Read existing parquet schema and countries (no data load)
    # ------------------------------------------------------------------
    if not output_parquet_path.exists():
        logger.error(
            "Canonical parquet not found: %s  "
            "Run s03_harmonize.py first.",
            output_parquet_path,
        )
        sys.exit(1)

    existing_countries = get_existing_countries(output_parquet_path)
    logger.info(
        "Existing parquet has %d distinct country codes.",
        len(existing_countries),
    )

    canonical_cols = _get_canonical_columns(output_parquet_path)
    logger.info("Canonical schema: %d columns.", len(canonical_cols))

    # ------------------------------------------------------------------
    # Step 3: Harmonize each eligible survey
    # ------------------------------------------------------------------
    harmonize_mics_survey = _import_harmonizer()

    manifest_rows: list[dict] = []
    mics_dfs: list[pd.DataFrame] = []   # one aligned DataFrame per survey

    for idx, row in eligible.iterrows():
        iso3      = str(row.get("country_iso3", "")).strip().upper()
        country   = str(row.get("country_name", "")).strip() or iso3
        year      = row.get("survey_year", None)
        mics_rnd  = row.get("mics_round", None)
        zip_file  = str(row.get("zip_filename", "")).strip()
        who_rgn   = str(row.get("who_region", "")).strip()

        manifest_entry: dict = {
            "country_iso3":    iso3,
            "survey_year":     year,
            "mics_round":      mics_rnd,
            "zip_filename":    zip_file,
            "n_births":        0,
            "n_recorded_u5_events": 0,
            "recorded_u5_event_proportion": np.nan,
            "n_complete_binary": 0,
            "n_complete_u5_deaths": 0,
            "complete_u5_proportion": np.nan,
            "status":          "pending",
        }

        # -- Geographic-gaps-only guard --
        if iso3 in existing_countries:
            logger.warning(
                "  SKIP [%s year=%s]: already present in DHS parquet "
                "(geographic-gaps-only rule).",
                iso3, year,
            )
            manifest_entry["status"] = "skipped_dhs_overlap"
            manifest_rows.append(manifest_entry)
            continue

        logger.info(
            "  [%s year=%s round=%s] Harmonizing...", iso3, year, mics_rnd
        )

        try:
            zip_full = str(resolve_mics_zip_path(row))
            df_survey = harmonize_mics_survey(
                zip_path=zip_full,
                country_iso3=iso3,
                country_name=country,
                survey_year=int(year) if year is not None else None,
                who_region=who_rgn,
            )
        except Exception as exc:
            logger.error(
                "  [%s year=%s] harmonize_mics_survey raised: %s",
                iso3, year, exc, exc_info=True,
            )
            manifest_entry["status"] = "error"
            manifest_rows.append(manifest_entry)
            continue

        # Empty DataFrame means bh.sav was missing or survey yielded no rows
        if df_survey is None or df_survey.empty:
            logger.warning(
                "  [%s year=%s] Returned empty DataFrame. Skipping.",
                iso3, year,
            )
            manifest_entry["status"] = "skipped_empty"
            manifest_rows.append(manifest_entry)
            continue

        # -- Schema validation --
        schema_errors = validate_mics_schema(df_survey, iso3)
        if schema_errors:
            for err in schema_errors:
                logger.error("  SCHEMA ERROR: %s", err)
            manifest_entry["status"] = "error_schema"
            manifest_rows.append(manifest_entry)
            continue

        # -- Align columns to canonical schema --
        df_aligned = align_to_canonical(df_survey, canonical_cols)

        n_births   = len(df_aligned)
        recorded_col = df_aligned["died_under5"]
        n_recorded = int(recorded_col.sum(skipna=True))
        n_recorded_valid = int(recorded_col.notna().sum())
        recorded_prop = (
            n_recorded / n_recorded_valid if n_recorded_valid > 0 else np.nan
        )
        complete_col = df_aligned["died_under5_complete"]
        n_complete_deaths = int(complete_col.sum(skipna=True))
        n_complete = int(complete_col.notna().sum())
        complete_prop = (
            n_complete_deaths / n_complete if n_complete > 0 else np.nan
        )

        manifest_entry.update({
            "n_births":       n_births,
            "n_recorded_u5_events": n_recorded,
            "recorded_u5_event_proportion": (
                round(recorded_prop, 6) if np.isfinite(recorded_prop) else np.nan
            ),
            "n_complete_binary": n_complete,
            "n_complete_u5_deaths": n_complete_deaths,
            "complete_u5_proportion": (
                round(complete_prop, 6) if np.isfinite(complete_prop) else np.nan
            ),
            "status":         "success",
        })
        manifest_rows.append(manifest_entry)

        mics_dfs.append(df_aligned)
        logger.info(
            "  [%s year=%s] %d births, exact-complete proportion=%.4f",
            iso3, year, n_births,
            complete_prop if np.isfinite(complete_prop) else float("nan"),
        )

        del df_survey, df_aligned
        gc.collect()

    # ------------------------------------------------------------------
    # Step 4: Bail early if nothing to append
    # ------------------------------------------------------------------
    n_surveys_ok = len(mics_dfs)
    if n_surveys_ok == 0:
        logger.warning(
            "No MICS surveys produced usable data. Parquet not modified."
        )
        _write_manifest(manifest_rows)
        return

    # ------------------------------------------------------------------
    # Step 5: Validate cross-survey stack (type compatibility check)
    # ------------------------------------------------------------------
    logger.info(
        "Stacking %d MICS DataFrames for type-compatibility check...",
        n_surveys_ok,
    )
    # Lightweight check: collect dtypes across all survey dfs
    for i, df in enumerate(mics_dfs):
        bad_cols = [
            c for c in df.columns
            if df[c].dtype.kind not in ("i", "u", "f", "b", "O", "U", "c")
        ]
        if bad_cols:
            logger.warning(
                "  Survey %d has unusual dtypes in: %s", i, bad_cols
            )

    # ------------------------------------------------------------------
    # Step 6: Stream-append to canonical parquet (atomic rename pattern)
    # ------------------------------------------------------------------
    logger.info("Building unified PyArrow schema for append pass...")

    existing_schema = _normalize_pa_schema(pq.read_schema(str(output_parquet_path)))

    # Build PyArrow tables for each MICS survey (still in memory as pandas)
    mics_pa_tables: list[pa.Table] = []
    for df in mics_dfs:
        try:
            t = pa.Table.from_pandas(df, preserve_index=False)
            t = _conform_pa_table(_normalize_pa_schema_table(t), existing_schema)
            mics_pa_tables.append(t)
        except Exception as exc:
            logger.error("  Error converting survey to PyArrow table: %s", exc)
        finally:
            del df
            gc.collect()

    del mics_dfs

    if not mics_pa_tables:
        logger.error("All MICS tables failed PyArrow conversion. Aborting.")
        _write_manifest(manifest_rows)
        return

    total_new_rows = sum(len(t) for t in mics_pa_tables)
    logger.info(
        "MICS tables ready: %d surveys, %d total new rows.",
        len(mics_pa_tables), total_new_rows,
    )

    # Write to a temp file then atomically rename
    temp_path = output_parquet_path.with_suffix(".tmp_mics_append.parquet")

    logger.info(
        "Streaming existing parquet + MICS rows -> %s ...", temp_path
    )

    rows_copied = 0
    try:
        with pq.ParquetWriter(
            str(temp_path), schema=existing_schema, compression="snappy"
        ) as writer:
            # Copy all existing row groups (O(1) memory per batch).
            # Use try/finally so the source ParquetFile is closed before
            # os.replace -- on Windows, an open handle blocks the rename.
            pf = pq.ParquetFile(str(output_parquet_path))
            try:
                for batch in pf.iter_batches(batch_size=100_000):
                    t_batch = pa.Table.from_batches([batch])
                    t_batch = _conform_pa_table(
                        _normalize_pa_schema_table(t_batch), existing_schema
                    )
                    writer.write_table(t_batch)
                    rows_copied += len(t_batch)
                    del t_batch
                    gc.collect()
            finally:
                # PyArrow's ParquetFile in older builds has no .close(); ensure
                # the underlying file handle is released by dropping the ref.
                close_fn = getattr(pf, "close", None)
                if callable(close_fn):
                    close_fn()
                del pf
                gc.collect()

            logger.info("  Copied %d existing rows.", rows_copied)

            # Append MICS tables one survey at a time
            rows_added = 0
            for t in mics_pa_tables:
                writer.write_table(t)
                rows_added += len(t)
                del t
                gc.collect()

            logger.info("  Appended %d MICS rows.", rows_added)

    except Exception as exc:
        logger.error("Write failed: %s", exc, exc_info=True)
        # Clean up partial temp file
        if temp_path.exists():
            temp_path.unlink()
        _write_manifest(manifest_rows)
        raise

    # Verify row count via metadata before promoting
    meta = pq.read_metadata(str(temp_path))
    expected = rows_copied + rows_added
    if meta.num_rows != expected:
        logger.error(
            "Row-count mismatch in temp file: on-disk=%d, expected=%d. "
            "Aborting rename — canonical parquet unchanged.",
            meta.num_rows, expected,
        )
        temp_path.unlink()
        _write_manifest(manifest_rows)
        return

    # Atomic rename
    os.replace(str(temp_path), str(output_parquet_path))
    logger.info(
        "Atomic rename complete. Canonical parquet updated: %s",
        output_parquet_path,
    )

    # ------------------------------------------------------------------
    # Step 7: Write manifest
    # ------------------------------------------------------------------
    _write_manifest(manifest_rows)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    n_success  = sum(1 for r in manifest_rows if r["status"] == "success")
    n_skipped  = sum(1 for r in manifest_rows if r["status"].startswith("skipped"))
    n_errors   = sum(1 for r in manifest_rows if r["status"].startswith("error"))

    size_mb = output_parquet_path.stat().st_size / (1024 ** 2)
    elapsed = time.time() - t0

    logger.info("")
    logger.info("=" * 60)
    logger.info("MICS Integration Summary")
    logger.info("  Countries processed (success) : %d", n_success)
    logger.info("  Skipped (overlap/empty)       : %d", n_skipped)
    logger.info("  Errors                        : %d", n_errors)
    logger.info("  New births added              : %d", rows_added)
    logger.info("  Total rows in parquet now     : %d", meta.num_rows)
    logger.info("  New parquet size              : %.1f MB", size_mb)
    logger.info("  Elapsed                       : %.0f s", elapsed)
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# PyArrow helper: normalize schema on an existing pa.Table
# (mirrors _normalize_pa_schema for Table objects)
# ---------------------------------------------------------------------------

def _normalize_pa_schema_table(t: pa.Table) -> pa.Table:
    """
    Return t with its schema normalized (dict -> string, int -> float64).

    Needed before _conform_pa_table so the source and target schemas are
    in the same normal form for type-matching.
    """
    new_schema = _normalize_pa_schema(t.schema)
    cols = {}
    for field in new_schema:
        col = t.column(field.name)
        if pa.types.is_dictionary(col.type):
            col = col.combine_chunks().dictionary_decode()
        if col.type != field.type:
            try:
                col = col.cast(field.type)
            except Exception:
                col = pa.nulls(len(t), type=field.type)
        cols[field.name] = col
    return pa.table(cols, schema=new_schema)


# ---------------------------------------------------------------------------
# Manifest writer
# ---------------------------------------------------------------------------

def _write_manifest(manifest_rows: list[dict]) -> None:
    """Write the per-survey run manifest to output/mics_integration_manifest.csv."""
    logger = _get_logger()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUTPUT_DIR / "mics_integration_manifest.csv"
    manifest_df = pd.DataFrame(manifest_rows, columns=[
        "country_iso3",
        "survey_year",
        "mics_round",
        "zip_filename",
        "n_births",
        "n_recorded_u5_events",
        "recorded_u5_event_proportion",
        "n_complete_binary",
        "n_complete_u5_deaths",
        "complete_u5_proportion",
        "status",
    ])
    manifest_df.to_csv(manifest_path, index=False)
    logger.info("Manifest written: %s (%d rows)", manifest_path, len(manifest_df))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "MICS integration pipeline: harmonize MICS surveys and append "
            "to the canonical global_analysis.parquet."
        )
    )
    parser.add_argument(
        "--catalog",
        default=str(_PROJECT_ROOT / "data" / "mics_survey_catalog.csv"),
        help="Path to mics_survey_catalog.csv "
             "(default: data/mics_survey_catalog.csv)",
    )
    parser.add_argument(
        "--output",
        default=str(GLOBAL_PARQUET),
        help="Path to canonical output parquet "
             "(default: data/02_harmonized/global_analysis.parquet)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be processed without writing anything.",
    )
    parser.add_argument(
        "--country",
        default=None,
        metavar="ISO3",
        help="Process only this ISO3 country code (for testing).",
    )
    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = _build_parser().parse_args()
    run_mics_pipeline(
        catalog_path=args.catalog,
        output_parquet_path=args.output,
        dry_run=args.dry_run,
        only_country=args.country,
    )
