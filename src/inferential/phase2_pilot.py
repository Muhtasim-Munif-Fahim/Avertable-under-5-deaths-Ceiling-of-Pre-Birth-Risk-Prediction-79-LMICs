"""Phase 2 diagnostic pilot for the exact under-five rebuild.

The public interface keeps data construction, model-state preparation,
comparison fitting, uncertainty, and stop-rule evaluation behind one pilot
seam. Generated individual-level pilot data remain under the ignored data
directory; only aggregate diagnostic outputs are publication candidates.
"""

from __future__ import annotations

import gc
import hashlib
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from config.paths import HARMONIZED_DIR, OUTPUT_DIR, PARQUET_DIR, RAW_MICS_DIR, RNG_SEED


PHASE2_OUTPUT_DIR = OUTPUT_DIR / "phase2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

PILOT_COLUMNS = [
    "country_code",
    "who_region",
    "survey_year",
    "birth_year",
    "_source_file",
    "_source_row",
    "data_source",
    "sample_weight",
    "psu",
    "stratum",
    "died_under5",
    "died_under5_legacy_frozen",
    "died_under5_complete",
    "birth_cmc",
    "interview_cmc",
    "followup_months",
    "age_at_death_months",
    "event_time_months",
    "u5_event",
    "u5_censored",
    "u5_binary_eligible",
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


@dataclass(frozen=True)
class PilotConfig:
    """Versioned configuration for one reproducible diagnostic run."""

    sample_step: int = 20
    bootstrap_reps: int = 200
    seed: int = RNG_SEED
    rebuild_data: bool = False
    data_path_override: Path | None = None
    manifest_path_override: Path | None = None

    @property
    def data_path(self) -> Path:
        if self.data_path_override is not None:
            return Path(self.data_path_override)
        return HARMONIZED_DIR / (
            f"global_analysis_v2_pilot_step{self.sample_step}.parquet"
        )

    @property
    def manifest_path(self) -> Path:
        if self.manifest_path_override is not None:
            return Path(self.manifest_path_override)
        return PHASE2_OUTPUT_DIR / f"phase2_pilot_manifest_step{self.sample_step}.csv"


@dataclass(frozen=True)
class PilotBuildResult:
    data_path: Path
    manifest_path: Path
    n_rows: int
    n_surveys: int


@dataclass(frozen=True)
class _DhsRawSample:
    """Materialized raw DHS rows plus their full-survey provenance."""

    path: Path
    source_rows: np.ndarray
    n_source_births: int
    sample_offset: int


def _logger() -> logging.Logger:
    logger = logging.getLogger("phase2_pilot")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def _select_pilot_columns(frame: pd.DataFrame, *, data_source: str) -> pd.DataFrame:
    selected = frame.copy()
    selected["data_source"] = data_source
    for column in PILOT_COLUMNS:
        if column not in selected.columns:
            selected[column] = np.nan
    return selected[PILOT_COLUMNS].copy()


def _manifest_row(
    sampled: pd.DataFrame,
    *,
    data_source: str,
    survey_id: str,
    sample_step: int,
    sample_offset: int,
    n_source_births: int,
) -> dict:
    return {
        "data_source": data_source,
        "survey_id": survey_id,
        "country_code": str(sampled["country_code"].iloc[0]),
        "survey_year": int(float(sampled["survey_year"].iloc[0])),
        "sample_step": int(sample_step),
        "sample_offset": int(sample_offset),
        "n_source_births": int(n_source_births),
        "n_pilot_births": int(len(sampled)),
        "n_exact_binary": int(sampled["died_under5_complete"].notna().sum()),
        "n_exact_deaths": int(sampled["died_under5_complete"].sum(skipna=True)),
        "n_survival_known": int(sampled["u5_event"].notna().sum()),
    }


def _sample_offset(survey_id: str, step: int) -> int:
    if step < 1:
        raise ValueError("step must be a positive integer")
    return sum(str(survey_id).encode("utf-8")) % step


def _stream_dhs_raw_sample(
    source_path: Path,
    sampled_path: Path,
    *,
    survey_id: str,
    step: int,
    batch_size: int = 250_000,
) -> _DhsRawSample:
    """Stream only harmonizer inputs into a small per-survey raw sample.

    The returned positions are zero-based row numbers in the full converted
    survey. The temporary parquet keeps the original filename so the existing
    DHS harmonizer preserves its source-file contract.
    """
    from src.data.s03_harmonize import (
        _CRITICAL_FOLLOWUP_RAW_COLS,
        _RAW_COLS_NEEDED,
    )

    parquet = pq.ParquetFile(str(source_path))
    source_names = list(parquet.schema_arrow.names)
    missing = sorted(_CRITICAL_FOLLOWUP_RAW_COLS - set(source_names))
    if missing:
        raise ValueError(
            f"{source_path.name} lacks exact-follow-up fields {missing}"
        )
    columns = [name for name in source_names if name in _RAW_COLS_NEEDED]
    offset = _sample_offset(survey_id, step)
    n_source = int(parquet.metadata.num_rows)
    sampled_path.parent.mkdir(parents=True, exist_ok=True)
    sampled_path.unlink(missing_ok=True)

    writer: pq.ParquetWriter | None = None
    source_rows: list[np.ndarray] = []
    row_start = 0
    failed = False
    try:
        for batch in parquet.iter_batches(
            batch_size=batch_size,
            columns=columns,
            use_threads=True,
        ):
            positions = np.arange(
                row_start, row_start + len(batch), dtype=np.int64
            )
            keep = positions % step == offset
            if keep.any():
                selected = batch.filter(pa.array(keep))
                table = pa.Table.from_batches([selected])
                if writer is None:
                    writer = pq.ParquetWriter(
                        str(sampled_path), table.schema, compression="zstd"
                    )
                writer.write_table(table)
                source_rows.append(positions[keep])
            row_start += len(batch)
    except Exception:
        failed = True
        raise
    finally:
        if writer is not None:
            writer.close()
        if failed:
            sampled_path.unlink(missing_ok=True)

    if not source_rows:
        sampled_path.unlink(missing_ok=True)
        raise ValueError(
            f"sampling {source_path.name} at step={step} selected no rows"
        )
    if row_start != n_source:
        sampled_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"streamed {row_start} rows from {source_path.name}; expected {n_source}"
        )
    return _DhsRawSample(
        path=sampled_path,
        source_rows=np.concatenate(source_rows),
        n_source_births=n_source,
        sample_offset=offset,
    )


def _dhs_survey_year(path: Path) -> int | None:
    """Read only the survey-year column to gate scope before harmonization."""
    from src.data.s03_harmonize import normalize_survey_year

    schema = set(pq.read_schema(str(path)).names)
    if "v007" not in schema:
        return None
    year_column = pq.read_table(str(path), columns=["v007"]).column("v007")
    modes = pc.mode(year_column, n=1).to_pylist()
    if not modes:
        return None
    return normalize_survey_year(modes[0]["mode"], path.stem[:2].upper())


def _attach_frozen_legacy_outcome(
    pilot: pd.DataFrame,
    canonical_path: Path,
    *,
    expected_source_counts: dict[str, int],
) -> pd.DataFrame:
    """Join frozen legacy outcomes by exact source-file row position.

    The canonical parquet is streamed and never rewritten. A separate match
    mask distinguishes a legitimate missing legacy outcome from a missing key.
    """
    if not canonical_path.exists():
        raise FileNotFoundError(
            f"frozen canonical parquet not found: {canonical_path}"
        )
    required = {"_source_file", "died_under5"}
    schema = set(pq.read_schema(str(canonical_path)).names)
    missing = sorted(required - schema)
    if missing:
        raise ValueError(
            f"frozen canonical schema lacks required fields: {missing}"
        )

    wanted: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for source, group in pilot.groupby("_source_file", sort=False):
        rows = pd.to_numeric(group["_source_row"], errors="coerce").to_numpy()
        if (
            not np.isfinite(rows).all()
            or (rows < 0).any()
            or (rows % 1 != 0).any()
        ):
            raise ValueError(f"invalid source-row keys for {source}")
        order = np.argsort(rows, kind="stable")
        wanted[str(source)] = (
            rows[order].astype(np.int64),
            group.index.to_numpy(dtype=np.int64)[order],
        )

    frozen = np.full(len(pilot), np.nan, dtype=float)
    matched = np.zeros(len(pilot), dtype=bool)
    source_counts: dict[str, int] = {}
    parquet = pq.ParquetFile(str(canonical_path))
    for batch in parquet.iter_batches(
        batch_size=250_000,
        columns=["_source_file", "died_under5"],
        use_threads=True,
    ):
        sources = np.asarray(batch.column(0).to_pylist(), dtype=object)
        outcomes = pc.cast(batch.column(1), pa.float64()).to_numpy(
            zero_copy_only=False
        )
        if len(sources) == 0:
            continue
        starts = np.concatenate(
            ([0], np.flatnonzero(sources[1:] != sources[:-1]) + 1)
        )
        ends = np.concatenate((starts[1:], [len(sources)]))
        for start, end in zip(starts, ends):
            source = str(sources[start])
            base = source_counts.get(source, 0)
            stop = base + int(end - start)
            target = wanted.get(source)
            if target is not None:
                target_rows, target_indices = target
                left = int(np.searchsorted(target_rows, base, side="left"))
                right = int(np.searchsorted(target_rows, stop, side="left"))
                if right > left:
                    selected_indices = target_indices[left:right]
                    if matched[selected_indices].any():
                        raise RuntimeError(
                            f"duplicate frozen-canonical matches for {source}"
                        )
                    local = target_rows[left:right] - base + int(start)
                    frozen[selected_indices] = outcomes[local]
                    matched[selected_indices] = True
            source_counts[source] = stop

    count_mismatches = {
        source: (source_counts.get(source, 0), int(expected))
        for source, expected in expected_source_counts.items()
        if source_counts.get(source, 0) != int(expected)
    }
    if count_mismatches:
        examples = list(count_mismatches.items())[:5]
        raise RuntimeError(
            "frozen canonical source-row count mismatch "
            f"for {len(count_mismatches)} surveys; examples={examples}"
        )
    if not matched.all():
        missing_rows = pilot.loc[
            ~matched, ["_source_file", "_source_row"]
        ].head(5).to_dict("records")
        raise RuntimeError(
            f"frozen canonical join missed {int((~matched).sum())} pilot rows; "
            f"examples={missing_rows}"
        )

    result = pilot.copy()
    result["died_under5_legacy_frozen"] = frozen
    return result[PILOT_COLUMNS]


def _add_frozen_legacy_audit(
    manifest: pd.DataFrame,
    pilot: pd.DataFrame,
) -> pd.DataFrame:
    grouped = pilot.groupby("_source_file", sort=False)[
        "died_under5_legacy_frozen"
    ]
    totals = grouped.size()
    observed = grouped.count()
    deaths = grouped.sum(min_count=1).fillna(0)
    audit = pd.DataFrame(
        {
            "survey_id": totals.index.astype(str),
            "n_legacy_frozen_observed": observed.to_numpy(dtype=int),
            "n_legacy_frozen_deaths": deaths.to_numpy(dtype=int),
            "n_legacy_frozen_missing": (totals - observed).to_numpy(dtype=int),
        }
    )
    return manifest.merge(audit, on="survey_id", how="left", validate="one_to_one")


def _existing_build(config: PilotConfig) -> PilotBuildResult | None:
    """Return a reusable build only when its data and manifest contracts hold."""
    if not config.data_path.exists() or not config.manifest_path.exists():
        return None
    schema = set(pq.read_schema(str(config.data_path)).names)
    if set(PILOT_COLUMNS) - schema:
        return None
    manifest = pd.read_csv(config.manifest_path)
    required_manifest = {
        "survey_id",
        "sample_step",
        "sample_offset",
        "n_source_births",
        "n_pilot_births",
        "n_legacy_frozen_observed",
        "n_legacy_frozen_deaths",
        "n_legacy_frozen_missing",
        "frozen_canonical_sha256",
    }
    if required_manifest - set(manifest.columns) or manifest.empty:
        return None
    steps = pd.to_numeric(manifest["sample_step"], errors="coerce")
    if not steps.eq(config.sample_step).all():
        return None
    canonical_hashes = manifest["frozen_canonical_sha256"].dropna().astype(str).unique()
    canonical_path = HARMONIZED_DIR / "global_analysis.parquet"
    if (
        len(canonical_hashes) != 1
        or not canonical_path.exists()
        or canonical_hashes[0].lower() != _sha256(canonical_path).lower()
    ):
        return None
    metadata = pq.read_metadata(str(config.data_path))
    if int(manifest["n_pilot_births"].sum()) != metadata.num_rows:
        return None
    return PilotBuildResult(
        data_path=config.data_path,
        manifest_path=config.manifest_path,
        n_rows=int(metadata.num_rows),
        n_surveys=int(manifest["survey_id"].nunique()),
    )


def _write_build_atomically(
    pilot: pd.DataFrame,
    manifest: pd.DataFrame,
    config: PilotConfig,
) -> None:
    """Validate temporary artifacts before promoting either final path."""
    data_building = config.data_path.with_suffix(".building.parquet")
    manifest_building = config.manifest_path.with_suffix(".building.csv")
    data_building.unlink(missing_ok=True)
    manifest_building.unlink(missing_ok=True)
    try:
        pilot.to_parquet(data_building, index=False, compression="zstd")
        metadata = pq.read_metadata(str(data_building))
        missing = sorted(
            set(PILOT_COLUMNS) - set(pq.read_schema(str(data_building)).names)
        )
        if missing:
            raise RuntimeError(f"pilot parquet is missing required columns: {missing}")
        if metadata.num_rows != len(pilot):
            raise RuntimeError(
                f"pilot row-count mismatch: disk={metadata.num_rows}, memory={len(pilot)}"
            )
        manifest.to_csv(manifest_building, index=False)
        os.replace(data_building, config.data_path)
        os.replace(manifest_building, config.manifest_path)
    except Exception:
        data_building.unlink(missing_ok=True)
        manifest_building.unlink(missing_ok=True)
        raise


def build_phase2_dataset(config: PilotConfig) -> PilotBuildResult:
    """Build or reuse a fixed v2 diagnostic sample without touching canonical data."""
    from src.data.s03_harmonize import SKIP_SURVEYS, process_single_survey
    from src.data.s03_mics_harmonize import harmonize_mics_survey
    from src.data.s03_mics_pipeline import resolve_mics_zip_path

    if config.sample_step < 1:
        raise ValueError("sample_step must be a positive integer")
    canonical = HARMONIZED_DIR / "global_analysis.parquet"
    if config.data_path.resolve() == canonical.resolve():
        raise ValueError("Phase 2 data path must not be the canonical parquet")

    PHASE2_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    HARMONIZED_DIR.mkdir(parents=True, exist_ok=True)
    if not config.rebuild_data:
        existing = _existing_build(config)
        if existing is not None:
            return existing

    logger = _logger()
    frames: list[pd.DataFrame] = []
    manifest: list[dict] = []

    dhs_paths = [
        path
        for path in sorted(PARQUET_DIR.glob("*.parquet"))
        if not path.stem.startswith("_") and path.stem not in SKIP_SURVEYS
    ]
    logger.info("Building fixed DHS pilot sample from %d survey files", len(dhs_paths))
    with tempfile.TemporaryDirectory(
        prefix="_phase2_dhs_", dir=HARMONIZED_DIR
    ) as temp_name:
        temp_dir = Path(temp_name)
        for number, path in enumerate(dhs_paths, start=1):
            survey_year = _dhs_survey_year(path)
            if survey_year is None or not 2000 <= survey_year <= 2024:
                continue
            raw_sample = _stream_dhs_raw_sample(
                path,
                temp_dir / path.name,
                survey_id=path.stem,
                step=config.sample_step,
            )
            try:
                harmonized = process_single_survey(raw_sample.path, logger)
            finally:
                raw_sample.path.unlink(missing_ok=True)
            if harmonized is None or harmonized.empty:
                raise RuntimeError(f"DHS harmonization failed for {path.name}")
            if len(harmonized) != len(raw_sample.source_rows):
                raise RuntimeError(
                    f"{path.name}: harmonizer changed sampled row count "
                    f"from {len(raw_sample.source_rows)} to {len(harmonized)}"
                )
            harmonized["_source_row"] = raw_sample.source_rows
            selected = _select_pilot_columns(harmonized, data_source="DHS")
            frames.append(selected)
            manifest.append(
                _manifest_row(
                    selected,
                    data_source="DHS",
                    survey_id=path.stem,
                    sample_step=config.sample_step,
                    sample_offset=raw_sample.sample_offset,
                    n_source_births=raw_sample.n_source_births,
                )
            )
            if number % 50 == 0:
                logger.info("DHS progress %d/%d", number, len(dhs_paths))
            del harmonized, selected
            gc.collect()

    catalog_path = HARMONIZED_DIR.parent / "mics_survey_catalog.csv"
    catalog = pd.read_csv(catalog_path)
    eligible = catalog[
        pd.to_numeric(catalog["use_for_expansion"], errors="coerce").eq(1)
    ]
    logger.info("Adding %d eligible MICS surveys", len(eligible))
    for _, row in eligible.iterrows():
        survey_id = str(row["zip_filename"])
        harmonized = harmonize_mics_survey(
            zip_path=str(resolve_mics_zip_path(row, raw_mics_dir=RAW_MICS_DIR)),
            country_iso3=str(row["country_iso3"]),
            country_name=str(row["country_name"]),
            survey_year=int(row["survey_year"]),
            who_region=str(row["who_region"]),
        )
        if harmonized.empty:
            raise RuntimeError(f"MICS harmonization failed for {survey_id}")
        n_source_births = len(harmonized)
        sampled = deterministic_systematic_sample(
            harmonized, survey_id=survey_id, step=config.sample_step
        )
        selected = _select_pilot_columns(sampled, data_source="MICS")
        frames.append(selected)
        manifest.append(
            _manifest_row(
                selected,
                data_source="MICS",
                survey_id=survey_id,
                sample_step=config.sample_step,
                sample_offset=_sample_offset(survey_id, config.sample_step),
                n_source_births=n_source_births,
            )
        )
        del harmonized, sampled, selected
        gc.collect()

    if not frames:
        raise RuntimeError("No survey rows were available for the Phase 2 pilot")
    pilot = pd.concat(frames, ignore_index=True)
    duplicate_keys = pilot.duplicated(["_source_file", "_source_row"])
    if duplicate_keys.any():
        raise RuntimeError(
            f"pilot contains {int(duplicate_keys.sum())} duplicate source-row keys"
        )
    manifest_frame = pd.DataFrame(manifest).sort_values(
        ["data_source", "country_code", "survey_year", "survey_id"]
    )
    expected_source_counts = dict(
        zip(
            manifest_frame["survey_id"].astype(str),
            manifest_frame["n_source_births"].astype(int),
        )
    )
    pilot = _attach_frozen_legacy_outcome(
        pilot,
        canonical,
        expected_source_counts=expected_source_counts,
    )
    manifest_frame = _add_frozen_legacy_audit(manifest_frame, pilot)
    manifest_frame["frozen_canonical_sha256"] = _sha256(canonical)
    _write_build_atomically(pilot, manifest_frame, config)
    logger.info(
        "Wrote Phase 2 pilot: %s rows across %s surveys",
        f"{len(pilot):,}",
        pilot["_source_file"].nunique(),
    )
    return PilotBuildResult(
        data_path=config.data_path,
        manifest_path=config.manifest_path,
        n_rows=len(pilot),
        n_surveys=int(pilot["_source_file"].nunique()),
    )


def deterministic_systematic_sample(
    frame: pd.DataFrame,
    *,
    survey_id: str,
    step: int,
) -> pd.DataFrame:
    """Select fixed rows and retain their zero-based source positions."""
    offset = _sample_offset(survey_id, step)
    source_rows = np.arange(offset, len(frame), step, dtype=np.int64)
    sampled = frame.iloc[source_rows].copy()
    sampled["_source_row"] = source_rows
    return sampled.reset_index(drop=True)


def prepare_analysis_states(
    frame: pd.DataFrame,
    *,
    legacy: bool,
) -> pd.DataFrame:
    """Return explicit categorical states used by Phase 2 models.

    The v2 interval design maps first births to the 24--47 reference because
    first-birth status is already represented by birth-order terms. The
    separate ``birth_interval_state`` column retains the scientifically
    distinct first-birth state without creating a duplicate design column.
    """
    states = frame.copy()

    male = pd.to_numeric(frame["child_male"], errors="coerce")
    states["sex_cat"] = np.select(
        [male.eq(0), male.eq(1)],
        ["Female", "Male"],
        default="Female" if legacy else "Missing",
    )

    multiple = pd.to_numeric(frame["multiple_birth"], errors="coerce")
    states["birth_type"] = np.select(
        [multiple.eq(0), multiple.gt(0)],
        ["Singleton", "Multiple"],
        default="Singleton" if legacy else "Missing",
    )

    order = pd.to_numeric(frame["birth_order"], errors="coerce")
    states["bo_cat"] = np.select(
        [order.eq(1), order.between(2, 3), order.between(4, 5), order.ge(6)],
        ["1st", "2nd-3rd", "4th-5th", "6th+"],
        default="1st" if legacy else "Missing",
    )

    interval = pd.to_numeric(frame["birth_interval_months"], errors="coerce")
    interval_state = np.select(
        [
            order.eq(1),
            order.gt(1) & interval.isna(),
            order.gt(1) & interval.ge(0) & interval.lt(24),
            order.gt(1) & interval.between(24, 47),
            order.gt(1) & interval.ge(48),
        ],
        ["First birth", "Missing non-first", "<24 mo", "24-47 mo", "48+ mo"],
        default="Missing non-first",
    )
    states["birth_interval_state"] = interval_state
    if legacy:
        states["bi_model_cat"] = np.where(
            np.isin(interval_state, ["First birth", "Missing non-first"]),
            "First/missing",
            interval_state,
        )
    else:
        states["bi_model_cat"] = np.where(
            interval_state == "First birth", "24-47 mo", interval_state
        )

    maternal_age = pd.to_numeric(frame["maternal_age_at_birth"], errors="coerce")
    states["ma_cat"] = np.select(
        [maternal_age.between(20, 34), maternal_age.lt(20), maternal_age.ge(35)],
        ["20-34", "<20", "35+"],
        default="20-34" if legacy else "Missing",
    )

    education = pd.to_numeric(frame["edu_level"], errors="coerce")
    states["edu_cat"] = np.select(
        [education.eq(0), education.eq(1), education.eq(2), education.ge(3)],
        ["None", "Primary", "Secondary", "Higher"],
        default="None" if legacy else "Missing",
    )

    wealth = pd.to_numeric(frame["wealth_quintile"], errors="coerce")
    states["wq_cat"] = np.select(
        [wealth.eq(1), wealth.eq(2), wealth.eq(3), wealth.eq(4), wealth.eq(5)],
        ["Q1", "Q2", "Q3", "Q4", "Q5"],
        default="Q1" if legacy else "Missing",
    )

    married = pd.to_numeric(frame["currently_married"], errors="coerce")
    states["marital_cat"] = np.select(
        [married.eq(0), married.eq(1)],
        ["Not married", "Married"],
        default="Not married" if legacy else "Missing",
    )

    urban = pd.to_numeric(frame["urban"], errors="coerce")
    states["urban_cat"] = np.select(
        [urban.eq(0), urban.eq(1)],
        ["Rural", "Urban"],
        default="Rural" if legacy else "Missing",
    )

    survey_year = pd.to_numeric(frame["survey_year"], errors="coerce")
    states["decade"] = np.select(
        [survey_year.between(2000, 2009), survey_year.between(2010, 2019), survey_year.ge(2020)],
        ["2000-2009", "2010-2019", "2020-2024"],
        default="Missing",
    )
    states["survey_id"] = frame["_source_file"].astype(str)
    states["country_code"] = frame["country_code"].astype(str)
    return states
