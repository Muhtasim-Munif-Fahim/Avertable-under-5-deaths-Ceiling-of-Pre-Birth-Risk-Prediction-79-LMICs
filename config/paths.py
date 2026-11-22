"""
Central path configuration for the Global DHS Mortality Study.

All file paths used by pipeline scripts are defined here.
To run on a different machine, update RAW_DHS_DIR only.

References
----------
- Decision D006: Temporal scope (2000-2024)
- Decision D007: Survey deduplication
"""
from pathlib import Path
import os

# ---------------------------------------------------------------------------
# Project root (auto-detected from this file's location)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# External data source (set DHS_DATA_DIR env var or update fallback below)
# ---------------------------------------------------------------------------
def _resolve_dhs_dir() -> Path:
    """Resolve DHS data directory: env var > workspace data dir > sibling dir > hardcoded fallback."""
    env = os.environ.get("DHS_DATA_DIR")
    if env:
        return Path(env)
    # Workspace layout since 2026-09-24: <workspace>/01_data/raw/ and <workspace>/02_analysis/<repo>/
    workspace = PROJECT_ROOT.parents[1] / "01_data" / "raw" / "DHS MultiCountry Data"
    if workspace.is_dir():
        return workspace
    sibling = PROJECT_ROOT.parent / "DHS MultiCountry Data"
    if sibling.is_dir():
        return sibling
    return Path(r"J:\Survey Research\03 Temporal NAS\DHS MultiCountry Data")

RAW_DHS_DIR = _resolve_dhs_dir()

# ---------------------------------------------------------------------------
# Data layer hierarchy: raw DHS (01_data) -> 01_parquet -> 02_harmonized
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"

# Raw external datasets (MICS surveys and UN IGME estimates)
RAW_MICS_DIR    = DATA_DIR / "raw" / "mics"
RAW_UN_IGME_DIR = DATA_DIR / "raw" / "un_igme"

# Level 1: Per-survey Parquet files (output of s02_convert.py)
PARQUET_DIR = DATA_DIR / "01_parquet"

# Level 2: Final harmonized analysis datasets (output of s03_harmonize.py)
HARMONIZED_DIR = DATA_DIR / "02_harmonized"
# Harmonised DHS + MICS base written by the data build (s03_harmonize / s03_mics_pipeline). The v2
# analysis cohort (global_analysis_v2_candidate.parquet) is built from it and its publication runs
# check it is unchanged, so it stays even though no model reads it directly.
GLOBAL_PARQUET = HARMONIZED_DIR / "global_analysis.parquet"

# Catalog: dataset metadata
CATALOG_DIR = DATA_DIR / "catalog"
CATALOG_PARQUET = CATALOG_DIR / "dataset_catalog.parquet"
CATALOG_CSV = CATALOG_DIR / "dataset_catalog.csv"

# Logs: processing logs
LOG_DIR = DATA_DIR / "logs"

# ---------------------------------------------------------------------------
# Output directories (git-tracked, small files only)
# ---------------------------------------------------------------------------
OUTPUT_DIR = PROJECT_ROOT / "output"
TABLES_DIR = OUTPUT_DIR / "tables"
# Figures live outside the repo since 2026-09-24, as the standalone workspace folder
# <workspace>/04_figures/: ONE set (BMJ style), files named by submitted figure number.
FIGURES_DIR = PROJECT_ROOT.parents[1] / "04_figures"
# How output/*.tex reach that folder in \includegraphics (relative, so no spaces in the path).
TEX_FIGURES_PREFIX = "../../../04_figures/"
# MODELS_DIR removed 2026-09-02: it pointed at output/models/, which never existed and
# had no readers. Its subject (persisted ML boosters) left with the prediction arm on
# 2026-08-03 — see ../../_archive/2026-08-03_ml_arm_withdrawn/.

# ---------------------------------------------------------------------------
# Documentation
# ---------------------------------------------------------------------------
DOCS_DIR = PROJECT_ROOT / "docs"

# ---------------------------------------------------------------------------
# Reproducibility (Fix #32): single seed for all RNG-using code.
# Format YYYYMMDD locks the seed to a date so re-runs are deterministic
# but a deliberate refresh (new date) gives a fresh permutation.
# ---------------------------------------------------------------------------
RNG_SEED = 20260505
