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
RAW_DHS_DIR = Path(
    os.environ.get(
        "DHS_DATA_DIR",
        str(Path.home() / "dhs_data"),  # set DHS_DATA_DIR to your local DHS .DTA folder
    )
)

# ---------------------------------------------------------------------------
# Data layer hierarchy: 00_raw -> 01_parquet -> 02_harmonized
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"

# Raw external datasets (MICS surveys and UN IGME estimates)
RAW_MICS_DIR    = DATA_DIR / "raw" / "mics"
RAW_UN_IGME_DIR = DATA_DIR / "raw" / "un_igme"

# Level 0: Raw DHS .DTA files (symlink or reference)
RAW_DATA_DIR = DATA_DIR / "00_raw"

# Level 1: Per-survey Parquet files (output of s02_convert.py)
PARQUET_DIR = DATA_DIR / "01_parquet"

# Level 2: Final harmonized analysis datasets (output of s03_harmonize.py)
HARMONIZED_DIR = DATA_DIR / "02_harmonized"
# Single canonical analysis file (all years + in_primary_scope flag).
# Downstream code should use src.utils.io.load_primary() which applies
# the D006 scope filter at load time.
GLOBAL_PARQUET = HARMONIZED_DIR / "global_analysis.parquet"
# Country-level WDI covariates (output of s03d_country_covariates.py)
COUNTRY_COVARIATES = HARMONIZED_DIR / "country_covariates.parquet"

# Legacy aliases (DEPRECATED — will be removed after migration)
GLOBAL_FULL_PARQUET = GLOBAL_PARQUET
GLOBAL_PRIMARY_PARQUET = GLOBAL_PARQUET

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
FIGURES_DIR = OUTPUT_DIR / "figures"
MODELS_DIR = OUTPUT_DIR / "models"

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
