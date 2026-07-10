"""
01_batch_convert_to_parquet.py — DHS Stata → Parquet Conversion
================================================================

Reads raw DHS Birth Recode .DTA files and converts them to compressed
Parquet format, extracting only the ~40 DHS v-code variables needed
for the global child mortality analysis.

Design Decisions
----------------
  - D002: Skips India sub-national files (is_subnational=True in catalog)
  - D003: Parquet chosen over CSV for 10× compression and type preservation
  - D004: Extracts only required v-codes to reduce storage by ~95%

Variable Selection
------------------
  The variable list follows the Mosley & Chen (1984) proximate determinants
  framework and aligns with the IPUMS DHS harmonized variable list.
  See DECISIONS.md for rationale.

  Variables are requested in lowercase. Older DHS files may use uppercase;
  this script handles case normalization automatically.

References
----------
  - Mosley & Chen (1984). An analytical framework for the study of child
    survival in developing countries. Population and Development Review.
  - DHS Recode Manual v7 (ICF International, 2018)
"""

import logging
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import pyreadstat

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.paths import RAW_DHS_DIR, CATALOG_PARQUET, PARQUET_DIR, LOG_DIR

# Required DHS v-code variables (lowercase).
# Grouped by Mosley & Chen framework domain.
REQUIRED_VARIABLES: list[str] = [
    # ── Identifiers & Metadata ──
    "v000",    # Country code (e.g., "BD7")
    "v001",    # Cluster number (PSU)
    "v002",    # Household number
    "v003",    # Respondent line number
    "v005",    # Sample weight (divide by 1e6 for proportion)
    "v007",    # Year of interview
    "v006",    # Month of interview
    "v021",    # Primary sampling unit
    "v022",    # Sample strata for SE estimation
    "v023",    # Sample stratum number
    "v024",    # Region
    "v025",    # Type of place of residence (urban/rural)

    # ── Maternal Demographic ──
    "v012",    # Respondent's current age
    "v013",    # Age in 5-year groups
    "v106",    # Highest educational level
    "v107",    # Highest year of education
    "v133",    # Education in single years
    "v190",    # Wealth index combined
    "v190a",   # Wealth index for urban/rural
    "v191",    # Wealth index factor score (combined)
    "v501",    # Current marital status
    "v511",    # Age at first cohabitation

    # ── Household ──
    "v136",    # Number of household members
    "v137",    # Number of children 5 and under in HH
    "v445",    # Body mass index (respondent)

    # ── Child / Birth Record (b-variables) ──
    "bidx",    # Birth index in birth history
    "bord",    # Birth order number
    "b0",      # Child is twin (0=single, 1=1st twin, ...)
    "b1",      # Month of birth
    "b2",      # Year of birth
    "b3",      # Date of birth (CMC)
    "b4",      # Sex of child (1=male, 2=female)
    "b5",      # Child is alive (0=no, 1=yes)
    "b6",      # Age at death (imputed, units vary)
    "b7",      # Age at death in months (completed)
    "b8",      # Current age of child
    "b9",      # Child lives with whom
    "b10",     # Completeness of information
    "b11",     # Preceding birth interval (months)
    "b12",     # Succeeding birth interval (months)

    # ── Maternal & Child Health (m-variables) ──
    "m1",      # Number of tetanus injections before birth
    "m2a",     # Prenatal care: doctor
    "m2b",     # Prenatal care: nurse/midwife
    "m2c",     # Prenatal care: auxiliary midwife
    "m2d",     # Prenatal care: health assistant
    "m2n",     # Prenatal care: no one
    "m14",     # Number of antenatal visits during pregnancy
    "m15",     # Place of delivery
    "m17",     # Delivery by caesarean section
    "m18",     # Size of child at birth (mother's perception)
    "m19",     # Birth weight in kilograms (grams actually)
    "m19a",    # Weight at birth from card or recall
]

# Secondary variables: useful if available, not critical
SECONDARY_VARIABLES: list[str] = [
    "v008",    # Date of interview (CMC)
    "v010",    # Respondent's year of birth
    "v130",    # Religion
    "v131",    # Ethnicity
    "v151",    # Sex of household head
    "v152",    # Age of household head
    "v201",    # Total children ever born
    "v218",    # Number of living children
    "v228",    # Ever had a terminated pregnancy
    "v437",    # Respondent's weight in kilograms
    "v438",    # Respondent's height in centimeters
    "m2e",     # Prenatal: trained birth attendant
    "m2f",     # Prenatal: traditional birth attendant
    "m2g",     # Prenatal: community health worker
    "m3a",     # Assistance at delivery: doctor
    "m3b",     # Assistance at delivery: nurse/midwife
    "m3n",     # Assistance at delivery: no one
]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging() -> logging.Logger:
    """Configure logging to file and console."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / "01_convert_to_parquet.log"

    logger = logging.getLogger("parquet_convert")
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(str(log_file), encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


# ---------------------------------------------------------------------------
# Core Conversion
# ---------------------------------------------------------------------------

def find_available_columns(
    all_columns: list[str],
    requested: list[str],
) -> tuple[list[str], list[str]]:
    """
    Match requested variable names against available columns.

    Handles case-insensitive matching since older DHS files
    may use uppercase variable names.

    Parameters
    ----------
    all_columns : list[str]
        Column names present in the .DTA file.
    requested : list[str]
        Desired variable names (lowercase).

    Returns
    -------
    tuple[list[str], list[str]]
        (found_columns, missing_columns)
        found_columns uses the ORIGINAL casing from the file.
    """
    # Build case-insensitive lookup: lowercase → original
    col_map = {c.lower(): c for c in all_columns}

    found = []
    missing = []
    for var in requested:
        original = col_map.get(var.lower())
        if original is not None:
            found.append(original)
        else:
            missing.append(var)

    return found, missing


def convert_single_dataset(
    dta_path: Path,
    output_path: Path,
    logger: logging.Logger,
) -> Optional[dict]:
    """
    Convert a single DHS .DTA file to Parquet.

    Parameters
    ----------
    dta_path : Path
        Path to the .DTA (Stata) file.
    output_path : Path
        Path for the output .parquet file.
    logger : logging.Logger

    Returns
    -------
    dict or None
        Conversion metadata, or None on failure.
    """
    t0 = time.time()
    folder_name = dta_path.parent.name

    try:
        # Step 1: Read metadata only to discover columns
        _, meta = pyreadstat.read_dta(str(dta_path), metadataonly=True)
        all_cols = meta.column_names

        # Step 2: Determine which variables are available
        all_requested = REQUIRED_VARIABLES + SECONDARY_VARIABLES
        found_cols, missing_req = find_available_columns(
            all_cols, REQUIRED_VARIABLES
        )
        found_sec, _ = find_available_columns(
            all_cols, SECONDARY_VARIABLES
        )
        use_cols = found_cols + found_sec

        if not use_cols:
            logger.error(
                "  %s: No matching columns found!", folder_name
            )
            return None

        # Step 3: Read only the required columns
        df, _ = pyreadstat.read_dta(
            str(dta_path), usecols=use_cols
        )

        # Step 4: Normalize column names to lowercase
        df.columns = [c.lower() for c in df.columns]

        # Step 5: Add source metadata
        df["_source_file"] = folder_name
        df["_dta_filename"] = dta_path.name

        # Step 6: Save as compressed Parquet
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(
            output_path,
            engine="pyarrow",
            compression="snappy",
            index=False,
        )

        elapsed = time.time() - t0
        file_size_mb = output_path.stat().st_size / (1024**2)

        return {
            "folder_name": folder_name,
            "n_rows": len(df),
            "n_cols_total": len(all_cols),
            "n_cols_extracted": len(use_cols),
            "n_missing_required": len(missing_req),
            "missing_required": missing_req,
            "parquet_size_mb": round(file_size_mb, 2),
            "dta_size_mb": round(
                dta_path.stat().st_size / (1024**2), 2
            ),
            "compression_ratio": round(
                dta_path.stat().st_size / max(1, output_path.stat().st_size),
                1,
            ),
            "elapsed_sec": round(elapsed, 1),
            "status": "OK",
        }

    except Exception as e:
        logger.error("  %s: FAILED — %s", folder_name, str(e))
        return {
            "folder_name": folder_name,
            "status": "FAILED",
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Batch convert all national DHS BR datasets to Parquet."""
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("DHS Stata -> Parquet Batch Conversion")
    logger.info("=" * 60)

    # Load catalog
    if not CATALOG_PARQUET.exists():
        logger.error("Catalog not found: %s", CATALOG_PARQUET)
        logger.error("Run 00_catalog_datasets.py first.")
        sys.exit(1)

    catalog = pd.read_parquet(CATALOG_PARQUET)
    logger.info("Loaded catalog: %d datasets", len(catalog))

    # Filter: national only, with DTA files (Decision D002)
    national = catalog[
        (~catalog["is_subnational"])
        & (catalog["dta_found"] == True)  # noqa: E712
    ].copy()
    logger.info(
        "National datasets with .DTA: %d (skipping %d sub-national)",
        len(national),
        len(catalog) - len(national),
    )

    # Process each dataset
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    skipped = 0
    converted = 0
    failed = 0

    for i, (_, row) in enumerate(national.iterrows(), 1):
        folder = row["folder_name"]
        dta_path = Path(row["dta_path"])
        out_path = PARQUET_DIR / f"{folder}.parquet"

        # Resume support: skip if already converted
        if out_path.exists():
            skipped += 1
            logger.info(
                "[%d/%d] %s — SKIPPED (already exists)",
                i, len(national), folder,
            )
            results.append({
                "folder_name": folder,
                "status": "SKIPPED",
            })
            continue

        logger.info(
            "[%d/%d] %s — converting...",
            i, len(national), folder,
        )

        result = convert_single_dataset(dta_path, out_path, logger)
        if result is None:
            failed += 1
            continue

        results.append(result)
        if result["status"] == "OK":
            converted += 1
            logger.info(
                "  -> %s rows, %d/%d vars, "
                "%.1f MB -> %.1f MB (%.0fx), %.1fs",
                f"{result['n_rows']:,}",
                result["n_cols_extracted"],
                result["n_cols_total"],
                result["dta_size_mb"],
                result["parquet_size_mb"],
                result["compression_ratio"],
                result["elapsed_sec"],
            )
        else:
            failed += 1

    # Save conversion log
    results_df = pd.DataFrame(results)
    log_path = PARQUET_DIR / "_conversion_log.parquet"
    results_df.to_parquet(log_path, index=False)
    results_df.to_csv(
        PARQUET_DIR / "_conversion_log.csv", index=False
    )

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("CONVERSION SUMMARY")
    logger.info("=" * 60)
    logger.info("  Converted:  %d", converted)
    logger.info("  Skipped:    %d (already existed)", skipped)
    logger.info("  Failed:     %d", failed)
    logger.info("  Total:      %d", len(national))

    if converted > 0:
        ok_results = results_df[results_df["status"] == "OK"]
        total_parquet = ok_results["parquet_size_mb"].sum()
        total_dta = ok_results["dta_size_mb"].sum()
        total_rows = ok_results["n_rows"].sum()
        avg_ratio = ok_results["compression_ratio"].mean()
        logger.info("")
        logger.info("  Total births converted: %s", f"{total_rows:,.0f}")
        logger.info(
            "  Total DTA size:     %.1f GB", total_dta / 1024
        )
        logger.info(
            "  Total Parquet size: %.1f GB", total_parquet / 1024
        )
        logger.info("  Avg compression:    %.1f×", avg_ratio)

        # Report commonly missing required variables
        all_missing = []
        for _, r in ok_results.iterrows():
            if isinstance(r.get("missing_required"), list):
                all_missing.extend(r["missing_required"])
        if all_missing:
            from collections import Counter
            missing_counts = Counter(all_missing).most_common(15)
            logger.info("")
            logger.info("  Most commonly missing required vars:")
            for var, count in missing_counts:
                pct = count / len(ok_results) * 100
                logger.info(
                    "    %-8s missing in %3d/%d datasets (%.0f%%)",
                    var, count, len(ok_results), pct,
                )

    logger.info("")
    logger.info("Output: %s", PARQUET_DIR)
    logger.info("Done.")


if __name__ == "__main__":
    main()
