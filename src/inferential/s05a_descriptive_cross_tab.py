"""
s05a_descriptive_cross_tab.py — Table 1: Mega Cross-Tabulation
================================================================

Produces a Lancet-tier descriptive table crossing every categorical
risk factor level against U5MR and sample size, stratified by
WHO region. All estimates are survey-weighted.

Output: output/tables/table_1_cross_tabulation.csv (~50 rows)

References
----------
  - DHS Analysis Guide (Rutstein & Rojas, 2006)
  - Mosley & Chen (1984) — Proximate determinants framework
"""
import sys
import gc
import functools
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

print = functools.partial(print, flush=True)

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent.parent)
)
from config.paths import GLOBAL_PARQUET, TABLES_DIR
from src.utils.io import load_primary

TABLES_DIR.mkdir(parents=True, exist_ok=True)
PARQUET = GLOBAL_PARQUET

# Columns needed for cross-tabulation
LOAD_COLS = [
    "died_under5", "who_region", "sample_weight",
    "child_male", "multiple_birth", "birth_order",
    "birth_interval_months", "maternal_age_at_birth",
    "edu_level", "currently_married", "wealth_quintile",
    "urban", "survey_year",
]

# Variable definitions: (column, label, categorizer_fn)
# categorizer_fn: maps raw array -> (labels_array, level_order)
VAR_DEFS: list[tuple[str, str, Callable]] = []


def _cat_sex(x: np.ndarray) -> tuple[np.ndarray, list]:
    labels = np.where(x == 1, "Male", "Female")
    return labels, ["Male", "Female"]


def _cat_birth_type(x: np.ndarray) -> tuple[np.ndarray, list]:
    labels = np.where(x == 1, "Multiple", "Singleton")
    return labels, ["Singleton", "Multiple"]


def _cat_birth_order(x: np.ndarray) -> tuple[np.ndarray, list]:
    labels = np.full(len(x), "", dtype=object)
    labels[x == 1] = "1st"
    labels[(x >= 2) & (x <= 3)] = "2nd-3rd"
    labels[(x >= 4) & (x <= 5)] = "4th-5th"
    labels[x >= 6] = "6th+"
    return labels, ["1st", "2nd-3rd", "4th-5th", "6th+"]


def _cat_birth_interval(x: np.ndarray) -> tuple[np.ndarray, list]:
    labels = np.full(len(x), "", dtype=object)
    labels[np.isnan(x)] = "First birth / missing"
    labels[(~np.isnan(x)) & (x < 24)] = "<24 months"
    labels[(~np.isnan(x)) & (x >= 24) & (x < 48)] = "24-47 months"
    labels[(~np.isnan(x)) & (x >= 48)] = "48+ months"
    return labels, [
        "First birth / missing", "<24 months",
        "24-47 months", "48+ months",
    ]


def _cat_mat_age(x: np.ndarray) -> tuple[np.ndarray, list]:
    labels = np.full(len(x), "", dtype=object)
    labels[x < 20] = "<20"
    labels[(x >= 20) & (x < 35)] = "20-34"
    labels[x >= 35] = "35+"
    return labels, ["<20", "20-34", "35+"]


def _cat_education(x: np.ndarray) -> tuple[np.ndarray, list]:
    labels = np.full(len(x), "", dtype=object)
    labels[x == 0] = "None"
    labels[x == 1] = "Primary"
    labels[x == 2] = "Secondary"
    labels[x == 3] = "Higher"
    return labels, ["None", "Primary", "Secondary", "Higher"]


def _cat_married(x: np.ndarray) -> tuple[np.ndarray, list]:
    labels = np.where(x == 1, "Married", "Not married")
    return labels, ["Married", "Not married"]


def _cat_wealth(x: np.ndarray) -> tuple[np.ndarray, list]:
    mapping = {1: "Poorest", 2: "Poorer", 3: "Middle",
               4: "Richer", 5: "Richest"}
    labels = np.array([mapping.get(int(v), "") if np.isfinite(v)
                       else "" for v in x], dtype=object)
    return labels, [
        "Poorest", "Poorer", "Middle", "Richer", "Richest",
    ]


def _cat_urban(x: np.ndarray) -> tuple[np.ndarray, list]:
    labels = np.where(x == 1, "Urban", "Rural")
    return labels, ["Urban", "Rural"]


def _cat_decade(x: np.ndarray) -> tuple[np.ndarray, list]:
    labels = np.full(len(x), "", dtype=object)
    labels[(x >= 2000) & (x <= 2009)] = "2000-2009"
    labels[(x >= 2010) & (x <= 2019)] = "2010-2019"
    labels[(x >= 2020) & (x <= 2025)] = "2020-2024"
    return labels, ["2000-2009", "2010-2019", "2020-2024"]


# Register all variables
VAR_DEFS = [
    ("child_male", "Child sex", _cat_sex),
    ("multiple_birth", "Birth type", _cat_birth_type),
    ("birth_order", "Birth order", _cat_birth_order),
    ("birth_interval_months", "Birth interval", _cat_birth_interval),
    ("maternal_age_at_birth", "Maternal age", _cat_mat_age),
    ("edu_level", "Education", _cat_education),
    ("currently_married", "Marital status", _cat_married),
    ("wealth_quintile", "Wealth quintile", _cat_wealth),
    ("urban", "Residence", _cat_urban),
    ("survey_year", "Survey decade", _cat_decade),
]


def weighted_stats(
    died: np.ndarray,
    weights: np.ndarray,
) -> dict:
    """Compute weighted N, U5MR for a subgroup."""
    mask = np.isfinite(died) & np.isfinite(weights) & (weights > 0)
    if mask.sum() == 0:
        return {"n": 0, "u5mr": np.nan}
    w = weights[mask]
    d = died[mask]
    n_weighted = w.sum()
    u5mr = np.average(d, weights=w) * 1000
    return {"n": int(round(n_weighted)), "u5mr": round(u5mr, 1)}


def main() -> None:
    """Build the mega cross-tabulation table."""
    print("=" * 60)
    print("Table 1: Mega Cross-Tabulation")
    print("=" * 60)

    # Load primary-scope dataset only (D006: 2000-2024). Without this
    # filter the cross-tab silently includes pre-2000 supplement births,
    # inflating N and biasing every stratum-specific U5MR.
    print("Loading data (primary scope, D006)...")
    df = load_primary(columns=LOAD_COLS, scope="primary")
    gc.collect()
    print(f"  Loaded {len(df):,} primary-scope rows")

    # Ensure weights are valid
    df["sample_weight"] = df["sample_weight"].fillna(1.0)
    died = df["died_under5"].values
    weights = df["sample_weight"].values

    # Get regions
    regions = sorted(df["who_region"].dropna().unique())

    rows = []
    total_n = weights[np.isfinite(died)].sum()

    for col, var_label, cat_fn in VAR_DEFS:
        print(f"  Processing: {var_label}...")
        raw = df[col].values.astype(float)
        cat_labels, level_order = cat_fn(raw)

        for level in level_order:
            mask_level = cat_labels == level
            if mask_level.sum() == 0:
                continue

            row = {
                "variable": var_label,
                "level": level,
            }

            # All LMICs
            sub_died = died[mask_level]
            sub_w = weights[mask_level]
            stats = weighted_stats(sub_died, sub_w)
            valid_mask = (
                np.isfinite(sub_died) & np.isfinite(sub_w)
            )
            row["N_All"] = f"{int(valid_mask.sum()):,}"
            row["pct_All"] = round(
                sub_w[valid_mask].sum() / total_n * 100, 1
            )
            row["U5MR_All"] = stats["u5mr"]

            # Per region
            for region in regions:
                reg_mask = (
                    mask_level
                    & (df["who_region"].values == region)
                )
                sub_d = died[reg_mask]
                sub_ww = weights[reg_mask]
                s = weighted_stats(sub_d, sub_ww)
                valid_r = (
                    np.isfinite(sub_d) & np.isfinite(sub_ww)
                )
                row[f"N_{region}"] = f"{int(valid_r.sum()):,}"
                row[f"U5MR_{region}"] = s["u5mr"]

            rows.append(row)

    result = pd.DataFrame(rows)

    # Save
    out = TABLES_DIR / "table_1_cross_tabulation.csv"
    result.to_csv(out, index=False)
    print(f"\nSaved: {out.name}")
    print(f"  Rows: {len(result)}")
    print(result.to_string(index=False))

    del df
    gc.collect()


if __name__ == "__main__":
    main()
