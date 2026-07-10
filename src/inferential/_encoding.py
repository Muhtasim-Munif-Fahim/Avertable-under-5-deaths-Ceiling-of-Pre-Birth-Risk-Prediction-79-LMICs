"""Shared categorical encoding of DHS features for inferential scripts.

Used by s05b (GEE regressions), s05c (counterfactual PAF), and s05g
(fully-exposed sensitivity) to ensure consistent category definitions,
reference groups, and label spellings across all Table 2/3 output.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def encode_cats(df: pd.DataFrame) -> None:
    """Create standardized categorical columns in-place.

    Columns created
    ---------------
    bo_cat : birth order (1st, 2nd-3rd, 4th-5th, 6th+)
    bi_cat : birth interval (24-47 mo, First/missing, <24 mo, 48+ mo)
    ma_cat : maternal age (20-34, <20, 35+)
    edu_cat : education (None, Primary, Secondary, Higher)
    wq_cat : wealth quintile (Poorest, Poorer, Middle, Richer, Richest)
    sex_cat : child sex (Female, Male)
    birth_type : singleton/multiple (Singleton, Multiple)
    marital_cat : marital status (Not married, Married)
    urban_cat : urban/rural residence (Rural, Urban)
    decade : survey decade (2000-2009, 2010-2019, 2020-2024)
    """
    # Birth order
    bo = df["birth_order"].values
    bo_cat = np.full(len(df), "1st", dtype=object)
    bo_cat[(bo >= 2) & (bo <= 3)] = "2nd-3rd"
    bo_cat[(bo >= 4) & (bo <= 5)] = "4th-5th"
    bo_cat[bo >= 6] = "6th+"
    df["bo_cat"] = pd.Categorical(
        bo_cat, categories=["1st", "2nd-3rd", "4th-5th", "6th+"])

    # Birth interval
    bi = df["birth_interval_months"].values
    bi_cat = np.full(len(df), "24-47 mo", dtype=object)
    bi_cat[np.isnan(bi)] = "First/missing"
    bi_cat[(~np.isnan(bi)) & (bi < 24)] = "<24 mo"
    bi_cat[(~np.isnan(bi)) & (bi >= 48)] = "48+ mo"
    df["bi_cat"] = pd.Categorical(
        bi_cat,
        categories=["24-47 mo", "First/missing", "<24 mo", "48+ mo"])

    # Maternal age
    ma = df["maternal_age_at_birth"].values
    ma_cat = np.full(len(df), "20-34", dtype=object)
    ma_cat[ma < 20] = "<20"
    ma_cat[ma >= 35] = "35+"
    df["ma_cat"] = pd.Categorical(
        ma_cat, categories=["20-34", "<20", "35+"])

    # Education
    edu = df["edu_level"].fillna(0).values
    edu_cat = np.full(len(df), "None", dtype=object)
    edu_cat[edu == 1] = "Primary"
    edu_cat[edu == 2] = "Secondary"
    edu_cat[edu >= 3] = "Higher"
    df["edu_cat"] = pd.Categorical(
        edu_cat,
        categories=["None", "Primary", "Secondary", "Higher"])

    # Wealth
    wq = df["wealth_quintile"].fillna(1).values
    wq_map = {1: "Poorest", 2: "Poorer", 3: "Middle",
              4: "Richer", 5: "Richest"}
    df["wq_cat"] = pd.Categorical(
        [wq_map.get(int(v), "Poorest") for v in wq],
        categories=["Poorest", "Poorer", "Middle",
                     "Richer", "Richest"])

    # Binary vars as categorical
    df["sex_cat"] = pd.Categorical(
        np.where(df["child_male"] == 1, "Male", "Female"),
        categories=["Female", "Male"])
    df["birth_type"] = pd.Categorical(
        np.where(df["multiple_birth"] == 1, "Multiple", "Singleton"),
        categories=["Singleton", "Multiple"])
    df["marital_cat"] = pd.Categorical(
        np.where(df["currently_married"] == 1, "Married", "Not married"),
        categories=["Not married", "Married"])
    df["urban_cat"] = pd.Categorical(
        np.where(df["urban"] == 1, "Urban", "Rural"),
        categories=["Rural", "Urban"])

    # Decade
    sy = df["survey_year"].values
    dec = np.full(len(df), "2000-2009", dtype=object)
    dec[(sy >= 2010) & (sy <= 2019)] = "2010-2019"
    dec[sy >= 2020] = "2020-2024"
    df["decade"] = pd.Categorical(
        dec, categories=["2000-2009", "2010-2019", "2020-2024"])
