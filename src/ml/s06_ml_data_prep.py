"""
s06_ml_data_prep.py - ML Spatio-Temporal Data Splitting
=======================================================

Creates the ML data matrix with the required Spatio-Temporal validation splits
to prevent data leakage and prove forward-predictive viability.
- Target: died_under5

Outputs: 
  - output/ml/X_train.parquet, y_train.parquet
  - output/ml/X_val_time.parquet, y_val_time.parquet
  - output/ml/X_val_space.parquet, y_val_space.parquet

References
----------
  - Nature Medicine ML Standards (OOT Validation)
"""

import sys
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent.parent)
)
from src.utils.print_util import print
from config.paths import GLOBAL_PARQUET, OUTPUT_DIR, RNG_SEED

ML_DIR = OUTPUT_DIR / "ml"
ML_DIR.mkdir(parents=True, exist_ok=True)

PARQUET = GLOBAL_PARQUET

# Sampling step for the ML pipeline. The full parquet (12.6M rows) is too
# large to hold in RAM together with XGBoost/LightGBM DMatrices on the 32 GB
# workstation, so we read every ML_SAMPLING_STEP-th row.  At step=7 this is a
# 14.2% systematic sample (~1.5M rows after the in_primary_scope filter),
# which the manuscript reports as the ML denominator (Methods §2.4).
# Set ML_FULL_SAMPLE=1 to use every row (full 100% cohort) instead of the
# 14.2% systematic sample. SHAP (s08/s08b) already subsamples internally, so
# full-data is RAM-safe; XGBoost/LightGBM hist handle the larger DMatrices.
import os as _os
ML_SAMPLING_STEP = 1 if _os.environ.get("ML_FULL_SAMPLE") == "1" else 7
SEED = RNG_SEED

# Main prediction features audited for measurement timing relative to
# died_under5. The publication main model uses only conservative
# demographic and household predictors available before or independent of
# the child's survival outcome. Healthcare/delivery and birth-size variables
# are excluded from the main ML pipeline and retained only for inferential or
# explicitly labelled sensitivity analyses.
BANNED_MAIN_ML_FEATURES = {
    "maternal_bmi",
    "birth_weight_kg",
    "small_at_birth",
    "caesarean",
    "facility_delivery",
    "anc_visits",
    "tetanus_protected",
}

FEATURES = [
    "child_male", "multiple_birth", "birth_order",
    "birth_interval_months", "maternal_age_at_birth",
    "edu_level", "currently_married", "wealth_quintile",
    "urban", "hh_members",
]

META_COLS = [
    "died_under5",
    "survey_year",
    "country_code",
    "sample_weight",
    "in_primary_scope",
]


def main():
    print("=" * 60)
    print("Phase 4: ML Spatio-Temporal Data Splitting")
    print("=" * 60)
    
    # Check total rows
    pf = pq.ParquetFile(str(PARQUET))
    total_rows = pf.metadata.num_rows
    print(f"Total dataset rows: {total_rows:,}")
    
    full_sample = ML_SAMPLING_STEP == 1
    if full_sample:
        print(f"Loading FULL cohort (ML_FULL_SAMPLE=1, every row)...")
    else:
        print(f"Loading {100/ML_SAMPLING_STEP:.1f}% systematic sample "
              f"(every {ML_SAMPLING_STEP}th row)...")

    # Load data in batches to avoid peak memory
    frames = []
    row_offset = 0
    for batch in pf.iter_batches(
        batch_size=500_000, columns=META_COLS + FEATURES
    ):
        batch_df = batch.to_pandas()
        batch_len = len(batch_df)
        if full_sample:
            # take every row — skip the O(n) index-set membership test
            frames.append(batch_df)
        else:
            local_idx = list(range(
                (-row_offset) % ML_SAMPLING_STEP, batch_len, ML_SAMPLING_STEP))
            if local_idx:
                frames.append(batch_df.iloc[local_idx])
        row_offset += batch_len
        del batch_df, batch
    
    df = pd.concat(frames, ignore_index=True)
    del frames
    gc.collect()
    
    df = df.dropna(subset=["died_under5"]).copy()
    df = df[df["in_primary_scope"] == 1].copy()
    print(f"Loaded {len(df):,} valid primary-scope rows.")
    
    # 1. Stratified Spatial Split: Hold out ~20% of countries per
    #    WHO region to ensure geographic representativeness (W1)
    np.random.seed(SEED)
    
    # Get country-to-region mapping
    region_frames = []
    for batch in pq.ParquetFile(str(PARQUET)).iter_batches(
        batch_size=1_000_000, columns=["country_code", "who_region"]
    ):
        region_frames.append(
            batch.to_pandas().drop_duplicates("country_code")
        )
    country_region = pd.concat(region_frames).drop_duplicates(
        "country_code"
    ).set_index("country_code")["who_region"].to_dict()
    del region_frames
    gc.collect()

    unique_countries = sorted(df["country_code"].unique())
    
    # Group countries by WHO region
    from collections import defaultdict
    region_to_countries = defaultdict(list)
    for c in unique_countries:
        r = country_region.get(c, "Unknown")
        region_to_countries[r].append(c)
    
    # Hold out ~20% from each region (min 1 country)
    space_holdout_countries = []
    for region in sorted(region_to_countries.keys()):
        countries = sorted(region_to_countries[region])
        n_holdout = max(1, round(len(countries) * 0.20))
        held = np.random.choice(
            countries, size=n_holdout, replace=False
        ).tolist()
        space_holdout_countries.extend(held)
        print(f"  {region}: {len(countries)} total, "
              f"{n_holdout} held out -> {held}")
    
    train_countries = [c for c in unique_countries
                       if c not in space_holdout_countries]
    
    with open(ML_DIR / "country_splits.json", "w") as f:
        json.dump({
            "training_countries": train_countries,
            "spatial_holdout_countries": space_holdout_countries,
            "stratification": "WHO_region_proportional",
        }, f, indent=2)
        
    print(f"\nStratified Spatial Split:")
    print(f"  Training Countries: {len(train_countries)}")
    print(f"  Spatial Holdout (Unseen): {len(space_holdout_countries)}")
    
    # 2. Extract Splits
    # Spatial Test: All years, but only in holdout countries
    mask_space = df["country_code"].isin(space_holdout_countries)
    df_space_test = df[mask_space].copy()
    
    # Training Data Pool
    df_train_pool = df[~mask_space].copy()
    
    # Train: <= 2015
    mask_train = df_train_pool["survey_year"] <= 2015
    df_train = df_train_pool[mask_train].copy()
    
    # Internal Val (for early stopping): 2016-2018
    mask_val = (df_train_pool["survey_year"] > 2015) & (df_train_pool["survey_year"] <= 2018)
    df_val_internal = df_train_pool[mask_val].copy()
    
    # Temporal Test: >= 2019 (PURE HOLDOUT)
    mask_test = df_train_pool["survey_year"] >= 2019
    df_time_test = df_train_pool[mask_test].copy()
    
    # 3. Print Stats
    def print_stats(name, d):
        w = d["sample_weight"].fillna(1.0).values
        y = d["died_under5"].values
        u5mr = np.average(y, weights=w) * 1000
        print(f"  {name:15}: {len(d):>8,} rows, U5MR = {u5mr:5.1f}")
        
    print("\nDataset Cohorts:")
    print_stats("Train", df_train)
    print_stats("Val (Internal)", df_val_internal)
    print_stats("Test (Temporal)", df_time_test)
    print_stats("Test (Spatial)", df_space_test)
    
    # 4. Save to disk
    print("\nSaving matricies...")
    for name, d in zip(
        ["train", "val_internal", "test_time", "test_space"], 
        [df_train, df_val_internal, df_time_test, df_space_test]
    ):
        X = d[FEATURES]
        y = d["died_under5"].astype(int)
        
        # Persist the DHS sample weight for descriptive cross-checks (e.g. so
        # we can confirm the cohort's survey-weighted prevalence). Per D010
        # the ML pipeline does NOT use this column for training or for
        # headline discrimination/calibration metrics.
        y_df = pd.DataFrame({"y": y, "weight": d["sample_weight"].fillna(1.0)})
        
        X.to_parquet(ML_DIR / f"X_{name}.parquet", index=False)
        y_df.to_parquet(ML_DIR / f"y_{name}.parquet", index=False)
        
    print(f"\nMatrices saved to {ML_DIR}")


if __name__ == "__main__":
    main()
