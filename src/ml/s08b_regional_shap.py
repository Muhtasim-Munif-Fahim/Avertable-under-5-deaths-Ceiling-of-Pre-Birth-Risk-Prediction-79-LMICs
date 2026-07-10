"""
s08b_regional_shap.py — Regional SHAP Comparison
==================================================

Generates SHAP summary plots stratified by WHO region to show
how determinant rankings shift between SSA, South/SE Asia, 
Latin America, and MENA. This is a key flagship finding that
demonstrates contextual heterogeneity in risk factor importance.

Outputs:
  - output/figures/figure_12_regional_shap.png

References
----------
  - Improvement Plan W4: Regional SHAP comparison
"""

import sys
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent.parent)
)
from src.utils.print_util import print
from config.paths import OUTPUT_DIR, GLOBAL_PARQUET, RNG_SEED
from src.ml.s06_ml_data_prep import (
    BANNED_MAIN_ML_FEATURES, FEATURES, ML_SAMPLING_STEP,
)

ML_DIR = OUTPUT_DIR / "ml"
FIG_DIR = OUTPUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Lancet-tier figure styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.labelsize": 10,
    "axes.titlesize": 12,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
})

RENAME_MAP = {
    "child_male": "Male Child",
    "multiple_birth": "Multiple Birth",
    "birth_order": "Birth Order",
    "birth_interval_months": "Birth Interval (mo)",
    "maternal_age_at_birth": "Maternal Age",
    "edu_level": "Maternal Education",
    "currently_married": "Married",
    "wealth_quintile": "Wealth Quintile",
    "urban": "Urban Residence",
    "hh_members": "Household Members",
}


# Must match s06_ml_data_prep.FEATURES. Timing-sensitive healthcare,
# delivery, birth-size, and maternal anthropometry variables are not allowed
# in the publication main ML model.

# WHO region labels
REGION_LABELS = {
    "SSA": "Sub-Saharan\nAfrica",
    "SSEA": "South/SE\nAsia",
    "LAC": "Latin America\n& Caribbean",
    "MENA": "Middle East\n& N. Africa",
}


def load_regional_data(n_per_region: int = 5000):
    """Load a stratified sample from temporal and spatial holdout sets with
    region labels from the harmonized dataset.
    
    Parameters
    ----------
    n_per_region : int
        Max samples per WHO region for SHAP computation.
    
    Returns
    -------
    dict : region -> (X_sample, region_label)
    """
    def load_x(split: str) -> pd.DataFrame:
        X = pd.read_parquet(ML_DIR / f"X_{split}.parquet")
        if list(X.columns) != FEATURES:
            raise ValueError(
                f"X_{split}.parquet feature schema does not match the "
                "leakage-screened main ML FEATURES."
            )
        banned = sorted(set(X.columns) & BANNED_MAIN_ML_FEATURES)
        if banned:
            raise ValueError(
                f"Leaky features found in regional SHAP input {split}: {banned}"
            )
        for col in X.columns:
            if X[col].dtype.name == "category":
                X[col] = X[col].cat.codes.astype("float32")
                X.loc[X[col] == -1, col] = np.nan
            else:
                X[col] = X[col].astype("float32")
        return X

    X_time = load_x("test_time")
    X_space = load_x("test_space")

    # We need region labels. Load country_code + who_region from
    # the full dataset for the same indices used in ML prep.
    # The ML splits don't include region, so we load from source.
    pf = pq.ParquetFile(str(GLOBAL_PARQUET))
    n_total = pf.metadata.num_rows
    sample_indices = set(range(0, n_total, ML_SAMPLING_STEP))  # match s06

    meta_cols = [
        "country_code",
        "who_region",
        "survey_year",
        "died_under5",
        "in_primary_scope",
    ]
    frames = []
    row_offset = 0
    for batch in pf.iter_batches(batch_size=500_000, columns=meta_cols):
        batch_df = batch.to_pandas()
        batch_len = len(batch_df)
        local_idx = [i - row_offset for i in sample_indices
                     if row_offset <= i < row_offset + batch_len]
        if local_idx:
            frames.append(batch_df.iloc[local_idx])
        row_offset += batch_len
        del batch_df, batch

    meta = pd.concat(frames, ignore_index=True)
    del frames
    gc.collect()

    meta = meta.dropna(subset=["died_under5"])
    meta = meta[meta["in_primary_scope"] == 1].reset_index(drop=True)

    import json
    splits_path = ML_DIR / "country_splits.json"
    with open(splits_path) as f:
        splits = json.load(f)
    train_countries = set(splits["training_countries"])
    holdout_countries = set(splits["spatial_holdout_countries"])

    mask_time = (
        (meta["survey_year"] >= 2019) &
        (meta["country_code"].isin(train_countries))
    )
    mask_space = meta["country_code"].isin(holdout_countries)
    meta_time = meta[mask_time].reset_index(drop=True)
    meta_space = meta[mask_space].reset_index(drop=True)

    min_time = min(len(X_time), len(meta_time))
    min_space = min(len(X_space), len(meta_space))
    X_test = pd.concat(
        [X_time.iloc[:min_time], X_space.iloc[:min_space]],
        ignore_index=True,
    )
    meta_test = pd.concat(
        [meta_time.iloc[:min_time], meta_space.iloc[:min_space]],
        ignore_index=True,
    )

    result = {}
    np.random.seed(RNG_SEED)
    for region in REGION_LABELS:
        region_mask = meta_test["who_region"] == region
        X_region = X_test[region_mask.values]
        if len(X_region) < 100:
            print(f"  Skipping {region}: only {len(X_region)} rows")
            continue
        if len(X_region) > n_per_region:
            idx = np.random.choice(
                len(X_region), size=n_per_region, replace=False
            )
            X_region = X_region.iloc[idx]
        result[region] = X_region
        print(f"  {region}: {len(X_region)} samples")

    return result


def main():
    print("=" * 60)
    print("Regional SHAP Comparison")
    print("=" * 60)

    model_path = ML_DIR / "xgb_model.json"
    if not model_path.exists():
        print("ERROR: xgb_model.json not found")
        return

    print("Loading XGBoost model...")
    model = xgb.Booster()
    model.load_model(str(model_path))

    print("Loading regional samples...")
    regional_data = load_regional_data(n_per_region=5000)

    if len(regional_data) < 2:
        print("ERROR: Need at least 2 regions for comparison")
        return

    print("Computing SHAP values per region...")
    explainer = shap.TreeExplainer(model)

    # Compute SHAP for each region
    regional_shap = {}
    for region, X in regional_data.items():
        print(f"  Computing SHAP for {region}...")
        sv = explainer.shap_values(X)
        regional_shap[region] = (sv, X)

    # Generate mean |SHAP| ranking table per region. This replaces the
    # previous oversized beeswarm panels with a compact supplemental display.
    rows = []
    for region, (sv, X) in regional_shap.items():
        mean_abs = np.abs(sv).mean(axis=0)
        for feat, val in zip(X.columns, mean_abs):
            rows.append({
                "Region": region, "Feature": RENAME_MAP.get(feat, feat),
                "Mean_abs_SHAP": round(float(val), 6),
            })
    rank_df = pd.DataFrame(rows)
    rank_out = ML_DIR / "regional_shap_rankings.csv"
    rank_df.to_csv(rank_out, index=False)
    print(f"Saved: {rank_out.name}")

    pivot = rank_df.pivot_table(
        index="Feature", columns="Region", values="Mean_abs_SHAP",
        aggfunc="mean"
    ).fillna(0)
    top_features = (
        pivot.mean(axis=1)
        .sort_values(ascending=False)
        .head(10)
        .index
    )
    plot_df = (
        pivot.loc[top_features]
        .reset_index()
        .melt(id_vars="Feature", var_name="Region", value_name="Mean_abs_SHAP")
    )
    feature_order = list(reversed(top_features))
    y_lookup = {feature: i for i, feature in enumerate(feature_order)}
    x_lookup = {region: i for i, region in enumerate(pivot.columns)}

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for _, row in plot_df.iterrows():
        ax.scatter(
            x_lookup[row["Region"]],
            y_lookup[row["Feature"]],
            s=35 + row["Mean_abs_SHAP"] * 250,
            color="#2166ac",
            alpha=0.72,
            edgecolor="white",
            linewidth=0.4,
        )
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(
        [REGION_LABELS.get(region, region).replace("\n", " ")
         for region in pivot.columns],
        rotation=20,
        ha="right",
    )
    ax.set_yticks(range(len(feature_order)))
    ax.set_yticklabels(feature_order)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color="#d8dee3", lw=0.5)
    plt.tight_layout()
    out = FIG_DIR / "figure_12_regional_shap.png"
    plt.savefig(out, dpi=600)
    plt.close()
    print(f"Saved: {out.name}")

    # Print top-5 per region
    print("\n--- Top 5 Features by Region ---")
    for region in regional_shap:
        sub = rank_df[rank_df["Region"] == region].nlargest(
            5, "Mean_abs_SHAP"
        )
        feats = ", ".join(sub["Feature"].values)
        print(f"  {region}: {feats}")

    print("\nDONE.")


if __name__ == "__main__":
    main()
