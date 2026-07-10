"""
s06b_loro.py — Leave-One-Region-Out (LORO) Sensitivity
=======================================================

Trains a separate XGBoost model for each fold where one entire WHO
region is held out as spatial test data. Quantifies cross-region
generalizability beyond the main single-holdout analysis.

5 folds: SSA, SSEA, LAC, ECA, MENA.
Each fold: train on all other regions (years <=2015), test on held-out
region (all years, within primary scope).

Outputs:
  output/ml/loro/loro_results.csv
  output/figures/figure_15_loro_auc.png

Runtime: ~30-90 min (5 x XGBoost retrains on ~1M rows each)
"""
import sys, gc
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb
from sklearn.metrics import roc_auc_score, brier_score_loss
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.print_util import print
from config.paths import OUTPUT_DIR, GLOBAL_PARQUET, RNG_SEED
from src.ml.s06_ml_data_prep import BANNED_MAIN_ML_FEATURES, FEATURES

ML_DIR = OUTPUT_DIR / "ml"
LORO_DIR = ML_DIR / "loro"
LORO_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = OUTPUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SEED = RNG_SEED
META_COLS = ["died_under5", "survey_year", "country_code",
             "who_region", "sample_weight", "in_primary_scope"]


def load_sample() -> pd.DataFrame:
    """Load from cached parquet (built by the pre-run above)."""
    cache = LORO_DIR / "loro_data_cache.parquet"
    if cache.exists():
        data = pd.read_parquet(cache)
        required = set(META_COLS) - {"in_primary_scope"} | set(FEATURES)
        stale_reasons = []
        if not required.issubset(data.columns):
            stale_reasons.append("missing current columns")
        if set(data.columns) & BANNED_MAIN_ML_FEATURES:
            stale_reasons.append("contains banned main-ML features")
        if "in_primary_scope" in data.columns:
            data = data[data["in_primary_scope"] == 1].drop(
                columns=["in_primary_scope"]
            )
            data.to_parquet(cache, index=False)
        if stale_reasons:
            print(f"  Rebuilding stale cache: {', '.join(stale_reasons)}")
            cache.unlink()
        else:
            print(f"  Loaded {len(data):,} rows from cache")
            return data
    # Fallback: build cache from scratch (float32).
    # LORO uses every-14th row (~7.1% systematic sample) -- intentionally
    # 2x coarser than the main ML pipeline's every-7th step (s06,
    # ML_SAMPLING_STEP=7, ~14.2%). LORO retrains the model five times (once
    # per held-out region), so the coarser sample keeps total compute
    # tractable; it is a supplementary external-validation table, not a
    # headline metric. Sampling rate is documented here and in DECISIONS.md.
    LORO_SAMPLING_STEP = 14  # == 2 * s06 ML_SAMPLING_STEP
    pf = pq.ParquetFile(str(GLOBAL_PARQUET))
    n_total = pf.metadata.num_rows
    indices = set(range(0, n_total, LORO_SAMPLING_STEP))
    frames = []
    row_offset = 0
    for batch in pf.iter_batches(
            batch_size=200_000, columns=META_COLS + FEATURES):
        df = batch.to_pandas()
        n = len(df)
        local_idx = [i - row_offset for i in indices
                     if row_offset <= i < row_offset + n]
        if local_idx:
            sub = df.iloc[local_idx].copy()
            for col in sub.columns:
                if sub[col].dtype == "float64":
                    sub[col] = sub[col].astype("float32")
            frames.append(sub)
        row_offset += n
        del df, batch
    gc.collect()
    data = pd.concat(frames, ignore_index=True)
    del frames
    data = data[data["in_primary_scope"] == 1].drop(
        columns=["in_primary_scope"])
    data = data.dropna(subset=["died_under5"]).copy()
    data.to_parquet(cache, index=False)
    print(f"  Loaded {len(data):,} rows (saved cache)")
    return data


def train_predict(df_train, df_test):
    X_tr = df_train[FEATURES].copy()
    y_tr = df_train["died_under5"].astype(int).values
    X_te = df_test[FEATURES].copy()
    y_te = df_test["died_under5"].astype(int).values

    for col in X_tr.columns:
        if X_tr[col].dtype.name == "category":
            X_tr[col] = X_tr[col].cat.codes.astype("float32")
            X_tr.loc[X_tr[col] == -1, col] = np.nan
            X_te[col] = X_te[col].cat.codes.astype("float32")
            X_te.loc[X_te[col] == -1, col] = np.nan
        else:
            X_tr[col] = X_tr[col].astype("float32")
            X_te[col] = X_te[col].astype("float32")

    # D010: ML training does not apply DHS sample weights. Class imbalance
    # is handled through scale_pos_weight only.
    scale = float(np.sum(y_tr == 0)) / float(np.sum(y_tr == 1))
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dtest = xgb.DMatrix(X_te, label=y_te)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": scale,
        "seed": SEED,
        "tree_method": "hist",
        "device": "cpu",
    }
    model = xgb.train(
        params, dtrain,
        num_boost_round=500,
        verbose_eval=False,
    )
    p = model.predict(dtest)
    return y_te, p


def main():
    print("=" * 60)
    print("LORO — Leave-One-Region-Out Sensitivity")
    print("=" * 60)

    print("\nLoading data sample...")
    data = load_sample()

    regions = sorted(data["who_region"].dropna().unique())
    print(f"\nRegions: {regions}")

    # Reference AUC from main model on spatial holdout
    ref_auc = None
    try:
        X_ts = pd.read_parquet(ML_DIR / "X_test_space.parquet")
        y_ts_df = pd.read_parquet(ML_DIR / "y_test_space.parquet")
        for col in X_ts.columns:
            if X_ts[col].dtype.name == "category":
                X_ts[col] = X_ts[col].cat.codes.astype("float32")
                X_ts.loc[X_ts[col] == -1, col] = np.nan
            else:
                X_ts[col] = X_ts[col].astype("float32")
        y_ts = y_ts_df["y"].astype(int).values if "y" in y_ts_df.columns else y_ts_df["died_under5"].astype(int).values
        booster = xgb.Booster()
        booster.load_model(str(ML_DIR / "xgb_model.json"))
        ref_auc = float(roc_auc_score(y_ts, booster.predict(xgb.DMatrix(X_ts))))
        print(f"\nReference AUC (main spatial holdout): {ref_auc:.4f}")
    except Exception as e:
        print(f"  Could not load reference model: {e}")

    rows = []
    for region in regions:
        print(f"\n--- LORO fold: hold out {region} ---")
        mask_held = data["who_region"] == region
        df_held = data[mask_held]
        df_rest = data[~mask_held]

        # Training: all-but-held-out, years <= 2015
        df_train = df_rest[df_rest["survey_year"] <= 2015]
        n_events = int(df_train["died_under5"].sum())
        print(f"  Train: {len(df_train):,} rows, {n_events:,} events")
        print(f"  Test (held-out {region}): {len(df_held):,} rows, "
              f"{int(df_held['died_under5'].sum()):,} events")

        if len(df_train) < 1000 or n_events < 100:
            print(f"  SKIP: insufficient training data")
            continue
        if df_held["died_under5"].sum() < 20:
            print(f"  SKIP: insufficient test events")
            continue

        y_te, p = train_predict(df_train, df_held)
        auc = float(roc_auc_score(y_te, p))
        brier = float(brier_score_loss(y_te, p))
        print(f"  AUC = {auc:.4f}, Brier = {brier:.4f}")

        rows.append({
            "held_out_region": region,
            "n_train": len(df_train),
            "n_test": len(df_held),
            "n_test_events": int(df_held["died_under5"].sum()),
            "test_prevalence": float(df_held["died_under5"].mean()),
            "auc": auc,
            "brier": brier,
        })

    if not rows:
        print("No LORO results produced.")
        return

    results_df = pd.DataFrame(rows)
    out_csv = LORO_DIR / "loro_results.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")
    print(results_df.to_string(index=False))

    # Figure
    fig, ax = plt.subplots(figsize=(6, 4))
    x = range(len(results_df))
    ax.bar(x, results_df["auc"], color="#377eb8", alpha=0.8, width=0.6)
    if ref_auc is not None:
        ax.axhline(ref_auc, color="#e41a1c", linestyle="--",
                   linewidth=1.5, label=f"Main model (ref AUC={ref_auc:.3f})")
        ax.legend(fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels(results_df["held_out_region"], rotation=20, ha="right")
    ax.set_ylabel("AUC on held-out region")
    ax.set_title("LORO: Cross-Region Generalizability")
    ax.set_ylim(0.5, 0.85)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_fig = FIG_DIR / "figure_15_loro_auc.png"
    fig.savefig(out_fig, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_fig.name}")


if __name__ == "__main__":
    main()
