"""
s06c_loro_percountry.py — Full-data Leave-One-Region-Out predictions
=====================================================================

Produces genuine OUT-OF-REGION predictions for EVERY child, so that the
per-country risk-stratification supplement (s09f) can cover all 79
countries — including the 33 training countries whose latest survey
predates 2019 and therefore never enter the pooled model's temporal/
spatial holdout (see s09e).

Design (a disaggregation of the existing region-level LORO, s06b):
  For each of the 5 WHO regions R:
    - train  = all OTHER regions, survey_year <= 2015   (temporal-honest)
    - val    = all OTHER regions, 2016 <= year <= 2018  (recalibration fit)
    - test   = region R, ALL years (primary scope)      (out-of-region)
  Each child in region R is therefore scored exactly once, by a model
  that never saw its region. Concatenated across the 5 folds this yields
  a complete prediction vector over all in-scope countries.

XGBoost recipe is identical to s06b (max_depth 6, lr 0.05, 500 rounds,
scale_pos_weight, hist, no DHS weights per D010). Raw predictions are
isotonic-recalibrated on the held-in val slice so calibration slope is
meaningful; ranking metrics (AUROC, quintile stratification) use raw and
are invariant to the monotone recalibration.

Full data, run SOLO (never concurrently with another full-data batch —
OOM). Trains 5 XGBoost models on multi-million-row matrices.

Output:
  output/ml/loro/loro_percountry_predictions.parquet
    columns: country_code, who_region, y, p_raw, p_iso

Run:
  set ML_FULL_SAMPLE=1 && python -m src.ml.s06c_loro_percountry   (cmd)
  $env:ML_FULL_SAMPLE=1; python -m src.ml.s06c_loro_percountry    (pwsh)
"""
from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.print_util import print
from config.paths import OUTPUT_DIR, GLOBAL_PARQUET, RNG_SEED
from src.ml.s06_ml_data_prep import FEATURES, BANNED_MAIN_ML_FEATURES

ML_DIR = OUTPUT_DIR / "ml"
LORO_DIR = ML_DIR / "loro"
LORO_DIR.mkdir(parents=True, exist_ok=True)
OUT_PARQUET = LORO_DIR / "loro_percountry_predictions.parquet"

SEED = RNG_SEED
META_COLS = ["died_under5", "survey_year", "country_code",
             "who_region", "in_primary_scope"]
REGION_ORDER = ["SSA", "SSEA", "MENA", "LAC", "ECA"]


def load_full() -> pd.DataFrame:
    """Stream the full parquet into a float32 frame (primary scope, valid
    outcome, non-null region). No subsample."""
    assert not (set(FEATURES) & BANNED_MAIN_ML_FEATURES), \
        "FEATURES must not contain banned leakage-prone variables"
    pf = pq.ParquetFile(str(GLOBAL_PARQUET))
    frames = []
    for batch in pf.iter_batches(batch_size=500_000,
                                 columns=META_COLS + FEATURES):
        df = batch.to_pandas()
        df = df.dropna(subset=["died_under5"])
        df = df[df["in_primary_scope"] == 1]
        df = df.dropna(subset=["who_region"])
        for col in FEATURES:
            if df[col].dtype.name == "category":
                codes = df[col].cat.codes.astype("float32")
                codes[codes == -1] = np.nan
                df[col] = codes
            else:
                df[col] = df[col].astype("float32")
        df["died_under5"] = df["died_under5"].astype("int8")
        df["survey_year"] = df["survey_year"].fillna(0).astype("int16")
        df["country_code"] = df["country_code"].astype("category")
        df["who_region"] = df["who_region"].astype("category")
        frames.append(df.drop(columns=["in_primary_scope"]))
        del batch
    data = pd.concat(frames, ignore_index=True)
    del frames
    gc.collect()
    print(f"  Loaded {len(data):,} primary-scope rows, "
          f"{data['country_code'].nunique()} countries, "
          f"{int(data['died_under5'].sum()):,} deaths.")
    return data


def fit_fold(df_train: pd.DataFrame, df_val: pd.DataFrame,
             df_test: pd.DataFrame):
    """Train on df_train, recalibrate on df_val, predict df_test.
    Returns (p_raw, p_iso) for df_test rows."""
    X_tr = df_train[FEATURES].to_numpy(dtype="float32")
    y_tr = df_train["died_under5"].to_numpy().astype(int)
    scale = float(np.sum(y_tr == 0)) / float(np.sum(y_tr == 1))
    params = {
        "objective": "binary:logistic", "eval_metric": "auc",
        "max_depth": 6, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "scale_pos_weight": scale, "seed": SEED,
        "tree_method": "hist", "device": "cpu",
    }
    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=FEATURES)
    model = xgb.train(params, dtrain, num_boost_round=500, verbose_eval=False)
    del dtrain, X_tr
    gc.collect()

    def _predict(df):
        d = xgb.DMatrix(df[FEATURES].to_numpy(dtype="float32"),
                        feature_names=FEATURES)
        return model.predict(d)

    p_test_raw = _predict(df_test)
    # isotonic recalibration fit on held-in val
    if len(df_val) > 0 and df_val["died_under5"].sum() > 0:
        p_val_raw = _predict(df_val)
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p_val_raw, df_val["died_under5"].to_numpy().astype(int))
        p_test_iso = iso.transform(p_test_raw)
    else:
        p_test_iso = p_test_raw.copy()
    del model
    gc.collect()
    return p_test_raw, p_test_iso


def main():
    print("=" * 64)
    print("s06c — Full-data Leave-One-Region-Out per-country predictions")
    print("=" * 64)
    t0 = time.time()
    print("\nLoading full cohort...")
    data = load_full()

    present = set(data["who_region"].astype(str).unique())
    regions = [r for r in REGION_ORDER if r in present]
    regions += sorted(present - set(regions))
    print(f"Regions ({len(regions)}): {regions}")

    records = []
    for region in regions:
        is_R = (data["who_region"] == region).to_numpy()
        yr = data["survey_year"].to_numpy()
        df_test = data[is_R]
        df_train = data[(~is_R) & (yr <= 2015)]
        df_val = data[(~is_R) & (yr >= 2016) & (yr <= 2018)]
        n_te_ev = int(df_test["died_under5"].sum())
        print(f"\n--- Fold: hold out {region} ---")
        print(f"  train {len(df_train):,} rows / "
              f"{int(df_train['died_under5'].sum()):,} deaths | "
              f"val {len(df_val):,} | test {len(df_test):,} / {n_te_ev:,} deaths")
        if len(df_train) < 1000 or df_train["died_under5"].sum() < 100 \
                or n_te_ev < 20:
            print("  SKIP: insufficient data")
            continue

        ft0 = time.time()
        p_raw, p_iso = fit_fold(df_train, df_val, df_test)
        y_te = df_test["died_under5"].to_numpy().astype(int)
        auc = float(roc_auc_score(y_te, p_raw))
        print(f"  out-of-region AUROC = {auc:.4f}  ({time.time()-ft0:.0f}s)")

        records.append(pd.DataFrame({
            "country_code": df_test["country_code"].astype(str).to_numpy(),
            "who_region": df_test["who_region"].astype(str).to_numpy(),
            "y": y_te, "p_raw": p_raw, "p_iso": p_iso,
        }))
        del df_test, df_train, df_val, p_raw, p_iso, y_te
        gc.collect()

    if not records:
        print("No folds produced predictions.")
        return
    out = pd.concat(records, ignore_index=True)
    out.to_parquet(OUT_PARQUET, index=False)
    print(f"\nSaved {len(out):,} out-of-region predictions across "
          f"{out['country_code'].nunique()} countries -> {OUT_PARQUET}")
    print(f"Total runtime: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
