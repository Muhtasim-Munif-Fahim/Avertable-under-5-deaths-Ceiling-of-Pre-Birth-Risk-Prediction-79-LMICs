"""
s07e_prebirth_only_sensitivity.py - Strict pre-birth-only feature ablation
==========================================================================

Reviewer sensitivity (R3.1, Lancet GH review 2026-06-13): the main ML model
retains four characteristics measured at interview rather than strictly before
birth (wealth quintile, urban residence, current marital status, household
size). This script quantifies how much discrimination is lost when those four
are dropped, leaving a strictly pre-birth feature set of six.

It is DELIBERATELY non-destructive:
  - It reuses the canonical train/holdout matrices saved by s06_ml_data_prep.py
    (output/ml/X_*.parquet), so the splits are identical to the production run.
  - It trains with the EXACT production hyperparameters from
    s07_xgboost_training.py.
  - It writes ONLY output/ml/sensitivity_prebirth_only.csv. It does not touch
    xgb_model.json, validation_metrics.csv, or any other canonical artefact.

The FULL (10-feature) arm is retrained here as a sanity check: it should land on
the published baseline (AUROC ~0.698 temporal / ~0.714 spatial). The headline
output is the delta between FULL and PRE-BIRTH-ONLY under an identical recipe.

Run:  python -m src.ml.s07e_prebirth_only_sensitivity
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.paths import OUTPUT_DIR, RNG_SEED

ML_DIR = OUTPUT_DIR / "ml"

# The 4 survey-time characteristics measured at interview, not before birth.
SURVEY_TIME_FEATURES = ["wealth_quintile", "urban", "currently_married", "hh_members"]


def load_split(name: str):
    """Identical to s07_xgboost_training.load_split (categories -> codes,
    float32), minus the unused weight return."""
    X = pd.read_parquet(ML_DIR / f"X_{name}.parquet")
    for col in X.columns:
        if X[col].dtype.name == "category":
            X[col] = X[col].cat.codes.astype("float32")
            X.loc[X[col] == -1, col] = np.nan
        else:
            X[col] = X[col].astype("float32")
    y = pd.read_parquet(ML_DIR / f"y_{name}.parquet")["y"].values
    return X, y


def train_eval(cols, label):
    """Train XGBoost on `cols` with production params; return AUROC/AUPRC on
    both external holdouts."""
    Xtr, ytr = load_split("train")
    Xvi, yvi = load_split("val_internal")
    Xtt, ytt = load_split("test_time")
    Xts, yts = load_split("test_space")
    Xtr, Xvi, Xtt, Xts = (df[cols] for df in (Xtr, Xvi, Xtt, Xts))

    spw = min(np.sum(ytr == 0) / np.sum(ytr == 1), 10.0)
    params = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "aucpr"],
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": spw,
        "tree_method": "hist",
        "n_jobs": 4,
        "random_state": RNG_SEED,
        "seed": RNG_SEED,
    }
    dtrain = xgb.DMatrix(Xtr, label=ytr)
    dval = xgb.DMatrix(Xvi, label=yvi)
    model = xgb.train(
        params, dtrain, num_boost_round=500,
        evals=[(dtrain, "train"), (dval, "val_internal")],
        early_stopping_rounds=50, verbose_eval=False,
    )

    out = {"feature_set": label, "n_features": len(cols)}
    for cohort, X, y in [("temporal", Xtt, ytt), ("geographic", Xts, yts)]:
        p = model.predict(xgb.DMatrix(X, label=y))
        out[f"AUROC_{cohort}"] = round(float(roc_auc_score(y, p)), 4)
        out[f"AUPRC_{cohort}"] = round(float(average_precision_score(y, p)), 4)
    print(f"  {label:16} ({len(cols)} feats): "
          f"AUROC temporal={out['AUROC_temporal']:.4f}, "
          f"geographic={out['AUROC_geographic']:.4f}")
    return out


def main():
    print("=" * 64)
    print("R3.1 sensitivity: strict pre-birth-only feature ablation")
    print("=" * 64)

    full_cols = list(pd.read_parquet(ML_DIR / "X_train.parquet").columns)
    prebirth_cols = [c for c in full_cols if c not in SURVEY_TIME_FEATURES]
    print(f"Full feature set ({len(full_cols)}): {full_cols}")
    print(f"Dropped (survey-time): {SURVEY_TIME_FEATURES}")
    print(f"Pre-birth-only set ({len(prebirth_cols)}): {prebirth_cols}\n")

    rows = [
        train_eval(full_cols, "full"),
        train_eval(prebirth_cols, "prebirth_only"),
    ]
    df = pd.DataFrame(rows)

    # Delta row (full - prebirth_only)
    f, p = df.iloc[0], df.iloc[1]
    print("\nDelta (full - pre-birth-only):")
    for c in ["temporal", "geographic"]:
        d = f[f"AUROC_{c}"] - p[f"AUROC_{c}"]
        print(f"  AUROC {c:11}: {f[f'AUROC_{c}']:.4f} -> {p[f'AUROC_{c}']:.4f}  (delta {d:+.4f})")

    out_csv = ML_DIR / "sensitivity_prebirth_only.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved -> {out_csv}")


if __name__ == "__main__":
    main()
