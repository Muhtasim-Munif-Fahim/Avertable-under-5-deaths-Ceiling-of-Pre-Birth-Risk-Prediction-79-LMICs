"""
s07_xgboost_training.py - Train Spatio-Temporal Prediction Model
=================================================================

Trains an XGBoost classifier to predict Under-5 Mortality.
Uses the Spatio-Temporal validation framework to prove robustness
against class imbalance and concept drift.
Evaluates model calibration (Brier Score) and precision (AUPRC),
which are standard requirements for Nature Medicine / Lancet Digital.

Outputs:
  - output/ml/xgb_model.json
  - output/ml/validation_metrics.csv
"""

import sys
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score, average_precision_score, 
    brier_score_loss, f1_score
)

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent.parent)
)
from src.utils.print_util import print
from config.paths import OUTPUT_DIR, RNG_SEED

ML_DIR = OUTPUT_DIR / "ml"


def load_split(name: str):
    """Load X, y, and the persisted DHS sample weight (kept for descriptive
    reporting only). Per decision D010 the ML pipeline does NOT use DHS
    sample weights for training or for headline discrimination/calibration
    metrics: the prediction objective is conditional on the joint feature
    distribution of the deployment population, not the survey-weighted
    population mean. The weight vector is loaded and returned only so it
    can be inspected by downstream diagnostic scripts.
    """
    X = pd.read_parquet(ML_DIR / f"X_{name}.parquet")
    # Force float32 for all columns for memory efficiency and xgboost
    for col in X.columns:
        if X[col].dtype.name == 'category':
            X[col] = X[col].cat.codes.astype("float32")
            X.loc[X[col] == -1, col] = np.nan
        else:
            X[col] = X[col].astype("float32")

    y_df = pd.read_parquet(ML_DIR / f"y_{name}.parquet")
    w = (y_df["weight"].values if "weight" in y_df.columns else
         np.ones(len(y_df), dtype="float32"))
    return X, y_df["y"].values, w


def evaluate(y_true, y_prob, weights, name="Cohort"):
    """Evaluate classifier using medical benchmark metrics.

    All headline metrics (AUROC, AUPRC, Brier) are UNWEIGHTED per
    decision~D010. The `weights` argument is accepted only for API
    compatibility with the rest of the pipeline; it is not used.
    """
    del weights  # decision D010: do not apply DHS sample weights to ML metrics
    auc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)

    # F1 score at the cohort prevalence threshold (unweighted)
    prevalence = float(np.mean(y_true))
    y_pred = (y_prob >= prevalence).astype(int)
    tp = int(np.sum(y_true * y_pred))
    fp = int(np.sum((1 - y_true) * y_pred))
    fn = int(np.sum(y_true * (1 - y_pred)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    print(f"  {name:15}: AUC={auc:.3f} | AUPRC={auprc:.3f} | Brier={brier:.3f} | F1(@prev)={f1:.3f}")
    return {"Cohort": name, "AUC": auc, "AUPRC": auprc, "Brier": brier, "F1": f1}


def main():
    print("=" * 60)
    print("Phase 4: XGBoost Spatio-Temporal Training")
    print("=" * 60)

    print("Loading datasets...")
    X_train, y_train, w_train = load_split("train")
    X_val_i, y_val_i, w_val_i = load_split("val_internal")
    X_test_t, y_test_t, w_test_t = load_split("test_time")
    X_test_s, y_test_s, w_test_s = load_split("test_space")
    
    print(f"  Train:         X={X_train.shape}, y={y_train.shape}")
    print(f"  Internal Val:  X={X_val_i.shape}, y={y_val_i.shape}")
    print(f"  Temporal Test: X={X_test_t.shape}, y={y_test_t.shape}")
    print(f"  Spatial Test:  X={X_test_s.shape}, y={y_test_s.shape}")

    pos_cases = np.sum(y_train == 1)
    neg_cases = np.sum(y_train == 0)
    spw = min(neg_cases / pos_cases, 10.0) 
    
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

    print(f"\nTraining base XGBoost model (scale_pos_weight={spw:.1f}, no DHS weights per D010)...")

    # D010: DHS sample weights are not used in ML training. Class imbalance
    # is handled via scale_pos_weight only. Removing the design weight makes
    # the predicted-probability scale consistent with the joint feature
    # distribution of the deployment population (the conditional prediction
    # objective the manuscript reports), and is a precondition for the
    # post-hoc Platt / isotonic recalibrator built by s09a_recalibration.py.
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval_i = xgb.DMatrix(X_val_i, label=y_val_i)
    
    evals = [(dtrain, "train"), (dval_i, "val_internal")]
    
    model = xgb.train(
        params, 
        dtrain, 
        num_boost_round=500,
        evals=evals,
        early_stopping_rounds=50,
        verbose_eval=50
    )
    
    out_model = ML_DIR / "xgb_model.json"
    model.save_model(str(out_model))
    print(f"\nModel saved to {out_model.name}")
    
    print("\nEvaluating cohorts:")
    res = []
    
    p_train = model.predict(dtrain)
    res.append(evaluate(y_train, p_train, w_train, "Train"))
    
    p_val_i = model.predict(dval_i)
    res.append(evaluate(y_val_i, p_val_i, w_val_i, "Internal Val"))
    
    dtest_t = xgb.DMatrix(X_test_t, label=y_test_t)
    p_test_t = model.predict(dtest_t)
    res.append(evaluate(y_test_t, p_test_t, w_test_t, "Temporal Test"))

    dtest_s = xgb.DMatrix(X_test_s, label=y_test_s)
    p_test_s = model.predict(dtest_s)
    res.append(evaluate(y_test_s, p_test_s, w_test_s, "Spatial Test"))
    
    # Save metrics
    metrics_df = pd.DataFrame(res)
    out_metrics = ML_DIR / "validation_metrics.csv"
    metrics_df.to_csv(out_metrics, index=False)
    
    # Print feature importance top 10
    print("\nTop 10 Important Features (Gain):")
    importance = model.get_score(importance_type="gain")
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
    for feat, score in sorted_imp:
        print(f"  {feat:25}: {score:.2f}")


if __name__ == "__main__":
    main()
