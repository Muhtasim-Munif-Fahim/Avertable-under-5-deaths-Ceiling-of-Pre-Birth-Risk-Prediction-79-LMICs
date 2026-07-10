"""
s07c_lr_baseline.py — Logistic Regression baseline + DeLong test
=================================================================

Trains a logistic regression on the same X_train/y_train as the
XGBoost model, evaluates on the temporal and spatial holdouts, and
compares AUCs against XGBoost using DeLong's test.

Outputs:
  output/ml/lr_baseline_metrics.csv
  output/ml/lr_vs_xgb_delong.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, brier_score_loss

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.print_util import print
from config.paths import OUTPUT_DIR, RNG_SEED

ML_DIR = OUTPUT_DIR / "ml"


def _load(name: str):
    X = pd.read_parquet(ML_DIR / f"X_{name}.parquet")
    y = pd.read_parquet(ML_DIR / f"y_{name}.parquet")
    for col in X.columns:
        if X[col].dtype.name == "category":
            X[col] = X[col].cat.codes.astype("float32")
            X.loc[X[col] == -1, col] = np.nan
        else:
            X[col] = X[col].astype("float32")
    y_col = "died_under5" if "died_under5" in y.columns else "y"
    w_col = "sample_weight" if "sample_weight" in y.columns else (
        "weight" if "weight" in y.columns else None)
    return (X, y[y_col].astype(int).values,
            y[w_col].fillna(1.0).values if w_col else None)


def delong_test(y, p1, p2):
    """DeLong test for paired AUCs.

    Returns (auc1, auc2, z, p_value). Uses fast O(n log n) algorithm
    of Sun & Xu (2014).
    """
    y = np.asarray(y, dtype=int)
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)

    pos = y == 1
    neg = y == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan, np.nan, np.nan, np.nan

    def _fast_auc_components(scores_pos, scores_neg):
        # Component-wise V10 / V01 using midrank.
        all_s = np.concatenate([scores_pos, scores_neg])
        all_r = _midrank(all_s)
        s_pos = scores_pos
        s_neg = scores_neg
        r_pos_in_pos = _midrank(s_pos)
        r_neg_in_neg = _midrank(s_neg)
        r_pos_in_all = all_r[:n_pos]
        r_neg_in_all = all_r[n_pos:]
        v10 = (r_pos_in_all - r_pos_in_pos) / n_neg
        v01 = 1.0 - (r_neg_in_all - r_neg_in_neg) / n_pos
        auc = (r_pos_in_all.sum() / n_pos - (n_pos + 1) / 2.0) / n_neg
        return auc, v10, v01

    auc1, v10_1, v01_1 = _fast_auc_components(p1[pos], p1[neg])
    auc2, v10_2, v01_2 = _fast_auc_components(p2[pos], p2[neg])
    s10 = np.cov(np.vstack([v10_1, v10_2]), bias=False)
    s01 = np.cov(np.vstack([v01_1, v01_2]), bias=False)
    sigma = s10 / n_pos + s01 / n_neg
    var = float(sigma[0, 0] - 2 * sigma[0, 1] + sigma[1, 1])
    if var <= 0:
        return float(auc1), float(auc2), np.nan, np.nan
    z = (auc1 - auc2) / np.sqrt(var)
    from scipy.stats import norm
    p = 2.0 * (1.0 - norm.cdf(abs(z)))
    return float(auc1), float(auc2), float(z), float(p)


def _midrank(x):
    order = np.argsort(x, kind="mergesort")
    x_sorted = x[order]
    n = len(x)
    rank = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n - 1 and x_sorted[j + 1] == x_sorted[i]:
            j += 1
        avg = 0.5 * (i + j) + 1
        rank[i:j + 1] = avg
        i = j + 1
    out = np.empty(n, dtype=float)
    out[order] = rank
    return out


def main():
    print("=" * 60)
    print("Logistic Regression baseline + DeLong vs XGBoost")
    print("=" * 60)

    X_tr, y_tr, w_tr = _load("train")
    print(f"Train: {len(y_tr):,} rows, {y_tr.sum():,} events "
          f"({y_tr.mean()*100:.2f}%)")

    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(
            solver="lbfgs", max_iter=2000,
            random_state=RNG_SEED,
        )),
    ])
    print("Fitting LR (no DHS sample weights per D010)...")
    pipe.fit(X_tr, y_tr)
    print("  done.")

    booster = xgb.Booster()
    booster.load_model(str(ML_DIR / "xgb_model.json"))

    rows_metrics = []
    rows_delong = []
    for split in ["test_time", "test_space"]:
        try:
            X, y, _ = _load(split)
        except FileNotFoundError as e:
            print(f"  Skipping {split}: {e}")
            continue
        p_lr = pipe.predict_proba(X)[:, 1]
        p_xgb = booster.predict(xgb.DMatrix(X))

        auc_lr = float(roc_auc_score(y, p_lr))
        auc_xgb = float(roc_auc_score(y, p_xgb))
        rows_metrics.append({
            "split": split, "model": "LR",
            "n": len(y), "auc": auc_lr,
            "brier": float(brier_score_loss(y, p_lr)),
        })
        rows_metrics.append({
            "split": split, "model": "XGBoost",
            "n": len(y), "auc": auc_xgb,
            "brier": float(brier_score_loss(y, p_xgb)),
        })

        auc1, auc2, z, p = delong_test(y, p_xgb, p_lr)
        rows_delong.append({
            "split": split,
            "auc_xgb": auc1,
            "auc_lr": auc2,
            "delta": auc1 - auc2,
            "z": z,
            "p_value": p,
        })
        print(f"  {split}: AUC LR={auc_lr:.4f}, "
              f"XGB={auc_xgb:.4f}, dAUC={auc_xgb-auc_lr:+.4f}, "
              f"DeLong p={p:.2e}")

    metrics_df = pd.DataFrame(rows_metrics)
    delong_df = pd.DataFrame(rows_delong)
    metrics_df.to_csv(ML_DIR / "lr_baseline_metrics.csv", index=False)
    delong_df.to_csv(ML_DIR / "lr_vs_xgb_delong.csv", index=False)
    print(f"\nSaved: lr_baseline_metrics.csv, lr_vs_xgb_delong.csv")


if __name__ == "__main__":
    main()
