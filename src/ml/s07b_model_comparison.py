"""
s07b_model_comparison.py — Multi-Model Benchmark Table
=======================================================

Trains Logistic Regression and LightGBM alongside the existing
XGBoost model to produce a model comparison table. This is a
mandatory requirement for Nature Medicine / Lancet Digital Health
to justify the chosen architecture.

Uses the same 4-way Spatio-Temporal splits from s06_ml_data_prep.

Outputs:
  - output/ml/model_comparison.csv
  - output/figures/figure_9_roc_curves.png
  - output/figures/figure_10_pr_curves.png
  - output/figures/figure_11_calibration.png

References
----------
  - TRIPOD+AI: requires comparison against a baseline model
  - Improvement Plan items F5, W2, M5
"""

import sys
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    brier_score_loss, roc_curve, precision_recall_curve,
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent.parent)
)
from src.utils.print_util import print
from config.paths import OUTPUT_DIR, RNG_SEED

ML_DIR = OUTPUT_DIR / "ml"
FIG_DIR = OUTPUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Lancet-tier figure styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.labelsize": 11,
    "axes.titlesize": 13,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
})

# Color palette (accessible, colorblind-safe)
COLORS = {
    "Train": "#636363",
    "Internal Val": "#3182bd",
    "Temporal Test": "#e6550d",
    "Spatial Test": "#31a354",
}


def load_split(name: str):
    """Load X and y with weights, imputing NaN for LR."""
    X = pd.read_parquet(ML_DIR / f"X_{name}.parquet")
    for col in X.columns:
        if X[col].dtype.name == "category":
            X[col] = X[col].cat.codes.astype("float32")
            X.loc[X[col] == -1, col] = np.nan
        else:
            X[col] = X[col].astype("float32")
    y_df = pd.read_parquet(ML_DIR / f"y_{name}.parquet")
    return X, y_df["y"].values, y_df["weight"].values


def evaluate(y_true, y_prob, weights, name="Cohort"):
    """Compute standard medical ML metrics. Per D010 the ML pipeline does
    NOT apply DHS sample weights for headline metrics; the `weights`
    argument is accepted for API compatibility only.
    """
    del weights  # decision D010
    auc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)

    prevalence = float(np.mean(y_true))
    y_pred = (y_prob >= prevalence).astype(int)
    tp = int(np.sum(y_true * y_pred))
    fp = int(np.sum((1 - y_true) * y_pred))
    fn = int(np.sum(y_true * (1 - y_pred)))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    return {
        "Cohort": name, "AUC": round(auc, 4),
        "AUPRC": round(auprc, 4), "Brier": round(brier, 4),
        "F1": round(f1, 4),
    }


def train_logistic(X_train, y_train, w_train):
    """Train L2-regularized Logistic Regression with imputation. Per D010
    the LR fit does not use DHS sample weights; the `w_train` argument is
    kept for API compatibility only."""
    del w_train  # decision D010
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(
            C=1.0, max_iter=1000, solver="lbfgs",
            random_state=RNG_SEED, n_jobs=4,
        )),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def train_lgbm(X_train, y_train, w_train, X_val, y_val, w_val):
    """Train LightGBM with early stopping on internal val. Per D010 the
    LightGBM fit does not use DHS sample weights; class imbalance is
    handled through scale_pos_weight only."""
    del w_train, w_val  # decision D010
    pos = np.sum(y_train == 1)
    neg = np.sum(y_train == 0)
    spw = min(neg / pos, 5.0)  # Cap at 5 to avoid overweighting

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val)

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "max_depth": 6,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": spw,
        "n_jobs": 4,
        "random_state": RNG_SEED,
        "seed": RNG_SEED,
        "verbose": -1,
    }

    model = lgb.train(
        params, dtrain,
        num_boost_round=500,
        valid_sets=[dval],
        callbacks=[
            lgb.early_stopping(50, verbose=True,
                               first_metric_only=True),
            lgb.log_evaluation(100),
        ],
    )
    return model


def plot_roc_curves(results_by_model, cohort_preds):
    """Generate overlaid ROC curves per cohort for all models."""
    cohorts = ["Internal Val", "Temporal Test", "Spatial Test"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, cohort in zip(axes, cohorts):
        for model_name, preds in cohort_preds.items():
            if cohort in preds:
                y_true, y_prob, _w = preds[cohort]
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                auc_val = roc_auc_score(y_true, y_prob)
                ax.plot(fpr, tpr, label=f"{model_name} (AUC={auc_val:.3f})",
                        linewidth=1.5)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(cohort, fontweight="bold")
        ax.legend(fontsize=8, loc="lower right")
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.01)

    fig.suptitle("ROC Curves Across Validation Cohorts",
                 fontweight="bold", fontsize=14, y=1.02)
    plt.tight_layout()
    out = FIG_DIR / "figure_9_roc_curves.png"
    plt.savefig(out)
    plt.close()
    print(f"  Saved: {out.name}")


def plot_pr_curves(cohort_preds):
    """Generate overlaid Precision-Recall curves."""
    cohorts = ["Internal Val", "Temporal Test", "Spatial Test"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, cohort in zip(axes, cohorts):
        for model_name, preds in cohort_preds.items():
            if cohort in preds:
                y_true, y_prob, _w = preds[cohort]
                prec, rec, _ = precision_recall_curve(y_true, y_prob)
                auprc = average_precision_score(y_true, y_prob)
                ax.plot(rec, prec,
                        label=f"{model_name} (AUPRC={auprc:.3f})",
                        linewidth=1.5)
        # Baseline: prevalence line
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(cohort, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(0, 1.01)

    fig.suptitle("Precision-Recall Curves Across Validation Cohorts",
                 fontweight="bold", fontsize=14, y=1.02)
    plt.tight_layout()
    out = FIG_DIR / "figure_10_pr_curves.png"
    plt.savefig(out)
    plt.close()
    print(f"  Saved: {out.name}")


def plot_calibration(cohort_preds):
    """Generate calibration (reliability) diagrams."""
    cohorts = ["Temporal Test", "Spatial Test"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, cohort in zip(axes, cohorts):
        for model_name, preds in cohort_preds.items():
            if cohort in preds:
                y_true, y_prob, w = preds[cohort]
                fraction_pos, mean_pred = calibration_curve(
                    y_true, y_prob, n_bins=10, strategy="quantile"
                )
                ax.plot(mean_pred, fraction_pos,
                        marker="o", markersize=4, linewidth=1.5,
                        label=model_name)
        ax.plot([0, 0.5], [0, 0.5], "k--", alpha=0.3, linewidth=1)
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Observed Fraction of Positives")
        ax.set_title(f"Calibration: {cohort}", fontweight="bold")
        ax.legend(fontsize=8)
        ax.set_xlim(-0.01, 0.5)
        ax.set_ylim(-0.01, 0.5)

    fig.suptitle("Model Calibration (Reliability Diagrams)",
                 fontweight="bold", fontsize=14, y=1.02)
    plt.tight_layout()
    out = FIG_DIR / "figure_11_calibration.png"
    plt.savefig(out)
    plt.close()
    print(f"  Saved: {out.name}")


def main():
    print("=" * 60)
    print("Phase 4b: Multi-Model Comparison Benchmark")
    print("=" * 60)

    # Load splits
    print("Loading datasets...")
    X_tr, y_tr, w_tr = load_split("train")
    X_vi, y_vi, w_vi = load_split("val_internal")
    X_tt, y_tt, w_tt = load_split("test_time")
    X_ts, y_ts, w_ts = load_split("test_space")
    print(f"  Train: {X_tr.shape}, Val: {X_vi.shape}, "
          f"Temporal: {X_tt.shape}, Spatial: {X_ts.shape}")

    cohort_names = {
        "train": ("Train", y_tr, w_tr),
        "val_internal": ("Internal Val", y_vi, w_vi),
        "test_time": ("Temporal Test", y_tt, w_tt),
        "test_space": ("Spatial Test", y_ts, w_ts),
    }
    X_splits = {
        "train": X_tr, "val_internal": X_vi,
        "test_time": X_tt, "test_space": X_ts,
    }

    all_results = []
    cohort_preds = {}  # model_name -> cohort -> (y, prob, w)

    # ---- 1. Logistic Regression ----
    print("\n--- Logistic Regression ---")
    lr_model = train_logistic(X_tr, y_tr, w_tr)
    print("  Trained.")
    cohort_preds["Logistic Regression"] = {}
    for key, (cname, y, w) in cohort_names.items():
        X = X_splits[key]
        prob = lr_model.predict_proba(X)[:, 1]
        res = evaluate(y, prob, w, cname)
        res["Model"] = "Logistic Regression"
        all_results.append(res)
        cohort_preds["Logistic Regression"][cname] = (y, prob, w)
        print(f"  {cname}: AUC={res['AUC']:.3f} | "
              f"AUPRC={res['AUPRC']:.3f} | Brier={res['Brier']:.3f}")
    del lr_model
    gc.collect()

    # ---- 2. LightGBM ----
    print("\n--- LightGBM ---")
    lgb_model = train_lgbm(X_tr, y_tr, w_tr, X_vi, y_vi, w_vi)
    print("  Trained.")
    cohort_preds["LightGBM"] = {}
    for key, (cname, y, w) in cohort_names.items():
        X = X_splits[key]
        prob = lgb_model.predict(X)
        res = evaluate(y, prob, w, cname)
        res["Model"] = "LightGBM"
        all_results.append(res)
        cohort_preds["LightGBM"][cname] = (y, prob, w)
        print(f"  {cname}: AUC={res['AUC']:.3f} | "
              f"AUPRC={res['AUPRC']:.3f} | Brier={res['Brier']:.3f}")
    del lgb_model
    gc.collect()

    # ---- 3. XGBoost (load existing model) ----
    print("\n--- XGBoost ---")
    xgb_model_path = ML_DIR / "xgb_model.json"
    if xgb_model_path.exists():
        xgb_model = xgb.Booster()
        xgb_model.load_model(str(xgb_model_path))
        print("  Loaded existing model.")
        cohort_preds["XGBoost"] = {}
        for key, (cname, y, w) in cohort_names.items():
            X = X_splits[key]
            dmat = xgb.DMatrix(X, label=y)
            prob = xgb_model.predict(dmat)
            res = evaluate(y, prob, w, cname)
            res["Model"] = "XGBoost"
            all_results.append(res)
            cohort_preds["XGBoost"][cname] = (y, prob, w)
            print(f"  {cname}: AUC={res['AUC']:.3f} | "
                  f"AUPRC={res['AUPRC']:.3f} | Brier={res['Brier']:.3f}")
        del xgb_model
    else:
        print("  WARNING: xgb_model.json not found, skipping.")
    gc.collect()

    # ---- Save comparison table ----
    comp_df = pd.DataFrame(all_results)
    comp_df = comp_df[["Model", "Cohort", "AUC", "AUPRC", "Brier", "F1"]]
    out = ML_DIR / "model_comparison.csv"
    comp_df.to_csv(out, index=False)
    print(f"\nSaved: {out.name}")
    print(comp_df.to_string(index=False))

    # ---- Generate figures ----
    print("\nGenerating figures...")
    plot_roc_curves(all_results, cohort_preds)
    plot_pr_curves(cohort_preds)
    plot_calibration(cohort_preds)

    print("\nDONE.")


if __name__ == "__main__":
    main()
