"""
s09a_recalibration.py - Post-hoc Recalibration of the XGBoost Risk Model
========================================================================

The leakage-screened XGBoost classifier (s07) was trained with
`scale_pos_weight` to handle the ~6% class imbalance, which pushes its
raw predicted probabilities into the [0.16, 0.72] band even though the
true event rate is ~6-8%. The raw model therefore has good ranking
(AUROC ~0.69-0.70) but very poor absolute calibration
(slope ~0.94, intercept ~-2.6, Brier ~0.20 — almost 4x the constant-
prevalence baseline). Per TRIPOD+AI item 10b and Lancet decision-curve
expectations, a deployable risk score must be recalibrated before
absolute-probability metrics are reported.

This script:

  1. Loads the XGBoost model fitted in s07 and computes its raw
     log-odds on the internal validation cohort (`val_internal`).
  2. Fits TWO recalibrators on `val_internal`:
       - Platt scaling: a 1-parameter logistic on the raw log-odds.
       - Isotonic regression: a non-parametric monotone mapping from
         raw probability to observed probability.
  3. Applies each recalibrator to the temporal-holdout (`test_time`)
     and geographic-holdout (`test_space`) cohorts.
  4. Persists recalibrated probabilities and the recalibrator objects.
  5. Writes a metrics table comparing RAW / PLATT / ISOTONIC on AUROC
     (unchanged by monotone rescaling), Brier, calibration slope,
     calibration intercept, and ICI.

Outputs:
  - output/ml/p_test_time__{raw|platt|isotonic}.parquet
  - output/ml/p_test_space__{raw|platt|isotonic}.parquet
  - output/ml/recalibrator_platt.pkl
  - output/ml/recalibrator_isotonic.pkl
  - output/tables/table_5_calibration_recalibrated.csv
"""
from __future__ import annotations

import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.print_util import print
from config.paths import OUTPUT_DIR, RNG_SEED, TABLES_DIR

ML_DIR = OUTPUT_DIR / "ml"


def _load_X(split: str) -> pd.DataFrame:
    X = pd.read_parquet(ML_DIR / f"X_{split}.parquet")
    for col in X.columns:
        if X[col].dtype.name == "category":
            X[col] = X[col].cat.codes.astype("float32")
            X.loc[X[col] == -1, col] = np.nan
        else:
            X[col] = X[col].astype("float32")
    return X


def _load_y(split: str) -> np.ndarray:
    y_df = pd.read_parquet(ML_DIR / f"y_{split}.parquet")
    col = "y" if "y" in y_df.columns else "died_under5"
    return y_df[col].astype(int).to_numpy()


def _calibration_slope_intercept(y_true: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Logistic recalibration of y on logit(p). Slope==1, intercept==0
    indicates perfect calibration."""
    eps = 1e-6
    p = np.clip(p, eps, 1 - eps)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(C=1e10, solver="lbfgs", max_iter=2000)
    lr.fit(logit, y_true)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def _integrated_calibration_index(y_true: np.ndarray, p: np.ndarray) -> float:
    """Austin and Steyerberg (2019) ICI: mean absolute deviation between
    predicted probability and an isotonic smoother of y on p."""
    iso = IsotonicRegression(out_of_bounds="clip")
    p_smooth = iso.fit_transform(p, y_true)
    return float(np.mean(np.abs(p_smooth - p)))


def _metrics_row(label: str, split: str, y: np.ndarray, p: np.ndarray) -> dict:
    auc = float(roc_auc_score(y, p))
    brier = float(brier_score_loss(y, p))
    slope, intercept = _calibration_slope_intercept(y, p)
    ici = _integrated_calibration_index(y, p)
    return {
        "recalibration": label,
        "split": split,
        "n": int(len(y)),
        "n_events": int(y.sum()),
        "auc": round(auc, 4),
        "brier": round(brier, 4),
        "calibration_slope": round(slope, 4),
        "calibration_intercept": round(intercept, 4),
        "ici": round(ici, 4),
        "mean_predicted": round(float(np.mean(p)), 4),
        "mean_observed": round(float(np.mean(y)), 4),
    }


def main() -> None:
    print("=" * 60)
    print("Phase 5a: Post-hoc Recalibration (Platt + Isotonic)")
    print("=" * 60)

    booster = xgb.Booster()
    booster.load_model(str(ML_DIR / "xgb_model.json"))
    print(f"Loaded model: {ML_DIR / 'xgb_model.json'}")

    # Load val_internal (recalibrator fit) and held-out cohorts.
    X_val = _load_X("val_internal")
    y_val = _load_y("val_internal")
    X_t = _load_X("test_time")
    y_t = _load_y("test_time")
    X_s = _load_X("test_space")
    y_s = _load_y("test_space")
    print(f"  Val internal:  N={len(y_val):,}, events={int(y_val.sum()):,}")
    print(f"  Test temporal: N={len(y_t):,}, events={int(y_t.sum()):,}")
    print(f"  Test spatial:  N={len(y_s):,}, events={int(y_s.sum()):,}")

    p_val_raw = booster.predict(xgb.DMatrix(X_val))
    p_t_raw = booster.predict(xgb.DMatrix(X_t))
    p_s_raw = booster.predict(xgb.DMatrix(X_s))

    # ----------- Platt scaling -----------
    print("\nFitting Platt scaling on val_internal log-odds...")
    eps = 1e-6
    logit_val = np.log(np.clip(p_val_raw, eps, 1 - eps) / (1 - np.clip(p_val_raw, eps, 1 - eps))).reshape(-1, 1)
    platt = LogisticRegression(C=1e10, solver="lbfgs", max_iter=2000,
                                random_state=RNG_SEED)
    platt.fit(logit_val, y_val)

    def _platt_apply(p_raw: np.ndarray) -> np.ndarray:
        logit = np.log(np.clip(p_raw, eps, 1 - eps) / (1 - np.clip(p_raw, eps, 1 - eps))).reshape(-1, 1)
        return platt.predict_proba(logit)[:, 1]

    p_t_platt = _platt_apply(p_t_raw)
    p_s_platt = _platt_apply(p_s_raw)
    print(f"  Platt slope={platt.coef_[0, 0]:.4f}, intercept={platt.intercept_[0]:.4f}")

    # ----------- Isotonic regression -----------
    print("\nFitting isotonic regression on val_internal probabilities...")
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_val_raw, y_val)

    p_t_iso = iso.transform(p_t_raw)
    p_s_iso = iso.transform(p_s_raw)
    print(f"  Isotonic knots: {len(iso.X_thresholds_)}")

    # ----------- Persist probabilities -----------
    for split, p_raw, p_pl, p_is in [
        ("test_time", p_t_raw, p_t_platt, p_t_iso),
        ("test_space", p_s_raw, p_s_platt, p_s_iso),
    ]:
        for label, vec in [("raw", p_raw), ("platt", p_pl), ("isotonic", p_is)]:
            pd.DataFrame({"p": vec}).to_parquet(
                ML_DIR / f"p_{split}__{label}.parquet", index=False)
    print("\nSaved recalibrated probability vectors.")

    # Persist recalibrators
    with open(ML_DIR / "recalibrator_platt.pkl", "wb") as f:
        pickle.dump(platt, f)
    with open(ML_DIR / "recalibrator_isotonic.pkl", "wb") as f:
        pickle.dump(iso, f)

    # ----------- Metrics table -----------
    print("\nMetrics (RAW vs PLATT vs ISOTONIC):")
    rows = []
    for split, y, p_raw, p_pl, p_is in [
        ("test_time", y_t, p_t_raw, p_t_platt, p_t_iso),
        ("test_space", y_s, p_s_raw, p_s_platt, p_s_iso),
    ]:
        for label, vec in [("raw", p_raw), ("platt", p_pl), ("isotonic", p_is)]:
            row = _metrics_row(label, split, y, vec)
            rows.append(row)
            print(f"  {split:12} {label:8}: AUC={row['auc']:.3f} "
                  f"Brier={row['brier']:.3f} slope={row['calibration_slope']:.3f} "
                  f"intercept={row['calibration_intercept']:+.3f} ICI={row['ici']:.4f} "
                  f"mean_p={row['mean_predicted']:.3f} vs obs={row['mean_observed']:.3f}")

    out_csv = TABLES_DIR / "table_5_calibration_recalibrated.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    print("\nDone. Downstream scripts (s09b decision curve, s09c fairness,")
    print("s10 risk score, T4 generator) should read the isotonic-recalibrated")
    print("probabilities (p_{split}__isotonic.parquet) as the deployable estimate")
    print("and the raw probabilities (p_{split}__raw.parquet) as the ranking-only")
    print("baseline.")


if __name__ == "__main__":
    main()
