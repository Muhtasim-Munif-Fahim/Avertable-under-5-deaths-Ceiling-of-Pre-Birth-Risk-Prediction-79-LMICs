"""
Compute proper DeLong-variance 95% confidence intervals for the AUROC of
each (model, validation cohort) pair. Replaces the Hanley-McNeil
approximation used as a placeholder in T4.

Loads:
    output/ml/xgb_model.json   (trained XGBoost)
    output/ml/X_*.parquet      (per-cohort feature frames)
    output/ml/y_*.parquet      (per-cohort labels)
    output/ml/lr_baseline_metrics.csv (for the held-out cohorts the LR
                                       was already scored against)

For XGBoost, we re-score from the persisted model and compute the
DeLong covariance of the empirical AUROC (DeLong, DeLong &
Clarke-Pearson 1988; Sun & Xu 2014 fast implementation).

For logistic regression and LightGBM, we report the AUROC stored in
`model_comparison.csv` together with a DeLong-equivalent CI derived
from the same formula applied to the held-out predictions where the
score columns are persisted; where they are not, we report the
Hanley-McNeil approximation (this is accurate to within 0.001 at the
n>>10^5 sample sizes used here, but flagged in the column).

Writes:
    output/tables/table_auc_cis.csv
"""

from __future__ import annotations
import sys
from pathlib import Path
import math
import json

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ))
from src.utils.print_util import print
from config.paths import RNG_SEED
ML = PROJ / "output" / "ml"
T = PROJ / "output" / "tables"
OUT = T / "table_auc_cis.csv"

COHORTS = [
    ("Train",         "X_train.parquet",        "y_train.parquet"),
    ("Internal Val",  "X_val_internal.parquet", "y_val_internal.parquet"),
    ("Temporal Test", "X_test_time.parquet",    "y_test_time.parquet"),
    ("Spatial Test",  "X_test_space.parquet",   "y_test_space.parquet"),
]


# ---------- DeLong covariance ----------

def _midrank(x: np.ndarray) -> np.ndarray:
    """Mid-rank assignment for the DeLong U-statistic; ties get average rank."""
    j = np.argsort(x)
    z = x[j]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1) + 1.0   # 1-based mid-rank
        i = k
    out = np.empty(n)
    out[j] = t
    return out


def delong_auc_var(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Return (AUROC, variance of AUROC) using Sun-Xu fast DeLong."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    m = len(pos)
    n = len(neg)
    if m == 0 or n == 0:
        return (float("nan"), float("nan"))
    tx = _midrank(pos)
    ty = _midrank(neg)
    tz = _midrank(np.concatenate([pos, neg]))
    auc = (tz[:m].sum() - m * (m + 1) / 2.0) / (m * n)
    # placement values
    v01 = (tz[:m] - tx) / n
    v10 = 1.0 - (tz[m:] - ty) / m
    s01 = v01.var(ddof=1)
    s10 = v10.var(ddof=1)
    var = s01 / m + s10 / n
    return float(auc), float(var)


def fmt_auc_ci(auc: float, var: float) -> str:
    if math.isnan(auc) or math.isnan(var) or var < 0:
        return f"{auc:.3f} (--)"
    se = math.sqrt(var)
    lo = auc - 1.96 * se
    hi = auc + 1.96 * se
    return f"{auc:.3f} ({lo:.3f}--{hi:.3f})"


# ---------- Scoring ----------

def _load_xgb():
    import xgboost as xgb
    booster = xgb.Booster()
    booster.load_model(str(ML / "xgb_model.json"))
    return booster


def _score_xgb(booster, X: pd.DataFrame) -> np.ndarray:
    import xgboost as xgb
    # Drop non-numeric columns if any; xgb_model was trained on the feature
    # set persisted as the columns of X_train.parquet, so the column order
    # is already canonical.
    d = xgb.DMatrix(X.values, feature_names=list(X.columns))
    return booster.predict(d)


def _load_xy(x_path: Path, y_path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray | None]:
    X = pd.read_parquet(x_path)
    y_df = pd.read_parquet(y_path)
    y_col = "y" if "y" in y_df.columns else "died_under5"
    y = y_df[y_col].to_numpy().astype(int)
    w_col = "weight" if "weight" in y_df.columns else (
        "sample_weight" if "sample_weight" in y_df.columns else None
    )
    w = y_df[w_col].fillna(1.0).to_numpy() if w_col else None
    return X, y, w


def _load_X_categorical_cast(X: pd.DataFrame) -> pd.DataFrame:
    """Convert categorical columns to integer codes for sklearn/lightgbm.

    Mirrors the preprocessing in s07c_lr_baseline.py so the refit replicates
    the trained baseline exactly.
    """
    X = X.copy()
    for col in X.columns:
        if X[col].dtype.name == "category":
            codes = X[col].cat.codes.astype("float32")
            codes[codes == -1] = np.nan
            X[col] = codes
        else:
            X[col] = X[col].astype("float32")
    return X


def _fit_lr(X_train, y_train, w_train):
    """Refit LR with the same recipe as s07c (no DHS weights per D010)."""
    del w_train  # decision D010
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(solver="lbfgs", max_iter=2000, random_state=RNG_SEED)),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def _fit_lgbm(X_train, y_train, w_train, X_val, y_val, w_val):
    """Refit LGBM with the same recipe as s07b (no DHS weights per D010)."""
    del w_train, w_val  # decision D010
    import lightgbm as lgb
    pos = int(np.sum(y_train == 1))
    neg = int(np.sum(y_train == 0))
    spw = min(neg / pos, 5.0)
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
    booster = lgb.train(
        params, dtrain,
        num_boost_round=500,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(50, verbose=False, first_metric_only=True)],
    )
    return booster


def _hanley_var(auc: float, n_pos: int, n_neg: int) -> float:
    """Hanley-McNeil approximate variance for AUROC."""
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc * auc / (1.0 + auc)
    return (
        auc * (1.0 - auc)
        + (n_pos - 1) * (q1 - auc * auc)
        + (n_neg - 1) * (q2 - auc * auc)
    ) / (n_pos * n_neg)


def main() -> None:
    print("DeLong AUC variance computation (XGB / LR / LightGBM)")
    print("=" * 60)

    # ---- Load all cohorts up front (sklearn/lightgbm path expects numeric arrays) ----
    cohort_data: dict[str, tuple] = {}
    for cohort, x_file, y_file in COHORTS:
        try:
            X, y, w = _load_xy(ML / x_file, ML / y_file)
            X_num = _load_X_categorical_cast(X)
            cohort_data[cohort] = (X, X_num, y, w)
            print(f"  loaded {cohort}: {len(y):,} rows, {int(y.sum()):,} events")
        except Exception as e:
            print(f"  load failed for {cohort}: {e}")

    # ---- XGBoost: load + score on each cohort ----
    booster = _load_xgb()
    rows: list[dict] = []
    for cohort in cohort_data:
        X, _X_num, y, _w = cohort_data[cohort]
        p_xgb = _score_xgb(booster, X)
        auc, var = delong_auc_var(p_xgb, y)
        rows.append(_row("XGBoost", cohort, y, auc, var, "delong"))
        print(f"  XGB    {cohort}: AUC = {auc:.4f}  SE = {math.sqrt(var):.4f}")

    # ---- Logistic regression: refit on train, score on all 4 cohorts ----
    X_tr, X_tr_num, y_tr, w_tr = cohort_data["Train"]
    print("\nFitting logistic regression...")
    lr = _fit_lr(X_tr_num, y_tr, w_tr)
    for cohort in cohort_data:
        _X, X_num, y, _w = cohort_data[cohort]
        p_lr = lr.predict_proba(X_num)[:, 1]
        auc, var = delong_auc_var(p_lr, y)
        rows.append(_row("Logistic Regression", cohort, y, auc, var, "delong"))
        print(f"  LR     {cohort}: AUC = {auc:.4f}  SE = {math.sqrt(var):.4f}")

    # ---- LightGBM: refit on train+val (early stopping), score on all 4 cohorts ----
    print("\nFitting LightGBM (with early stopping on Internal Val)...")
    _X_v, X_v_num, y_v, w_v = cohort_data["Internal Val"]
    lgbm = _fit_lgbm(X_tr_num, y_tr, w_tr, X_v_num, y_v, w_v)
    for cohort in cohort_data:
        _X, X_num, y, _w = cohort_data[cohort]
        p_lg = lgbm.predict(X_num.values)
        auc, var = delong_auc_var(p_lg, y)
        rows.append(_row("LightGBM", cohort, y, auc, var, "delong"))
        print(f"  LGBM   {cohort}: AUC = {auc:.4f}  SE = {math.sqrt(var):.4f}")

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")
    print(df[["model", "cohort", "n", "n_pos", "auc", "ci_lo", "ci_hi", "method"]].to_string(index=False))


def _row(model: str, cohort: str, y: np.ndarray, auc: float, var: float, method: str) -> dict:
    se = math.sqrt(max(var, 0.0))
    return {
        "model": model,
        "cohort": cohort,
        "n": len(y),
        "n_pos": int(y.sum()),
        "auc": auc,
        "var": var,
        "se": se,
        "ci_lo": auc - 1.96 * se,
        "ci_hi": auc + 1.96 * se,
        "method": method,
        "formatted": fmt_auc_ci(auc, var),
    }


if __name__ == "__main__":
    main()
