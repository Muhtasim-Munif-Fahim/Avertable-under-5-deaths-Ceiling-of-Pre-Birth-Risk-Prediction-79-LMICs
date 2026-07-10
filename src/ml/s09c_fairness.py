"""
s09c_fairness.py — Subgroup performance / fairness audit
=========================================================

Per-subgroup AUC, prevalence, calibration slope/intercept on the
temporal holdout. Subgroups defined by features actually present in
the X test parquets (sex, wealth quintile, urban/rural, edu level).

Calibration metrics (slope, intercept, Brier, CITL) are computed on the
ISOTONIC-recalibrated probabilities persisted by s09a_recalibration
(`p_{split}__isotonic.parquet`) when available, since those are the
deployment-relevant probabilities; AUC is invariant to monotone
recalibration so it is unaffected. If the recalibrated probabilities are
not yet persisted, the audit falls back to the raw booster output and
records that in the `prob_basis` column.

Outputs:
  output/tables/table_7_subgroup_performance.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.print_util import print
from config.paths import OUTPUT_DIR, TABLES_DIR

ML_DIR = OUTPUT_DIR / "ml"
TABLES_DIR.mkdir(parents=True, exist_ok=True)


def _load(name: str):
    X = pd.read_parquet(ML_DIR / f"X_{name}.parquet").copy()
    y = pd.read_parquet(ML_DIR / f"y_{name}.parquet")
    X_num = X.copy()
    for col in X_num.columns:
        if X_num[col].dtype.name == "category":
            X_num[col] = X_num[col].cat.codes.astype("float32")
            X_num.loc[X_num[col] == -1, col] = np.nan
        else:
            X_num[col] = X_num[col].astype("float32")
    y_col = "died_under5" if "died_under5" in y.columns else "y"
    return X_num, y[y_col].astype(int).values


def _load_recalibrated(split: str, label: str = "isotonic"):
    """Load s09a-persisted recalibrated probabilities, or None if absent."""
    p_file = ML_DIR / f"p_{split}__{label}.parquet"
    if not p_file.exists():
        return None
    return pd.read_parquet(p_file)["p"].to_numpy()


def slope_intercept(y, p):
    if y.sum() == 0 or y.sum() == len(y):
        return np.nan, np.nan
    eps = 1e-6
    p = np.clip(p, eps, 1 - eps)
    lp = np.log(p / (1 - p)).reshape(-1, 1)
    try:
        lr = LogisticRegression(C=1e10, solver="lbfgs",
                                max_iter=2000)
        lr.fit(lp, y)
        return float(lr.coef_[0, 0]), float(lr.intercept_[0])
    except Exception:
        return np.nan, np.nan


SUBGROUPS = [
    ("sex", "child_male", {0: "Female", 1: "Male"}),
    ("urban", "urban", {0: "Rural", 1: "Urban"}),
    ("wealth_quintile", "wealth_quintile",
     {1: "Poorest", 2: "Poorer", 3: "Middle", 4: "Richer", 5: "Richest"}),
    ("education", "edu_level",
     {0: "None", 1: "Primary", 2: "Secondary", 3: "Higher"}),
]


def evaluate(split: str, booster: xgb.Booster) -> pd.DataFrame:
    X, y = _load(split)
    # Prefer the isotonic-recalibrated (deployment-relevant) probabilities
    # for calibration metrics; AUC is recalibration-invariant.
    p_iso = _load_recalibrated(split, "isotonic")
    if p_iso is not None and len(p_iso) == len(y):
        p = p_iso
        prob_basis = "isotonic"
    else:
        p = booster.predict(xgb.DMatrix(X))
        prob_basis = "raw"
        if p_iso is not None:
            print(f"  {split}: isotonic prob length {len(p_iso)} != "
                  f"{len(y)}; falling back to raw booster output")
    rows = []
    for label, col, mapping in SUBGROUPS:
        if col not in X.columns:
            continue
        vals = X[col].values
        for code, name in mapping.items():
            mask = vals == code
            if mask.sum() < 100:
                continue
            ys, ps = y[mask], p[mask]
            try:
                auc = float(roc_auc_score(ys, ps)) if len(np.unique(ys)) > 1 else np.nan
            except Exception:
                auc = np.nan
            slope, intercept = slope_intercept(ys, ps)
            mean_p = float(np.mean(ps))
            mean_y = float(np.mean(ys))
            citl_prob = mean_y - mean_p
            rows.append({
                "split": split,
                "subgroup": label,
                "level": name,
                "n": int(mask.sum()),
                "prevalence": float(ys.mean()),
                "auc": auc,
                "brier": float(brier_score_loss(ys, ps)),
                "calibration_slope": slope,
                "calibration_intercept_logit0": intercept,
                "citl_prob": citl_prob,
                "prob_basis": prob_basis,
            })
    return pd.DataFrame(rows)


def main():
    print("=" * 60)
    print("Subgroup Fairness Audit")
    print("=" * 60)

    booster = xgb.Booster()
    booster.load_model(str(ML_DIR / "xgb_model.json"))

    dfs = []
    for split in ["test_time", "test_space"]:
        try:
            df = evaluate(split, booster)
            dfs.append(df)
            print(f"\n{split}: {len(df)} subgroups")
        except FileNotFoundError as e:
            print(f"  Skipping {split}: {e}")

    if not dfs:
        return
    out = pd.concat(dfs, ignore_index=True)
    out_csv = TABLES_DIR / "table_7_subgroup_performance.csv"
    out.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv.name}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
