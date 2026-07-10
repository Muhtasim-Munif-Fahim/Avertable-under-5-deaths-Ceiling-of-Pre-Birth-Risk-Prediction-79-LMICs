"""
s10_risk_score.py — Clinical Risk Score (0–10)
================================================

Derives a simple integer risk score from GEE Model 3 log-OR
coefficients, following the Framingham point-based approach.
Each domain maps to 0–3 integer points; total 0–10.

Score domains (from table_2_sequential_models.csv, M3 column):
  - Birth type     : singleton=0, multiple=3  (OR 4.35 → ln=1.47 ≈ 3 units)
  - Birth interval : 24+m=0, first/missing=2, <24m=2  (OR 1.73–1.89)
  - Birth order    : 1st-3rd=0, 4th-5th=1, 6th+=2  (OR 1.60–1.62)
  - Maternal age   : 20-34=0, <20=1  (OR 1.51)
  - Education      : secondary/higher=0, primary=1, none=2  (OR 0.77–0.39)

Outputs:
  output/tables/table_8_risk_score_nns.csv
  output/figures/figure_13_risk_score_roc.png
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.print_util import print
from config.paths import OUTPUT_DIR, TABLES_DIR

ML_DIR = OUTPUT_DIR / "ml"
TABLES_DIR.mkdir(parents=True, exist_ok=True)


def compute_risk_score(X: pd.DataFrame) -> np.ndarray:
    """
    Integer risk score 0–10 from GEE M3 aOR log-coefficients.

    Points are: round(ln(aOR) / 0.50), capped so total max = 10.

    Domain          Category          aOR(M3)   ln(aOR)  pts
    ----------      --------          -------   -------  ---
    Birth type      Multiple birth    4.345     1.47      3
    Birth interval  <24 months        1.886     0.64      2
                    First/missing     1.733     0.55      2
    Birth order     6th+              1.617     0.48      2
                    4th-5th           1.595     0.47      1
    Maternal age    <20               1.507     0.41      1
    Education       None              ref (+2 vs higher=0.394→ln=0.93)
                    Primary           partial (vs higher: 0.774/0.394→ln=0.68)
    """
    score = np.zeros(len(X), dtype=np.int8)

    if "multiple_birth" in X.columns:
        score += 3 * (X["multiple_birth"].fillna(0) == 1).astype(np.int8)

    if "birth_interval_months" in X.columns:
        bi = X["birth_interval_months"].fillna(-1)
        # <24 months (high risk) or first/missing (negative sentinel or NaN)
        score += 2 * ((bi < 24) & (bi >= 0)).astype(np.int8)
        score += 2 * (bi < 0).astype(np.int8)

    if "birth_order" in X.columns:
        bo = X["birth_order"].fillna(0)
        score += 1 * ((bo >= 4) & (bo <= 5)).astype(np.int8)
        score += 2 * (bo >= 6).astype(np.int8)

    if "maternal_age_at_birth" in X.columns:
        ma = X["maternal_age_at_birth"].fillna(25)
        score += 1 * (ma < 20).astype(np.int8)

    if "edu_level" in X.columns:
        # 0=None, 1=Primary, 2=Secondary, 3=Higher
        el = X["edu_level"].fillna(2)
        score += 2 * (el == 0).astype(np.int8)
        score += 1 * (el == 1).astype(np.int8)

    return score.clip(0, 10)


def nns_table(y_true: np.ndarray, score: np.ndarray, thresholds=None):
    """Number Needed to Screen (NNS) at each threshold.

    NNS = N_screened / N_events_in_screened
        = 1 / positive predictive value (PPV)

    Also reports: sensitivity, specificity, PPV, NPV, fraction screened.
    """
    if thresholds is None:
        thresholds = list(range(0, 11))
    n = len(y_true)
    rows = []
    for t in thresholds:
        high = score >= t
        tp = int(((high == 1) & (y_true == 1)).sum())
        fp = int(((high == 1) & (y_true == 0)).sum())
        fn = int(((high == 0) & (y_true == 1)).sum())
        tn = int(((high == 0) & (y_true == 0)).sum())
        n_screened = tp + fp
        sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        ppv = tp / n_screened if n_screened > 0 else np.nan
        npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
        nns = 1 / ppv if ppv and ppv > 0 else np.nan
        rows.append({
            "threshold": t,
            "n_screened": n_screened,
            "frac_screened": n_screened / n,
            "sensitivity": sens,
            "specificity": spec,
            "ppv": ppv,
            "npv": npv,
            "nns": nns,
        })
    return pd.DataFrame(rows)


def _load(name: str):
    X = pd.read_parquet(ML_DIR / f"X_{name}.parquet")
    for col in X.columns:
        if X[col].dtype.name == "category":
            X[col] = X[col].cat.codes.astype("float32")
            X.loc[X[col] == -1, col] = np.nan
        else:
            X[col] = X[col].astype("float32")
    y = pd.read_parquet(ML_DIR / f"y_{name}.parquet")
    y_col = "died_under5" if "died_under5" in y.columns else "y"
    return X, y[y_col].astype(int).values


def main():
    print("=" * 60)
    print("Clinical Risk Score derivation")
    print("=" * 60)

    booster = xgb.Booster()
    booster.load_model(str(ML_DIR / "xgb_model.json"))

    all_rows = []

    for split in ["test_time", "test_space"]:
        try:
            X, y = _load(split)
        except FileNotFoundError as e:
            print(f"  Skipping {split}: {e}")
            continue

        score = compute_risk_score(X)
        print(f"\n{split}: n={len(y):,}, events={y.sum():,} "
              f"({y.mean()*100:.2f}%)")
        print(f"  Score distribution: "
              f"0-3={100*(score<=3).mean():.1f}%, "
              f"4-6={100*((score>=4)&(score<=6)).mean():.1f}%, "
              f"7+={100*(score>=7).mean():.1f}%")

        nns_df = nns_table(y, score)
        nns_df.insert(0, "split", split)
        all_rows.append(nns_df)

        # AUC of integer score vs the underlying XGBoost probability
        auc_score = float(roc_auc_score(y, score))
        auc_xgb = float(roc_auc_score(y, booster.predict(xgb.DMatrix(X))))
        print(f"  AUC: risk score={auc_score:.3f}, XGBoost={auc_xgb:.3f}")

        # Print NNS at a few clinically useful thresholds
        for t in [3, 4, 5]:
            row = nns_df[nns_df["threshold"] == t].iloc[0]
            print(f"  Score >= {t}: sensitivity={row['sensitivity']:.2f}, "
                  f"frac_screened={row['frac_screened']:.2f}, "
                  f"NNS={row['nns']:.1f}")

    if all_rows:
        out = pd.concat(all_rows, ignore_index=True)
        out.to_csv(TABLES_DIR / "table_8_risk_score_nns.csv", index=False)
        print(f"Saved: table_8_risk_score_nns.csv")


if __name__ == "__main__":
    main()
