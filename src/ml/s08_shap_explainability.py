"""
s08_shap_explainability.py - Model Explainability via SHAP
==========================================================

Computes SHAP (Shapley Additive exPlanations) global feature importance for the
trained leakage-screened XGBoost model on a 10,000-record sample of the temporal
holdout, and writes the mean absolute SHAP ranking to
output/ml/shap_feature_importance.csv.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import shap

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.print_util import print
from config.paths import OUTPUT_DIR, RNG_SEED
from src.ml.s06_ml_data_prep import BANNED_MAIN_ML_FEATURES, FEATURES

ML_DIR = OUTPUT_DIR / "ml"


def load_val_sample():
    """Load a 10k random sample from the temporal holdout for SHAP."""
    X = pd.read_parquet(ML_DIR / "X_test_time.parquet")
    if list(X.columns) != FEATURES:
        raise ValueError(
            "X_test_time.parquet feature schema does not match the "
            "leakage-screened main ML FEATURES."
        )
    banned = sorted(set(X.columns) & BANNED_MAIN_ML_FEATURES)
    if banned:
        raise ValueError(f"Leaky features found in SHAP input: {banned}")
    for col in X.columns:
        if X[col].dtype.name == 'category':
            X[col] = X[col].cat.codes.astype("float32")
            X.loc[X[col] == -1, col] = np.nan
        else:
            X[col] = X[col].astype("float32")

    np.random.seed(RNG_SEED)
    idx = np.random.choice(len(X), size=min(10000, len(X)), replace=False)
    return X.iloc[idx]


def main():
    print("=" * 60)
    print("Phase 4: SHAP Explainability")
    print("=" * 60)

    model_path = ML_DIR / "xgb_model.json"
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return

    print("Loading XGBoost model...")
    model = xgb.Booster()
    model.load_model(str(model_path))

    print("Loading validation sample...")
    X = load_val_sample()

    print("Computing SHAP values (TreeExplainer)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    print("Generating SHAP feature-importance ranking...")
    rename_map = {
        "child_male": "Male child",
        "multiple_birth": "Multiple birth",
        "birth_order": "Birth order",
        "birth_interval_months": "Birth interval",
        "maternal_age_at_birth": "Maternal age",
        "edu_level": "Maternal education",
        "currently_married": "Currently married",
        "wealth_quintile": "Wealth quintile",
        "urban": "Urban residence",
        "hh_members": "Household members",
    }
    importance = (
        pd.DataFrame(
            {
                "feature": X.columns,
                "label": [rename_map.get(col, col) for col in X.columns],
                "mean_abs_shap": np.abs(shap_values).mean(axis=0),
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    importance.to_csv(ML_DIR / "shap_feature_importance.csv", index=False)
    print(f"Saved SHAP importance -> {ML_DIR / 'shap_feature_importance.csv'}")
    print("DONE.")


if __name__ == "__main__":
    main()
