"""
s07d_optuna_tuning.py — Optuna XGBoost Hyperparameter Search
============================================================

30-trial Hyperband-pruned search on a 10% stratified subsample of
training data. Saves the best params for downstream use.

Outputs:
  output/ml/optuna_best_params.json
  output/ml/optuna_study.csv
"""
import sys, gc, json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.print_util import print
from config.paths import OUTPUT_DIR, RNG_SEED

ML_DIR = OUTPUT_DIR / "ml"
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _load_train_subsample(frac: float = 0.10, rng_seed: int = RNG_SEED):
    X = pd.read_parquet(ML_DIR / "X_train.parquet")
    y_df = pd.read_parquet(ML_DIR / "y_train.parquet")
    for col in X.columns:
        if X[col].dtype.name == "category":
            X[col] = X[col].cat.codes.astype("float32")
            X.loc[X[col] == -1, col] = np.nan
        else:
            X[col] = X[col].astype("float32")
    y_col = "died_under5" if "died_under5" in y_df.columns else "y"
    w_col = ("sample_weight" if "sample_weight" in y_df.columns
             else "weight" if "weight" in y_df.columns else None)
    y = y_df[y_col].astype(int).values
    w = y_df[w_col].fillna(1.0).values if w_col else None

    rng = np.random.default_rng(rng_seed)
    idx = rng.choice(len(y), size=int(len(y) * frac), replace=False)
    idx.sort()
    X_s = X.iloc[idx].reset_index(drop=True)
    y_s = y[idx]
    w_s = w[idx] if w is not None else None
    del X, y, w
    gc.collect()
    print(f"  Subsample: {len(y_s):,} rows, {y_s.sum():,} events "
          f"({y_s.mean()*100:.2f}%)")
    return X_s, y_s, w_s


def objective(trial, X_tr, y_tr, w_tr, X_val, y_val):
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "device": "cpu",
        "seed": RNG_SEED,
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
    }
    scale = float((y_tr == 0).sum()) / float(max(1, (y_tr == 1).sum()))
    params["scale_pos_weight"] = scale

    # D010: do not apply DHS sample weights to ML training.
    del w_tr
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_val, label=y_val)

    model = xgb.train(
        params, dtrain,
        num_boost_round=300,
        evals=[(dval, "eval")],
        early_stopping_rounds=30,
        verbose_eval=False,
    )
    p = model.predict(dval)
    return float(roc_auc_score(y_val, p))


def main():
    print("=" * 60)
    print("Optuna XGBoost Hyperparameter Search (30 trials)")
    print("=" * 60)

    print("\nLoading training subsample (10%)...")
    X_sub, y_sub, w_sub = _load_train_subsample(frac=0.10)

    # 80/20 internal train/val split
    rng = np.random.default_rng(RNG_SEED)
    n = len(y_sub)
    val_idx = rng.choice(n, size=int(n * 0.20), replace=False)
    val_mask = np.zeros(n, dtype=bool)
    val_mask[val_idx] = True
    X_tr = X_sub[~val_mask].reset_index(drop=True)
    y_tr = y_sub[~val_mask]
    w_tr = w_sub[~val_mask] if w_sub is not None else None
    X_val = X_sub[val_mask].reset_index(drop=True)
    y_val = y_sub[val_mask]
    print(f"  Train: {len(y_tr):,}, Val: {len(y_val):,}")

    sampler = TPESampler(seed=RNG_SEED)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
    )
    study.optimize(
        lambda trial: objective(trial, X_tr, y_tr, w_tr, X_val, y_val),
        n_trials=30,
        show_progress_bar=False,
        callbacks=[lambda study, trial: print(
            f"  Trial {trial.number:2d}: AUC={trial.value:.4f}"
            if trial.value is not None else
            f"  Trial {trial.number:2d}: pruned"
        )],
    )

    best = study.best_params
    best["scale_pos_weight"] = float(
        (y_tr == 0).sum()) / float(max(1, (y_tr == 1).sum()))
    best_auc = study.best_value

    print(f"\nBest AUC (val): {best_auc:.4f}")
    print("Best params:")
    for k, v in best.items():
        print(f"  {k}: {v}")

    out_json = ML_DIR / "optuna_best_params.json"
    with open(out_json, "w") as f:
        json.dump({"best_auc_val": best_auc, "params": best}, f, indent=2)
    print(f"\nSaved: {out_json.name}")

    study_df = study.trials_dataframe()
    study_df.to_csv(ML_DIR / "optuna_study.csv", index=False)
    print(f"Saved: optuna_study.csv")

    # Compare default params AUC on same val set
    default_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "device": "cpu",
        "seed": RNG_SEED,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": best["scale_pos_weight"],
    }
    # D010: do not apply DHS sample weights to ML training.
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_val, label=y_val)
    default_model = xgb.train(
        default_params, dtrain, num_boost_round=300,
        verbose_eval=False)
    default_auc = float(roc_auc_score(y_val, default_model.predict(dval)))
    print(f"\nDefault AUC (val):    {default_auc:.4f}")
    print(f"Optimised AUC (val):  {best_auc:.4f}  "
          f"(+{best_auc-default_auc:+.4f})")
    print("\nTo retrain the full model with best params, update")
    print("s07_xgboost_training.py to load optuna_best_params.json.")


if __name__ == "__main__":
    main()
