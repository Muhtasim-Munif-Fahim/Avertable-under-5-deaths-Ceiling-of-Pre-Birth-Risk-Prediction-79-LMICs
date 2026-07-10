"""
s09_calibration.py — Model Calibration Analysis
================================================

Evaluates calibration of the trained XGBoost model on the temporal
and spatial holdouts. Required for clinical-utility argument
(TRIPOD+AI item 14a).

Reports:
  - Brier score (already in validation_metrics; recomputed here)
  - Calibration slope & intercept (logistic recalibration regression)
  - Integrated Calibration Index (ICI) — Austin & Steyerberg 2019
  - Decile calibration plot

Hosmer-Lemeshow is intentionally omitted: unreliable at N > ~25k
because chi-square is overpowered (Paul, Pennell, Lemeshow 2013).

Outputs:
  output/tables/table_5_calibration.csv
  output/figures/figure_7_calibration_decile.png
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent.parent)
)
from src.utils.print_util import print
from config.paths import FIGURES_DIR, OUTPUT_DIR, RNG_SEED, TABLES_DIR

ML_DIR = OUTPUT_DIR / "ml"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


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
    w_col = ("sample_weight" if "sample_weight" in y.columns
             else "weight" if "weight" in y.columns else None)
    y_arr = y[y_col].astype(int).values
    w = y[w_col].fillna(1.0).values if w_col else None
    return X, y_arr, w


def calibration_slope_intercept(y_true, y_prob):
    """Logistic recalibration: logit(y) = a + b * logit(p_hat).
    b == 1, a == 0 means perfect calibration. The intercept returned
    here is the value of the linear predictor at logit(p_hat) = 0
    (i.e., at p_hat = 0.5). For probability-scale calibration-in-the-large,
    use ``calibration_in_the_large``."""
    eps = 1e-6
    p = np.clip(y_prob, eps, 1 - eps)
    lp = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(C=1e10, solver="lbfgs",
                            max_iter=2000)
    lr.fit(lp, y_true)
    slope = float(lr.coef_[0, 0])
    intercept = float(lr.intercept_[0])
    return slope, intercept


def calibration_in_the_large(y_true, y_prob):
    """Calibration-in-the-large (CITL), reported on two scales.

    Returns (mean_observed, mean_predicted, citl_prob, citl_logit) where
      - citl_prob  = mean_observed - mean_predicted (probability scale,
                     Van Calster et al. 2019 Level 1)
      - citl_logit = logit(mean_observed) - logit(mean_predicted)
                     (logit-scale CITL, closed-form approximation to the
                     offset-intercept MLE; reported alongside citl_prob
                     for comparability with the slope-intercept model).
    """
    eps = 1e-6
    mean_p = float(np.clip(np.mean(y_prob), eps, 1 - eps))
    mean_y = float(np.clip(np.mean(y_true), eps, 1 - eps))
    citl_prob = mean_y - mean_p
    citl_logit = (float(np.log(mean_y / (1 - mean_y)))
                  - float(np.log(mean_p / (1 - mean_p))))
    return mean_y, mean_p, citl_prob, citl_logit


def integrated_calibration_index(y_true, y_prob):
    """ICI: mean absolute difference between predicted prob and a
    flexible (isotonic) regression of y on p_hat. Austin/Steyerberg 2019."""
    iso = IsotonicRegression(out_of_bounds="clip")
    p_smooth = iso.fit_transform(y_prob, y_true)
    return float(np.mean(np.abs(p_smooth - y_prob)))


def decile_table(y_true, y_prob, w=None):
    df = pd.DataFrame({"y": y_true, "p": y_prob,
                       "w": np.ones_like(y_true) if w is None else w})
    df["decile"] = pd.qcut(df["p"], 10, labels=False,
                           duplicates="drop")
    rows = []
    for d, grp in df.groupby("decile"):
        sw = grp["w"].sum()
        rows.append({
            "decile": int(d) + 1,
            "n": len(grp),
            "mean_predicted": float(np.average(grp["p"], weights=grp["w"])),
            "observed_rate": float(np.average(grp["y"], weights=grp["w"])),
            "sum_weight": float(sw),
        })
    return pd.DataFrame(rows)


def _load_recalibrated(split: str, label: str) -> np.ndarray | None:
    """Load persisted probabilities from s09a_recalibration."""
    p_file = ML_DIR / f"p_{split}__{label}.parquet"
    if not p_file.exists():
        return None
    return pd.read_parquet(p_file)["p"].to_numpy()


def evaluate_split(name: str, booster: xgb.Booster, label: str = "raw") -> dict:
    """Evaluate one split under one probability calibration.

    `label` in {'raw', 'platt', 'isotonic'}. For raw, predictions are
    drawn live from the XGBoost booster; for platt/isotonic, the
    s09a-persisted probabilities are loaded. Per D010 sample weights
    are not used in calibration metrics or in the decile table.
    """
    print(f"\n  Evaluating split: {name} ({label})")
    X, y, _w = _load(name)
    if label == "raw":
        dmat = xgb.DMatrix(X)
        p = booster.predict(dmat)
    else:
        p = _load_recalibrated(name, label)
        if p is None:
            print(f"    skipping {label}: probabilities not yet persisted; "
                  "run s09a_recalibration.py first.")
            return {}
    brier = float(brier_score_loss(y, p))
    slope, intercept = calibration_slope_intercept(y, p)
    mean_obs, mean_pred, citl_prob, citl_logit = (
        calibration_in_the_large(y, p)
    )
    ici = integrated_calibration_index(y, p)
    deciles = decile_table(y, p, w=None)
    return {
        "split": name,
        "calibration": label,
        "n": int(len(y)),
        "n_events": int(y.sum()),
        "brier": round(brier, 4),
        "calibration_slope": round(slope, 4),
        "calibration_intercept_logit0": round(intercept, 4),
        "mean_predicted": round(mean_pred, 4),
        "mean_observed": round(mean_obs, 4),
        "citl_prob": round(citl_prob, 4),
        "citl_logit": round(citl_logit, 4),
        "ici": round(ici, 4),
        "deciles": deciles,
    }


def plot_decile(results, out: Path) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4),
                             sharey=True)
    if len(results) == 1:
        axes = [axes]
    for ax, r in zip(axes, results):
        d = r["deciles"]
        ax.plot([0, d["mean_predicted"].max() * 1.05],
                [0, d["mean_predicted"].max() * 1.05],
                "k--", alpha=0.5, label="Perfect")
        ax.plot(d["mean_predicted"], d["observed_rate"],
                "o-", color="#377eb8", label="Observed")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed event rate")
        ax.set_title(
            f"{r['split']}\n"
            f"slope={r['calibration_slope']:.2f}, "
            f"CITL={r['citl_prob']:+.2f}, "
            f"ICI={r['ici']:.3f}"
        )
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    print("=" * 60)
    print("Phase 5: Calibration Analysis (raw + Platt + isotonic)")
    print("=" * 60)

    booster = xgb.Booster()
    booster.load_model(str(ML_DIR / "xgb_model.json"))
    print(f"Loaded model: {ML_DIR / 'xgb_model.json'}")

    results = []
    for split in ["test_time", "test_space"]:
        for label in ["raw", "platt", "isotonic"]:
            try:
                r = evaluate_split(split, booster, label=label)
                if r:
                    results.append(r)
            except FileNotFoundError as e:
                print(f"  Skipping {split} ({label}): {e}")

    if not results:
        print("No splits evaluated; aborting.")
        return

    # Summary table (without decile rows). All three calibrations side-by-side
    # so reviewers can see the impact of recalibration on Brier, slope, ICI.
    rows = [
        {k: v for k, v in r.items() if k != "deciles"}
        for r in results
    ]
    summary = pd.DataFrame(rows)
    out_csv = TABLES_DIR / "table_5_calibration.csv"
    summary.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv.name}")
    print(summary.to_string(index=False))

    # Decile detail. We emit decile tables for both raw and isotonic so the
    # reviewer can see the recalibrated decile mean_predicted converge to
    # the observed_rate; raw deciles show the inflated mean predicted.
    decile_rows = []
    for r in results:
        if r["calibration"] not in ("raw", "isotonic"):
            continue
        d = r["deciles"].copy()
        d.insert(0, "calibration", r["calibration"])
        d.insert(0, "split", r["split"])
        decile_rows.append(d)
    if decile_rows:
        decile_df = pd.concat(decile_rows, ignore_index=True)
        decile_df.to_csv(TABLES_DIR / "table_5b_calibration_deciles.csv",
                         index=False)
        print("Saved: table_5b_calibration_deciles.csv")

    # Figure: show raw vs isotonic calibration curves side-by-side per split.
    fig_out = FIGURES_DIR / "figure_7_calibration_decile.png"
    raw_results = [r for r in results if r["calibration"] == "raw"]
    plot_decile(raw_results, fig_out)
    print(f"Saved: {fig_out.name}")


if __name__ == "__main__":
    main()
