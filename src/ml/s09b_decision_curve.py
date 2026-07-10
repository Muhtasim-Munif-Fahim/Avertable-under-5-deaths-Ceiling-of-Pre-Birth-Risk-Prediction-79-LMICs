"""
s09b_decision_curve.py - Decision Curve Analysis
=================================================

Net benefit (Vickers & Elkin 2006) across the clinically plausible
threshold range. We compare three strategies against the model:
  - Treat-all (intervene on every birth)
  - Treat-none (no intervention)

Net benefit at threshold p_t:
    NB = (TP/N) - (FP/N) * (p_t / (1 - p_t))

The XGBoost classifier is scored on its RAW probabilities (ranking
baseline) AND on its ISOTONIC-recalibrated probabilities (deployable
absolute-risk surface; see s09a_recalibration.py). At the prediction
prevalence (~6-8%) the raw model's probability scale is inflated by
the scale_pos_weight used during training, which mechanically suppresses
its net benefit above prevalence. Recalibrated probabilities restore
the correct decision-threshold mapping and are what Lancet readers
should use to judge clinical value.

Outputs:
  output/tables/table_6_decision_curve.csv
  output/figures/figure_8_decision_curve.png
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.print_util import print
from config.paths import FIGURES_DIR, OUTPUT_DIR, TABLES_DIR

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
    return X, y[y_col].astype(int).values


def _load_recalibrated(split: str, label: str) -> np.ndarray | None:
    """Load persisted probabilities from s09a_recalibration. `label` is one
    of {'raw', 'platt', 'isotonic'}. Returns None if not yet produced."""
    p_file = ML_DIR / f"p_{split}__{label}.parquet"
    if not p_file.exists():
        return None
    return pd.read_parquet(p_file)["p"].to_numpy()


def net_benefit(y_true, y_prob, thresholds, model_label: str):
    """Vickers & Elkin (2006) net benefit at each decision threshold."""
    n = len(y_true)
    prevalence = float(y_true.mean())
    rows = []
    for pt in thresholds:
        flag = y_prob >= pt
        tp = int(((flag == 1) & (y_true == 1)).sum())
        fp = int(((flag == 1) & (y_true == 0)).sum())
        nb_model = (tp / n) - (fp / n) * (pt / (1 - pt))
        nb_all = prevalence - (1 - prevalence) * (pt / (1 - pt))
        rows.append({
            "model": model_label,
            "threshold": pt,
            "nb_model": nb_model,
            "nb_treat_all": nb_all,
            "nb_treat_none": 0.0,
        })
    return pd.DataFrame(rows)


def main():
    print("=" * 60)
    print("Decision Curve Analysis (raw vs isotonic-recalibrated)")
    print("=" * 60)

    booster = xgb.Booster()
    booster.load_model(str(ML_DIR / "xgb_model.json"))

    # Threshold range tuned to typical childhood-mortality risk levels
    # (events are 5-10% of births globally so meaningful thresholds are
    # 1%-30%, finer below 10%).
    thresholds = np.r_[
        np.arange(0.005, 0.10, 0.005),
        np.arange(0.10, 0.30, 0.01),
    ]

    all_dfs = []
    for split in ["test_time", "test_space"]:
        try:
            X, y = _load(split)
        except FileNotFoundError as e:
            print(f"  Skipping {split}: {e}")
            continue
        p_raw = booster.predict(xgb.DMatrix(X))
        p_iso = _load_recalibrated(split, "isotonic")
        if p_iso is None:
            print(f"  WARNING: no recalibrated probabilities for {split}; "
                  "run s09a_recalibration.py first. Using raw only.")
            df = net_benefit(y, p_raw, thresholds, "xgb_raw")
        else:
            df_raw = net_benefit(y, p_raw, thresholds, "xgb_raw")
            df_iso = net_benefit(y, p_iso, thresholds, "xgb_isotonic")
            df = pd.concat([df_raw, df_iso], ignore_index=True)
        df.insert(0, "split", split)
        all_dfs.append(df)
        print(f"  {split}: prevalence={y.mean():.3f}, n={len(y):,}")

    if not all_dfs:
        return

    out = pd.concat(all_dfs, ignore_index=True)
    out_csv = TABLES_DIR / "table_6_decision_curve.csv"
    out.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv.name}")

    # Plot one panel per split with raw + recalibrated curves overlaid.
    splits = sorted({df["split"].iloc[0] for df in all_dfs})
    fig, axes = plt.subplots(1, len(splits), figsize=(6 * len(splits), 4),
                             sharey=True)
    if len(splits) == 1:
        axes = [axes]
    for ax, split in zip(axes, splits):
        df = next(d for d in all_dfs if d["split"].iloc[0] == split)
        for model_label, color, ls in [
            ("xgb_raw", "#999", "--"),
            ("xgb_isotonic", "#377eb8", "-"),
        ]:
            sub = df[df["model"] == model_label]
            if not sub.empty:
                ax.plot(sub["threshold"], sub["nb_model"], ls,
                        color=color, linewidth=2,
                        label=f"XGB ({model_label.replace('xgb_', '')})")
        # Treat-all is identical across models for a given split
        any_model_df = df[df["model"] == df["model"].iloc[0]]
        ax.plot(any_model_df["threshold"], any_model_df["nb_treat_all"], ":",
                color="#888", linewidth=1.5, label="Treat all")
        ax.axhline(0, color="black", linewidth=1, label="Treat none")
        ax.set_xlabel("Threshold probability")
        ax.set_ylabel("Net benefit")
        ax.set_title(split)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_ylim(min(0, any_model_df["nb_treat_all"].min() * 1.1),
                    df["nb_model"].max() * 1.2)
    fig.tight_layout()
    fig_out = FIGURES_DIR / "figure_8_decision_curve.png"
    fig.savefig(fig_out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_out.name}")


if __name__ == "__main__":
    main()
