"""
s11_sensitivity.py — Sensitivity Analyses
==========================================

Evaluates whether the XGBoost model's performance is sensitive to:
  1. Pre-COVID (2019-2021) vs COVID/post-COVID (2022-2024) period
  2. >=2010 cohort restriction (uses s06 same-every-7th sampling)
  3. Temporal-split year sensitivity (2018 vs 2019 vs 2020 cutoffs)

Strategy: loads survey_year meta from global parquet (every-7th
sample, same as s06) and aligns to X_test_time rows, then computes
per-stratum AUC without retraining the model.

Outputs:
  output/tables/table_s11_sensitivity.csv
  output/figures/figure_14_sensitivity.png
"""
import sys, gc, json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.print_util import print
from config.paths import OUTPUT_DIR, TABLES_DIR, FIGURES_DIR, GLOBAL_PARQUET
from src.ml.s06_ml_data_prep import ML_SAMPLING_STEP

ML_DIR = OUTPUT_DIR / "ml"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _load_test(name: str):
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


def load_meta_for_split(split: str) -> pd.Series:
    """Reconstruct survey_year for X_test_time rows using the same
    every-7th-row subsample and the same split filtering as s06."""
    with open(ML_DIR / "country_splits.json") as f:
        splits = json.load(f)

    pf = pq.ParquetFile(str(GLOBAL_PARQUET))
    n_total = pf.metadata.num_rows
    sample_indices = set(range(0, n_total, ML_SAMPLING_STEP))

    if split == "test_time":
        train_countries = set(splits["training_countries"])
        year_filter = lambda yr, cc: (yr >= 2019) and (cc in train_countries)
    else:
        holdout_countries = set(
            splits.get("spatial_holdout_countries", splits.get("holdout_countries", []))
        )
        year_filter = lambda yr, cc: cc in holdout_countries

    years = []
    row_offset = 0
    for batch in pf.iter_batches(
            batch_size=500_000,
            columns=["survey_year", "country_code", "died_under5", "in_primary_scope"]):
        df = batch.to_pandas()
        n = len(df)
        local_idx = [i - row_offset for i in sample_indices
                     if row_offset <= i < row_offset + n]
        if local_idx:
            sub = df.iloc[local_idx].dropna(subset=["died_under5"])
            for _, row in sub.iterrows():
                if row.get("in_primary_scope", 1) == 1:
                    yr = int(row["survey_year"]) if not pd.isna(row["survey_year"]) else 0
                    cc = row["country_code"]
                    if year_filter(yr, cc):
                        years.append(yr)
        row_offset += n
        del df, batch
    gc.collect()
    return pd.Series(years)


def main():
    print("=" * 60)
    print("Sensitivity Analyses")
    print("=" * 60)

    booster = xgb.Booster()
    booster.load_model(str(ML_DIR / "xgb_model.json"))

    rows = []

    # ── Temporal test: stratify by period ──────────────────────────
    print("\nLoading X_test_time + year metadata...")
    try:
        X_tt, y_tt = _load_test("test_time")
        years = load_meta_for_split("test_time")
        min_len = min(len(X_tt), len(years))
        X_tt = X_tt.iloc[:min_len]
        y_tt = y_tt[:min_len]
        years = years.iloc[:min_len].values
        p_tt = booster.predict(xgb.DMatrix(X_tt))
        print(f"  n={len(y_tt):,}, year range={years.min()}-{years.max()}")

        for label, mask in [
            ("Pre-COVID (2019-2021)", (years >= 2019) & (years <= 2021)),
            ("COVID/post-COVID (2022-2024)", (years >= 2022) & (years <= 2024)),
            ("Full temporal test", np.ones(len(years), dtype=bool)),
        ]:
            ym, pm = y_tt[mask], p_tt[mask]
            if mask.sum() < 200 or ym.sum() < 10:
                print(f"  Skipping {label}: n={mask.sum()}, events={ym.sum()}")
                continue
            auc = float(roc_auc_score(ym, pm))
            rows.append({
                "analysis": "COVID period split",
                "stratum": label,
                "n": int(mask.sum()),
                "events": int(ym.sum()),
                "prevalence": float(ym.mean()),
                "auc": auc,
            })
            print(f"  {label}: n={mask.sum():,}, auc={auc:.4f}")

        # ≥2010 restriction: compare to full training-era test
        mask_2010 = years >= 2010
        ym_2010, pm_2010 = y_tt[mask_2010], p_tt[mask_2010]
        if mask_2010.sum() > 200 and ym_2010.sum() > 10:
            auc_2010 = float(roc_auc_score(ym_2010, pm_2010))
            rows.append({
                "analysis": "Cohort restriction >=2010",
                "stratum": ">=2010 births (temporal test)",
                "n": int(mask_2010.sum()),
                "events": int(ym_2010.sum()),
                "prevalence": float(ym_2010.mean()),
                "auc": auc_2010,
            })
            print(f"  >=2010 restriction: n={mask_2010.sum():,}, auc={auc_2010:.4f}")

        # Temporal split year sensitivity
        for split_yr in [2018, 2019, 2020]:
            mask_yr = years >= split_yr
            ym_yr, pm_yr = y_tt[mask_yr], p_tt[mask_yr]
            if mask_yr.sum() < 200 or ym_yr.sum() < 10:
                continue
            auc_yr = float(roc_auc_score(ym_yr, pm_yr))
            rows.append({
                "analysis": "Temporal split sensitivity",
                "stratum": f"Test = survey_year >= {split_yr}",
                "n": int(mask_yr.sum()),
                "events": int(ym_yr.sum()),
                "prevalence": float(ym_yr.mean()),
                "auc": auc_yr,
            })
            print(f"  Split @{split_yr}: n={mask_yr.sum():,}, auc={auc_yr:.4f}")

    except FileNotFoundError as e:
        print(f"  Skipping test_time: {e}")

    # ── Spatial test: full holdout ──────────────────────────────────
    try:
        X_ts, y_ts = _load_test("test_space")
        p_ts = booster.predict(xgb.DMatrix(X_ts))
        auc_ts = float(roc_auc_score(y_ts, p_ts))
        rows.append({
            "analysis": "Reference",
            "stratum": "Spatial holdout (all countries)",
            "n": int(len(y_ts)),
            "events": int(y_ts.sum()),
            "prevalence": float(y_ts.mean()),
            "auc": auc_ts,
        })
        print(f"\nSpatial holdout: n={len(y_ts):,}, auc={auc_ts:.4f}")
    except FileNotFoundError as e:
        print(f"  Skipping test_space: {e}")

    out_df = pd.DataFrame(rows)
    out_csv = TABLES_DIR / "table_s11_sensitivity.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv.name}")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
