"""
s09d_regional_calibration.py -- Per-region calibration curves
==============================================================

Supplement analysis N3 (2026-06 rewrite): regional calibration of the
leakage-screened XGBoost model on both external holdouts, for raw and
isotonic-recalibrated predictions.

Uses the persisted prediction parquets from s09a (no re-training, no
re-prediction), and reconstructs the who_region label for each test row
with the same every-7th-row metadata alignment used by s11_sensitivity.

Outputs:
  output/tables/table_s_regional_calibration.csv
  output/figures/journal/eF_regional_calibration.png / .pdf

Run:  python -m src.ml.s09d_regional_calibration
"""
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.print_util import print
from config.paths import OUTPUT_DIR, TABLES_DIR, GLOBAL_PARQUET
from src.ml.s06_ml_data_prep import ML_SAMPLING_STEP
from src.ml.s09_calibration import (
    calibration_slope_intercept, calibration_in_the_large)

ML_DIR = OUTPUT_DIR / "ml"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

MIN_EVENTS = 100  # skip region-cohort cells with fewer outcome events

REGION_ORDER = ["SSA", "SSEA", "MENA", "LAC", "ECA"]  # burden-descending, SSA first


def load_meta(split: str) -> pd.DataFrame:
    """Reconstruct (who_region, country_code) for each test row using the
    same every-7th-row subsample and split filter as s06/s11."""
    with open(ML_DIR / "country_splits.json") as f:
        splits = json.load(f)

    pf = pq.ParquetFile(str(GLOBAL_PARQUET))
    if split == "test_time":
        train_countries = set(splits["training_countries"])
        keep = lambda yr, cc: (yr >= 2019) and (cc in train_countries)
    else:
        holdout = set(splits.get("spatial_holdout_countries",
                                 splits.get("holdout_countries", [])))
        keep = lambda yr, cc: cc in holdout

    recs = []
    row_offset = 0
    for batch in pf.iter_batches(
            batch_size=500_000,
            columns=["survey_year", "country_code", "who_region",
                     "died_under5", "in_primary_scope"]):
        df = batch.to_pandas()
        n = len(df)
        # every ML_SAMPLING_STEP-th global row
        local = np.arange(n) + row_offset
        sel = df.loc[local % ML_SAMPLING_STEP == 0]
        sel = sel.dropna(subset=["died_under5"])
        sel = sel[sel["in_primary_scope"] == 1]
        for _, row in sel.iterrows():
            yr = int(row["survey_year"]) if not pd.isna(row["survey_year"]) else 0
            cc = row["country_code"]
            if keep(yr, cc):
                recs.append((row["who_region"], cc))
        row_offset += n
        del df, batch
    gc.collect()
    return pd.DataFrame(recs, columns=["who_region", "country_code"])


def decile_curve(y, p, n_bins=10):
    """Observed event rate by predicted-risk decile."""
    q = pd.qcut(pd.Series(p), n_bins, labels=False, duplicates="drop")
    rows = []
    for b in sorted(pd.unique(q)):
        m = (q == b).values
        rows.append((float(np.mean(p[m])), float(np.mean(y[m])), int(m.sum())))
    return rows


def integrated_calibration_index(y, p):
    iso = IsotonicRegression(out_of_bounds="clip")
    cal = iso.fit_transform(p, y)
    return float(np.mean(np.abs(cal - p)))


def main():
    print("=" * 60)
    print("Regional calibration (N3)")
    print("=" * 60)

    table_rows = []
    curves = {}  # (split, kind, region) -> decile points

    for split, label in [("test_time", "Temporal holdout"),
                         ("test_space", "Geographic holdout")]:
        y = pd.read_parquet(ML_DIR / f"y_{split}.parquet")
        y_col = "died_under5" if "died_under5" in y.columns else "y"
        y = y[y_col].astype(int).values

        meta = load_meta(split)
        print(f"  {label}: y={len(y):,} rows, meta={len(meta):,} rows")
        n = min(len(y), len(meta))
        if abs(len(y) - len(meta)) > 0.01 * len(y):
            print("    WARNING: >1% length mismatch; truncating to align "
                  "(same convention as s11_sensitivity)")
        y = y[:n]
        regions = meta["who_region"].values[:n]

        for kind in ["raw", "isotonic"]:
            p = pd.read_parquet(
                ML_DIR / f"p_{split}__{kind}.parquet").iloc[:, 0].values[:n]
            for region in REGION_ORDER:
                m = regions == region
                ym, pm = y[m], p[m]
                if m.sum() == 0 or ym.sum() < MIN_EVENTS:
                    continue
                slope, intc = calibration_slope_intercept(ym, pm)
                mean_y, mean_p, citl_prob, citl_logit = \
                    calibration_in_the_large(ym, pm)
                ici = integrated_calibration_index(ym, pm)
                table_rows.append({
                    "cohort": label, "predictions": kind, "region": region,
                    "n": int(m.sum()), "events": int(ym.sum()),
                    "mean_observed": round(mean_y, 4),
                    "mean_predicted": round(mean_p, 4),
                    "citl_prob": round(citl_prob, 4),
                    "slope": round(slope, 3),
                    "ici": round(ici, 4),
                })
                curves[(split, kind, region)] = decile_curve(ym, pm)
                print(f"    {label[:8]} {kind:<8} {region}: n={m.sum():,}, "
                      f"slope={slope:.3f}, ICI={ici:.4f}")

    out = pd.DataFrame(table_rows)
    out_csv = TABLES_DIR / "table_s_regional_calibration.csv"
    out.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
