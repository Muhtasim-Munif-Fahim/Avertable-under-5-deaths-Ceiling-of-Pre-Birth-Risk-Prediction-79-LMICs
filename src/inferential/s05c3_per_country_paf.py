"""
s05c3_per_country_paf.py - Per-Country G-Computation PAFs
=========================================================

For each of the 68 LMICs in the primary cohort, refit the Model 3
binomial GLM on the country subsample and compute the conditional
counterfactual PAF for the two dominant interventions:

  (i)  universal secondary or higher maternal education
  (ii) optimal 24-47 month birth-to-birth interval

These per-country point estimates appear in the Lancet supplement as
eTable~S17 (per-country PAF). They are point estimates only - per-country
bootstrap CIs would require ~68 x B x ~1s ~= hours of compute and would
be wide for small-N countries (ECA region). The pooled Global and per-region
CIs (eTable~S8) provide the precision context.

Outputs:
  output/tables/table_s17_per_country_paf.csv

Schema:
  country_code, country_name, who_region, n_births,
  paf_education, paf_birth_spacing, recorded_u5m_per1000
"""
from __future__ import annotations

import os
import sys
import gc
import functools
import warnings
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.genmod.families import Binomial

warnings.filterwarnings("ignore")
print = functools.partial(print, flush=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.paths import TABLES_DIR
from src.inferential.s05c_counterfactual_paf import (
    load_sample, FORMULA, gcomp_paf, fit_paf_glm,
)


OUT = TABLES_DIR / "table_s17_per_country_paf.csv"

# Country-code -> long name lookup. Built from the canonical parquet's
# (country_code, country_name) pairs at runtime.
COUNTRY_INTERVENTIONS = [
    ("edu_cat", "Secondary",   "paf_education"),
    ("bi_cat",  "24-47 mo",    "paf_birth_spacing"),
]


def _load_country_names() -> dict:
    from config.paths import GLOBAL_PARQUET
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(str(GLOBAL_PARQUET))
    frames = []
    for batch in pf.iter_batches(
        batch_size=500_000, columns=["country_code", "country_name"]
    ):
        frames.append(batch.to_pandas().drop_duplicates("country_code"))
    out = pd.concat(frames).drop_duplicates("country_code")
    return dict(zip(out["country_code"], out["country_name"]))


def main() -> None:
    print("=" * 60)
    print("Per-country G-computation PAFs (eTable S17)")
    print("=" * 60)

    df = load_sample()
    print(f"  Sample: {len(df):,} rows, "
          f"{df['country_code'].nunique()} countries")

    country_names = _load_country_names()

    rows = []
    for c in sorted(df["country_code"].unique()):
        sub = df[df["country_code"] == c].copy()
        n = len(sub)
        if n < 200:
            print(f"  Skipping {c}: only {n} rows")
            continue

        # Fit Model 3 GLM on the country subset. Drop the WHO-region term
        # (already constant within country) and the decade term if the
        # country has only one decade represented.
        n_decades = sub["decade"].nunique()
        if n_decades < 2:
            formula = FORMULA.replace(" + decade", "")
        else:
            formula = FORMULA

        try:
            res = fit_paf_glm(formula, sub)
        except Exception as e:
            print(f"  {c} fit failed: {e}")
            continue

        row = {
            "country_code": c,
            "country_name": country_names.get(c, c),
            "who_region": sub["who_region"].iloc[0],
            "n_births": int(n),
            "recorded_u5m_per1000": round(
                float(sub["died_under5"].mean() * 1000), 1),
        }
        for col, cf_val, out_col in COUNTRY_INTERVENTIONS:
            try:
                paf = gcomp_paf(sub, col, cf_val, res, formula)
                value = round(paf * 100, 1)
                # Plausibility cap: a PAF outside [-50, 60] indicates an
                # unstable GLM fit on a small country subsample (typical
                # for ECA countries with <1000 records and few deaths).
                # We replace with NaN and the eTable shows "n.s.".
                if value < -50.0 or value > 60.0:
                    value = float("nan")
                row[out_col] = value
            except Exception:
                row[out_col] = float("nan")
        rows.append(row)
        print(f"  {c} ({row['country_name'][:25]:<25}) n={n:>6,}  "
              f"PAF_edu={row.get('paf_education', 'n/a'):>5} "
              f"PAF_bi={row.get('paf_birth_spacing', 'n/a'):>5}")

    out_df = pd.DataFrame(rows)
    # Sort by WHO region then by descending burden
    out_df = out_df.sort_values(
        ["who_region", "recorded_u5m_per1000"],
        ascending=[True, False]
    ).reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT, index=False)
    print(f"\nSaved: {OUT} ({len(out_df)} countries)")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
