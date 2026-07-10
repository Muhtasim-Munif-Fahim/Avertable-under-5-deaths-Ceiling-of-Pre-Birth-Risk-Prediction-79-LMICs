"""
s05d_sensitivity_logit_decomp.py — eTable S9b (logit sensitivity)
=================================================================

Runs a binomial-logit version of the Wagstaff concentration-index
decomposition on the same 5% systematic sample as ``s05d``, then
emits a side-by-side table comparing the LPM-based (primary) and
logit-based (sensitivity) contribution percentages by aggregated
factor and region.

This addresses reviewer concern that the LPM residual share in
several regions (notably ECA and SSEA) is large; under the logit
substitution, the marginal-effect-weighted contributions absorb a
larger share of the total concentration index because the nonlinear
link function better fits the binary outcome at low prevalence.

Outputs:
  output/tables/table_4b_inequality_decomp_logit.csv
  output/tables/table_4c_inequality_decomp_lpm_vs_logit.csv
"""
from __future__ import annotations
import sys, functools
from pathlib import Path

import numpy as np
import pandas as pd

print = functools.partial(print, flush=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.paths import TABLES_DIR
from src.inferential.s05d_inequality_decomposition import (
    load_sample, decompose_logit, decompose_wagstaff, aggregate_factors,
)


def _build_logit_table(df: pd.DataFrame) -> pd.DataFrame:
    groups = [
        ("All LMICs (2000-2024)", df),
    ]
    for region in ["ECA", "LAC", "MENA", "SSA", "SSEA"]:
        sub = df[(df["who_region"] == region) & (df["survey_year"] >= 2010)]
        if len(sub) >= 1000:
            groups.append((f"{region} (2010-2024)", sub))

    rows = []
    for name, sub in groups:
        print(f"  logit decomposition: {name} (n={len(sub):,})")
        rows.append(decompose_logit(sub, name))
    return pd.concat([r for r in rows if not r.empty], ignore_index=True)


def _build_comparison(lpm_agg: pd.DataFrame,
                      logit_agg: pd.DataFrame) -> pd.DataFrame:
    lpm_pct = lpm_agg.rename(
        columns={"Percent_Contribution": "Percent_LPM"}
    )[["Group", "Factor", "Percent_LPM"]]
    logit_pct = logit_agg.rename(
        columns={"Percent_Contribution": "Percent_Logit"}
    )[["Group", "Factor", "Percent_Logit"]]
    merged = lpm_pct.merge(logit_pct, on=["Group", "Factor"], how="outer")
    merged["Delta_pp"] = merged["Percent_Logit"] - merged["Percent_LPM"]
    factor_order = [
        "Education", "Residence", "Birth order", "Birth interval",
        "Maternal age", "Birth type", "Child sex", "Marital status",
        "Decade trend", "Residual", "TOTAL_CI",
    ]
    merged["Factor"] = pd.Categorical(
        merged["Factor"], categories=factor_order, ordered=True
    )
    merged = merged.sort_values(by=["Group", "Factor"]).reset_index(drop=True)
    for c in ["Percent_LPM", "Percent_Logit", "Delta_pp"]:
        merged[c] = merged[c].round(2)
    return merged


def main():
    print("=" * 60)
    print("S05d sensitivity: logit-based Wagstaff decomposition")
    print("=" * 60)
    df = load_sample()
    print(f"  Loaded 5% sample: {len(df):,} rows")

    logit_raw = _build_logit_table(df)
    out1 = TABLES_DIR / "table_4b_inequality_decomp_logit.csv"
    logit_raw.to_csv(out1, index=False)
    print(f"  Wrote {out1.name}")

    # Aggregate logit decomposition to factor-level percentages.
    # Re-use the same aggregator as the LPM path; it works on Predictor names.
    logit_for_agg = logit_raw[~logit_raw["Predictor"].isin(
        ["LOGIT_PSEUDO_R2"]
    )].copy()
    logit_agg = aggregate_factors(logit_for_agg)

    # Re-run the LPM decomposition once for the comparison.
    lpm_rows = []
    for name, sub in [
        ("All LMICs (2000-2024)", df),
        *[(f"{r} (2010-2024)",
           df[(df["who_region"] == r) & (df["survey_year"] >= 2010)])
          for r in ["ECA", "LAC", "MENA", "SSA", "SSEA"]
          if len(df[(df["who_region"] == r) & (df["survey_year"] >= 2010)]) >= 1000],
    ]:
        lpm_rows.append(decompose_wagstaff(sub, name))
    lpm_raw = pd.concat(lpm_rows, ignore_index=True)
    lpm_for_agg = lpm_raw[~lpm_raw["Predictor"].isin(["LPM_R2"])].copy()
    lpm_agg = aggregate_factors(lpm_for_agg)

    comp = _build_comparison(lpm_agg, logit_agg)
    out2 = TABLES_DIR / "table_4c_inequality_decomp_lpm_vs_logit.csv"
    comp.to_csv(out2, index=False)
    print(f"  Wrote {out2.name}")
    print(comp.to_string(index=False))


if __name__ == "__main__":
    main()
