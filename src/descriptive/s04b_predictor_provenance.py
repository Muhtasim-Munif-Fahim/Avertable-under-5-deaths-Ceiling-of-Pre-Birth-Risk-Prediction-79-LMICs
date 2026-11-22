"""s04b_predictor_provenance.py — measurement-timing provenance for every predictor.

External-review amendment (2026-08-03). The reviewers' most serious methodological
objection was that variables described as "pre-birth" may not all have been measured
at or before the index birth. They are right for four of the nine model covariates.

In a retrospective birth history, a variable can be:
  * **at birth** — fixed at or before the index birth and recorded as such
    (child sex, multiple birth, birth order, preceding interval);
  * **derived at birth** — reconstructed from century-month codes so that it refers
    to the birth even though the interview supplied the inputs (maternal age at
    birth = age at interview minus years since the birth);
  * **at interview** — the woman's or household's status when the survey was
    fielded, used here as a proxy for its value at the birth (education, current
    marital status, wealth quintile, urban residence).

The last group is the honest problem: a mother's highest education, household wealth
index, marital status, and place of residence can all change between a birth and the
interview, and DHS/MICS record only the interview value. This script emits the table
that lets a reader see exactly which is which, together with the observed missingness
in the analysis cohort, so the manuscript can label the covariate set *baseline*
accurately instead of asserting that all covariates are pre-birth.

This table is the sole source of truth for the covariate set (nine variables: five
at-birth, four at interview). It backs the Methods covariate paragraph, the
provenance eTable, and the regression guard in
tests/test_external_review_invariants.py. The machine-learning arm that originally
also consumed it was withdrawn 2026-08-03 and archived to
../../_archive/2026-08-03_ml_arm_withdrawn/, so no ML module reads this any more.

Output: output/phase4/manuscript/table_predictor_provenance.csv
Run:    python -m src.descriptive.s04b_predictor_provenance
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from config.paths import HARMONIZED_DIR, OUTPUT_DIR
from src.inferential.phase2_analysis import _read_pilot

OUT = OUTPUT_DIR / "phase4" / "manuscript"
PILOT = HARMONIZED_DIR / "global_analysis_v2_candidate.parquet"

# One row per harmonised model variable. `timing` is the field the whole amendment
# turns on; `known_at_birth` is its boolean form.
PROVENANCE: list[dict] = [
    dict(
        variable="child_male", label="Child sex",
        dhs_source="b4", mics_source="BH3",
        timing="At birth", known_at_birth=True, derived=False,
        harmonisation="Male=1, female=0; DHS b4 and MICS BH3 recoded to a common "
                      "indicator, MICS non-response set missing.",
        in_primary_model=True,
    ),
    dict(
        variable="multiple_birth", label="Multiple birth",
        dhs_source="b0", mics_source="BH2",
        timing="At birth", known_at_birth=True, derived=False,
        harmonisation="Any multiple birth=1, singleton=0.",
        in_primary_model=True,
    ),
    dict(
        variable="birth_order", label="Birth order",
        dhs_source="bord", mics_source="brthord",
        timing="At birth", known_at_birth=True, derived=False,
        harmonisation="Integer order. MICS truncates at 4 or more, coarsening the "
                      "parity contrast for the 11 MICS-only countries.",
        in_primary_model=True,
    ),
    dict(
        variable="birth_interval_months", label="Preceding birth interval",
        dhs_source="b11", mics_source="birthint (MICS6/7 ordinal) / raw months (MICS5)",
        timing="At birth", known_at_birth=True, derived=False,
        harmonisation="Months since the preceding birth; undefined for first births, "
                      "which are modelled as their own state rather than missing. "
                      "MICS6/7 ordinal categories mapped to category midpoints.",
        in_primary_model=True,
    ),
    dict(
        variable="maternal_age_at_birth", label="Maternal age at birth",
        dhs_source="v012 and b3 (via century-month codes)",
        mics_source="magebrt (ordinal, midpoint-mapped)",
        timing="Derived at birth", known_at_birth=True, derived=True,
        harmonisation="Age at interview minus years since the birth, clamped to "
                      "10-55. Refers to the birth, but is reconstructed from an "
                      "interview-reported age.",
        in_primary_model=True,
    ),
    dict(
        variable="edu_level", label="Maternal education",
        dhs_source="v106", mics_source="welevel",
        timing="At interview", known_at_birth=False, derived=False,
        harmonisation="Highest level attended (none/primary/secondary/higher); "
                      "post-secondary sub-categories collapsed to higher. Recorded "
                      "at interview: schooling completed after the index birth is "
                      "counted, so this is a proxy for education at the birth.",
        in_primary_model=True,
    ),
    dict(
        variable="currently_married", label="Current marital status",
        dhs_source="v501", mics_source="MSTATUS",
        timing="At interview", known_at_birth=False, derived=False,
        harmonisation="Married or cohabiting=1. Explicitly a *current* status; "
                      "union formation and dissolution after the birth are not "
                      "distinguishable.",
        in_primary_model=True,
    ),
    dict(
        variable="wealth_quintile", label="Household wealth quintile",
        dhs_source="v190", mics_source="windex5",
        timing="At interview", known_at_birth=False, derived=False,
        harmonisation="Survey-specific asset-index quintile, so quintiles are "
                      "relative within each survey and measured at interview, not "
                      "at the birth.",
        in_primary_model=True,
    ),
    dict(
        variable="urban", label="Urban residence",
        dhs_source="v025", mics_source="HH6",
        timing="At interview", known_at_birth=False, derived=False,
        harmonisation="Urban=1, rural=0, at the interview household. Migration "
                      "between the birth and the interview is not observed.",
        in_primary_model=True,
    ),
    dict(
        variable="hh_members", label="Household size",
        dhs_source="v136", mics_source="not harmonised",
        timing="At interview", known_at_birth=False, derived=False,
        harmonisation="Present in the DHS extract but absent from the harmonised "
                      "analysis file, so it is NOT a model feature and is listed "
                      "here only for completeness.",
        in_primary_model=False,
    ),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = _read_pilot(PILOT)
    eligible = pd.to_numeric(frame["died_under5_complete"], errors="coerce").notna()
    cohort = frame.loc[eligible]
    n = len(cohort)
    print(f"[provenance] exact covariate cohort: {n:,} births")

    is_mics = ~cohort["_source_file"].astype(str).str.contains(r"BR", case=True,
                                                               na=False)
    rows = []
    for spec in PROVENANCE:
        var = spec["variable"]
        if var in cohort.columns:
            col = cohort[var]
            miss_all = float(col.isna().mean() * 100.0)
            miss_dhs = float(col[~is_mics].isna().mean() * 100.0)
            miss_mics = (float(col[is_mics].isna().mean() * 100.0)
                         if is_mics.any() else np.nan)
            available = "Yes"
        else:
            miss_all = miss_dhs = miss_mics = np.nan
            available = "Not in v2 analysis file"
        rows.append({
            "variable": var,
            "label": spec["label"],
            "timing": spec["timing"],
            "known_at_index_birth": spec["known_at_birth"],
            "reconstructed": spec["derived"],
            "dhs_source_variable": spec["dhs_source"],
            "mics_source_variable": spec["mics_source"],
            "harmonisation_rule": spec["harmonisation"],
            "in_primary_model": spec["in_primary_model"],
            "available_in_v2_file": available,
            "pct_missing_all": None if np.isnan(miss_all) else round(miss_all, 2),
            "pct_missing_dhs": None if np.isnan(miss_dhs) else round(miss_dhs, 2),
            "pct_missing_mics": None if np.isnan(miss_mics) else round(miss_mics, 2),
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "table_predictor_provenance.csv", index=False)

    modelled = out[out["in_primary_model"]]
    n_at_birth = int(modelled["known_at_index_birth"].sum())
    print(f"[provenance] {len(modelled)} model covariates: {n_at_birth} known at the "
          f"index birth, {len(modelled) - n_at_birth} recorded at interview")
    print(out[["variable", "timing", "known_at_index_birth", "in_primary_model",
               "pct_missing_all"]].to_string(index=False))
    print(f"\n[provenance] wrote {OUT / 'table_predictor_provenance.csv'}")


if __name__ == "__main__":
    main()
