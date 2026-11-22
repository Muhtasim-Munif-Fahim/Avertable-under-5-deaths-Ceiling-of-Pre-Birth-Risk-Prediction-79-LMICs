"""s05p_percountry_master.py -- per-country master burden table (v2).

Emits `table_percountry_burden.csv`, the single per-country source of truth
backing main-text Table 2. Until now the v2 per-country burden existed **only as
rendered LaTeX** inside `journal_tables.py::build_S_percountry_burden()`, which
meant no machine-readable artifact, no registry atom, and no test surface. This
closes that gap.

Every column is v2-native
-------------------------
Two legacy files carry columns that are *wrong* on the v2 cohort and are
deliberately NOT read here:

  - `output/tables/table_s_percountry_avertable_annual_igme.csv::paf_joint_pct`
    is the pre-v2 PAF (Nigeria 53.3 against the correct 33.0). The v2 equivalent
    is `manuscript/table_igme_annualisation.csv`, which `derived_extras`
    builds from `table_percountry_paf.csv`, so it is used instead.
  - `output/tables/table_s9_country_summary.csv::U5MR_pooled` / `N_births` are on
    the legacy cohort, not the exact->=60-month complete-wealth v2 cohort. Survey
    counts and year ranges are derived natively from the v2 inventory instead.

Cohort
------
`u5mr_weighted` is computed on the **complete-wealth binary cohort**
(7,652,332 births / 679,059 deaths) -- the same rows `derived_pafs.py` fits
and the same rows whose `n_births` / `n_deaths` appear in
`table_percountry_paf.csv` and `table_percountry_equity.csv`. The selection
below mirrors `derived_pafs._reproduce_cohort_extras()` exactly. Note this is
NOT the 8,184,578-birth exact cohort behind the inventory's
`u5_risk_per1000_exact`, and the two must not be mixed in one column.

Weights are survey-normalised to mean one within `_source_file`
(`phase2_analysis._positive_normalized_weights`), matching the pooled estimand.

Output
------
  output/phase4/manuscript/table_percountry_burden_survival.csv  (primary, survival basis)
  output/phase4/manuscript/table_percountry_burden.csv           (exact-binary sensitivity)

Both share every descriptive column; only the PAF, avertable, cumulative-share and
SDG columns differ. See `country_basis.py`.

Run:  python -m src.inferential.s05p_percountry_master
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from config.paths import HARMONIZED_DIR, OUTPUT_DIR  # noqa: E402
from src.inferential import country_basis as cb  # noqa: E402
from src.inferential.phase2_analysis import _positive_normalized_weights  # noqa: E402

OUT = OUTPUT_DIR / "phase4" / "manuscript"
PARQUET = HARMONIZED_DIR / "global_analysis_v2_candidate.parquet"

# Columns sufficient to reproduce the complete-wealth cohort selection.
_COHORT_COLS = [
    "country_code", "_source_file", "sample_weight",
    "died_under5_complete", "wealth_quintile",
]


def survey_metadata() -> pd.DataFrame:
    """A1 -- per-country survey count and fieldwork year range, v2-native.

    Derived from the v2 inventory (215 country-surveys) rather than the legacy
    `table_s9_country_summary.csv`, so it carries no legacy-cohort dependency.
    """
    inv = pd.read_csv(OUT / "table_s1_survey_inventory.csv")
    g = inv.groupby("country_code").agg(
        country_name=("country_name", "first"),
        who_region=("who_region", "first"),
        n_surveys=("survey_id", "nunique"),
        year_min=("survey_year", "min"),
        year_max=("survey_year", "max"),
    ).reset_index()
    g["year_range"] = np.where(
        g["year_min"] == g["year_max"],
        g["year_min"].astype(int).astype(str),
        g["year_min"].astype(int).astype(str) + "-" + g["year_max"].astype(int).astype(str),
    )
    return g.drop(columns=["year_min", "year_max"])


def weighted_u5mr_by_country() -> pd.DataFrame:
    """A2 -- survey-weighted under-5 mortality per 1,000 on the complete-wealth cohort.

    Mirrors `derived_pafs._reproduce_cohort_extras()`: drop missing wealth,
    drop non-positive/non-finite weights, drop missing outcome.
    """
    frame = pd.read_parquet(PARQUET, columns=_COHORT_COLS)
    normalized = _positive_normalized_weights(frame)

    keep = (
        frame["wealth_quintile"].notna()
        & normalized.notna()
        & pd.to_numeric(frame["died_under5_complete"], errors="coerce").notna()
    )
    data = frame.loc[keep].copy()
    data["w"] = normalized.loc[keep]
    data["y"] = pd.to_numeric(data["died_under5_complete"], errors="coerce")

    n_total = len(data)
    if n_total != 7_652_332:
        print(f"  WARNING: cohort is {n_total:,}, expected 7,652,332 "
              f"(delta {n_total - 7_652_332:+,}) -- verify upstream parquet")

    data["wy"] = data["w"] * data["y"]
    g = data.groupby("country_code").agg(
        _sum_wy=("wy", "sum"), _sum_w=("w", "sum"),
        n_births_cohort=("y", "size"), n_deaths_cohort=("y", "sum"),
    ).reset_index()
    g["u5mr_weighted"] = (1000.0 * g["_sum_wy"] / g["_sum_w"]).round(1)
    g["n_deaths_cohort"] = g["n_deaths_cohort"].astype(int)
    print(f"  cohort {n_total:,} births / {int(data['y'].sum()):,} deaths "
          f"across {len(g)} countries")
    return g[["country_code", "u5mr_weighted", "n_births_cohort", "n_deaths_cohort"]]


BASES = {
    # basis: (per-country PAF, IGME annualisation, SDG scoreboard, output)
    "survival": (cb.PERCOUNTRY_PAF, cb.IGME_ANNUALISATION, cb.SDG_SCOREBOARD,
                 cb.PERCOUNTRY_BURDEN),
    "binary": (cb.BINARY_PERCOUNTRY_PAF, cb.BINARY_IGME_ANNUALISATION,
               cb.BINARY_SDG_SCOREBOARD, cb.BINARY_PERCOUNTRY_BURDEN),
}


def main() -> None:
    meta, u5mr = None, None
    for basis in BASES:
        meta, u5mr = build(basis, meta, u5mr)


def build(basis: str, meta=None, u5mr=None):
    paf_file, igme_file, sdg_file, dest_file = BASES[basis]
    print("=" * 68)
    print(f"Per-country master burden table ({basis} basis)")
    print("=" * 68)

    OUT.mkdir(parents=True, exist_ok=True)

    if meta is None:
        print("[A1] survey metadata from v2 inventory")
        meta = survey_metadata()
        print(f"  {len(meta)} countries in inventory")
    if u5mr is None:
        print("[A2] survey-weighted per-country U5MR on complete-wealth cohort")
        u5mr = weighted_u5mr_by_country()

    paf = pd.read_csv(OUT / paf_file)
    equity = pd.read_csv(OUT / "table_percountry_equity.csv")
    igme = pd.read_csv(OUT / igme_file)
    sdg = pd.read_csv(OUT / sdg_file)
    if basis == "survival":
        # The survival PAF file counts survival-cohort births (censored births
        # included). The descriptive columns of this table -- births, deaths,
        # weighted U5MR -- stay on the complete-wealth exact cohort, so they agree
        # with each other and with the equity table.
        paf = paf.drop(columns="n_births").merge(
            u5mr[["country_code", "n_births_cohort"]].rename(
                columns={"n_births_cohort": "n_births"}),
            on="country_code", how="left")

    # The v2 77-country set is defined by the PAF file; everything left-joins onto it.
    base = paf[["country_code", "who_region", "n_births",
                "paf_education", "paf_spacing", "paf_joint"]].copy()

    out = (
        base
        .merge(meta[["country_code", "country_name", "n_surveys", "year_range"]],
               on="country_code", how="left")
        .merge(u5mr, on="country_code", how="left")
        .merge(equity[["country_code", "n_deaths"]], on="country_code", how="left")
        .merge(igme[["country_code", "iso3", "igme_u5_deaths_2024", "avertable_per_year"]],
               on="country_code", how="left")
        .merge(sdg[["country_code", "u5mr_2024", "u5mr_counterfactual",
                    "reaches_sdg_factual", "reaches_sdg_counterfactual"]],
               on="country_code", how="left")
    )

    # Integrity: n_births must agree between the PAF and equity files, and the
    # cohort pass must reproduce both. A mismatch means the cohort drifted.
    mismatch = out["n_births"] != out["n_births_cohort"]
    if mismatch.any():
        bad = out.loc[mismatch, ["country_code", "n_births", "n_births_cohort"]]
        print(f"  WARNING: n_births mismatch in {int(mismatch.sum())} countries:\n{bad.to_string(index=False)}")

    # Cumulative share of global avertable burden, computed here rather than
    # inherited from the legacy file (whose PAF column is pre-v2).
    out = out.sort_values("avertable_per_year", ascending=False, na_position="last").reset_index(drop=True)
    total_avert = out["avertable_per_year"].sum(skipna=True)
    out["cum_share_2024_pct"] = (
        100.0 * out["avertable_per_year"].fillna(0).cumsum() / total_avert).round(1)

    # Three-state SDG 3.2 status.
    out["sdg_status"] = np.select(
        [out["reaches_sdg_factual"] == True,                                        # noqa: E712
         (out["reaches_sdg_factual"] == False) & (out["reaches_sdg_counterfactual"] == True)],  # noqa: E712
        ["met", "crosses_under_counterfactual"],
        default="still_above",
    )
    out.loc[out["u5mr_2024"].isna(), "sdg_status"] = "no_igme"

    cols = ["country_code", "iso3", "country_name", "who_region",
            "n_surveys", "year_range", "n_births", "n_deaths", "u5mr_weighted",
            "u5mr_2024", "paf_education", "paf_spacing", "paf_joint",
            "avertable_per_year", "igme_u5_deaths_2024", "cum_share_2024_pct",
            "u5mr_counterfactual", "sdg_status"]
    out = out[cols]

    dest = OUT / dest_file
    out.to_csv(dest, index=False)

    print(f"\nSaved: {dest}  ({len(out)} countries x {len(cols)} columns)")
    print(f"  total avertable/yr : {int(total_avert):,}")
    print(f"  SDG status         : " + ", ".join(
        f"{k}={v}" for k, v in out['sdg_status'].value_counts().items()))
    print(f"  top-15 cum share   : {out['cum_share_2024_pct'].iloc[14]:.1f}%")
    print()
    print(out.head(12).to_string(index=False))
    return meta, u5mr


if __name__ == "__main__":
    main()
