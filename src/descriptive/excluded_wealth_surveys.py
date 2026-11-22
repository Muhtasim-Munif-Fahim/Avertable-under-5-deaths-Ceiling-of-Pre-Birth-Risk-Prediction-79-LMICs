"""Profile of the survey rounds excluded for lack of a household wealth index.

The supervisor's review (2026-10-10) asked how many births and countries the
complete-case-on-wealth rule removes, and whether the removed rounds differ by region
or period. This module answers that from the data and writes it as a result file, so
the sentence in the Results is traceable like every other number.

Cohort: the exact >=60-month cohort of eTable S1 (``s04_descriptive``): births with
a non-missing ``died_under5_complete`` and a positive survey weight, weights normalised
to mean one within each survey. A round counts as excluded when no birth in it has a
wealth quintile.

Outputs (output/phase4/manuscript/)
    table_excluded_wealth_surveys.csv          one row per excluded round
    table_excluded_wealth_surveys_summary.csv  excluded vs retained (all, and 2000-2004)

Run:  python -m src.descriptive.excluded_wealth_surveys
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config.paths import HARMONIZED_DIR, OUTPUT_DIR
from src.inferential.phase2_analysis import _positive_normalized_weights

MV = OUTPUT_DIR / "phase4" / "manuscript"
COLS = ["country_code", "who_region", "survey_year", "data_source", "_source_file",
        "sample_weight", "died_under5_complete", "wealth_quintile"]


def main() -> None:
    frame = pd.read_parquet(HARMONIZED_DIR / "global_analysis_v2_candidate.parquet",
                            columns=COLS)
    w = _positive_normalized_weights(frame)
    y = pd.to_numeric(frame["died_under5_complete"], errors="coerce")
    keep = y.notna() & w.notna()
    d = frame.loc[keep].assign(_w=w[keep].to_numpy(float), _y=y[keep].to_numpy(float))
    if len(d) != 8_184_578:
        raise SystemExit(f"exact cohort is {len(d):,}, expected 8,184,578")

    has_wealth = d.groupby("_source_file", observed=True)["wealth_quintile"].transform(
        lambda s: s.notna().any())
    d["excluded_round"] = ~has_wealth.astype(bool)

    per_round = (d[d["excluded_round"]]
                 .groupby("_source_file", observed=True)
                 .apply(lambda g: pd.Series({
                     "country_code": str(g["country_code"].iloc[0]),
                     "who_region": str(g["who_region"].iloc[0]),
                     "survey_year": int(pd.to_numeric(g["survey_year"]).iloc[0]),
                     "programme": str(g["data_source"].iloc[0]),
                     "n_births_exact": int(len(g)),
                     "n_deaths": int(g["_y"].sum()),
                     "u5_risk_per1000": round(1000 * np.average(g["_y"], weights=g["_w"]), 1),
                 }), include_groups=False)
                 .reset_index().rename(columns={"_source_file": "survey_id"})
                 .sort_values(["who_region", "country_code", "survey_year"]))
    per_round.to_csv(MV / "table_excluded_wealth_surveys.csv", index=False)

    retained_countries = set(d.loc[~d["excluded_round"], "country_code"].astype(str))
    rows = []
    for label, g in (("excluded (no wealth index)", d[d["excluded_round"]]),
                     ("retained", d[~d["excluded_round"]]),
                     # Same-era comparator: the excluded rounds are all early, so part of
                     # their excess mortality is period rather than selection.
                     ("retained, fielded 2000-2004",
                      d[~d["excluded_round"]
                        & pd.to_numeric(d["survey_year"]).between(2000, 2004)])):
        countries = set(g["country_code"].astype(str))
        rows.append({
            "group": label,
            "n_rounds": int(g["_source_file"].nunique()),
            "n_countries": len(countries),
            "countries_entirely_lost": ";".join(sorted(countries - retained_countries))
            if label.startswith("excluded") else "",
            "programmes": ";".join(sorted(g["data_source"].astype(str).unique())),
            "survey_year_min": int(pd.to_numeric(g["survey_year"]).min()),
            "survey_year_max": int(pd.to_numeric(g["survey_year"]).max()),
            "n_births_exact": int(len(g)),
            "pct_of_exact_cohort": round(100 * len(g) / len(d), 1),
            "u5_risk_per1000": round(1000 * np.average(g["_y"], weights=g["_w"]), 1),
            "rounds_by_region": ";".join(
                f"{k}={v}" for k, v in g.groupby("who_region", observed=True)["_source_file"]
                .nunique().sort_values(ascending=False).items()),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(MV / "table_excluded_wealth_surveys_summary.csv", index=False)
    print(summary.T.to_string())


if __name__ == "__main__":
    main()
