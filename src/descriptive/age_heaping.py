"""Age-at-death heaping and follow-up boundary displacement (2026-09 reviewer round).

The computation behind eFigure S9 and two evidence-registry atoms: deaths reported at
12, 24 and 36 months relative to the mean of the neighbouring months, the share of
deaths at month 0, and births at exactly 60 months of follow-up relative to the mean of
59 and 61. Moved out of the eFigure S9 drawing module on 2026-11-21 so the
public deposit carries the code behind every reported number; the figure module now
only draws. Numbers are unchanged.

Outputs: output/phase4/sensitivity_2026-09/heaping/{heaping_index,
         age_at_death_distribution,followup_boundary}.csv
Run:     python -m src.descriptive.age_heaping
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.paths import HARMONIZED_DIR, OUTPUT_DIR

DATA_OUT = OUTPUT_DIR / "phase4" / "sensitivity_2026-09" / "heaping"
PERIODS = [(2000, 2009, "2000-2009"), (2010, 2019, "2010-2019"), (2020, 2024, "2020-2024")]


def load() -> pd.DataFrame:
    cols = ["data_source", "survey_year", "age_at_death_months", "followup_months",
            "u5_event", "wealth_quintile"]
    frame = pd.read_parquet(HARMONIZED_DIR / "global_analysis_v2_candidate.parquet",
                            columns=cols)
    # The analytic cohort: births with a household wealth index (77 countries).
    frame = frame[pd.to_numeric(frame["wealth_quintile"], errors="coerce").notna()]
    frame["programme"] = np.where(frame["data_source"].astype(str).str.upper()
                                  .str.contains("MICS"), "MICS", "DHS")
    return frame


def heap_ratio(counts: pd.Series, month: int) -> float:
    neighbours = (counts.get(month - 1, 0) + counts.get(month + 1, 0)) / 2
    return float(counts.get(month, 0) / neighbours) if neighbours > 0 else float("nan")


def compute(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    deaths = frame[pd.to_numeric(frame["u5_event"], errors="coerce").eq(1)].copy()
    deaths["age"] = pd.to_numeric(deaths["age_at_death_months"], errors="coerce")
    deaths = deaths[deaths["age"].between(0, 59)]
    rows = []
    groups = [("All", deaths)] + [(p, deaths[deaths.programme == p]) for p in ("DHS", "MICS")]
    for lo, hi, label in PERIODS:
        year = pd.to_numeric(deaths["survey_year"], errors="coerce")
        groups.append((label, deaths[year.between(lo, hi)]))
    for name, g in groups:
        counts = g["age"].astype(int).value_counts()
        rows.append({"group": name, "n_deaths": int(len(g)),
                     "share_month0_pct": round(100 * counts.get(0, 0) / len(g), 1),
                     **{f"heap_{m}": round(heap_ratio(counts, m), 2) for m in (12, 24, 36)}})
    index = pd.DataFrame(rows)

    post = deaths[deaths["age"].between(1, 59)]
    dist = (post.groupby(["programme", post["age"].astype(int)]).size()
            .rename("n").reset_index())
    dist["share_pct"] = dist["n"] / dist.groupby("programme")["n"].transform("sum") * 100

    follow = pd.to_numeric(frame["followup_months"], errors="coerce")
    band = follow[follow.between(48, 71)].astype(int).value_counts().sort_index()
    boundary = pd.DataFrame({"followup_month": band.index, "n_births": band.to_numpy()})
    boundary["share_pct"] = boundary["n_births"] / boundary["n_births"].sum() * 100
    ratio60 = float(band.get(60, 0) / ((band.get(59, 0) + band.get(61, 0)) / 2))
    index = pd.concat([index, pd.DataFrame([{"group": "followup_boundary",
                                             "n_deaths": np.nan,
                                             "heap_60_followup": round(ratio60, 3)}])],
                      ignore_index=True)
    return index, dist, boundary


def write() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    index, dist, boundary = compute(load())
    index.to_csv(DATA_OUT / "heaping_index.csv", index=False)
    dist.to_csv(DATA_OUT / "age_at_death_distribution.csv", index=False)
    boundary.to_csv(DATA_OUT / "followup_boundary.csv", index=False)
    return index, dist, boundary


if __name__ == "__main__":
    print(write()[0].to_string(index=False))
