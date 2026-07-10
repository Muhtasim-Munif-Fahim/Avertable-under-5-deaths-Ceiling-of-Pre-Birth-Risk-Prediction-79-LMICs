"""
run_igme_sdg_scoreboard.py — distance-to-SDG-3.2 scoreboard (P3 forward framing, C3.2/C3.3)
==========================================================================================

LGH forward / gap-to-target framing (benchmarking R3). Joins UN-IGME 2024 under-five
mortality RATES (per 1,000 live births) for the 79 study countries with each country's
joint education+birth-spacing counterfactual PAF, and asks how many countries have reached
(or would reach) the SDG 3.2 target of U5MR <=25 per 1,000.

Counterfactual U5MR = observed U5MR x (1 - joint PAF) — the conditional upper-bound
scenario in which both levers are fully achieved (same caveats as the avertable-deaths
framing: a ceiling, not a forecast).

Inputs:  UN IGME/*.csv (Under-five mortality rate, Total/Total), and
         output/tables/table_s_percountry_avertable_annual_igme.csv (iso3 + joint PAF).
Output:  output/tables/table_s_sdg_scoreboard.csv
         + printed headline counts for the manuscript / evidence_registry.
"""
from __future__ import annotations
import sys, glob
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.paths import TABLES_DIR

OUT = TABLES_DIR / "table_s_sdg_scoreboard.csv"
SDG_TARGET = 25.0  # SDG 3.2: under-five mortality <= 25 per 1,000 live births


def main():
    base = pd.read_csv(TABLES_DIR / "table_s_percountry_avertable_annual_igme.csv")
    base = base[["country_code", "iso3", "country_name", "who_region", "paf_joint_pct"]].copy()

    igme = pd.read_csv(glob.glob(str(Path(__file__).resolve().parents[2] / "data" / "raw" / "un_igme" / "*.csv"))[0],
                       low_memory=False)
    r = igme[(igme["Indicator"] == "Under-five mortality rate") &
             (igme["Sex"] == "Total") &
             (igme["Wealth Quintile"] == "Total")].copy()
    r["period"] = r["TIME_PERIOD"].astype(str)
    r = r[r["period"].str.startswith("2024")]
    # one modelled estimate per area at the latest 2024 period
    u5mr = (r.sort_values("period").groupby("REF_AREA")["OBS_VALUE"].last())

    base["u5mr_2024"] = base["iso3"].map(u5mr)
    miss = base[base["u5mr_2024"].isna()]
    if len(miss):
        print("WARNING unmatched IGME U5MR:", miss["country_code"].tolist())
    base = base.dropna(subset=["u5mr_2024"]).copy()
    base["cf_u5mr"] = base["u5mr_2024"] * (1.0 - base["paf_joint_pct"] / 100.0)
    base["reached_now"] = base["u5mr_2024"] <= SDG_TARGET
    base["reached_cf"] = base["cf_u5mr"] <= SDG_TARGET
    base["newly_reached"] = base["reached_cf"] & ~base["reached_now"]

    base = base.sort_values("u5mr_2024", ascending=False).reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    base.round(1).to_csv(OUT, index=False)

    n = len(base)
    now = int(base["reached_now"].sum())
    cf = int(base["reached_cf"].sum())
    print(f"Wrote {OUT.name} ({n} countries with IGME U5MR)")
    print(f"  SDG 3.2 reached now (U5MR<=25): {now}/{n}")
    print(f"  above target now:               {n - now}/{n}")
    print(f"  reached under joint counterfactual: {cf}/{n}  (+{cf - now} newly reaching)")
    print(f"  median U5MR now {base['u5mr_2024'].median():.1f} -> counterfactual {base['cf_u5mr'].median():.1f}")
    print(f"  countries still above 25 even under the counterfactual: {n - cf}")


if __name__ == "__main__":
    main()
