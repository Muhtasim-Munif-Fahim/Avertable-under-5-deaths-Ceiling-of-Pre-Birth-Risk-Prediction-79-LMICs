"""igme_uncertainty.py — propagate UN-IGME's own death-count uncertainty into
the avertable-burden headline (~1,075,380/yr), which until 2026-08-08 carried only
PAF bootstrap uncertainty and treated the IGME 2024 death counts as fixed.

UN-IGME publishes 90% uncertainty bounds (LOWER_BOUND/UPPER_BOUND) alongside its
point estimate for TMY0T4 (under-five deaths). This module reads that already-
downloaded raw file (data/raw/un_igme/, present since data acquisition — no new
external data) and combines it with the existing contemporary joint PAF bootstrap
interval under two explicit correlation regimes:

  independent   — PAF uncertainty and the IGME death-count uncertainty are
                   uncorrelated; combined relative SE via sqrt-sum-of-squares.
                   The regime a reader gets by default if not told otherwise.
  comonotonic   — the two move together in the worst case (PAF_lo x deaths_lo,
                   PAF_hi x deaths_hi); an explicit upper bound on how wide the
                   true interval could be, not a probabilistic claim.

Both are reported; neither is asserted as "the" interval. Point estimate is
unchanged — this widens the reported uncertainty, it does not move the headline.

Run: python -m src.inferential.igme_uncertainty
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config.paths import HARMONIZED_DIR, OUTPUT_DIR
from src.inferential.igme_crosswalk import build_crosswalk

OUT = OUTPUT_DIR / "phase4" / "manuscript"
IGME_CSV = HARMONIZED_DIR.parents[1] / "data" / "raw" / "un_igme" / "UNICEF-CME_DF_2021_WQ-1.0-download.csv"

# Contemporary (2018-24 exposure) joint PAF, survival primary — the headline the
# ~1,075,380/yr figure is built from. Read from the source table, not hardcoded,
# so this cannot silently drift from the number the paper reports.
RECENTDECADE_CSV = OUT / "table_recentdecade_paf_survival.csv"


def _z90() -> float:
    from scipy.stats import norm
    return float(norm.ppf(0.95))  # 90% two-sided, matching IGME's own UI convention


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    rd = pd.read_csv(RECENTDECADE_CSV)
    joint = rd[rd["scenario"] == "joint"].iloc[0]
    paf_mid = float(joint["paf_recent_prevalence_pct"]) / 100.0
    paf_lo = float(joint["ci_lo"]) / 100.0
    paf_hi = float(joint["ci_hi"]) / 100.0
    paf_rse = (paf_hi - paf_lo) / (2 * _z90() * paf_mid)  # relative SE at 90% level

    inv = pd.read_csv(OUT / "table_s1_survey_inventory.csv")
    igme = pd.read_csv(IGME_CSV, encoding="utf-8", engine="python")
    our_names = inv.groupby("country_code")["country_name"].first().reset_index()
    cw = build_crosswalk(our_names, igme[["REF_AREA", "Geographic area"]].drop_duplicates())

    d = igme[(igme["INDICATOR"] == "TMY0T4") & (igme["Sex"] == "Total")
             & (igme["Series Name"] == "UN IGME estimate")].copy()
    d["year"] = d["TIME_PERIOD"].astype(str).str[:4].astype(int)
    d2024 = d[d["year"] == 2024].set_index("REF_AREA")

    igme_pc = pd.read_csv(OUT / "table_igme_annualisation.csv")  # 77 crosswalked countries
    rows = []
    for _, r in igme_pc.iterrows():
        iso = r["iso3"]
        if iso not in d2024.index:
            continue
        row = d2024.loc[iso]
        rows.append({
            "country_code": r["country_code"], "iso3": iso,
            "deaths_mid": float(row["OBS_VALUE"]),
            "deaths_lo": float(row["LOWER_BOUND"]), "deaths_hi": float(row["UPPER_BOUND"]),
        })
    dd = pd.DataFrame(rows)
    if dd["iso3"].duplicated().any():
        raise ValueError(f"Duplicate ISO3 in IGME crosswalk: {dd.loc[dd.iso3.duplicated(), 'iso3'].tolist()}")
    n_countries = len(dd)
    if n_countries != len(igme_pc):
        raise ValueError(
            f"IGME uncertainty bounds resolved for {n_countries} countries but the "
            f"published annualisation table has {len(igme_pc)}; a country dropped "
            f"out of the crosswalk between the two reads."
        )

    deaths_mid = dd["deaths_mid"].sum()
    deaths_lo = dd["deaths_lo"].sum()
    deaths_hi = dd["deaths_hi"].sum()
    deaths_rse = (deaths_hi - deaths_lo) / (2 * _z90() * deaths_mid)

    avertable_mid = paf_mid * deaths_mid

    # independent regime: combine relative SEs in quadrature, apply to the point
    combined_rse = float(np.hypot(paf_rse, deaths_rse))
    indep_lo = avertable_mid * (1 - _z90() * combined_rse)
    indep_hi = avertable_mid * (1 + _z90() * combined_rse)

    # comonotonic regime: both uncertain quantities at their bound simultaneously —
    # an explicit worst-case width, not a probability statement
    comono_lo = paf_lo * deaths_lo
    comono_hi = paf_hi * deaths_hi

    out = pd.DataFrame([
        {"quantity": "Contemporary joint PAF (survival primary)",
         "value": round(paf_mid * 100, 2), "lo": round(paf_lo * 100, 2), "hi": round(paf_hi * 100, 2),
         "source": "bootstrap (B=1000), PAF uncertainty only"},
        {"quantity": "UN-IGME 2024 under-5 deaths, 77 crosswalked countries",
         "value": round(deaths_mid), "lo": round(deaths_lo), "hi": round(deaths_hi),
         "source": "UN-IGME published 90% uncertainty interval, summed across countries"},
        {"quantity": "Avertable deaths/year, PAF uncertainty only (as previously reported)",
         "value": round(avertable_mid), "lo": round(paf_lo * deaths_mid), "hi": round(paf_hi * deaths_mid),
         "source": "PAF bootstrap CI x fixed IGME point estimate"},
        {"quantity": "Avertable deaths/year, PAF + IGME uncertainty, independent regime",
         "value": round(avertable_mid), "lo": round(indep_lo), "hi": round(indep_hi),
         "source": "relative SEs combined in quadrature (Gaussian approx, 90% level)"},
        {"quantity": "Avertable deaths/year, PAF + IGME uncertainty, comonotonic regime",
         "value": round(avertable_mid), "lo": round(comono_lo), "hi": round(comono_hi),
         "source": "explicit worst case: both bounds hit simultaneously, not a probability interval"},
    ])
    out.to_csv(OUT / "table_igme_uncertainty.csv", index=False)

    print(f"[igme-uncertainty] PAF-only interval:      {round(paf_lo*deaths_mid):,} - {round(paf_hi*deaths_mid):,}")
    print(f"[igme-uncertainty] independent regime:      {round(indep_lo):,} - {round(indep_hi):,}")
    print(f"[igme-uncertainty] comonotonic regime:       {round(comono_lo):,} - {round(comono_hi):,}")
    print(f"[igme-uncertainty] wrote {OUT / 'table_igme_uncertainty.csv'}")


if __name__ == "__main__":
    main()
