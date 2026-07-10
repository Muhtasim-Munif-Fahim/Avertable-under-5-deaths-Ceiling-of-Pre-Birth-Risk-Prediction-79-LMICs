"""
run_igme_annualisation.py — Option B: UN-IGME annual avertable under-5 deaths
============================================================================

Companion to run_fulldata_joint_counterfactual.py. Takes the per-country JOINT
education+birth-spacing PAF (Option A output) and applies it to the UN-IGME
*absolute under-five death envelope* to express the avertable burden as deaths
PER YEAR — the policy-facing figure ("~X under-5 deaths/year avertable across
the 79 LMICs at current mortality").

This is a SEPARATE, clearly-labelled denominator from Option A (avertable share
of the 885,756 recorded cohort deaths). It is LIGHT (reads two CSVs, no model
fit, no full-data load) — safe to run alongside the bootstrap.

Inputs:
  output/tables/table_s_percountry_joint_paf.csv     (Option A; paf_joint per country)
  UN IGME/UNICEF-CME_DF_2021_WQ-1.0-download.csv      (TMY0T4 = Under-five deaths,
                                                       UN IGME estimate, Total sex)
  output/tables/table_s1_survey_inventory.csv         (survey years per country)

Two annualisation bases (both reported, clearly labelled):
  * latest year (2024)        -> "avertable deaths/year at current mortality"
  * coverage-window mean       -> mean over each country's survey years (matched
                                  to the period the PAF was estimated over)

Outputs:
  output/tables/table_s_percountry_avertable_annual_igme.csv   (per country)
  output/fulldata/table_joint_avertable_annual_igme.csv         (pooled + regional)

Run: python -m src.inferential.run_igme_annualisation
"""
from __future__ import annotations
import sys, re
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.paths import TABLES_DIR

ROOT = Path(__file__).resolve().parents[2]
IGME_CSV = ROOT / "data" / "raw" / "un_igme" / "UNICEF-CME_DF_2021_WQ-1.0-download.csv"
PAF_CSV = TABLES_DIR / "table_s_percountry_joint_paf.csv"
INV_CSV = TABLES_DIR / "table_s1_survey_inventory.csv"
PERCOUNTRY_OUT = TABLES_DIR / "table_s_percountry_avertable_annual_igme.csv"
POOLED_OUT = TABLES_DIR.parent / "fulldata" / "table_joint_avertable_annual_igme.csv"

LATEST_YEAR = 2024

# IGME calls some areas differently from our DHS country_name; map the residue.
NAME_ALIASES = {
    "drcongo": "COD", "congo": "COG", "cotedivoire": "CIV",
    "kyrgyzrepublic": "KGZ", "laopdr": "LAO", "moldova": "MDA",
    "stateofpalestine": "PSE", "vietnam": "VNM", "gambia": "GMB",
    "tanzania": "TZA", "eswatini": "SWZ", "bolivia": "BOL",
    "saotomeandprincipe": "STP", "timorleste": "TLS", "egypt": "EGY",
}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z]", "", str(s).lower())


def build_crosswalk(our_names: pd.DataFrame, igme_areas: pd.DataFrame) -> dict:
    """our country_code -> IGME ISO3 REF_AREA. Name-match + alias + 3-letter passthrough."""
    amap = {_norm(r["Geographic area"]): r["REF_AREA"] for _, r in igme_areas.iterrows()}
    iso_set = set(igme_areas["REF_AREA"])
    cw, unmatched = {}, []
    for _, r in our_names.iterrows():
        cc, nm = r["country_code"], r["country_name"]
        iso = amap.get(_norm(nm)) or NAME_ALIASES.get(_norm(nm))
        if iso is None and len(str(cc)) == 3 and str(cc) in iso_set:
            iso = str(cc)            # our code already an ISO3 (e.g. DZA, GNB, KIR)
        if iso is None:
            unmatched.append((cc, nm))
        else:
            cw[cc] = iso
    if unmatched:
        print("WARNING — unmatched countries (no IGME ISO3):")
        for cc, nm in unmatched:
            print(f"   {cc}  {nm}")
    print(f"crosswalk: matched {len(cw)}/{len(our_names)} countries to IGME ISO3")
    return cw


def main():
    paf = pd.read_csv(PAF_CSV)
    inv = pd.read_csv(INV_CSV)
    igme = pd.read_csv(IGME_CSV, encoding="utf-8", engine="python")

    # ---- IGME under-five DEATHS (absolute), UN IGME estimate, Total sex ----- #
    d = igme[(igme["INDICATOR"] == "TMY0T4") & (igme["Sex"] == "Total")
             & (igme["Series Name"] == "UN IGME estimate")].copy()
    d["year"] = d["TIME_PERIOD"].astype(str).str[:4].astype(int)
    d["deaths"] = pd.to_numeric(d["OBS_VALUE"], errors="coerce")
    deaths_by = d.groupby(["REF_AREA", "year"])["deaths"].first().reset_index()

    # ---- crosswalk + survey-year windows ----------------------------------- #
    our_names = inv.groupby("country_code")["country_name"].first().reset_index()
    cw = build_crosswalk(our_names, igme[["REF_AREA", "Geographic area"]].drop_duplicates())
    survey_years = inv.groupby("country_code")["survey_year"].apply(
        lambda s: sorted(set(int(y) for y in s)))

    def igme_latest(iso):
        s = deaths_by[(deaths_by["REF_AREA"] == iso) & (deaths_by["year"] == LATEST_YEAR)]
        return float(s["deaths"].iloc[0]) if len(s) else np.nan

    def igme_windowmean(iso, years):
        s = deaths_by[(deaths_by["REF_AREA"] == iso) & (deaths_by["year"].isin(years))]
        return float(s["deaths"].mean()) if len(s) else np.nan

    rows = []
    for _, r in paf.iterrows():
        cc = r["country_code"]
        iso = cw.get(cc)
        pj = r.get("paf_joint", np.nan)
        if iso is None or not np.isfinite(pj):
            continue
        yrs = survey_years.get(cc, [])
        dl = igme_latest(iso)
        dw = igme_windowmean(iso, yrs)
        rows.append({
            "country_code": cc, "iso3": iso,
            "country_name": r.get("country_name", cc),
            "who_region": r.get("who_region", ""),
            "paf_joint_pct": pj,
            "igme_u5_deaths_2024": None if not np.isfinite(dl) else int(round(dl)),
            "avertable_per_year_2024": None if not np.isfinite(dl) else int(round(pj / 100 * dl)),
            "igme_u5_deaths_windowmean": None if not np.isfinite(dw) else int(round(dw)),
            "avertable_per_year_windowmean": None if not np.isfinite(dw) else int(round(pj / 100 * dw)),
            "survey_years": ",".join(str(y) for y in yrs),
        })
    out = pd.DataFrame(rows).sort_values("avertable_per_year_2024",
                                         ascending=False, na_position="last")
    out["cum_share_2024_pct"] = (out["avertable_per_year_2024"].cumsum()
                                 / out["avertable_per_year_2024"].sum() * 100).round(1)
    out.to_csv(PERCOUNTRY_OUT, index=False)

    tot_2024 = int(out["avertable_per_year_2024"].sum())
    tot_win = int(out["avertable_per_year_windowmean"].dropna().sum())
    igme_tot_2024 = int(out["igme_u5_deaths_2024"].sum())
    top10 = float(out["cum_share_2024_pct"].iloc[9]) if len(out) >= 10 else np.nan

    # ---- pooled + regional summary ----------------------------------------- #
    reg = out.groupby("who_region").agg(
        n_countries=("country_code", "size"),
        igme_u5_deaths_2024=("igme_u5_deaths_2024", "sum"),
        avertable_per_year_2024=("avertable_per_year_2024", "sum"),
        avertable_per_year_windowmean=("avertable_per_year_windowmean", "sum"),
    ).reset_index()
    pooled = pd.DataFrame([{
        "who_region": "All", "n_countries": len(out),
        "igme_u5_deaths_2024": igme_tot_2024,
        "avertable_per_year_2024": tot_2024,
        "avertable_per_year_windowmean": tot_win,
    }])
    pd.concat([pooled, reg], ignore_index=True).to_csv(POOLED_OUT, index=False)

    print(f"\nwrote {PERCOUNTRY_OUT.name} ({len(out)} countries) and {POOLED_OUT.name}")
    print(f"  IGME 2024 under-5 deaths across {len(out)} covered countries: {igme_tot_2024:,}")
    print(f"  OPTION B headline — avertable/year @2024 mortality: {tot_2024:,} deaths/year")
    print(f"  OPTION B (coverage-window mean basis):              {tot_win:,} deaths/year")
    print(f"  top-10 countries carry {top10}% of the annual avertable burden")
    print("\n  top 10 countries (avertable/year @2024):")
    print(out.head(10)[["country_name", "paf_joint_pct",
                        "igme_u5_deaths_2024", "avertable_per_year_2024"]].to_string(index=False))


if __name__ == "__main__":
    main()
