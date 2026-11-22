"""Crosswalk from this study's country codes to UN IGME ISO3 codes (name match + aliases).

Moved out of the v1 `run_igme_annualisation.py` on 2026-09-24; used by `derived_extras`
and `igme_uncertainty`.
"""
from __future__ import annotations

import re

import pandas as pd

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
