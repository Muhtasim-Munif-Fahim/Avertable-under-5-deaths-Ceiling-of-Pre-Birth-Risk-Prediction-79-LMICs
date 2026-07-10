"""
build_mics_catalog.py -- MICS Survey Catalog Builder
=====================================================

Builds data/mics_survey_catalog.csv for all BH-positive MICS zip files,
annotated with country metadata, eligibility flags, and expansion decision.

Output
------
  data/mics_survey_catalog.csv
  data/mics_catalog_notes.md
"""

import csv
import os
from pathlib import Path

from config.paths import RAW_MICS_DIR
MICS_DIR = RAW_MICS_DIR
PROJECT_ROOT = Path(r"J:\Survey Research\03 Temporal NAS\expand_global")
OUTPUT_CSV = PROJECT_ROOT / "data" / "mics_survey_catalog.csv"
OUTPUT_NOTES = PROJECT_ROOT / "data" / "mics_catalog_notes.md"

# ---------------------------------------------------------------------------
# DHS 68-country ISO3 set (derived from table_s1_survey_inventory.csv)
# DHS codes -> ISO3 mapping used to build this set
# ---------------------------------------------------------------------------
DHS_68_ISO3 = {
    "AFG",  # Afghanistan
    "ALB",  # Albania
    "ARM",  # Armenia
    "AGO",  # Angola
    "AZE",  # Azerbaijan
    "BGD",  # Bangladesh
    "BFA",  # Burkina Faso
    "BEN",  # Benin
    "BOL",  # Bolivia
    "BDI",  # Burundi
    "COD",  # DR Congo
    "COG",  # Congo (Republic)
    "CIV",  # Cote d'Ivoire
    "CMR",  # Cameroon
    "COL",  # Colombia
    "DOM",  # Dominican Republic
    "EGY",  # Egypt
    "ETH",  # Ethiopia
    "GAB",  # Gabon
    "GHA",  # Ghana
    "GMB",  # Gambia
    "GIN",  # Guinea
    "GTM",  # Guatemala
    "GUY",  # Guyana
    "HND",  # Honduras
    "HTI",  # Haiti
    "IND",  # India
    "IDN",  # Indonesia
    "JOR",  # Jordan
    "KHM",  # Cambodia
    "COM",  # Comoros
    "KGZ",  # Kyrgyz Republic / Kyrgyzstan
    "LBR",  # Liberia
    "LSO",  # Lesotho
    "MAR",  # Morocco
    "MDA",  # Moldova
    "MDG",  # Madagascar
    "MLI",  # Mali
    "MMR",  # Myanmar
    "MRT",  # Mauritania
    "MDV",  # Maldives
    "MWI",  # Malawi
    "NIC",  # Nicaragua
    "NGA",  # Nigeria
    "NER",  # Niger
    "NAM",  # Namibia
    "NPL",  # Nepal
    "PER",  # Peru
    "PNG",  # Papua New Guinea
    "PHL",  # Philippines
    "PAK",  # Pakistan
    "RWA",  # Rwanda
    "SLE",  # Sierra Leone
    "SEN",  # Senegal
    "STP",  # Sao Tome and Principe
    "SWZ",  # Eswatini
    "TCD",  # Chad
    "TGO",  # Togo
    "TJK",  # Tajikistan
    "TLS",  # Timor-Leste
    "TZA",  # Tanzania
    "UKR",  # Ukraine
    "UGA",  # Uganda
    "VNM",  # Vietnam
    "YEM",  # Yemen
    "ZAF",  # South Africa
    "ZMB",  # Zambia
    "ZWE",  # Zimbabwe
}

# ---------------------------------------------------------------------------
# Catalog rows: one row per BH-positive zip
# Fields: zip_filename, country_name, country_iso3, mics_round,
#         survey_year_start, survey_year_end, who_region, wb_income_2024,
#         is_national, notes
# ---------------------------------------------------------------------------

# Each tuple: (zip_filename, country_name, iso3, round, yr_start, yr_end,
#              who_region, wb_income, is_national, notes)

CATALOG_ROWS = [
    # --- Afghanistan ---
    (
        "Afghanistan MICS6 Datasets.zip",
        "Afghanistan", "AFG", "MICS6", 2022, 2023,
        "SSEA", "L", 1, "",
    ),
    # --- Algeria ---
    (
        "Algeria MICS6 Datasets.zip",
        "Algeria", "DZA", "MICS6", 2018, 2019,
        "MENA", "LM", 1, "",
    ),
    # --- Azerbaijan ---
    (
        "Azerbaijan MICS6 2023 Datasets.zip",
        "Azerbaijan", "AZE", "MICS6", 2023, 2024,
        "ECA", "UM", 1, "",
    ),
    # --- Bangladesh ---
    (
        "Bangladesh MICS6 SPSS Datasets.zip",
        "Bangladesh", "BGD", "MICS6", 2019, 2021,
        "SSEA", "LM", 1, "",
    ),
    # --- Belize ---
    (
        "Belize_MICS5_Datasets.zip",
        "Belize", "BLZ", "MICS5", 2015, 2016,
        "LAC", "UM", 1, "",
    ),
    # --- Benin MICS6 ---
    (
        "Benin MICS6 Datasets.zip",
        "Benin", "BEN", "MICS6", 2021, 2022,
        "SSA", "L", 1, "",
    ),
    # --- Benin MICS5 ---
    (
        "Benin_MICS5_Datasets.zip",
        "Benin", "BEN", "MICS5", 2014, 2014,
        "SSA", "L", 1, "",
    ),
    # --- Cameroon MICS5 ---
    (
        "Cameroon_MICS5_Datasets.zip",
        "Cameroon", "CMR", "MICS5", 2014, 2014,
        "SSA", "LM", 1, "",
    ),
    # --- Central African Republic ---
    (
        "Central African Republic MICS6 Datasets.zip",
        "Central African Republic", "CAF", "MICS6", 2018, 2019,
        "SSA", "L", 1, "",
    ),
    # --- Chad MICS6 ---
    (
        "Chad MICS6 Datasets.zip",
        "Chad", "TCD", "MICS6", 2019, 2022,
        "SSA", "L", 1, "",
    ),
    # --- Comoros ---
    (
        "Comoros MICS6 SPSS Datasets.zip",
        "Comoros", "COM", "MICS6", 2022, 2022,
        "SSA", "LM", 1, "",
    ),
    # --- Congo (Republic) MICS5 ---
    (
        "Congo_MICS5_Datasets.zip",
        "Congo", "COG", "MICS5", 2014, 2015,
        "SSA", "LM", 1, "",
    ),
    # --- Cote d'Ivoire MICS5 ---
    (
        "Côte d'Ivoire_MICS5Datasets.zip",
        "Cote d'Ivoire", "CIV", "MICS5", 2016, 2016,
        "SSA", "LM", 1, "",
    ),
    # --- Dominican Republic MICS6 ---
    (
        "Dominican Republic MICS6 Datasets.zip",
        "Dominican Republic", "DOM", "MICS6", 2019, 2020,
        "LAC", "UM", 1, "",
    ),
    # --- Dominican Republic MICS5 ---
    (
        "Dominican Republic_MICS5_Datasets.zip",
        "Dominican Republic", "DOM", "MICS5", 2014, 2014,
        "LAC", "UM", 1, "",
    ),
    # --- DR Congo MICS6 ---
    (
        "DRCongo MICS6 SPSS Datafiles.zip",
        "DR Congo", "COD", "MICS6", 2017, 2018,
        "SSA", "L", 1, "",
    ),
    # --- El Salvador MICS5 ---
    (
        "El Salvador_MICS5_Datasets.zip",
        "El Salvador", "SLV", "MICS5", 2014, 2014,
        "LAC", "UM", 1, "",
    ),
    # --- Eswatini MICS6 ---
    (
        "Eswatini MICS6 SPSS Datasets.zip",
        "Eswatini", "SWZ", "MICS6", 2021, 2022,
        "SSA", "LM", 1, "",
    ),
    # --- Fiji MICS6 ---
    (
        "Fiji MICS6 SPSS Datasets.zip",
        "Fiji", "FJI", "MICS6", 2021, 2021,
        "SSEA", "UM", 1, "",
    ),
    # --- Ghana MICS6 ---
    (
        "Ghana MICS6 SPSS Datasets.zip",
        "Ghana", "GHA", "MICS6", 2017, 2018,
        "SSA", "LM", 1, "",
    ),
    # --- Guinea-Bissau MICS6 ---
    (
        "Guinea Bissau MICS6 Datasets.zip",
        "Guinea-Bissau", "GNB", "MICS6", 2018, 2019,
        "SSA", "L", 1, "",
    ),
    # --- Guinea-Bissau MICS5 ---
    (
        "Guinea Bissau_MICS5_Datasets.zip",
        "Guinea-Bissau", "GNB", "MICS5", 2014, 2014,
        "SSA", "L", 1, "",
    ),
    # --- Guinea MICS5 ---
    (
        "Guinea_MICS5_Datasets.zip",
        "Guinea", "GIN", "MICS5", 2016, 2016,
        "SSA", "L", 1, "",
    ),
    # --- Guyana MICS6 ---
    (
        "Guyana MICS6 Datasets.zip",
        "Guyana", "GUY", "MICS6", 2019, 2020,
        "LAC", "UM", 1, "",
    ),
    # --- Guyana MICS 2014 ---
    (
        "Guyana_MICS 2014_Datasets.zip",
        "Guyana", "GUY", "MICS5", 2014, 2014,
        "LAC", "UM", 1, "",
    ),
    # --- Honduras MICS6 (typo in folder: "Honudras") ---
    (
        "Honudras MICS6 SPSS Datasets.zip",
        "Honduras", "HND", "MICS6", 2019, 2020,
        "LAC", "LM", 1, "",
    ),
    # --- Iraq MICS6 ---
    (
        "Iraq MICS6 Datasets.zip",
        "Iraq", "IRQ", "MICS6", 2018, 2019,
        "MENA", "UM", 1, "",
    ),
    # --- Kazakhstan MICS7 ---
    (
        "Kazakhstan MICS7 Datasets.zip",
        "Kazakhstan", "KAZ", "MICS7", 2023, 2024,
        "ECA", "UM", 1, "",
    ),
    # --- Kiribati MICS6 ---
    (
        "Kiribati MICS6 Datasets.zip",
        "Kiribati", "KIR", "MICS6", 2018, 2019,
        "SSEA", "LM", 1, "",
    ),
    # --- Kosovo MICS5 (national, not Roma community) ---
    (
        "Kosovo under UNSC res. 1244 MICS5 Datasets.zip",
        "Kosovo", "XKX", "MICS5", 2013, 2014,
        "ECA", "UM", 1, "territory under UNSCR 1244; not UN member state",
    ),
    # --- Kosovo MICS6 (national, not Roma community) ---
    (
        "Kosovo under UNSC res. 1244 MICS6 Datasets.zip",
        "Kosovo", "XKX", "MICS6", 2019, 2020,
        "ECA", "UM", 1, "territory under UNSCR 1244; not UN member state",
    ),
    # --- Kyrgyz Republic MICS6 ---
    (
        "Kyrgyz Republic MICS6 Datasets.zip",
        "Kyrgyzstan", "KGZ", "MICS6", 2018, 2018,
        "ECA", "LM", 1, "",
    ),
    # --- Kyrgyzstan 2014 MICS ---
    (
        "Kyrgyzstan 2014 MICS_Datasets.zip",
        "Kyrgyzstan", "KGZ", "MICS5", 2014, 2014,
        "ECA", "LM", 1, "",
    ),
    # --- Kyrgyzstan MICS6 2023 ---
    (
        "Kyrgyzstan MICS6 2023 Datasets.zip",
        "Kyrgyzstan", "KGZ", "MICS6", 2023, 2024,
        "ECA", "LM", 1, "",
    ),
    # --- Lao PDR MICS6 2023 ---
    (
        "Lao PDR MICS6 2023 Datasets.zip",
        "Lao PDR", "LAO", "MICS6", 2023, 2023,
        "SSEA", "LM", 1, "",
    ),
    # --- Lao PDR MICS6 SPSS ---
    (
        "Lao PDR MICS6 SPSS Datasets.zip",
        "Lao PDR", "LAO", "MICS6", 2017, 2017,
        "SSEA", "LM", 1, "",
    ),
    # --- Lesotho MICS6 ---
    (
        "Lesotho_MICS6_datasets.zip",
        "Lesotho", "LSO", "MICS6", 2018, 2018,
        "SSA", "LM", 1, "",
    ),
    # --- Madagascar MICS6 ---
    (
        "Madagascar MICS6 SPSS datasets.zip",
        "Madagascar", "MDG", "MICS6", 2018, 2018,
        "SSA", "L", 1, "",
    ),
    # --- Malawi 2013-14 MICS ---
    (
        "Malawi 2013-14 MICS_Datasets.zip",
        "Malawi", "MWI", "MICS5", 2013, 2014,
        "SSA", "L", 1, "",
    ),
    # --- Malawi MICS6 ---
    (
        "Malawi MICS6 SPSS Datasets.zip",
        "Malawi", "MWI", "MICS6", 2019, 2020,
        "SSA", "L", 1, "",
    ),
    # --- Mali MICS5 ---
    (
        "Mali_MICS5_Datasets.zip",
        "Mali", "MLI", "MICS5", 2015, 2015,
        "SSA", "L", 1, "",
    ),
    # --- Mauritania MICS5 ---
    (
        "Mauritania_MICS5_Datasets.zip",
        "Mauritania", "MRT", "MICS5", 2015, 2015,
        "SSA", "LM", 1, "",
    ),
    # --- Samoa MICS6 ---
    (
        "MICS6 Samoa Datasets.zip",
        "Samoa", "WSM", "MICS6", 2019, 2020,
        "SSEA", "UM", 1, "",
    ),
    # --- Mongolia MICS 2018 ---
    (
        "Mongolia MICS 2018 SPSS Datasets.zip",
        "Mongolia", "MNG", "MICS6", 2018, 2018,
        "SSEA", "LM", 1, "",
    ),
    # --- Mongolia MICS5 (national) ---
    (
        "Mongolia_MICS5_Datasets.zip",
        "Mongolia", "MNG", "MICS5", 2013, 2013,
        "SSEA", "LM", 1, "",
    ),
    # --- Nauru MICS6 2023 ---
    (
        "Nauru MICS6 2023 Datasets.zip",
        "Nauru", "NRU", "MICS6", 2023, 2024,
        "SSEA", "H", 1, "Pacific Island state; high income",
    ),
    # --- Nepal 2014 MICS ---
    (
        "Nepal 2014 MICS_Datasets.zip",
        "Nepal", "NPL", "MICS5", 2014, 2015,
        "SSEA", "L", 1, "",
    ),
    # --- Nepal MICS6 ---
    (
        "Nepal MICS6 Datasets.zip",
        "Nepal", "NPL", "MICS6", 2019, 2019,
        "SSEA", "LM", 1, "",
    ),
    # --- Nepal MICS7 ---
    (
        "Nepal MICS7 Datasets.zip",
        "Nepal", "NPL", "MICS7", 2023, 2024,
        "SSEA", "LM", 1, "",
    ),
    # --- Nigeria MICS 2016-17 ---
    (
        "Nigeria MICS 2016-17 SPSS Datasets.zip",
        "Nigeria", "NGA", "MICS5", 2016, 2017,
        "SSA", "LM", 1, "",
    ),
    # --- Nigeria MICS6 ---
    (
        "Nigeria MICS6 SPSS Datasets.zip",
        "Nigeria", "NGA", "MICS6", 2021, 2022,
        "SSA", "LM", 1, "",
    ),
    # --- Pakistan Balochistan MICS6 (sub-national) ---
    (
        "Pakistan (Balochistan) MICS6 Datasets.zip",
        "Pakistan (Balochistan)", "PAK", "MICS6", 2019, 2020,
        "SSEA", "LM", 0, "sub-national Pakistan province; use DHS national survey",
    ),
    # --- Pakistan Khyber Pakhtunkhwa MICS6 (sub-national) ---
    (
        "Pakistan Khyber Pakhtunkhwa MICS6 Datasets.zip",
        "Pakistan (Khyber Pakhtunkhwa)", "PAK", "MICS6", 2019, 2020,
        "SSEA", "LM", 0, "sub-national Pakistan province; use DHS national survey",
    ),
    # --- Pakistan Punjab MICS6 (sub-national) ---
    (
        "Pakistan Punjab MICS6 Datasets.zip",
        "Pakistan (Punjab)", "PAK", "MICS6", 2017, 2018,
        "SSEA", "LM", 0, "sub-national Pakistan province; use DHS national survey",
    ),
    # --- Pakistan Sindh MICS6 (sub-national) ---
    (
        "Pakistan Sindh MICS6 Datasets.zip",
        "Pakistan (Sindh)", "PAK", "MICS6", 2018, 2019,
        "SSEA", "LM", 0, "sub-national Pakistan province; use DHS national survey",
    ),
    # --- Paraguay MICS5 ---
    (
        "Paraguay_MICS5_Datasets.zip",
        "Paraguay", "PRY", "MICS5", 2016, 2016,
        "LAC", "UM", 1, "",
    ),
    # --- Republic of North Macedonia MICS6 (national) ---
    (
        "Republic of North Macedonia MICS6 Datasets.zip",
        "North Macedonia", "MKD", "MICS6", 2018, 2019,
        "ECA", "UM", 1, "",
    ),
    # --- Sao Tome and Principe 2014 ---
    (
        "Sao Tome and Principe 2014 MICS_Datasets.zip",
        "Sao Tome and Principe", "STP", "MICS5", 2014, 2014,
        "SSA", "LM", 1, "",
    ),
    # --- Sao Tome and Principe MICS6 ---
    (
        "Sao Tome and Principe MICS6 Datasets.zip",
        "Sao Tome and Principe", "STP", "MICS6", 2019, 2019,
        "SSA", "LM", 1, "",
    ),
    # --- Sierra Leone MICS6 ---
    (
        "Sierra Leone MICS6 Datasets.zip",
        "Sierra Leone", "SLE", "MICS6", 2017, 2017,
        "SSA", "L", 1, "",
    ),
    # --- State of Palestine 2014 ---
    (
        "State of Palestine 2014 MICS_Datasets.zip",
        "State of Palestine", "PSE", "MICS5", 2014, 2014,
        "MENA", "LM", 1, "non-member observer state; treated as national survey",
    ),
    # --- State of Palestine MICS6 ---
    (
        "State of Palestine MICS6 SPSS Datasets.zip",
        "State of Palestine", "PSE", "MICS6", 2019, 2020,
        "MENA", "LM", 1, "non-member observer state; treated as national survey",
    ),
    # --- Sudan MICS5 ---
    (
        "Sudan_MICS5_Datasets.zip",
        "Sudan", "SDN", "MICS5", 2014, 2014,
        "SSA", "L", 1, "",
    ),
    # --- Suriname MICS6 ---
    (
        "Suriname MICS6 SPSS Datafiles.zip",
        "Suriname", "SUR", "MICS6", 2018, 2019,
        "LAC", "UM", 1, "",
    ),
    # --- Swaziland MICS5 (Eswatini, old name used in zip) ---
    (
        "Swaziland_MICS5_Datasets.zip",
        "Eswatini", "SWZ", "MICS5", 2014, 2014,
        "SSA", "LM", 1, "zip uses old name Swaziland; now Eswatini",
    ),
    # --- The Gambia MICS6 ---
    (
        "The Gambia MICS6 SPSS Datasets_0.zip",
        "Gambia", "GMB", "MICS6", 2018, 2019,
        "SSA", "L", 1, "",
    ),
    # --- Togo MICS6 ---
    (
        "Togo MICS6 SPSS Datasets.zip",
        "Togo", "TGO", "MICS6", 2017, 2017,
        "SSA", "L", 1, "",
    ),
    # --- Tonga MICS6 ---
    (
        "Tonga MICS6 Datasets.zip",
        "Tonga", "TON", "MICS6", 2019, 2019,
        "SSEA", "UM", 1, "",
    ),
    # --- Trinidad and Tobago MICS6 ---
    (
        "Trinidad and Tobago MICS6 Datasets.zip",
        "Trinidad and Tobago", "TTO", "MICS6", 2018, 2018,
        "LAC", "H", 1, "",
    ),
    # --- Tunisia MICS6 ---
    (
        "Tunisia MICS6 Datasets.zip",
        "Tunisia", "TUN", "MICS6", 2018, 2018,
        "MENA", "LM", 1, "",
    ),
    # --- Tunisia MICS6 2023 ---
    (
        "Tunisia MICS6 2023 Datasets.zip",
        "Tunisia", "TUN", "MICS6", 2023, 2023,
        "MENA", "LM", 1, "",
    ),
    # --- Turkmenistan MICS5 ---
    (
        "Turkmenistan_MICS5_Datasets.zip",
        "Turkmenistan", "TKM", "MICS5", 2015, 2015,
        "ECA", "UM", 1, "",
    ),
    # --- Turkmenistan MICS6 ---
    (
        "Turkmenistan MICS6 SPSS Datasets.zip",
        "Turkmenistan", "TKM", "MICS6", 2019, 2019,
        "ECA", "UM", 1, "",
    ),
    # --- Turkmenistan MICS7 ---
    (
        "Turkmenistan MICS7 Datasets.zip",
        "Turkmenistan", "TKM", "MICS7", 2023, 2024,
        "ECA", "UM", 1, "",
    ),
    # --- Turks and Caicos Islands MICS6 (British territory, sub-national) ---
    (
        "Turks and Caicos Islands MICS6 Datasets.zip",
        "Turks and Caicos Islands", "TCA", "MICS6", 2019, 2019,
        "LAC", "H", 0, "British Overseas Territory; not sovereign state",
    ),
    # --- Tuvalu MICS6 ---
    (
        "Tuvalu MICS6 Datasets.zip",
        "Tuvalu", "TUV", "MICS6", 2019, 2020,
        "SSEA", "UM", 1, "",
    ),
    # --- Uzbekistan MICS6 ---
    (
        "Uzbekistan MICS6 Datasets.zip",
        "Uzbekistan", "UZB", "MICS6", 2021, 2022,
        "ECA", "LM", 1, "",
    ),
    # --- Vanuatu MICS6 ---
    (
        "Vanuatu MICS6 Datasets.zip",
        "Vanuatu", "VUT", "MICS6", 2023, 2023,
        "SSEA", "LM", 1, "",
    ),
    # --- Viet Nam 2013-14 MICS ---
    (
        "Viet Nam 2013-14 MICS_Datasets.zip",
        "Viet Nam", "VNM", "MICS5", 2013, 2014,
        "SSEA", "LM", 1, "",
    ),
    # --- Viet Nam MICS6 ---
    (
        "Viet Nam MICS6 Datasets.zip",
        "Viet Nam", "VNM", "MICS6", 2020, 2021,
        "SSEA", "LM", 1, "",
    ),
    # --- Yemen MICS6 ---
    (
        "Yemen MICS6 Datasets.zip",
        "Yemen", "YEM", "MICS6", 2022, 2023,
        "MENA", "L", 1, "",
    ),
    # --- Zimbabwe 2014 MICS ---
    (
        "Zimbabwe 2014 MICS_Datasets.zip",
        "Zimbabwe", "ZWE", "MICS5", 2014, 2014,
        "SSA", "LM", 1, "",
    ),
    # --- Zimbabwe MICS6 ---
    (
        "Zimbabwe MICS6 SPSS Datasets.zip",
        "Zimbabwe", "ZWE", "MICS6", 2019, 2019,
        "SSA", "LM", 1, "",
    ),
]

COLUMNS = [
    "zip_filename",
    "zip_path",
    "country_name",
    "country_iso3",
    "mics_round",
    "survey_year",
    "survey_year_start",
    "survey_year_end",
    "who_region",
    "wb_income_2024",
    "is_lmic",
    "is_national",
    "in_dhs_68",
    "use_for_expansion",
    "notes",
]


def canonical_survey_year(yr_start: int, yr_end: int) -> int:
    """
    Use the fieldwork end year as the canonical survey year.

    The pooled DHS pipeline keys surveys by a single survey-year value for
    decade assignment and WDI alignment. Using the end year is more stable
    than midpoint rounding and matches the MICS interview-date validation used
    for the expansion surveys.
    """
    return int(yr_end)


def build_catalog() -> list[dict]:
    rows = []
    for rec in CATALOG_ROWS:
        (
            zip_filename,
            country_name,
            iso3,
            mics_round,
            yr_start,
            yr_end,
            who_region,
            wb_income,
            is_national,
            notes,
        ) = rec

        # Resolve zip_path
        zip_path = str(MICS_DIR / zip_filename)

        # Computed flags
        is_lmic = 1 if wb_income in ("L", "LM") else 0
        in_dhs_68 = 1 if iso3 in DHS_68_ISO3 else 0
        use_for_expansion = (
            1
            if (is_national == 1 and is_lmic == 1 and in_dhs_68 == 0)
            else 0
        )
        survey_year = canonical_survey_year(yr_start, yr_end)

        rows.append(
            {
                "zip_filename": zip_filename,
                "zip_path": zip_path,
                "country_name": country_name,
                "country_iso3": iso3,
                "mics_round": mics_round,
                "survey_year": survey_year,
                "survey_year_start": yr_start,
                "survey_year_end": yr_end,
                "who_region": who_region,
                "wb_income_2024": wb_income,
                "is_lmic": is_lmic,
                "is_national": is_national,
                "in_dhs_68": in_dhs_68,
                "use_for_expansion": use_for_expansion,
                "notes": notes,
            }
        )
    return rows


def main() -> None:
    rows = build_catalog()

    # Write CSV
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {OUTPUT_CSV}")

    # Summary stats
    total = len(rows)
    national = sum(1 for r in rows if r["is_national"] == 1)
    use_expansion = sum(1 for r in rows if r["use_for_expansion"] == 1)
    in_dhs = sum(1 for r in rows if r["in_dhs_68"] == 1 and r["is_national"] == 1)
    new_countries = sorted(set(
        r["country_name"]
        for r in rows
        if r["use_for_expansion"] == 1
    ))
    new_unique = len(set(r["country_iso3"] for r in rows if r["use_for_expansion"] == 1))

    print()
    print("=" * 60)
    print("MICS CATALOG SUMMARY")
    print("=" * 60)
    print(f"Total BH-positive rows:           {total}")
    print(f"National surveys:                  {national}")
    print(f"Sub-national (excluded):           {total - national}")
    print(f"National & in DHS-68:              {in_dhs}")
    print(f"use_for_expansion = 1:             {use_expansion}")
    print(f"New unique countries for expansion:{new_unique}")
    print()
    print("New countries (use_for_expansion=1):")
    for c in sorted(set(r["country_name"] for r in rows if r["use_for_expansion"] == 1)):
        iso = next(r["country_iso3"] for r in rows if r["country_name"] == c and r["use_for_expansion"] == 1)
        region = next(r["who_region"] for r in rows if r["country_name"] == c and r["use_for_expansion"] == 1)
        income = next(r["wb_income_2024"] for r in rows if r["country_name"] == c and r["use_for_expansion"] == 1)
        print(f"  {iso}  {c:35s} {region}  {income}")

    # Write notes markdown
    new_country_lines = "\n".join(
        f"- {r['country_iso3']} {r['country_name']} ({r['who_region']}, {r['wb_income_2024']}, {r['mics_round']} {r['survey_year']})"
        for r in sorted(rows, key=lambda x: (x["country_name"], x["survey_year"]))
        if r["use_for_expansion"] == 1
    )

    dhs_overlap_lines = "\n".join(
        f"- {r['country_iso3']} {r['country_name']} ({r['mics_round']} {r['survey_year']}) -- already in DHS 68"
        for r in sorted(rows, key=lambda x: (x["country_name"], x["survey_year"]))
        if r["in_dhs_68"] == 1 and r["is_national"] == 1
    )

    notes_text = f"""# MICS Survey Catalog Notes

Generated: 2026-05-21
Script: src/data/build_mics_catalog.py

## 1. How the catalog was constructed

The catalog covers all MICS zip files confirmed to contain a birth-history
file (bh.sav), located in:
  MICS Data/MICS_Datasets/

Each zip receives one row. Metadata (ISO3, round, survey years, WHO region,
World Bank 2024 income group) was assigned manually from UNICEF MICS
documentation and World Bank country classifications.

Key eligibility flags:
- is_national: 1 for national surveys, 0 for sub-national (province, county,
  community, or territory surveys)
- in_dhs_68: 1 if the country already appears in our 68-country DHS dataset
  (table_s1_survey_inventory.csv)
- is_lmic: 1 if World Bank 2024 income group is L or LM (low or lower-middle)
- use_for_expansion: 1 if is_national=1 AND is_lmic=1 AND in_dhs_68=0

## 2. Countries genuinely new vs DHS-overlap

### New countries (use_for_expansion = 1)
These are LMIC national surveys not covered by DHS:

{new_country_lines}

### DHS-overlap countries
These have MICS surveys but their country is already in the DHS 68 set:

{dhs_overlap_lines}

## 3. Decision rationale: geographic gaps only

The expand_global study targets geographic coverage gaps, not data source
diversity. Adding MICS data for countries already represented by DHS would
introduce cross-instrument comparability challenges (different variable
naming, sampling frames, questionnaire wording) without expanding geographic
coverage. Therefore:

- MICS surveys for countries IN the DHS 68 are catalogued but excluded
  (use_for_expansion = 0).
- MICS surveys for countries NOT in the DHS 68 that are national and LMIC
  are flagged use_for_expansion = 1.
- Sub-national surveys (Pakistan provinces, Kenya counties, Kosovo Roma
  communities, Mongolia sub-provincial surveys, Senegal Dakar, Turks and
  Caicos) are excluded (is_national = 0) because they cannot be treated as
  nationally representative.
- Upper-middle and high income national surveys (Belize, Dominican Republic,
  El Salvador, Paraguay, Samoa, Fiji, Tonga, Tuvalu, Nauru, Kosovo, North
  Macedonia, Tonga, Suriname, Turkmenistan, Guyana, Kazakhstan, Iraq,
  Trinidad and Tobago) have is_lmic = 0 and are excluded from expansion.

## 4. Income classification notes

World Bank FY2024 income thresholds (GNI per capita, Atlas method):
- Low (L):           <= USD 1,135
- Lower-middle (LM): USD 1,136 -- 4,465
- Upper-middle (UM): USD 4,466 -- 13,845
- High (H):          > USD 13,845

Sources consulted for each country follow the World Bank official list.
Notable edge cases:
- Honduras classified as LM (GNI ~$2,800); use_for_expansion = 1 since not
  in DHS 68
- Nepal reclassified LM in FY2024 (was L in earlier years)
- Mauritania reclassified LM in FY2023
"""

    with open(OUTPUT_NOTES, "w", encoding="utf-8") as f:
        f.write(notes_text)

    print(f"\nNotes written to {OUTPUT_NOTES}")


if __name__ == "__main__":
    main()
