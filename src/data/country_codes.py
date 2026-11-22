"""Country-code normalization at DHS/MICS integration seams."""

from __future__ import annotations


# DHS recode prefixes are not uniformly ISO alpha-2. This explicit map covers
# the study inventory and avoids comparing two-character DHS codes with MICS
# alpha-3 codes in overlap checks.
DHS_CODE_TO_ISO3 = {
    "AF": "AFG", "AL": "ALB", "AM": "ARM", "AO": "AGO", "AZ": "AZE",
    "BD": "BGD", "BF": "BFA", "BJ": "BEN", "BO": "BOL", "BR": "BRA",
    "BU": "BDI", "CD": "COD", "CF": "CAF", "CG": "COG", "CI": "CIV",
    "CM": "CMR", "CO": "COL", "DR": "DOM", "EC": "ECU", "EG": "EGY",
    "ER": "ERI", "ET": "ETH", "GA": "GAB", "GH": "GHA", "GM": "GMB",
    "GN": "GIN", "GU": "GTM", "GY": "GUY", "HN": "HND", "HT": "HTI",
    "IA": "IND", "ID": "IDN", "JO": "JOR", "KE": "KEN", "KH": "KHM",
    "KK": "KAZ", "KM": "COM", "KY": "KGZ", "LA": "LAO", "LB": "LBR",
    "LS": "LSO", "MA": "MAR", "MB": "MDA", "MD": "MDG", "ML": "MLI",
    "MM": "MMR", "MR": "MRT", "MV": "MDV", "MW": "MWI", "MZ": "MOZ",
    "NC": "NIC", "NG": "NGA", "NI": "NER", "NM": "NAM", "NP": "NPL",
    "PE": "PER", "PG": "PNG", "PH": "PHL", "PK": "PAK", "PY": "PRY",
    "RW": "RWA", "SD": "SDN", "SL": "SLE", "SN": "SEN", "SR": "SUR",
    "ST": "STP", "SZ": "SWZ", "TD": "TCD", "TG": "TGO", "TJ": "TJK",
    "TL": "TLS", "TM": "TKM", "TN": "TUN", "TR": "TUR", "TZ": "TZA",
    "UA": "UKR", "UG": "UGA", "UZ": "UZB", "VN": "VNM", "YE": "YEM",
    "ZA": "ZAF", "ZM": "ZMB", "ZW": "ZWE",
}


def normalize_country_iso3(code: object) -> str | None:
    """Return a canonical alpha-3 code for study inventory values."""
    if code is None:
        return None
    value = str(code).strip().upper()
    if not value:
        return None
    if len(value) == 3:
        return value
    return DHS_CODE_TO_ISO3.get(value)
