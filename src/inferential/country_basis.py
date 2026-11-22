"""The surface behind every country- and region-level display.

Since 2026-11-17 (supervisor review of 2026-10-10: one primary estimand in every
main display) the country and regional fractions come from the discrete-time
survival surface, the same surface as the pooled headline, evaluated at each
country's own observed exposures (``survival_country_pafs``). The exact-binary
country files (``table_percountry_paf.csv`` and siblings) are kept as the
sensitivity analysis and are read only where a display compares the two.

Every builder takes its file names from here, so the figures, tables, registry
and Word edits cannot drift onto different surfaces.
"""

PERCOUNTRY_PAF = "table_percountry_paf_survival.csv"
PERCOUNTRY_PAF_CI = "table_percountry_paf_ci_survival.csv"
REGIONAL_PAF = "table_regional_paf_survival.csv"
IGME_ANNUALISATION = "table_igme_annualisation_survival.csv"
SDG_SCOREBOARD = "table_sdg_scoreboard_survival.csv"
PERCOUNTRY_BURDEN = "table_percountry_burden_survival.csv"

# Exact-binary sensitivity counterparts.
BINARY_PERCOUNTRY_PAF = "table_percountry_paf.csv"
BINARY_PERCOUNTRY_PAF_CI = "table_percountry_paf_ci.csv"
BINARY_REGIONAL_PAF = "table_regional_paf.csv"
BINARY_IGME_ANNUALISATION = "table_igme_annualisation.csv"
BINARY_SDG_SCOREBOARD = "table_sdg_scoreboard.csv"
BINARY_PERCOUNTRY_BURDEN = "table_percountry_burden.csv"

BASIS_LABEL = "discrete-time survival surface, each country's own observed exposures"
