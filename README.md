# Global Under-5 Mortality Risk Stratification — 79 LMICs (DHS + MICS, 2000–2024)

Analysis code, derived aggregate results, evidence registry, and the trained
pre-birth prediction model for a harmonised study of under-5 mortality across 79
low- and middle-income countries (10,669,397 birth-history records from 215 DHS
and UNICEF MICS country-surveys, 2000–2024).

The pipeline covers: DHS/MICS **data harmonization**, **sequential GEE**
associations, **g-computation** single-factor and joint population attributable
fractions and avertable deaths, **Wagstaff decomposition** of wealth-related
inequality, and a **leakage-screened, externally validated** pre-birth
machine-learning risk model (temporal / geographic / leave-one-region-out).

## What is (and is not) here

**Included**
- `src/` — the analysis pipeline:
  - `src/data/` — full DHS/MICS **harmonization** (catalog → convert → harmonize → integrate).
  - `src/inferential/` — sequential GEE, g-computation PAFs, Wagstaff decomposition, and sensitivity analyses.
  - `src/ml/` — leakage-safe splits, XGBoost training, temporal/geographic/LORO validation, calibration/recalibration, decision curves, fairness, SHAP, DeLong CIs, integer risk score.
  - `src/utils/`, `config/` — shared loaders and path/seed configuration.
- `output/evidence_registry.csv` — maps every numerical claim in the paper to the exact source CSV and generating script.
- `output/tables/`, `output/fulldata/*.csv`, `output/ml/*.csv` — derived **aggregate** results (country-, region-, or model-level; no individual records).
- `output/ml/xgb_model.json` + `recalibrator_isotonic.pkl` / `recalibrator_platt.pkl` + `country_splits.json` + `optuna_best_params.json` — the trained leakage-screened model and its recalibrators.
- `data/catalog/`, `data/mics_survey_catalog.csv` — survey inventories (metadata only).
- `tests/` — pipeline and integrity tests (invariants, DeLong, inference smoke, leakage).

**Not included — and why**
- **No individual-level microdata.** DHS and MICS data-use agreements prohibit redistribution. Register (free) at the [DHS Program](https://dhsprogram.com) and [UNICEF MICS](https://mics.unicef.org) for the identical inputs, then rebuild the cohort with the harmonization pipeline.
- No manuscript, figures, or typesetting artifacts — this repository is the analysis, not the paper.

## Reproducing the analysis

```bash
conda env create -f environment.yml      # or: pip install -r requirements.txt
export DHS_DATA_DIR=/path/to/DHS_dta_folder   # downloaded DHS .DTA files
# MICS zips go under data/raw/mics/ (see src/data/s03_mics_pipeline.py)

python src/data/s01_catalog.py           # inventory raw surveys
python src/data/s02_convert.py           # .DTA -> per-survey parquet
python src/data/s03_harmonize.py         # harmonised analysis cohort (DHS)
python src/data/s03_mics_pipeline.py     # MICS harmonisation + integration
```

Downstream stages read the harmonized cohort via `src/utils/io.py:load_primary()`
(`src/inferential/` for the statistics, `src/ml/` for the prediction pipeline).
All randomness is seeded (`config/paths.py:RNG_SEED`); full-cohort refits need
~64 GB RAM. Every reported number is reproducible from the script named for it in
`output/evidence_registry.csv`.

**Note on `.pkl` files:** the two recalibrators are scikit-learn pickles — load
them only from this repository and only if you trust it (standard pickle caveat).
`xgb_model.json` is a plain-JSON XGBoost dump with no such caveat.

## Data availability

DHS and MICS microdata: free from the providers above after registration; not
redistributable here. UN-IGME national death envelopes:
[childmortality.org](https://childmortality.org). Everything generated *by* this
study (code, aggregates, registry, model) is in this repository and archived on
Zenodo ([10.5281/zenodo.21302199](https://doi.org/10.5281/zenodo.21302199)).

## License and citation

Code and derived outputs: [MIT](LICENSE). See [CITATION.cff](CITATION.cff).
