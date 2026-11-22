# Maternal education, birth spacing, and under-5 mortality across 79 LMICs — analysis code

Analysis code and derived aggregate results for:

> **Maternal Education, Birth Spacing, and Under-5 Mortality in 77 Low- and
> Middle-Income Countries: A Joint Counterfactual Analysis of Attributable Burden.**
> Md Muhtasim Munif Fahim, Md Rezaul Karim.
> Data Science Research Lab, Department of Statistics and Data Science, University of Rajshahi,
> Rajshahi 6205, Bangladesh.

A harmonised analysis of **215 DHS and UNICEF MICS surveys across 79 countries,
2000–2024**. The primary cohort is the 8,184,578 births with exact 60-month
follow-up; the fitted surfaces use the 7,652,332 of those with a complete wealth
index (193 surveys, 77 countries), and the discrete-time survival analysis
retains births censored before 60 months.

## What is here

```
src/data/          harmonisation of DHS Birth Recode / IPUMS-DHS and MICS
src/descriptive/   cohort description, survey inventory, predictor provenance,
                   the survey rounds excluded for lack of a wealth index,
                   age-at-death heaping
src/inferential/   the estimation core: survey-weighted logistic and
                   discrete-time survival surfaces, g-computation PAFs,
                   Wagstaff decomposition, the estimand-shift decomposition,
                   sensitivity and ablation grids, per-country results
src/synthesis/     the evidence-registry builder
config/            paths and the fixed RNG seed
output/            derived aggregate results (CSV) and the evidence registry
```

## What is deliberately not here

- **Individual-level microdata.** DHS and MICS terms of use forbid
  redistribution. Both are free to any researcher after registration:
  <https://dhsprogram.com> and <https://mics.unicef.org>. The harmonisation code
  that turns those files into the analysis cohort *is* included, so the cohort is
  reconstructable from the sources.
- **The manuscript, its LaTeX, and the figure/table rendering layer.** The paper
  is submitted as a Word document and its display items are published with the
  article; a parallel copy here would only drift. Every computation behind a
  reported number is included; only the drawing code is left out.

## Every reported number is traceable

`output/evidence_registry.csv` maps **all 79 reported quantities** to the source
file and the script that produced them. That is the intended entry point for
anyone checking a number in the paper: find the claim, follow `source_file` and
`source_script`.

Headline results, for orientation:

| Quantity | Value |
|---|---|
| Joint counterfactual, 2018–24 exposures (primary) | 25·1% (95% CI 22·4–27·3) |
| — as annual deaths, applied to UN-IGME 2024 totals | ≈1·08 million |
| Joint counterfactual, 2000–24 exposures | 30·5% (27·5–32·1) |
| Education alone / birth spacing alone (2018–24) | 16·9% / 9·9% |
| Wealth-related concentration index | −0·134 (−0·155 to −0·105) |
| Conventional pooled approach on the same cohort | 47·6%, against 32·2% corrected |

These are **conditional counterfactual quantities** under measured-confounding
assumptions — what the fitted surface implies if a covariate distribution were
shifted — not causal effects and not projected programme effects.

## Reproducing

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Python 3.11+ (developed on 3.13). The RNG seed is fixed at `RNG_SEED = 20260505`
in `config/paths.py`; no seed is hard-coded anywhere else. Bootstraps are
country-cluster resamples at B=1000.

Point `config/paths.py` at a directory holding the DHS/MICS source files, then
run the harmonisation under `src/data/`, then the estimation modules under
`src/inferential/`. Most modules are runnable as `python -m src.<package>.<module>`.
Several are run-once scripts that produced a specific published CSV; each says so
in its docstring, and the CSV it writes is in `output/`.

The survival estimation uses a GPU where one is available (`compute_backend="cuda"`)
and falls back to CPU. Coefficients and standardised risks agree across backends
to ≤1e-9.

## Licence and citation

MIT (see `LICENSE`). Please cite the article; `CITATION.cff` carries the software
citation. The DHS Program, IPUMS-DHS, UNICEF MICS, national statistical offices,
and the participating women and households are the source of the underlying data.
