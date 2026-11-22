"""reconcile_survival_primary_burden.py — survival-primary burden reconciliation.

The author decision is that the discrete-time SURVIVAL joint PAF (30.5%) is the
manuscript's primary headline avertable *fraction*, with the exact-binary
G-computation (32.2%) as the sensitivity analysis. This script re-anchors the
absolute burden translations to that primary estimand so the abstract's fraction
and its absolute figures are internally consistent, WITHOUT refitting any model or
perturbing the anchor-locked binary decomposition — it reads published outputs only.

Design note: since 2026-11-17 the geographic *distribution* of avertable burden
(which countries / regions) also comes from the survival surface, evaluated at each
country's own exposures (`survival_country_pafs.py`), so the headline, the country
rows and the SDG scenario share one estimand. The exact-binary per-country
decomposition is kept as a sensitivity row. Until then the distribution had been
taken from the exact-binary surface, which the supervisor's review (2026-10-10)
flagged as mixing two estimands in the main displays.

Reads:
  output/phase4/phase3_publication/phase3_bootstrap_summary.csv   (survival, primary)
  output/phase4/phase2_publication/phase2_bootstrap_summary.csv   (binary, sensitivity)
  output/phase4/manuscript/table_igme_annualisation_survival.csv  (survival per-country)
  output/phase4/manuscript/table_igme_annualisation.csv           (binary per-country)
  output/phase4/manuscript/table_avertable_burden.csv             (cohort-anchored, both)
  output/phase4/manuscript/table_sdg_scoreboard_survival.csv      (survival per-country)
  output/phase4/manuscript/table_sdg_scoreboard.csv               (binary per-country)

Writes:
  output/phase4/manuscript/table_burden_reconciled.csv

Run:
  python -m src.inferential.reconcile_survival_primary_burden
"""
from __future__ import annotations

import pandas as pd

from config.paths import OUTPUT_DIR
from src.inferential import country_basis as cb

PUB = OUTPUT_DIR / "phase4"
OUT = OUTPUT_DIR / "phase4" / "manuscript"
SDG_TARGET = 25.0  # SDG 3.2 under-five mortality target, per 1,000 live births


def _joint_survival() -> tuple[float, float, float]:
    df = pd.read_csv(PUB / "phase3_publication" / "phase3_bootstrap_summary.csv")
    r = df[(df["scenario"] == "joint") & (df["metric"] == "paf_pct")].iloc[0]
    return float(r["point_estimate"]), float(r["ci_lower"]), float(r["ci_upper"])


def _joint_binary() -> tuple[float, float, float]:
    df = pd.read_csv(PUB / "phase2_publication" / "phase2_bootstrap_summary.csv")
    r = df[(df["scenario"] == "joint") & (df["metric_type"] == "paf_pct")].iloc[0]
    return float(r["point_estimate"]), float(r["ci_lower"]), float(r["ci_upper"])


def _joint_recent(filename: str) -> tuple[float, float, float] | None:
    """Joint PAF re-standardised to the 2018-2024 exposure mix, if computed."""
    path = OUT / filename
    if not path.exists():
        return None
    df = pd.read_csv(path)
    r = df[df["scenario"] == "joint"].iloc[0]
    return (float(r["paf_recent_prevalence_pct"]), float(r["ci_lo"]),
            float(r["ci_hi"]))


def main() -> None:
    surv_p, surv_lo, surv_hi = _joint_survival()
    bin_p, bin_lo, bin_hi = _joint_binary()

    ann = pd.read_csv(OUT / cb.BINARY_IGME_ANNUALISATION)
    ann_s = pd.read_csv(OUT / cb.IGME_ANNUALISATION)
    mapped_total = int(ann["igme_u5_deaths_2024"].sum())
    if int(ann_s["igme_u5_deaths_2024"].sum()) != mapped_total:
        raise SystemExit("survival and binary annualisations map different IGME totals")
    n_mapped = int(len(ann))
    avert_binary_percountry = int(ann["avertable_per_year"].sum())
    eff_binary_pct = 100.0 * avert_binary_percountry / mapped_total
    avert_surv_percountry = int(ann_s["avertable_per_year"].sum())
    eff_surv_pct = 100.0 * avert_surv_percountry / mapped_total

    def annualise(p, lo, hi):
        return (int(round(p / 100 * mapped_total)),
                int(round(lo / 100 * mapped_total)),
                int(round(hi / 100 * mapped_total)))

    surv_ann = annualise(surv_p, surv_lo, surv_hi)
    bin_ann_flat = annualise(bin_p, bin_lo, bin_hi)

    # cohort-anchored avertable deaths (already both estimands)
    burden = pd.read_csv(OUT / "table_avertable_burden.csv")
    surv_cohort = int(burden.loc[burden["model"].str.startswith("Survival"),
                                 "avertable_deaths"].iloc[0])
    bin_cohort = int(burden.loc[burden["model"].str.startswith("Exact-binary"),
                                "avertable_deaths"].iloc[0])
    recorded = int(burden["recorded_deaths_cohort"].iloc[0])

    # SDG 3.2 scoreboard: survival per-country decomposition (primary) + binary
    sdg_s = pd.read_csv(OUT / cb.SDG_SCOREBOARD)
    sdg = pd.read_csv(OUT / cb.BINARY_SDG_SCOREBOARD)
    n_sdg = int(len(sdg_s))
    fact = int(sdg_s["reaches_sdg_factual"].sum())
    if fact != int(sdg["reaches_sdg_factual"].sum()):
        raise SystemExit("factual SDG counts differ between bases")
    cf_surv = int(sdg_s["reaches_sdg_counterfactual"].sum())
    cf_binary = int(sdg["reaches_sdg_counterfactual"].sum())

    rows = [
        {"quantity": "Joint PAF (global)", "estimand": "survival (PRIMARY)",
         "basis": "discrete-time survival, full birth history",
         "value": round(surv_p, 2), "ci_low": round(surv_lo, 2), "ci_high": round(surv_hi, 2),
         "unit": "percent"},
        {"quantity": "Joint PAF (global)", "estimand": "exact-binary (sensitivity)",
         "basis": ">=60-mo exact-binary G-computation",
         "value": round(bin_p, 2), "ci_low": round(bin_lo, 2), "ci_high": round(bin_hi, 2),
         "unit": "percent"},
        {"quantity": "Avertable deaths per year (UN-IGME 2024)", "estimand": "survival (PRIMARY)",
         "basis": f"survival global PAF x {mapped_total:,} mapped IGME 2024 U5 deaths ({n_mapped} countries)",
         "value": surv_ann[0], "ci_low": surv_ann[1], "ci_high": surv_ann[2], "unit": "deaths/year"},
        {"quantity": "Avertable deaths per year (UN-IGME 2024)", "estimand": "exact-binary, flat",
         "basis": f"binary global PAF x {mapped_total:,} mapped IGME 2024 U5 deaths",
         "value": bin_ann_flat[0], "ci_low": bin_ann_flat[1], "ci_high": bin_ann_flat[2],
         "unit": "deaths/year"},
        {"quantity": "Avertable deaths per year (UN-IGME 2024)", "estimand": "survival, per-country decomposition",
         "basis": "sum of per-country survival joint PAF x per-country IGME 2024 deaths (GEOGRAPHIC BASIS)",
         "value": avert_surv_percountry, "ci_low": "", "ci_high": "",
         "unit": f"deaths/year (effective {eff_surv_pct:.1f}%)"},
        {"quantity": "Avertable deaths per year (UN-IGME 2024)", "estimand": "exact-binary, per-country decomposition (sensitivity)",
         "basis": "sum of per-country binary joint PAF x per-country IGME 2024 deaths",
         "value": avert_binary_percountry, "ci_low": "", "ci_high": "",
         "unit": f"deaths/year (effective {eff_binary_pct:.1f}%)"},
        {"quantity": "Avertable deaths (cohort-anchored)", "estimand": "survival (PRIMARY)",
         "basis": f"of {recorded:,} recorded deaths in the complete-wealth cohort",
         "value": surv_cohort, "ci_low": "", "ci_high": "", "unit": "deaths"},
        {"quantity": "Avertable deaths (cohort-anchored)", "estimand": "exact-binary (sensitivity)",
         "basis": f"of {recorded:,} recorded deaths in the complete-wealth cohort",
         "value": bin_cohort, "ci_low": "", "ci_high": "", "unit": "deaths"},
        {"quantity": "SDG 3.2 countries reaching <=25/1,000", "estimand": "factual (observed)",
         "basis": f"{n_sdg} countries with IGME 2024 U5MR",
         "value": fact, "ci_low": "", "ci_high": "", "unit": "countries"},
        {"quantity": "SDG 3.2 countries reaching <=25/1,000", "estimand": "counterfactual, survival (GEOGRAPHIC BASIS)",
         "basis": "per-country survival joint PAF applied to per-country IGME 2024 U5MR",
         "value": cf_surv, "ci_low": "", "ci_high": "", "unit": "countries"},
        {"quantity": "SDG 3.2 countries reaching <=25/1,000", "estimand": "counterfactual, exact-binary (sensitivity)",
         "basis": "per-country binary joint PAF applied to per-country IGME 2024 U5MR",
         "value": cf_binary, "ci_low": "", "ci_high": "", "unit": "countries"},
    ]
    # --- contemporary (2018-2024 exposure mix) -------------------------------
    # The pooled fractions above average exposures over 2000-2024. Because schooling
    # and birth spacing both improved over that window, they overstate the share
    # attributable at today's exposure prevalence. These rows carry the
    # re-standardised estimand, which is the appropriate one for present-day policy.
    rec_surv = _joint_recent("table_recentdecade_paf_survival.csv")
    rec_bin = _joint_recent("table_recentdecade_paf_reweighted.csv")
    if rec_surv:
        rs_ann = annualise(*rec_surv)
        rows += [
            {"quantity": "Joint PAF (global, 2018-2024 exposure mix)",
             "estimand": "survival (PRIMARY, contemporary)",
             "basis": "survival surface re-standardised to recent exposure prevalence",
             "value": round(rec_surv[0], 2), "ci_low": round(rec_surv[1], 2),
             "ci_high": round(rec_surv[2], 2), "unit": "percent"},
            {"quantity": "Avertable deaths per year (UN-IGME 2024, contemporary)",
             "estimand": "survival (PRIMARY, contemporary)",
             "basis": f"contemporary survival PAF x {mapped_total:,} mapped IGME 2024 "
                      f"U5 deaths ({n_mapped} countries)",
             "value": rs_ann[0], "ci_low": rs_ann[1], "ci_high": rs_ann[2],
             "unit": "deaths/year"},
        ]
    if rec_bin:
        rows.append(
            {"quantity": "Joint PAF (global, 2018-2024 exposure mix)",
             "estimand": "exact-binary (sensitivity, contemporary)",
             "basis": "binary surface re-standardised to recent exposure prevalence",
             "value": round(rec_bin[0], 2), "ci_low": round(rec_bin[1], 2),
             "ci_high": round(rec_bin[2], 2), "unit": "percent"})

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "table_burden_reconciled.csv", index=False)
    if rec_surv:
        print(f"[reconcile] CONTEMPORARY (survival, 2018-24 exposures) joint PAF "
              f"{rec_surv[0]:.2f}% (CI {rec_surv[1]:.2f}-{rec_surv[2]:.2f})")
        print(f"[reconcile]   -> avertable/year ~ {rs_ann[0]:,} "
              f"(CI {rs_ann[1]:,}-{rs_ann[2]:,})")

    print(f"[reconcile] mapped IGME 2024 U5 deaths (n={n_mapped}): {mapped_total:,}")
    print(f"[reconcile] PRIMARY (survival) joint PAF {surv_p:.2f}% (CI {surv_lo:.2f}-{surv_hi:.2f})")
    print(f"[reconcile]   -> avertable/year ~ {surv_ann[0]:,} (CI {surv_ann[1]:,}-{surv_ann[2]:,})")
    print(f"[reconcile] sensitivity (binary) joint PAF {bin_p:.2f}%; per-country decomp {avert_binary_percountry:,}/yr (eff {eff_binary_pct:.1f}%)")
    print(f"[reconcile] cohort-anchored: survival {surv_cohort:,} / binary {bin_cohort:,} of {recorded:,} recorded")
    print(f"[reconcile] SDG 3.2: {fact} factual -> {cf_surv} survival-cf / {cf_binary} binary-cf (of {n_sdg})")
    print(f"[reconcile] survival per-country decomp {avert_surv_percountry:,}/yr (eff {eff_surv_pct:.1f}%)")
    print(f"[reconcile] wrote {OUT / 'table_burden_reconciled.csv'}")


if __name__ == "__main__":
    main()
