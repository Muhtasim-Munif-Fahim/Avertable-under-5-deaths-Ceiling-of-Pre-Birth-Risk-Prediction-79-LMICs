"""build_evidence_registry.py — wholesale v2 evidence registry.

Assembles every v2 headline number from the regenerated outputs into a single
registry (claim, value, source_file, source_script), mirroring the legacy
output/evidence_registry.csv but sourced entirely from the v2 exact-cohort
publications and ports. Written to output/phase4/manuscript/evidence_registry.csv
so the frozen legacy registry stays untouched until the manuscript switches over.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from config.paths import OUTPUT_DIR

MV = OUTPUT_DIR / "phase4" / "manuscript"
P2 = OUTPUT_DIR / "phase4" / "phase2_publication"
P3 = OUTPUT_DIR / "phase4" / "phase3_publication"


def main() -> None:
    rows: list[dict] = []

    def add(claim, value, src, script):
        rows.append({"claim": claim, "value": value, "source_file": src, "source_script": script})

    # ---- cohort / descriptive ----
    s1 = pd.read_csv(MV / "table_s1_survey_inventory.csv")
    s2 = pd.read_csv(MV / "table_s2_characteristics.csv")
    allrow = s2[s2["region"] == "All"].iloc[0]
    add("Exact analysis cohort",
        f"{int(s1['n_births_exact'].sum()):,} births (>=60-mo exact), "
        f"{s1['country_code'].nunique()} countries, {len(s1)} surveys",
        "manuscript/table_s1_survey_inventory.csv", "src/descriptive/s04_descriptive.py")
    add("Overall U5 mortality risk (exact cohort, survey-weighted)",
        f"{allrow['u5_risk_per1000']} per 1,000",
        "manuscript/table_s2_characteristics.csv", "src/descriptive/s04_descriptive.py")

    # ---- Table 3 PAFs (binary + survival) ----
    b = pd.read_csv(P2 / "phase2_bootstrap_summary.csv")
    s = pd.read_csv(P3 / "phase3_bootstrap_summary.csv")
    for scen in ["education", "spacing", "joint"]:
        bp = b[(b.metric_type == "paf_pct") & (b.scenario == scen)].iloc[0]
        add(f"{scen.capitalize()} PAF (exact-binary, B=1000)",
            f"{bp.point_estimate:.2f} ({bp.ci_lower:.2f}-{bp.ci_upper:.2f})",
            "phase4/phase2_publication/phase2_bootstrap_summary.csv", "src/inferential/phase4_full_analysis.py")
        sp = s[(s.metric == "paf_pct") & (s.scenario == scen)].iloc[0]
        add(f"{scen.capitalize()} PAF (survival, B=1000)",
            f"{sp.point_estimate:.2f} ({sp.ci_lower:.2f}-{sp.ci_upper:.2f})",
            "phase4/phase3_publication/phase3_bootstrap_summary.csv", "src/inferential/phase3_survival.py")

    # ---- Table 4 inequality ----
    t4m = json.loads((MV / "table_4_inequality_decomp_manifest.json").read_text())
    add("Wealth concentration index (U5 mortality)",
        f"{t4m['total_concentration_index']:.4f} ({t4m['total_ci_lo']:.4f},{t4m['total_ci_hi']:.4f})",
        "manuscript/table_4_inequality_decomp.csv", "src/inferential/s05d_inequality.py")
    add("Education share of wealth-related inequality",
        f"{t4m['education_share_pct']:.1f}% ({t4m['education_share_ci'][0]:.1f}-{t4m['education_share_ci'][1]:.1f})",
        "manuscript/table_4_inequality_decomp.csv", "src/inferential/s05d_inequality.py")

    # ---- mediation ----
    med = pd.read_csv(MV / "table_mediation_education_paf.csv")
    add("Education total-effect PAF (no wealth mediator) vs adjusted",
        f"{med.iloc[1]['paf_pct']} vs {med.iloc[0]['paf_pct']} (mediated {med.iloc[2]['paf_pct']} pp)",
        "manuscript/table_mediation_education_paf.csv", "src/inferential/derived_extras.py")

    # ---- avertable burden + IGME + SDG ----
    av = pd.read_csv(MV / "table_avertable_burden.csv")
    add("Joint avertable deaths (cohort-anchored)",
        f"{av.iloc[0]['avertable_deaths']:,} (binary) / {av.iloc[1]['avertable_deaths']:,} (survival) "
        f"of {av.iloc[0]['recorded_deaths_cohort']:,} recorded",
        "manuscript/table_avertable_burden.csv", "src/inferential/derived_extras.py")
    igme_s = pd.read_csv(MV / "table_igme_annualisation_survival.csv")
    add("Joint avertable deaths per year (UN-IGME 2024; per-country survival, GEOGRAPHIC BASIS)",
        f"{int(igme_s['avertable_per_year'].sum()):,} per year "
        f"(sum of per-country survival joint PAF at own exposures x per-country IGME 2024 deaths)",
        "manuscript/table_igme_annualisation_survival.csv", "src/inferential/derived_extras.py")
    igme = pd.read_csv(MV / "table_igme_annualisation.csv")
    # RELABELLED 2026-08-07: this is the per-country exact-binary sum on the GEOGRAPHIC
    # basis, NOT the survival-primary long-window figure the manuscript headlines
    # (1,305,079). The two were previously shipped under one label; see the
    # long-window atom added below and docs/REWRITE_FLAGS_2026-08-07.md defect 1.
    add("Joint avertable deaths per year (UN-IGME 2024; per-country exact-binary, sensitivity)",
        f"{int(igme['avertable_per_year'].sum()):,} per year "
        f"(sum of per-country binary joint PAF x per-country IGME 2024 deaths)",
        "manuscript/table_igme_annualisation.csv", "src/inferential/derived_extras.py")
    sdg = pd.read_csv(MV / "table_sdg_scoreboard_survival.csv")
    add("SDG 3.2 under joint counterfactual",
        f"{int(sdg['reaches_sdg_factual'].sum())} reach <=25/1000 factually -> "
        f"{int(sdg['reaches_sdg_counterfactual'].sum())} under counterfactual",
        "manuscript/table_sdg_scoreboard_survival.csv", "src/inferential/derived_extras.py")

    # ---- external-review amendments (2026-08-03) ----
    # NOTE: the machine-learning arm (discrimination, LORO, targeting yield,
    # country-clustered AUROC, split stability, calibration) was withdrawn from the
    # manuscript by author decision, same day as the amendments below (honest null
    # result superseded by the estimand-shift decomposition; see
    # docs/reviews/EXTERNAL_REVIEW_RESPONSE_2026-08-03.md). Its atoms are removed
    # from this registry along with it; the archived output lives at
    # ../../_archive/2026-08-03_ml_arm_withdrawn/output_ml_v2/ if ever needed again.
    rec = pd.read_csv(MV / "table_burden_reconciled.csv")

    def _rec_row(q_prefix, e_prefix):
        m = rec[rec["quantity"].str.startswith(q_prefix)
                & rec["estimand"].str.startswith(e_prefix)]
        return m.iloc[0] if len(m) else None

    rs = _rec_row("Joint PAF (global, 2018-2024", "survival")
    if rs is not None:
        add("CONTEMPORARY joint PAF (2018-24 exposures, survival PRIMARY)",
            f"{rs['value']:.2f} ({rs['ci_low']:.2f}-{rs['ci_high']:.2f})",
            "manuscript/table_recentdecade_paf_survival.csv",
            "src/inferential/run_fulldata_recentdecade_paf_survival.py")
    rb = _rec_row("Joint PAF (global, 2018-2024", "exact-binary")
    if rb is not None:
        add("CONTEMPORARY joint PAF (2018-24 exposures, exact-binary sensitivity)",
            f"{rb['value']:.2f} ({rb['ci_low']:.2f}-{rb['ci_high']:.2f})",
            "manuscript/table_recentdecade_paf_reweighted.csv",
            "src/inferential/run_fulldata_recentdecade_paf.py")
    ry = _rec_row("Avertable deaths per year (UN-IGME 2024, contemporary)", "survival")
    if ry is not None:
        add("CONTEMPORARY avertable deaths per year (UN-IGME 2024, survival PRIMARY)",
            f"~{int(ry['value']):,} per year (CI {int(ry['ci_low']):,}-{int(ry['ci_high']):,})",
            "manuscript/table_burden_reconciled.csv",
            "src/inferential/reconcile_survival_primary_burden.py")

    prov = pd.read_csv(MV / "table_predictor_provenance.csv")
    mdl = prov[prov["in_primary_model"]]
    add("Predictor measurement timing",
        f"{len(mdl)} model covariates: {int(mdl['known_at_index_birth'].sum())} known at "
        f"the index birth, {int((~mdl['known_at_index_birth']).sum())} recorded at "
        f"interview (education, marital status, wealth, urban residence)",
        "manuscript/table_predictor_provenance.csv",
        "src/descriptive/s04b_predictor_provenance.py")

    add("Inequality index, bounded-outcome variants",
        f"Wagstaff-normalised {t4m['wagstaff_normalised_index']:.4f} "
        f"({t4m['wagstaff_ci_lo']:.4f},{t4m['wagstaff_ci_hi']:.4f}); "
        f"Erreygers {t4m['erreygers_index']:.4f} "
        f"({t4m['erreygers_ci_lo']:.4f},{t4m['erreygers_ci_hi']:.4f}); "
        f"attainable range {t4m['index_attainable_range'][0]:.3f} to "
        f"{t4m['index_attainable_range'][1]:.3f}",
        "manuscript/table_4_inequality_decomp_manifest.json",
        "src/inferential/s05d_inequality.py")

    sens = pd.read_csv(MV / "table_sensitivity.csv")
    sj = sens[sens["scenario"] == "joint"]
    base = float(sj[sj["analysis"] == "primary"]["paf_pct"].iloc[0])
    add("Joint PAF sensitivity envelope",
        f"{len(sj)} specifications, max deviation "
        f"{(sj['paf_pct'] - base).abs().max():.2f} pp from primary {base:.2f}%",
        "manuscript/table_sensitivity.csv", "src/inferential/sensitivity.py")

    wf = pd.read_csv(MV / "table_paf_waterfall.csv")
    legacy = wf.iloc[0]
    primary = wf.iloc[-1]
    country_fe_row = wf[wf["step"].str.contains("country/survey", case=False)]
    policy_row = wf[wf["step"].str.contains("policy-eligible", case=False)]
    add("Estimand-shift decomposition (conventional pooled vs primary; Figure 4)",
        f"conventional {legacy['joint_paf_pct']:.1f}% -> primary {primary['joint_paf_pct']:.2f}%; "
        f"dominant correction country/survey adjustment "
        f"{country_fe_row['delta_from_prev'].iloc[0]:+.2f} pp, "
        f"policy-eligible contract {policy_row['delta_from_prev'].iloc[0]:+.2f} pp",
        "manuscript/table_paf_waterfall.csv",
        "src/inferential/decompose_paf_shift.py")

    # ------------------------------------------------------------------
    # Atoms added 2026-08-07 to cover every number reported in
    # output/manuscript_rewrite.tex. Before this, Table 1, the mortality
    # trends, the contemporary per-lever PAFs and most of the inequality
    # decomposition had no atom, so the Data Sharing claim that the registry
    # maps every reported number to its source was not true.
    # See docs/REWRITE_FLAGS_2026-08-07.md.
    # ------------------------------------------------------------------

    # ---- cohort denominators (published values, not re-derived) ----
    g3 = pd.read_csv(MV / "table_3_gcomp_pafs.csv")
    n_bin = int(g3[g3["model"].str.startswith("Exact-binary")]["n_births"].iloc[0])
    n_surv = int(g3[g3["model"].str.startswith("Discrete-time")]["n_births"].iloc[0])
    add("Analytic cohort denominators (complete-wealth binary vs survival)",
        f"{n_bin:,} births fitted by the exact-binary model (complete-case on wealth); "
        f"{n_surv:,} by the discrete-time survival model (retains births censored before 60 months)",
        "manuscript/table_3_gcomp_pafs.csv", "src/inferential/phase4_full_analysis.py")

    # ---- survey inventory: programme split and repeat-round structure ----
    is_mics = s1["survey_id"].str.upper().str.contains("MICS")
    mics_countries = set(s1.loc[is_mics, "country_code"])
    dhs_countries = set(s1.loc[~is_mics, "country_code"])
    overlap = mics_countries & dhs_countries
    if overlap:
        raise ValueError(f"Countries contribute to BOTH programmes, which the paper denies: {sorted(overlap)}")
    rounds = s1.groupby("country_code").size()
    add("Survey programme composition",
        f"{int((~is_mics).sum())} DHS country-surveys covering {len(dhs_countries)} countries; "
        f"{int(is_mics.sum())} MICS country-surveys covering {len(mics_countries - dhs_countries)} "
        f"further countries; no country contributes to both programmes",
        "manuscript/table_s1_survey_inventory.csv", "src/descriptive/s04_descriptive.py")
    add("Repeat-survey structure",
        f"{int((rounds > 1).sum())} of {len(rounds)} countries contribute more than one round "
        f"(maximum {int(rounds.max())}), so births are not independent observations",
        "manuscript/table_s1_survey_inventory.csv", "src/descriptive/s04_descriptive.py")

    # ---- complete-wealth analytic base ----
    # These come from the fitted model's OWN diagnostics rather than being re-derived
    # from the 8M-row cohort: phase2_model_diagnostics.csv is written by the estimation
    # run itself, so it cannot drift from the model the paper reports. (Until
    # 2026-08-07 this atom re-read the raw parquet, which was slower, depended on raw
    # data being present, and could in principle disagree with the fitted surface.)
    diag = pd.read_csv(P2 / "phase2_model_diagnostics.csv")
    prim = diag[diag["role"] == "primary_observed_wealth"].iloc[0]
    published_ctry = int(sens[sens["analysis"] == "primary"]["n_countries"].iloc[0])
    if int(prim["n_countries"]) != published_ctry or int(prim["n_births"]) != n_bin:
        raise ValueError(
            f"Primary-surface diagnostics disagree with the published tables: "
            f"diagnostics n_births={int(prim['n_births'])}/n_countries="
            f"{int(prim['n_countries'])} vs table_3 n_births={n_bin} and "
            f"table_sensitivity n_countries={published_ctry}."
        )
    add("Complete-wealth analytic base (surveys and countries)",
        f"{int(prim['n_surveys'])} surveys in {int(prim['n_countries'])} of "
        f"{int(s1['country_code'].nunique())} countries; "
        f"{len(s1) - int(prim['n_surveys'])} rounds excluded for lacking a household "
        f"wealth index; {int(prim['n_deaths']):,} deaths",
        "phase4/phase2_publication/phase2_model_diagnostics.csv",
        "src/inferential/phase4_full_analysis.py")

    # ---- Table 1: primary adjusted odds ratios ----
    aor = pd.read_csv(MV / "table_2_primary_aor.csv").set_index("feature")
    AOR_REPORTED = {
        "edu_cat_Primary": "Maternal primary education (vs none)",
        "edu_cat_Secondary": "Maternal secondary education (vs none)",
        "edu_cat_Higher": "Maternal higher education (vs none)",
        "bi_model_cat_<24 mo": "Preceding birth interval <24 months (vs 24-47)",
        "bi_model_cat_48+ mo": "Preceding birth interval 48+ months (vs 24-47)",
        "birth_type_Multiple": "Multiple birth (vs singleton)",
        "ma_cat_<20": "Maternal age <20 at birth (vs 20-34)",
        "wq_cat_Q5": "Richest wealth quintile (vs poorest)",
        "sex_cat_Male": "Male child (vs female)",
    }
    for feat, label in AOR_REPORTED.items():
        r = aor.loc[feat]
        add(f"Adjusted OR, {label}",
            f"{r['aOR']:.3f} ({r['ci_lo']:.3f}-{r['ci_hi']:.3f})",
            "manuscript/table_2_primary_aor.csv", "src/inferential/phase4_full_analysis.py")

    # ---- regional education gradient ----
    rg = pd.read_csv(MV / "table_2_regional_aor.csv")
    for feat, tag in [("edu_cat_Secondary", "secondary"), ("edu_cat_Higher", "higher")]:
        sub = rg[rg["feature"] == feat]
        lo, hi = sub.loc[sub["aOR"].idxmin()], sub.loc[sub["aOR"].idxmax()]
        add(f"Regional education gradient ({tag}), 5 study regions",
            f"protective in all {len(sub)} regions, {lo['aOR']:.3f} ({lo['region']}) to "
            f"{hi['aOR']:.3f} ({hi['region']}); weakest region CR1 interval "
            f"{hi['ci_lo_cr1']:.3f}-{hi['ci_hi_cr1']:.3f}",
            "manuscript/table_2_regional_aor.csv", "src/inferential/s05b_regional_aor.py")
    clus = rg.groupby("region")["n_clusters"].first().sort_values(ascending=False)
    add("Regional model cluster counts",
        "; ".join(f"{k} {int(v)}" for k, v in clus.items()) + f" (minimum {int(clus.min())})",
        "manuscript/table_2_regional_aor.csv", "src/inferential/s05b_regional_aor.py")

    # ---- mortality trends by survey period ----
    tr = pd.read_csv(MV / "table_s3_mortality_trends_ci.csv")
    tra = tr[tr["region"] == "All"].sort_values("decade")
    first, last = tra.iloc[0], tra.iloc[-1]
    add("U5 mortality by survey period (pooled, exact cohort)",
        "; ".join(
            f"{r['decade']} {r['u5_risk_per1000']:.1f} ({r['ci_lo']:.1f}-{r['ci_hi']:.1f}) "
            f"per 1,000, {int(r['n_countries'])} countries"
            for _, r in tra.iterrows()
        ) + f"; relative reduction "
            f"{(first['u5_risk_per1000'] - last['u5_risk_per1000']) / first['u5_risk_per1000'] * 100:.1f}%",
        "manuscript/table_s3_mortality_trends_ci.csv", "src/descriptive/s04_descriptive.py")
    lac = tr[(tr["region"] == "LAC") & (tr["u5_risk_per1000"].isna())]
    add("Survey-period country composition caveat",
        f"Latin America and the Caribbean contributed no surveys to "
        f"{'/'.join(lac['decade'].astype(str))}, so part of the apparent decline reflects "
        f"changing country composition",
        "manuscript/table_s3_mortality_trends_ci.csv", "src/descriptive/s04_descriptive.py")

    # ---- contemporary per-lever PAFs + the exposure shift that drives them ----
    rd = pd.read_csv(MV / "table_recentdecade_paf_survival.csv").set_index("scenario")
    for scen in ["education", "spacing"]:
        r = rd.loc[scen]
        add(f"CONTEMPORARY {scen} PAF (2018-24 exposures, survival PRIMARY)",
            f"{r['paf_recent_prevalence_pct']:.2f} ({r['ci_lo']:.2f}-{r['ci_hi']:.2f})",
            "manuscript/table_recentdecade_paf_survival.csv",
            "src/inferential/run_fulldata_recentdecade_paf_survival.py")
    j = rd.loc["joint"]
    add("Exposure prevalence shift, pooled 2000-24 vs contemporary 2018-24",
        f"below-secondary education {j['pct_below_secondary_pooled']:.1f}% -> "
        f"{j['pct_below_secondary_recent']:.1f}%; short interval "
        f"{j['pct_short_interval_pooled']:.1f}% -> {j['pct_short_interval_recent']:.1f}%; "
        f"{int(j['n_recent_2018_2024']):,} births in the recent window; "
        f"{j['pct_recent_weight_unsupported']:.1f}% of recent weight outside supported cells",
        "manuscript/table_recentdecade_paf_survival.csv",
        "src/inferential/run_fulldata_recentdecade_paf_survival.py")

    # ---- long-window burden on the SURVIVAL-PRIMARY basis (the headline figure) ----
    rl = _rec_row("Avertable deaths per year (UN-IGME 2024)", "survival")
    if rl is not None:
        add("LONG-WINDOW avertable deaths per year (UN-IGME 2024, survival PRIMARY)",
            f"{int(rl['value']):,} per year (CI {int(rl['ci_low']):,}-{int(rl['ci_high']):,}); "
            f"basis: {rl['basis']}",
            "manuscript/table_burden_reconciled.csv",
            "src/inferential/reconcile_survival_primary_burden.py")
    add("UN-IGME 2024 death denominator (crosswalked countries)",
        f"{int(igme['igme_u5_deaths_2024'].sum()):,} under-5 deaths across "
        f"{len(igme)} crosswalked countries",
        "manuscript/table_igme_annualisation.csv", "src/inferential/derived_extras.py")
    top = igme_s.nlargest(15, "avertable_per_year")["avertable_per_year"].sum()
    tot = igme_s["avertable_per_year"].sum()
    add("Burden concentration in the highest-burden countries",
        f"top 15 countries account for {int(top):,} of {int(tot):,} avertable deaths per year "
        f"({top / tot * 100:.1f}%), on the per-country survival geographic basis",
        "manuscript/table_igme_annualisation_survival.csv", "src/inferential/derived_extras.py")
    pcs = pd.read_csv(MV / "table_percountry_paf_survival.csv")
    add("Range of country joint fractions (survival, own exposures)",
        f"{pcs['paf_joint'].min():.1f}% to {pcs['paf_joint'].max():.1f}% across "
        f"{len(pcs)} countries",
        "manuscript/table_percountry_paf_survival.csv",
        "src/inferential/survival_country_pafs.py")

    # ---- non-additivity of the single-lever fractions ----
    sj_s = {sc: s[(s.metric == "paf_pct") & (s.scenario == sc)].iloc[0].point_estimate
            for sc in ["education", "spacing", "joint"]}
    add("Non-additivity of single-lever PAFs (survival, long-window)",
        f"education {sj_s['education']:.2f} + spacing {sj_s['spacing']:.2f} = "
        f"{sj_s['education'] + sj_s['spacing']:.2f} naive sum vs joint {sj_s['joint']:.2f}; "
        f"joint sits {sj_s['education'] + sj_s['spacing'] - sj_s['joint']:.2f} pp below",
        "phase4/phase3_publication/phase3_bootstrap_summary.csv", "src/inferential/phase3_survival.py")

    # ---- agreement between the two estimands ----
    bj = b[(b.metric_type == "paf_pct") & (b.scenario == "joint")].iloc[0].point_estimate
    if rs is not None and rb is not None:
        add("Agreement between primary survival and exact-binary sensitivity estimands",
            f"long-window {abs(bj - sj_s['joint']):.2f} pp apart "
            f"({sj_s['joint']:.2f} vs {bj:.2f}); contemporary "
            f"{abs(float(rb['value']) - float(rs['value'])):.2f} pp apart "
            f"({float(rs['value']):.2f} vs {float(rb['value']):.2f})",
            "manuscript/table_burden_reconciled.csv",
            "src/inferential/reconcile_survival_primary_burden.py")

    # ---- inequality decomposition: contributions beyond education ----
    dec = pd.read_csv(MV / "table_4_inequality_decomp.csv").set_index("factor")
    for fac in ["Residence", "Birth interval"]:
        r = dec.loc[fac]
        add(f"{fac} share of wealth-related inequality",
            f"{r['share_pct']:.1f}% ({r['share_ci_lo']:.1f}-{r['share_ci_hi']:.1f})",
            "manuscript/table_4_inequality_decomp.csv", "src/inferential/s05d_inequality.py")
    add("Unexplained residual of the inequality decomposition",
        f"{dec.loc['Residual', 'share_pct']:.1f}%",
        "manuscript/table_4_inequality_decomp.csv", "src/inferential/s05d_inequality.py")
    add("Weighted mean U5 risk in the complete-wealth cohort",
        f"{t4m['weighted_u5_risk'] * 100:.1f}% (this is what bounds the concentration index "
        f"at {t4m['index_attainable_range'][1]:.3f} rather than 1)",
        "manuscript/table_4_inequality_decomp_manifest.json", "src/inferential/s05d_inequality.py")

    # ---- per-country wealth gradient ----
    eq = pd.read_csv(MV / "table_percountry_equity.csv")
    ng = eq[eq["country_name"] == "Nigeria"].iloc[0]
    ne = eq[eq["country_name"] == "Niger"].iloc[0]
    add("Per-country wealth gradient direction",
        f"{int((eq['concentration_index'] < 0).sum())} of {len(eq)} countries have a negative "
        f"concentration index (mortality concentrated among poorer households)",
        "manuscript/table_percountry_equity.csv", "src/inferential/percountry_equity.py")
    add("Steepest vs flattest wealth gradient contrast (Nigeria, Niger)",
        f"Nigeria index {ng['concentration_index']:.4f}, poorest-to-richest ratio "
        f"{ng['ratio_q1_q5']:.2f}; Niger {ne['concentration_index']:.4f}, ratio {ne['ratio_q1_q5']:.2f}",
        "manuscript/table_percountry_equity.csv", "src/inferential/percountry_equity.py")

    # ---- E-values ----
    ev = pd.read_csv(MV / "sensitivity" / "sensitivity_evalues.csv").set_index("feature")
    add("E-values for the primary adjusted odds ratios",
        f"range {ev['e_value_point'].min():.2f} to {ev['e_value_point'].max():.2f} across "
        f"{len(ev)} reported determinants; secondary education "
        f"{ev.loc['edu_cat_Secondary', 'e_value_point']:.2f}, multiple birth "
        f"{ev.loc['birth_type_Multiple', 'e_value_point']:.2f}",
        "manuscript/sensitivity/sensitivity_evalues.csv", "src/inferential/sensitivity.py")

    # ---- named sensitivity scenarios ----
    add("Joint PAF sensitivity scenarios, individual deviations",
        "; ".join(
            f"{r['label']} {r['paf_pct'] - base:+.2f} pp"
            for _, r in sj[sj["analysis"] != "primary"].iterrows()
        ),
        "manuscript/table_sensitivity.csv", "src/inferential/sensitivity.py")

    # ---- IGME death-count uncertainty propagated into the burden headline ----
    igu = pd.read_csv(MV / "table_igme_uncertainty.csv").set_index("quantity")
    indep = igu.loc["Avertable deaths/year, PAF + IGME uncertainty, independent regime"]
    comono = igu.loc["Avertable deaths/year, PAF + IGME uncertainty, comonotonic regime"]
    add("Avertable burden interval including UN-IGME's own death-count uncertainty",
        f"independent regime {int(indep['lo']):,}-{int(indep['hi']):,} per year; "
        f"comonotonic (worst-case) regime {int(comono['lo']):,}-{int(comono['hi']):,}; "
        f"point estimate unchanged at ~1,075,380",
        "manuscript/table_igme_uncertainty.csv", "src/inferential/igme_uncertainty.py")

    # ---- full waterfall, every step ----
    add("Estimand-shift decomposition, all steps",
        "; ".join(
            f"{r['step']} {r['joint_paf_pct']:.2f}%"
            + ("" if pd.isna(r["delta_from_prev"]) else f" ({r['delta_from_prev']:+.2f} pp)")
            for _, r in wf.iterrows()
        ) + "; run on the exact-binary surface, so it terminates at the exact-binary fraction",
        "manuscript/table_paf_waterfall.csv", "src/inferential/decompose_paf_shift.py")

    # ---- country-resolved master tables (main-text Tables 2 and 3, 2026-08-16) ----
    burden = pd.read_csv(MV / "table_percountry_burden_survival.csv")
    b_sorted = burden.sort_values("avertable_per_year", ascending=False)
    add("Per-country master burden table (main-text Table 2)",
        f"{len(burden)} countries; {int(burden['n_births'].sum()):,} births and "
        f"{int(burden['n_deaths'].sum()):,} deaths on the complete-wealth cohort; "
        f"{int(burden['avertable_per_year'].sum()):,} potentially avertable deaths/year; "
        f"top 15 countries carry {b_sorted['cum_share_2024_pct'].iloc[14]:.1f}% of that total",
        "manuscript/table_percountry_burden_survival.csv",
        "src/inferential/s05p_percountry_master.py")
    sdg_counts = burden["sdg_status"].value_counts()
    add("SDG 3.2 scoreboard resolved by country (Table 2 final column)",
        f"{int(sdg_counts.get('met', 0))} already at or below 25 per 1,000; "
        f"{int(sdg_counts.get('crosses_under_counterfactual', 0))} cross only under the "
        f"joint counterfactual; {int(sdg_counts.get('still_above', 0))} remain above",
        "manuscript/table_percountry_burden_survival.csv",
        "src/inferential/s05p_percountry_master.py")

    ci = pd.read_csv(MV / "table_percountry_paf_ci_survival.csv")
    width = (ci["paf_joint_hi"] - ci["paf_joint_lo"])
    add("Per-country joint PAF confidence intervals",
        f"country-cluster refit bootstrap, B=1000; median {int(ci['n_draws'].median())} "
        f"draws per country ({ci['pct_replicates_present'].median():.1f}% of replicates, "
        f"a country being absent from the rest); joint interval width median "
        f"{width.median():.2f} pp (range {width.min():.2f}-{width.max():.2f}); "
        f"surface uncertainty at each country's fixed exposure profile, NOT an "
        f"independent national estimate",
        "manuscript/table_percountry_paf_ci_survival.csv",
        "src/inferential/survival_country_pafs.py")

    # ---- supervisor review (2026-11-17): excluded rounds, refit contemporary CI ----
    ex = pd.read_csv(MV / "table_excluded_wealth_surveys_summary.csv").set_index("group")
    e, k, s0 = (ex.loc["excluded (no wealth index)"], ex.loc["retained"],
                ex.loc["retained, fielded 2000-2004"])
    add("Survey rounds excluded for lack of a wealth index",
        f"{int(e['n_rounds'])} DHS rounds fielded {int(e['survey_year_min'])}-"
        f"{int(e['survey_year_max'])} in {int(e['n_countries'])} countries "
        f"({e['rounds_by_region']}); {int(e['n_births_exact']):,} exact-cohort births "
        f"({e['pct_of_exact_cohort']}%); countries lost entirely: {e['countries_entirely_lost']}; "
        f"under-5 risk {e['u5_risk_per1000']} per 1000 vs {k['u5_risk_per1000']} retained and "
        f"{s0['u5_risk_per1000']} in retained rounds fielded 2000-2004",
        "manuscript/table_excluded_wealth_surveys_summary.csv",
        "src/descriptive/excluded_wealth_surveys.py")
    rdc = pd.read_csv(MV / "table_recentdecade_paf_survival.csv").set_index("scenario")
    add("Contemporary survival PAF intervals (refit bootstrap, reported)",
        "; ".join(f"{sc} {rdc.loc[sc, 'paf_recent_prevalence_pct']:.2f} "
                  f"({rdc.loc[sc, 'ci_lo']:.2f}-{rdc.loc[sc, 'ci_hi']:.2f}); earlier fixed-surface "
                  f"{rdc.loc[sc, 'ci_lo_fixed_surface']:.2f}-{rdc.loc[sc, 'ci_hi_fixed_surface']:.2f}"
                  for sc in ("joint", "education", "spacing")),
        "manuscript/table_recentdecade_paf_survival.csv",
        "src/inferential/contemporary_refit_ci.py")

    # ---- ablation grid (eTable, 2026-08-16) ----
    abl = pd.read_csv(MV / "table_ablation_grid.csv")
    contract = abl[abl["axis"] == "counterfactual contract"]
    adjust = abl[abl["axis"] == "adjustment set"]
    add("Ablation: counterfactual contract",
        f"joint PAF spans {contract['delta_joint_vs_primary'].min():+.2f} to "
        f"{contract['delta_joint_vs_primary'].max():+.2f} pp against the primary "
        f"({contract['paf_joint'].min():.2f}-{contract['paf_joint'].max():.2f}%); "
        f"the fitted surface is held fixed, so these need no refit",
        "manuscript/table_ablation_grid.csv",
        "src/inferential/s05s_ablation_grid.py")
    add("Ablation: adjustment set",
        f"joint PAF spans {adjust['delta_joint_vs_primary'].min():+.2f} to "
        f"{adjust['delta_joint_vs_primary'].max():+.2f} pp across six dropped covariate "
        f"blocks; the two largest are household wealth and birth period, both arguably "
        f"on the pathway from schooling to survival rather than confounding it",
        "manuscript/table_ablation_grid.csv",
        "src/inferential/s05s_ablation_grid.py")
    dw = float(adjust[adjust["ablation"] == "drop_wealth"]["paf_education"].iloc[0])
    add("Cross-check: wealth-unadjusted education PAF by two independent routes",
        f"{dw:.2f}% from dropping wq_cat from the design (ablation grid) against "
        f"{float(med.loc[med['estimand'].str.contains('total_effect'), 'paf_pct'].iloc[0]):.2f}% "
        f"from refitting with wealth_mode='none_complete' (mediation analysis)",
        "manuscript/table_ablation_grid.csv + table_mediation_education_paf.csv",
        "src/inferential/s05s_ablation_grid.py")

    # ---- 2026-09-15 reviewer round: sample flow, reporting quality, sensitivity ----
    # Appended after every existing claim so the earlier rows are unchanged.
    sens = OUTPUT_DIR / "phase4" / "sensitivity_2026-09"
    flow = pd.read_csv(P3 / "phase3_cohort_flow.csv").set_index("scope")
    n_exact = int(s1["n_births_exact"].sum())
    diag = pd.read_csv(P2 / "phase2_model_diagnostics.csv")
    n_bin = int(diag.loc[diag["surface"] == "exact_wave_fe_complete_wealth", "n_births"].iloc[0])
    pilot, known, wealth = (flow.loc["pilot"], flow.loc["known_survival_positive_weight"],
                            flow.loc["analysis_observed_wealth"])
    add("Sample flow: harmonised surveys",
        f"{int(pilot['n_births']):,} births and {int(pilot['n_events']):,} under-5 deaths in "
        f"{int(pilot['n_surveys'])} surveys, {int(pilot['n_countries'])} countries",
        "phase4/phase3_publication/phase3_cohort_flow.csv", "src/inferential/phase3_survival.py")
    add("Sample flow: survival cohort",
        f"{int(known['n_births']):,} births with known survival and positive weight; "
        f"{int(wealth['n_births']):,} complete-wealth births with {int(wealth['n_events']):,} "
        f"deaths in {int(wealth['n_surveys'])} surveys, {int(wealth['n_countries'])} countries",
        "phase4/phase3_publication/phase3_cohort_flow.csv", "src/inferential/phase3_survival.py")
    add("Sample flow: exclusions (derived from the counts above)",
        f"binary: {int(pilot['n_births']) - n_exact:,} without 60 months of follow-up, known "
        f"outcome and positive weight, then {n_exact - n_bin:,} without wealth data; survival: "
        f"{int(pilot['n_births']) - int(known['n_births']):,} without known survival or positive "
        f"weight, then {int(known['n_births']) - int(wealth['n_births']):,} without wealth data",
        "phase3_cohort_flow.csv + table_s1_survey_inventory.csv + phase2_model_diagnostics.csv",
        "src/synthesis/build_evidence_registry.py")
    heap = pd.read_csv(sens / "heaping" / "heaping_index.csv").set_index("group")
    add("Age-at-death heaping at 12 months",
        f"deaths reported at 12 months are {heap.loc['All', 'heap_12']:.2f} times the mean of 11 "
        f"and 13 months; month 0 holds {heap.loc['All', 'share_month0_pct']:.1f}% of deaths",
        "phase4/sensitivity_2026-09/heaping/heaping_index.csv",
        "src/descriptive/age_heaping.py")
    add("Follow-up boundary displacement",
        f"births with exactly 60 months of follow-up are "
        f"{heap.loc['followup_boundary', 'heap_60_followup']:.3f} times the mean of 59 and 61",
        "phase4/sensitivity_2026-09/heaping/heaping_index.csv",
        "src/descriptive/age_heaping.py")
    evs = pd.read_csv(MV / "sensitivity" / "sensitivity_evalues.csv").set_index("feature")
    add("E-values at the confidence limit nearer the null",
        f"secondary education {evs.loc['edu_cat_Secondary', 'e_value_ci']:.2f}; "
        f"multiple birth {evs.loc['birth_type_Multiple', 'e_value_ci']:.2f}",
        "manuscript/sensitivity/sensitivity_evalues.csv", "src/inferential/sensitivity.py")
    scripts = {"survival": "src/inferential/sens2026_survival.py",
               "binary": "src/inferential/sens2026_binary.py"}
    for model_dir in ("survival_main", "survival_interaction", "survival_ageband",
                      "binary_main", "binary_interaction"):
        path = sens / model_dir / "summary.csv"
        path = path if path.exists() else sens / model_dir / "point.csv"
        if not path.exists():
            continue
        res = pd.read_csv(path)
        parts = []
        for r in res.itertuples(index=False):
            ci = (f" ({r.ci_lo:.2f}-{r.ci_hi:.2f})"
                  if hasattr(r, "ci_lo") and pd.notna(getattr(r, "ci_lo", None)) else "")
            parts.append(f"{r.estimand} {r.scenario} {r.paf_pct:.2f}{ci}")
        add(f"Sensitivity 2026-09: {model_dir.replace('_', ' ')} (joint, education, spacing PAF %)",
            "; ".join(parts), f"phase4/sensitivity_2026-09/{model_dir}/{path.name}",
            scripts[model_dir.split("_")[0]])
    shp = sens / "shapley" / "shapley_values.csv"
    if shp.exists():
        table = pd.read_csv(shp)
        add("Shapley decomposition of the estimand shift (pp of joint PAF)",
            "; ".join(f"{r.choice} {r.shapley_pp:+.2f} (range {r.min_marginal_pp:+.2f} to "
                      f"{r.max_marginal_pp:+.2f})" for r in table.itertuples(index=False)),
            "phase4/sensitivity_2026-09/shapley/shapley_values.csv",
            "src/inferential/sens2026_shapley.py")

    reg = pd.DataFrame(rows)
    if reg["claim"].duplicated().any():
        raise ValueError(f"Duplicate claims: {sorted(reg.loc[reg['claim'].duplicated(), 'claim'])}")
    reg.to_csv(MV / "evidence_registry.csv", index=False)
    print(f"Wrote {len(reg)} v2 registry rows -> {MV / 'evidence_registry.csv'}")
    print(reg.to_string(index=False))


if __name__ == "__main__":
    main()
