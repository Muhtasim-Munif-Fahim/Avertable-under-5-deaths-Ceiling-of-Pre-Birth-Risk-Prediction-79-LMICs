"""
_fulldata_table4c_evalues.py — full-data table_4c (LPM vs logit) + E-values
============================================================================

s05d_sensitivity_logit_decomp and s05f OOM in statsmodels at full N. Both
quantities are computable memory-safe: the Wagstaff LPM/logit-AME contributions
via sklearn weighted logit + closed-form WLS, and E-values via s05f's own
compute_evalue() applied to the full-data Model-3 aORs (identical methodology).

Writes (5% saved as .bak-5pct):
  output/tables/table_4c_inequality_decomp_lpm_vs_logit.csv
  output/tables/table_s7_evalues.csv
"""
from __future__ import annotations
import sys, shutil
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.paths import TABLES_DIR, OUTPUT_DIR
from src.inferential import _fulldata as FD
from src.inferential.run_fulldata_all import build_dummies
from src.inferential.s05f_sensitivity_analyses import compute_evalue

FDIR = OUTPUT_DIR / "fulldata"
WAG = ["male", "is_multiple", "bi_short", "bi_long", "bi_firstmissing",
       "bo_2_3", "bo_4_5", "bo_6plus", "age_under20", "age_over35", "married",
       "edu_primary", "edu_secondary", "edu_higher", "is_urban",
       "decade_2010", "decade_2020"]
FACTOR = {"male": "Child sex", "is_multiple": "Birth type",
          "bi_short": "Birth interval", "bi_long": "Birth interval",
          "bi_firstmissing": "Birth interval", "bo_2_3": "Birth order",
          "bo_4_5": "Birth order", "bo_6plus": "Birth order",
          "age_under20": "Maternal age", "age_over35": "Maternal age",
          "married": "Marital status", "edu_primary": "Education",
          "edu_secondary": "Education", "edu_higher": "Education",
          "is_urban": "Residence", "decade_2010": "Decade trend",
          "decade_2020": "Decade trend"}
EV_PARAM = {"male": "sex_cat_Male", "is_multiple": "birth_type_Multiple",
            "bo_2_3": "bo_cat_2nd-3rd", "bo_4_5": "bo_cat_4th-5th", "bo_6plus": "bo_cat_6th+",
            "bi_short": "bi_cat_<24 mo", "bi_long": "bi_cat_48+ mo",
            "bi_firstmissing": "bi_cat_First/missing", "age_under20": "ma_cat_<20",
            "age_over35": "ma_cat_35+", "edu_primary": "edu_cat_Primary",
            "edu_secondary": "edu_cat_Secondary", "edu_higher": "edu_cat_Higher",
            "wq_2": "wq_cat_Poorer", "wq_3": "wq_cat_Middle", "wq_4": "wq_cat_Richer",
            "wq_5": "wq_cat_Richest", "married": "marital_cat_Married",
            "is_urban": "urban_cat_Urban", "decade_2010": "decade_2010-2019",
            "decade_2020": "decade_2020-2024"}


def backup(p):
    b = p.with_suffix(p.suffix + ".bak-5pct")
    if p.exists() and not b.exists():
        shutil.copy(p, b)


def table4c():
    df = FD.load_full(["died_under5", "wealth_quintile", "sample_weight",
                       "child_male", "multiple_birth", "birth_order",
                       "birth_interval_months", "maternal_age_at_birth",
                       "edu_level", "currently_married", "urban", "survey_year"],
                      scope="primary")
    d = build_dummies(df); n = len(df)
    y = df["died_under5"].astype(float).values
    w = df["sample_weight"].fillna(1.0).values
    rank = df["wealth_quintile"].fillna(3.0).values
    o = np.argsort(rank, kind="stable"); ws = w[o]; cum = np.cumsum(ws)
    fr = np.empty(n); fr[o] = (cum - 0.5 * ws) / ws.sum()
    muy = np.average(y, weights=w); mr = np.average(fr, weights=w)
    tot = (2 / muy) * np.average((y - muy) * (fr - mr), weights=w)

    def ci(x):
        mu = np.average(x, weights=w)
        return (2 / mu) * np.average((x - mu) * (fr - mr), weights=w)

    X = np.column_stack([np.ones(n)] + [d[c] for c in WAG])
    # logit-AME contributions (LPM comes from the canonical s05d aggregated table
    # so table_4c's LPM column matches table_4 exactly).
    beta_lo = FD.logit_coefs(X, y, weights=w)
    p = 1 / (1 + np.exp(-np.clip(X @ beta_lo, -30, 30)))
    smean = np.average(p * (1 - p), weights=w)
    log = {}
    for j, c in enumerate(WAG, 1):
        mu_x = np.average(d[c], weights=w); cix = ci(d[c])
        log[c] = (smean * beta_lo[j] * mu_x / muy) * cix

    # authoritative LPM percentages from the canonical full-data Wagstaff table
    agg = pd.read_csv(TABLES_DIR / "table_4_inequality_decomp_aggregated.csv")
    agg = agg[agg["Group"] == "All LMICs (2000-2024)"].set_index("Factor")["Percent_Contribution"]
    rows = []
    for fac in ["Education", "Residence", "Birth order", "Birth interval",
                "Maternal age", "Marital status", "Child sex", "Birth type",
                "Decade trend"]:
        keys = [c for c in WAG if FACTOR[c] == fac]
        pl = float(agg.get(fac, np.nan))
        pg = sum(log[c] for c in keys) / tot * 100
        rows.append({"Group": "All LMICs (2000-2024)", "Factor": fac,
                     "Percent_LPM": round(pl, 2), "Percent_Logit": round(pg, 2),
                     "Delta_pp": round(pg - pl, 2)})
    out = TABLES_DIR / "table_4c_inequality_decomp_lpm_vs_logit.csv"
    backup(out); pd.DataFrame(rows).to_csv(out, index=False)
    print(f"table_4c: written ({len(rows)} factors); education LPM "
          f"{rows[0]['Percent_LPM']} vs logit {rows[0]['Percent_Logit']}")


def evalues():
    full = pd.read_csv(FDIR / "table_2_sequential_models_fulldata.csv")
    m3 = full[full["Model"] == "Model3"]
    rows = []
    for _, r in m3.iterrows():
        nm = EV_PARAM.get(r["Variable"])
        if nm is None:
            continue
        # E-value for the CI must use the confidence limit NEAREST the null (OR=1):
        # CI_upper for protective effects (OR<1), CI_lower for harmful (OR>1).
        or_v = float(r["OR"])
        ci_near = float(r["CI_upper"]) if or_v < 1.0 else float(r["CI_lower"])
        ev = compute_evalue(or_v, ci_near)
        rows.append({"param": nm, "aOR": round(float(r["OR"]), 3),
                     "CI_lo": round(float(r["CI_lower"]), 3),
                     "CI_hi": round(float(r["CI_upper"]), 3),
                     "E_value_point": ev["E_value_point"],
                     "E_value_CI": ev["E_value_CI"]})
    out = TABLES_DIR / "table_s7_evalues.csv"
    backup(out); pd.DataFrame(rows).to_csv(out, index=False)
    print(f"table_s7_evalues: written ({len(rows)} params)")


if __name__ == "__main__":
    table4c()
    evalues()
