"""
run_fulldata_percountry_paf.py — per-country PAFs on FULL data (eTable S17 / F3)
================================================================================

Full-data replacement for s05c3_per_country_paf.py (which fits statsmodels GLM
on the 5% sample). For each country, fits a Model-3 logistic surface on ALL of
that country's births and g-computes the two dominant whole-population PAFs:
education (shift below-secondary to secondary+) and birth spacing (shift all
non-24--47-month intervals to the reference window) — same estimand as s05c.

Writes the canonical schema figure_paf_country.py / journal_tables.py expect:
  country_code, country_name, who_region, n_births,
  paf_education, paf_birth_spacing, recorded_u5m_per1000

Output: output/tables/table_s17_per_country_paf.csv
"""
from __future__ import annotations
import os, sys
from pathlib import Path
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.paths import TABLES_DIR
from src.inferential import _fulldata as FD
from src.inferential.run_fulldata_all import build_dummies, design_from, M3, _paf_one

OUT = TABLES_DIR / "table_s17_per_country_paf.csv"
EDU_SHIFT = {"edu_primary": 0, "edu_secondary": 1, "edu_higher": 0}
BI_SHIFT = {"bi_firstmissing": 0, "bi_short": 0, "bi_long": 0}


def main():
    df = FD.load_full(
        ["country_code", "who_region", "survey_year", "died_under5",
         "child_male", "multiple_birth", "birth_order", "birth_interval_months",
         "maternal_age_at_birth", "edu_level", "wealth_quintile",
         "currently_married", "urban"], scope="primary")
    try:
        inv = pd.read_csv(TABLES_DIR / "table_s1_survey_inventory.csv")
        names = inv.groupby("country_code")["country_name"].first()
    except Exception:
        names = pd.Series(dtype=object)

    rows = []
    for cc, sub in df.groupby("country_code", sort=True):
        sub = sub.reset_index(drop=True)
        n = len(sub)
        y = sub["died_under5"].astype(np.float64).values
        d = build_dummies(sub)
        keep = [k for k in M3 if d[k].std() > 0]
        X = design_from(d, keep, n)
        colidx = {nm: i + 1 for i, nm in enumerate(keep)}
        try:
            beta = FD.logit_coefs(X, y)
            paf_edu = _paf_one(X, beta, colidx,
                               {k: v for k, v in EDU_SHIFT.items() if k in colidx})
            paf_bi = _paf_one(X, beta, colidx,
                              {k: v for k, v in BI_SHIFT.items() if k in colidx})
        except Exception as e:
            paf_edu = paf_bi = np.nan
            print(f"  {cc}: fit failed ({e})")
        rows.append({
            "country_code": cc,
            "country_name": names.get(cc, cc),
            "who_region": sub["who_region"].iloc[0],
            "n_births": int(n),
            "paf_education": round(float(paf_edu) * 100, 1) if np.isfinite(paf_edu) else np.nan,
            "paf_birth_spacing": round(float(paf_bi) * 100, 1) if np.isfinite(paf_bi) else np.nan,
            "recorded_u5m_per1000": round(float(y.mean()) * 1000, 1),
        })
    out = pd.DataFrame(rows).sort_values(["who_region", "country_code"])
    out.to_csv(OUT, index=False)
    print(f"Wrote {OUT.name} ({len(out)} countries)")
    print(out[["country_code", "n_births", "paf_education",
               "paf_birth_spacing"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
