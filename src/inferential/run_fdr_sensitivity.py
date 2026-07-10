"""
run_fdr_sensitivity.py — Benjamini-Hochberg false-discovery-rate sensitivity
============================================================================

Multiple-comparisons sensitivity for the inferential associations. The primary
analysis reports nominal 95% CIs and unadjusted p-values; reviewers asked whether
the associations survive a multiplicity correction. This script applies the
Benjamini-Hochberg (1995) FDR procedure to three pre-defined test families and
reports how many nominally significant associations remain significant at
q < 0.05.

Because the published tables carry adjusted odds ratios and 95% CIs (not raw
p-values to full precision), two-sided p-values are recovered from each OR and
its 95% CI on the log scale by the standard normal approximation
(Altman & Bland, BMJ 2011: SE = (ln(upper) - ln(lower)) / (2 x 1.96);
z = ln(OR) / SE). This is exact for Wald intervals and a close approximation
for the bootstrap/sandwich intervals used here; the only borderline case is a
single regional test at p ~ 0.05, which the qualitative conclusion ("FDR control
leaves the picture essentially unchanged") does not hinge on.

Families:
  1. Pooled Model 3   - the 21 non-reference covariate levels (Table 2).
  2. Regional GEE     - region x covariate tests (eTable S8 regional GEE).
  3. Per-country aOR   - estimable country x lead-determinant tests (eTable S41).

Output : output/tables/table_s_fdr_sensitivity.csv
Run    : python -m src.inferential.run_fdr_sensitivity
"""
from __future__ import annotations
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.paths import TABLES_DIR

OUT = TABLES_DIR / "table_s_fdr_sensitivity.csv"
Z95 = 1.959963985


def _two_sided_p(z: float) -> float:
    """Two-sided p-value from a z-statistic (no scipy dependency)."""
    return math.erfc(abs(z) / math.sqrt(2.0))


def _p_from_or_ci(or_, lo, hi) -> np.ndarray:
    or_ = np.asarray(or_, float); lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    b = np.log(or_)
    se = (np.log(hi) - np.log(lo)) / (2.0 * Z95)
    z = b / se
    return np.array([_two_sided_p(zi) for zi in z])


def bh_qvalues(pvals) -> np.ndarray:
    """Benjamini-Hochberg adjusted q-values (monotone-enforced)."""
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0.0, 1.0)
    return out


def _summarise(name: str, p: np.ndarray) -> dict:
    q = bh_qvalues(p)
    raw_sig = int((p < 0.05).sum())
    bh_sig = int((q < 0.05).sum())
    lost = int(((p < 0.05) & (q >= 0.05)).sum())
    return {
        "family": name,
        "n_tests": len(p),
        "raw_sig_p_lt_0.05": raw_sig,
        "bh_sig_q_lt_0.05": bh_sig,
        "lost_to_fdr": lost,
        "max_raw_p": round(float(p.max()), 4),
    }


def _parse_or_cell(cell):
    """Parse an 'OR (lo--hi)' display cell (per-country aOR table)."""
    if not isinstance(cell, str) or cell.strip() in {"--", "", "nan"}:
        return None
    m = re.match(r"([\d.]+)\s*\(([\d.]+)--([\d.]+)\)", cell.replace("$", ""))
    if not m:
        return None
    o, lo, hi = (float(x) for x in m.groups())
    if min(o, lo, hi) <= 0:
        return None
    return float(_p_from_or_ci([o], [lo], [hi])[0])


def main() -> None:
    rows = []

    # 1. Pooled Model 3 (21 covariate levels)
    t2 = pd.read_csv(TABLES_DIR / "table_2_sequential_models.csv")
    m3 = t2[(t2["OR_Model3"] != 1.0) & t2["CIlo_Model3"].notna()
            & (t2["CIlo_Model3"] != 1.0) & (t2["CIlo_Model3"] > 0)]
    rows.append(_summarise("Pooled Model 3 (21 covariates)",
                           _p_from_or_ci(m3["OR_Model3"], m3["CIlo_Model3"], m3["CIhi_Model3"])))

    # 2. Regional GEE family
    reg = pd.read_csv(TABLES_DIR / "table_s8_regional_gee.csv")
    reg = reg[reg["OR"].notna() & reg["CI_lower"].notna()
              & (reg["CI_lower"] > 0) & (reg["OR"] != 1.0)]
    rows.append(_summarise("Regional GEE (region x covariate)",
                           _p_from_or_ci(reg["OR"], reg["CI_lower"], reg["CI_upper"])))

    # 3. Per-country aOR family
    pc = pd.read_csv(TABLES_DIR / "table_s_percountry_aor.csv")
    cols = ["aor_edu_higher", "aor_edu_secondary", "aor_bi_short", "aor_bo_6plus"]
    pcp = [r for c in cols for r in (_parse_or_cell(v) for v in pc[c]) if r is not None]
    rows.append(_summarise("Per-country aOR (country x determinant)", np.array(pcp)))

    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    pd.set_option("display.width", 160)
    print(out.to_string(index=False))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
