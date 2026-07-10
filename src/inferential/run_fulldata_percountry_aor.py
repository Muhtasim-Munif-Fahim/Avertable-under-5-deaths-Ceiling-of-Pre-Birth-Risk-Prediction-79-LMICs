"""
run_fulldata_percountry_aor.py — per-country adjusted ORs on FULL data (eTable, P1/C1.6)
========================================================================================

LGH-depth country-resolution layer (benchmarking R1): every headline is backed by a
per-country table. This fits the Model-3 logistic surface on ALL of each country's births
(same full-data engine + design as run_fulldata_all / run_fulldata_percountry_paf — NOT the
5% s05c3 path) and reports the adjusted odds ratios (with model-based 95% CIs) for the
paper's lead social/demographic determinants:

  Education: Higher  ·  Education: Secondary  ·  Birth interval <24 mo  ·  Birth order 6+

Within a single country there is no country-cluster structure, so CIs use the model-based
logistic information matrix (inverse Fisher information), not the pooled cluster-robust
sandwich. Point estimates reproduce the per-country Model-3 fit.

Output: output/tables/table_s_percountry_aor.csv
Schema: country_code, country_name, who_region, n_births, n_deaths,
        aor_edu_higher, aor_edu_secondary, aor_bi_short, aor_bo_6plus
        (each formatted "x.xx (lo--hi)", or "--" when not estimable in that country)
"""
from __future__ import annotations
import os, sys
from pathlib import Path
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.paths import TABLES_DIR
from src.inferential import _fulldata as FD
from src.inferential.run_fulldata_all import build_dummies, design_from, M3

OUT = TABLES_DIR / "table_s_percountry_aor.csv"

# Lead determinants to surface (dummy name -> output column). Portrait-friendly subset
# of M3 matching the determinant narrative (education gradient + spacing + parity).
TARGETS = [
    ("edu_higher",    "aor_edu_higher"),
    ("edu_secondary", "aor_edu_secondary"),
    ("bi_short",      "aor_bi_short"),
    ("bo_6plus",      "aor_bo_6plus"),
]


def _fmt_or(beta_j: float, se_j: float) -> str:
    """exp(beta) (95% CI) as 'x.xx (lo--hi)', guarding unstable small-country fits."""
    if not (np.isfinite(beta_j) and np.isfinite(se_j)) or se_j <= 0 or se_j > 3.0:
        return "--"
    orr = float(np.exp(beta_j))
    lo = float(np.exp(beta_j - 1.96 * se_j))
    hi = float(np.exp(beta_j + 1.96 * se_j))
    # Implausible range => unstable separation on a small subsample; suppress.
    if not (0.02 <= orr <= 30.0) or hi / max(lo, 1e-9) > 200:
        return "--"
    return f"{orr:.2f} ({lo:.2f}--{hi:.2f})"


def _model_se(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Model-based logistic SE = sqrt(diag((X' W X)^-1)), W = p(1-p)."""
    eta = np.clip(X @ beta, -30, 30)
    p = 1.0 / (1.0 + np.exp(-eta))
    w = np.clip(p * (1.0 - p), 1e-9, None)
    XtWX = X.T @ (w[:, None] * X)
    try:
        cov = np.linalg.inv(XtWX)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(XtWX)
    return np.sqrt(np.clip(np.diag(cov), 0, None))


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
        colidx = {nm: i + 1 for i, nm in enumerate(keep)}  # col 0 = intercept
        row = {
            "country_code": cc,
            "country_name": names.get(cc, cc),
            "who_region": sub["who_region"].iloc[0],
            "n_births": int(n),
            "n_deaths": int(y.sum()),
        }
        try:
            beta = FD.irls_logit(X, y)
            se = _model_se(X, beta)
            for dummy, outcol in TARGETS:
                if dummy in colidx:
                    j = colidx[dummy]
                    row[outcol] = _fmt_or(beta[j], se[j])
                else:
                    row[outcol] = "--"
        except Exception as e:
            print(f"  {cc}: fit failed ({e})")
            for _, outcol in TARGETS:
                row[outcol] = "--"
        rows.append(row)
        print(f"  {cc} n={n:>7,} d={int(y.sum()):>5}  "
              f"eduH={row['aor_edu_higher']:>16}  bi<24={row['aor_bi_short']:>16}")

    out = pd.DataFrame(rows).sort_values(
        ["who_region", "n_births"], ascending=[True, False]).reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"\nWrote {OUT} ({len(out)} countries)")


if __name__ == "__main__":
    main()
