"""s05b_table2.py — Table 2 adjusted odds ratios with country-clustered CIs.

The v2 §6 primary model is a single survey-weighted logistic wave-FE surface
(exact_wave_fe_complete_wealth), not the legacy sequential GEE. This fits it once
(the same fit behind the published PAFs) and computes country-cluster-robust
(sandwich) 95% CIs for the substantive covariate aORs — one fit, not a 1000x
bootstrap. Uncertainty reflects country-level clustering (§6).

Sandwich: bread = (X' W p(1-p) X)^-1 ; meat = sum_c s_c s_c',
s_c = sum_{i in c} w_i (y_i - p_i) x_i ; V = bread @ meat @ bread.

Output: output/phase4/manuscript/table_2_primary_aor.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from config.paths import HARMONIZED_DIR, OUTPUT_DIR
from src.inferential.phase2_analysis import _fit_surface, _read_pilot

OUT = OUTPUT_DIR / "phase4" / "manuscript"


def _is_nuisance(name: str) -> bool:
    return name == "(intercept)" or name.startswith("survey_id") or name.startswith("birth_period")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = _read_pilot(HARMONIZED_DIR / "global_analysis_v2_candidate.parquet")
    result = _fit_surface(
        surface="exact_wave_fe_complete_wealth", role="primary_observed_wealth",
        frame=frame, outcome_column="died_under5_complete", legacy_states=False,
        fixed_effects="survey", wealth_mode="complete", weighted=True)

    coef = np.r_[float(result.model.intercept_[0]), result.model.coef_[0]]
    names = ["(intercept)"] + [str(n) for n in result.feature_names]
    n = result.matrix.shape[0]
    Xa = sparse.hstack([sparse.csr_matrix(np.ones((n, 1))), result.matrix]).tocsr()
    w = result.weights.astype(float)
    y = result.outcome.astype(float)
    p = result.factual_predictions.astype(float)

    # bread = inv(X' diag(w p(1-p)) X)
    V = w * p * (1.0 - p)
    XVX = (Xa.T @ sparse.diags(V) @ Xa).toarray()
    bread = np.linalg.inv(XVX)

    # meat = sum_c s_c s_c' with s_c = X_c' (w (y-p))_c
    g = w * (y - p)
    country = result.frame["country_code"].astype(str).to_numpy()
    codes, inv = np.unique(country, return_inverse=True)
    k = Xa.shape[1]
    S = np.zeros((len(codes), k))
    for ci in range(len(codes)):
        rows = np.flatnonzero(inv == ci)
        S[ci] = np.asarray(Xa[rows].T @ g[rows]).ravel()
    meat = S.T @ S
    cov = bread @ meat @ bread
    se = np.sqrt(np.clip(np.diag(cov), 0, None))

    rows = []
    for j, nm in enumerate(names):
        if _is_nuisance(nm):
            continue
        b = coef[j]
        rows.append({
            "feature": nm,
            "aOR": round(float(np.exp(b)), 3),
            "ci_lo": round(float(np.exp(b - 1.96 * se[j])), 3),
            "ci_hi": round(float(np.exp(b + 1.96 * se[j])), 3),
            "coefficient": round(float(b), 4),
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "table_2_primary_aor.csv", index=False)
    print(f"Country-clustered sandwich CIs ({len(codes)} country clusters, n={n:,}):")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
