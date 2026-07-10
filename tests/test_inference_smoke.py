"""Smoke tests for the inferential pipeline: G-computation PAF, Wagstaff
decomposition, and bootstrap CI.

Verifies correct behavior of the G-computation PAF, concentration index,
and bootstrap CI pipeline without requiring hours of computation.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.paths import GLOBAL_PARQUET, RNG_SEED

import pyarrow.parquet as pq


@pytest.fixture(scope="module")
def sample_5pct():
    """5% systematic sample (every 20th row) matching D015/D016 sampling."""
    if not GLOBAL_PARQUET.exists():
        pytest.skip(f"Canonical parquet not found: {GLOBAL_PARQUET}")

    cols = [
        "died_under5", "country_code", "who_region",
        "sample_weight", "survey_year",
        "child_male", "multiple_birth", "birth_order",
        "birth_interval_months", "maternal_age_at_birth",
        "edu_level", "currently_married", "wealth_quintile",
        "urban", "in_primary_scope",
    ]

    pf = pq.ParquetFile(str(GLOBAL_PARQUET))
    n = pf.metadata.num_rows
    idx = set(range(0, n, 20))
    frames = []
    offset = 0
    for batch in pf.iter_batches(batch_size=500_000, columns=cols):
        bdf = batch.to_pandas()
        local = [
            i - offset for i in idx if offset <= i < offset + len(bdf)
        ]
        if local:
            frames.append(bdf.iloc[local])
        offset += len(bdf)

    df = pd.concat(frames, ignore_index=True)
    df = df[df["in_primary_scope"] == 1].drop(columns=["in_primary_scope"])
    df = df[df["died_under5"].notna()].copy()
    df["died_under5"] = df["died_under5"].astype(int)
    return df


def test_sample_loads(sample_5pct):
    """Test 1: 5% systematic sample loads with expected structure."""
    df = sample_5pct
    assert len(df) > 100_000, (
        f"Expected >100K rows, got {len(df):,}"
    )

    expected_cols = {
        "died_under5", "country_code", "who_region",
        "sample_weight", "survey_year", "child_male",
    }
    missing = expected_cols - set(df.columns)
    assert not missing, f"Missing expected columns: {missing}"

    vals = df["died_under5"].dropna().unique()
    assert set(vals).issubset({0, 1}), (
        f"Non-binary died_under5 values: {vals}"
    )

    assert df["country_code"].nunique() >= 50, (
        f"Only {df['country_code'].nunique()} unique countries, expected >=50"
    )


def test_sample_row_count_about_512k(sample_5pct):
    """Test 2: Primary-scope sample is ~512K rows (D015 target)."""
    n = len(sample_5pct)
    assert 450_000 <= n <= 600_000, (
        f"Row count {n:,} outside expected 450K-600K range"
    )


def test_paf_values_in_range(sample_5pct):
    """Test 3: G-computation PAFs in plausible range on a 2000-row subset."""
    from src.inferential.s05c_counterfactual_paf import (
        gcomp_paf, RISK_FACTORS, FORMULA, _encode_cats, fit_paf_glm,
    )

    subset = sample_5pct.sample(n=2000, random_state=RNG_SEED).copy()
    _encode_cats(subset)

    result = fit_paf_glm(FORMULA, subset)

    any_positive = False
    for col, label, cf_val in RISK_FACTORS:
        paf_pct = gcomp_paf(subset, col, cf_val, result, FORMULA) * 100
        assert -10 <= paf_pct <= 100, (
            f"PAF '{label}' = {paf_pct:.2f}% outside plausible [-10, 100]"
        )
        if paf_pct > 0:
            any_positive = True

    assert any_positive, (
        "No PAF > 0: expected positive signal for at least one risk factor"
    )


def test_concentration_index_negative(sample_5pct):
    """Test 4: Wealth CI for U5MR is negative (pro-poor)."""
    from src.inferential.s05d_inequality_decomposition import (
        calc_concentration_index,
    )

    sub = sample_5pct.dropna(subset=["wealth_quintile"]).copy()
    sub = sub.sort_values("wealth_quintile")

    w = sub["sample_weight"].fillna(1.0).values
    total_w = w.sum()
    frac_rank = (np.cumsum(w) - 0.5 * w) / total_w

    y = sub["died_under5"].values
    ci = calc_concentration_index(y, frac_rank, w)

    assert ci < 0, (
        f"CI = {ci:.4f}, expected < 0 (pro-poor U5MR concentration)"
    )


def test_bootstrap_ci_computes(sample_5pct):
    """Test 5: Minimal cluster bootstrap (10 reps) yields valid CIs."""
    from src.inferential.s05c_counterfactual_paf import (
        gcomp_paf, RISK_FACTORS, FORMULA, _encode_cats, fit_paf_glm,
    )

    subset = sample_5pct.sample(n=5000, random_state=RNG_SEED + 1).copy()
    _encode_cats(subset)

    result_full = fit_paf_glm(FORMULA, subset)
    point_estimates = {}
    for col, _, cf_val in RISK_FACTORS:
        point_estimates[col] = gcomp_paf(
            subset, col, cf_val, result_full, FORMULA,
        )

    countries = subset["country_code"].unique()
    n_countries = len(countries)
    boot_pafs = {col: [] for col, _, _ in RISK_FACTORS}

    np.random.seed(RNG_SEED)
    n_success = 0
    for _ in range(10):
        boot_cc = np.random.choice(
            countries, size=n_countries, replace=True,
        )
        df_boot = pd.concat(
            [subset[subset["country_code"] == c] for c in boot_cc],
            ignore_index=True,
        )
        try:
            r_boot = fit_paf_glm(FORMULA, df_boot)
            for col, _, cf_val in RISK_FACTORS:
                boot_pafs[col].append(
                    gcomp_paf(df_boot, col, cf_val, r_boot, FORMULA),
                )
            n_success += 1
        except Exception:
            pass

    assert n_success >= 8, (
        f"Only {n_success}/10 bootstrap iterations succeeded, need >=8"
    )

    ci_valid = False
    for col, _, _ in RISK_FACTORS:
        bvals = np.array(boot_pafs[col])
        if len(bvals) < 5:
            continue
        lo = np.percentile(bvals, 2.5)
        hi = np.percentile(bvals, 97.5)
        pt = point_estimates[col]
        if lo < pt < hi:
            ci_valid = True
            break

    assert ci_valid, (
        "No risk factor had valid CI (lo < point_estimate < hi)"
    )



def test_paf_glm_is_unweighted(sample_5pct):
    """Verify fit_paf_glm fits the GLM unweighted.

    The PAF estimand is the pooled, population-averaged association across
    the cohort, fitted on unweighted birth-history records (manuscript
    Methods; decision D015). gcomp_paf likewise averages predictive margins
    unweighted, so model and margins target the same population. This
    configuration reproduces the published Table 3 PAFs (e.g. education
    38.3, birth spacing 22.4). The test confirms fit_paf_glm matches an
    explicit unweighted GLM fit and is NOT survey-weighted.
    """
    from src.inferential.s05c_counterfactual_paf import (
        fit_paf_glm, _encode_cats,
    )
    import numpy as np
    import statsmodels.api as sm
    from statsmodels.genmod.families import Binomial

    FORMULA = (
        "died_under5 ~ sex_cat + birth_type + bo_cat + bi_cat "
        "+ ma_cat + edu_cat + wq_cat + marital_cat + urban_cat "
        "+ decade"
    )

    df_sub = sample_5pct.sample(n=500, random_state=20260505).copy()
    _encode_cats(df_sub)

    result = fit_paf_glm(FORMULA, df_sub)
    result_unweighted = sm.GLM.from_formula(
        FORMULA, data=df_sub, family=Binomial(),
    ).fit()

    for name in result.params.index:
        diff = abs(result.params[name] - result_unweighted.params[name])
        assert diff < 1e-8, (
            f"fit_paf_glm is not unweighted; coefficient {name} "
            f"diverges by {diff:.6f}"
        )