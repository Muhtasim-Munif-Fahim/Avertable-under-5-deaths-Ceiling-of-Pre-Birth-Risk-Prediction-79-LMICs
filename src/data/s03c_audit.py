"""
s03c_audit.py — Post-Harmonization Data Audit
===============================================

Runs invariant checks on global_analysis.parquet. Each check reads only
the columns it needs (never the full dataset) to stay within memory limits.

Writes a per-check report to output/tables/table_s0b_audit_report.csv
and exits non-zero on any FAIL.

Design Decisions Verified
--------------------------
  D006: in_primary_scope ⇔ survey_year in [2000, 2024]
  D007/D008: re-release / sub-national exclusions complete
  D011: edu_level in {0, 1, 2, 3}
  D012: birth_interval_months >= 0
"""
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent.parent)
)
from config.paths import (
    GLOBAL_PARQUET, CATALOG_PARQUET, TABLES_DIR,
)
from src.utils.print_util import print


# India sub-national DHS 2-letter codes (must NOT appear in country_code)
INDIA_STATE_CODES = {
    "AP", "AR", "AS", "BH", "DL", "GJ", "GO", "HP", "HR",
    "JM", "KA", "MG", "MH", "MN", "MP", "OR", "PJ", "RJ",
    "SK", "TN", "TR", "UP", "WB", "NA", "MZ", "KE",
}


def _read(cols: list[str]) -> pd.DataFrame:
    """Read only the specified columns from the canonical parquet."""
    return pq.read_table(str(GLOBAL_PARQUET), columns=cols).to_pandas()


def run_audit() -> pd.DataFrame:
    """
    Run all audit checks. Returns a DataFrame of results.
    Each row: check_name, scope, n_violations, status, detail.
    """
    results: list[dict] = []

    def record(name: str, scope: str, n_viol: int, detail: str = ""):
        status = "PASS" if n_viol == 0 else "FAIL"
        results.append({
            "check_name": name,
            "scope": scope,
            "n_violations": n_viol,
            "status": status,
            "detail": detail,
        })
        icon = "OK" if n_viol == 0 else "FAIL"
        print(f"  [{icon}] {name}: {n_viol} violations {detail}")

    def record_warn(name: str, scope: str, n_viol: int, detail: str = ""):
        status = "PASS" if n_viol == 0 else "WARN"
        results.append({
            "check_name": name,
            "scope": scope,
            "n_violations": n_viol,
            "status": status,
            "detail": detail,
        })
        icon = "OK" if n_viol == 0 else "WARN"
        print(f"  [{icon}] {name}: {n_viol} violations {detail}")

    # Metadata (free — no data read)
    meta = pq.read_metadata(str(GLOBAL_PARQUET))
    n_rows = meta.num_rows
    print(f"  File: {GLOBAL_PARQUET}")
    print(f"  Rows: {n_rows:,}   Row groups: {meta.num_row_groups}")

    # ---------------------------------------------------------------
    # Check 1: edu_level in {0, 1, 2, 3} (D011)
    # ---------------------------------------------------------------
    print("  [1/12] edu_level_bounded ...")
    edu = _read(["edu_level"])["edu_level"].dropna()
    bad = (~edu.isin({0, 1, 2, 3})).sum()
    record("edu_level_bounded", "all", int(bad),
           f"(expected {{0,1,2,3}}, found unique={sorted(edu.unique())})")
    del edu

    # ---------------------------------------------------------------
    # Check 2: birth_interval_months >= 0 (D012)
    # ---------------------------------------------------------------
    print("  [2/12] birth_interval_nonneg ...")
    bim = _read(["birth_interval_months"])["birth_interval_months"].dropna()
    neg = int((bim < 0).sum())
    record("birth_interval_nonneg", "all", neg,
           f"(min={bim.min():.0f})" if neg > 0 else "")
    del bim

    # ---------------------------------------------------------------
    # Check 3: wealth_quintile in {1, 2, 3, 4, 5}
    # ---------------------------------------------------------------
    print("  [3/12] wealth_quintile_bounded ...")
    wq = _read(["wealth_quintile"])["wealth_quintile"].dropna()
    bad_wq = int((~wq.isin({1.0, 2.0, 3.0, 4.0, 5.0})).sum())
    record("wealth_quintile_bounded", "all", bad_wq,
           f"(unique={sorted(wq.unique()[:20])})" if bad_wq > 0 else "")
    del wq

    # ---------------------------------------------------------------
    # Check 4: No India sub-national codes
    # ---------------------------------------------------------------
    print("  [4/12] no_india_subnational ...")
    cc_ser = _read(["country_code"])["country_code"]
    cc_set = set(cc_ser.dropna().unique())
    india_found = cc_set & INDIA_STATE_CODES
    record("no_india_subnational", "all", len(india_found),
           f"(found: {sorted(india_found)})" if india_found else "")
    del cc_ser, cc_set

    # ---------------------------------------------------------------
    # Check 5: No NaN in key identifiers
    # ---------------------------------------------------------------
    print("  [5/12] no_nan_identifiers ...")
    id_df = _read(["country_code", "survey_year", "who_region"])
    for col in ["country_code", "survey_year", "who_region"]:
        n_na = int(id_df[col].isna().sum())
        record(f"no_nan_{col}", "all", n_na)
    del id_df

    # ---------------------------------------------------------------
    # Check 6: years_before_survey == survey_year - birth_year
    # ---------------------------------------------------------------
    print("  [6/12] ybs_equals_sy_minus_by ...")
    ybs_df = _read(["birth_year", "years_before_survey", "survey_year"])
    mask6 = ybs_df["birth_year"].notna() & ybs_df["years_before_survey"].notna()
    sub6 = ybs_df[mask6]
    expected6 = sub6["survey_year"] - sub6["birth_year"]
    diff = (sub6["years_before_survey"] - expected6).abs()
    bad6 = int((diff > 0.5).sum())
    record("ybs_equals_sy_minus_by", "all", bad6,
           f"(max_diff={diff.max():.2f})" if bad6 > 0 else "")
    del ybs_df, sub6, expected6, diff

    # ---------------------------------------------------------------
    # Check 7: birth_year in [1950, survey_year + 1]
    # WARN (not FAIL): NP-1995 has ~10 k rows with impossible b2 values
    # in the raw DHS file (Phase I calendar coding artefact, one survey).
    # ---------------------------------------------------------------
    print("  [7/12] birth_year_plausible ...")
    by_df = _read(["birth_year", "survey_year", "country_code"])
    sub7 = by_df[by_df["birth_year"].notna()]
    out_lo = int((sub7["birth_year"] < 1950).sum())
    hi_mask = sub7["birth_year"] > sub7["survey_year"] + 1
    out_hi = int(hi_mask.sum())
    bad7 = out_lo + out_hi
    surveys_hi = (
        sub7[hi_mask]
        .groupby(["country_code", "survey_year"])
        .size()
        .sort_values(ascending=False)
        .head(5)
    )
    detail7 = f"(low={out_lo}, high={out_hi})"
    if out_hi:
        detail7 += " surveys: " + "; ".join(
            f"{cc}-{int(yr)}:{n}" for (cc, yr), n in surveys_hi.items()
        )
    record_warn("birth_year_plausible", "all", bad7, detail7)
    del by_df, sub7

    # ---------------------------------------------------------------
    # Check 8: U5MR per (country, survey_year) in [5, 500]/1000 (WARN)
    # ---------------------------------------------------------------
    print("  [8/12] u5mr_plausible_per_survey ...")
    u5_df = _read(["country_code", "survey_year", "died_under5"])
    grp = u5_df.groupby(["country_code", "survey_year"]).agg(
        n=("died_under5", "count"),
        deaths=("died_under5", "sum"),
    )
    grp["u5mr"] = grp["deaths"] / grp["n"] * 1000
    outliers = grp[(grp["u5mr"] < 5) | (grp["u5mr"] > 500)]
    detail8 = ""
    if len(outliers):
        detail8 = "; ".join(
            f"{cc}-{yr}: {r['u5mr']:.0f}/1000 (n={r['n']})"
            for (cc, yr), r in outliers.head(5).iterrows()
        )
        if len(outliers) > 5:
            detail8 += f" ... +{len(outliers)-5} more"
    record_warn("u5mr_plausible_per_survey", "all", len(outliers),
                f"({detail8})" if detail8 else "")
    del u5_df, grp, outliers

    # ---------------------------------------------------------------
    # Check 9: sample_weight mean per survey ≈ 1 (WARN)
    # ---------------------------------------------------------------
    print("  [9/12] sample_weight_mean_near_1 ...")
    sw_df = _read(["country_code", "survey_year", "sample_weight"])
    grp_w = sw_df.groupby(["country_code", "survey_year"])["sample_weight"].mean()
    bad_w = int(((grp_w < 0.5) | (grp_w > 1.5)).sum())
    record_warn("sample_weight_mean_near_1", "all", bad_w,
                f"(out of {len(grp_w)} survey-year groups)")
    del sw_df, grp_w

    # ---------------------------------------------------------------
    # Check 10: in_primary_scope ⇔ 2000 ≤ survey_year ≤ 2024 (D006)
    # ---------------------------------------------------------------
    print("  [10/12] d006_scope_invariant ...")
    sc_df = _read(["survey_year", "in_primary_scope"])
    expected_scope = (
        (sc_df["survey_year"] >= 2000) & (sc_df["survey_year"] <= 2024)
    ).astype(int)
    bad10 = int((sc_df["in_primary_scope"] != expected_scope).sum())
    record("d006_scope_invariant", "all", bad10)
    del sc_df, expected_scope

    # ---------------------------------------------------------------
    # Check 11: D007/D008 re-release exclusion completeness
    #   A re-release is when two files share (country, phase, survey_year).
    #   Different survey_years in the same phase are LEGITIMATE separate
    #   surveys (e.g. BD 1994+1996, Peru annual ENDES) and are not flagged.
    # ---------------------------------------------------------------
    print("  [11/12] d007_rerelease_complete ...")
    from src.data.s03_harmonize import SKIP_SURVEYS
    if CATALOG_PARQUET.exists():
        cat = pd.read_parquet(CATALOG_PARQUET)
        cat = cat[cat["file_type"] == "BR"]
        unaccounted = []
        for (cc, phase), grp in cat.groupby(["country_code", "dhs_phase"]):
            # Cluster by exact survey_year match — same year = re-release
            for yr, yr_grp in grp.groupby("survey_year"):
                stems = set(yr_grp["folder_name"])
                kept = stems - SKIP_SURVEYS
                if len(kept) > 1:
                    unaccounted.append(
                        f"{cc}-phase{phase}-{int(yr)}: {sorted(kept)}"
                    )
        record("d007_rerelease_complete", "catalog",
               len(unaccounted),
               f"({'; '.join(unaccounted)})" if unaccounted else "")
    else:
        record_warn("d007_rerelease_complete", "catalog", 0,
                    "(catalog not found, check skipped)")

    # ---------------------------------------------------------------
    # Check 12: Source file count
    # ---------------------------------------------------------------
    print("  [12/12] source_file_count ...")
    sf_df = _read(["_source_file"])
    n_sources = sf_df["_source_file"].nunique()
    record_warn("source_file_count", "all", 0,
                f"({n_sources} unique source files)")
    del sf_df

    return pd.DataFrame(results)


def main() -> None:
    """Run audit and write report."""
    print("=" * 60)
    print("Post-Harmonization Data Audit")
    print("=" * 60)

    report = run_audit()

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TABLES_DIR / "table_s0b_audit_report.csv"
    report.to_csv(out_path, index=False)
    print(f"\nReport: {out_path}")

    n_fail = (report["status"] == "FAIL").sum()
    n_warn = (report["status"] == "WARN").sum()
    n_pass = (report["status"] == "PASS").sum()
    print(f"Summary: {n_pass} PASS, {n_warn} WARN, {n_fail} FAIL")

    if n_fail > 0:
        print("\nAUDIT FAILED — fix issues before proceeding.")
        sys.exit(1)
    else:
        print("\nAudit passed.")


if __name__ == "__main__":
    main()
