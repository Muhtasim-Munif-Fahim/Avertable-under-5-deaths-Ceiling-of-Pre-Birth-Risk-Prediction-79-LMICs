"""
run_fulldata_recentdecade_paf.py — Recent-EXPOSURE-PREVALENCE PAF sensitivity
=============================================================================
Reviewer (epidemiology, major revision) asked whether the ~15-year pooled
education and birth-spacing PAFs (37.5% / 20.8%) validly transport onto current
death totals, given that exposure prevalence (e.g. secondary schooling) has
risen across the observation window.

A literal "2020-2024 births only" PAF is NOT valid for under-5 mortality: every
2020-2024 birth has <5 years of follow-up (mean 1.40 yr before survey), so
`died_under5` is right-censored to infancy and the raw rate (43/1000) is an
artefact, not a real decline; only 30 of 79 countries even contribute such
births. See the diagnostic in the 2026-07-02 session log.

Instead this isolates the reviewer's actual concern — the EXPOSURE-PREVALENCE
channel — in a censoring-immune way:

  * associations (the Model-3 logit surface) are estimated on the FULL cohort
    (valid, fully-observed outcomes; pooled population-averaged estimand, s05c);
  * the g-computation PAF is then re-standardised to the exposure distribution
    of the MOST RECENT birth cohort (2018-2024 births). Education and birth
    interval are measured at/before birth, so they are complete for these births
    regardless of mortality censoring.

The PAF therefore changes ONLY through the exposure-prevalence channel, directly
answering "given today's exposure mix, what fraction is still attributable?".
Reported alongside the pooled value (which this script reproduces as an internal
check). Uncertainty via the same country-cluster bootstrap as
run_fulldata_all.table3 (associations refit + recent prevalence recomputed on
each resample, so both channels are propagated).

Outputs (output/fulldata/):
  table_recentdecade_paf_fulldata.csv

Run (SOLO — full-data job, never concurrent with another; background):
  python -m src.inferential.run_fulldata_recentdecade_paf
"""
from __future__ import annotations
import os, sys, gc, time
from datetime import datetime
from pathlib import Path
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
try:  # Windows console/file default cp1252 cannot encode Δ / en-dash
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from config.paths import RNG_SEED, TABLES_DIR
from src.inferential import _fulldata as FD
from src.inferential.run_fulldata_all import (
    build_dummies, design_from, _paf_one, M3,
)

OUTDIR = TABLES_DIR.parent / "fulldata"
OUTDIR.mkdir(parents=True, exist_ok=True)
LOG = OUTDIR / "run_recentdecade.log"

N_BOOT_PAF = 500          # match pooled Table 3 bootstrap methodology
RECENT_LO, RECENT_HI = 2018, 2024      # most recent birth cohort (exposure only)

# Only the two headline levers the reviewer asked about.
PAF_FACTORS = [
    ("Education < Secondary -> Secondary+",
     {"edu_primary": 0, "edu_secondary": 1, "edu_higher": 0}),
    ("Birth interval != 24-47mo -> 24-47mo",
     {"bi_firstmissing": 0, "bi_short": 0, "bi_long": 0}),
]
POOLED_CANONICAL = {
    "Education < Secondary -> Secondary+": "37.5 (32.6-43.1)",
    "Birth interval != 24-47mo -> 24-47mo": "20.8 (17.4-23.7)",
}


def log(msg: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _below_secondary(d) -> np.ndarray:
    """1 if mother's education is below secondary (none or primary)."""
    return 1.0 - (d["edu_secondary"] + d["edu_higher"])


def main() -> None:
    log("=" * 70)
    log(f"RECENT-PREVALENCE PAF START (recent cohort = {RECENT_LO}-{RECENT_HI} births)")
    cols = ["country_code", "who_region", "survey_year", "birth_year",
            "died_under5", "child_male", "multiple_birth", "birth_order",
            "birth_interval_months", "maternal_age_at_birth", "edu_level",
            "wealth_quintile", "currently_married", "urban", "sample_weight"]
    df = FD.load_full(cols, scope="primary")
    df = df.sort_values("country_code").reset_index(drop=True)
    n = len(df)
    log(f"Full cohort: {n:,} rows, {df['country_code'].nunique()} countries")

    dummies = build_dummies(df)                       # decade dummies kept (full cohort)
    keep = [k for k in M3 if dummies[k].std() > 0]
    y = df["died_under5"].astype(np.float64).values
    X = design_from(dummies, keep, n)                 # ~n x (len(keep)+1) float64
    colidx = {nm: i + 1 for i, nm in enumerate(keep)}
    beta = FD.logit_coefs(X, y)                        # pooled associations
    log(f"Fitted pooled Model-3 surface on full cohort ({len(keep)} covariates)")

    by = df["birth_year"].values
    recent_mask = (by >= RECENT_LO) & (by <= RECENT_HI)
    n_rec = int(recent_mask.sum())
    log(f"Recent cohort ({RECENT_LO}-{RECENT_HI}): {n_rec:,} births, "
        f"{df.loc[recent_mask, 'country_code'].nunique()} countries")

    # Exposure prevalence shift (unweighted, matches estimand)
    bsec = _below_secondary(dummies)
    prev = {
        "pct_below_secondary_pooled": round(100 * float(bsec.mean()), 1),
        "pct_below_secondary_recent": round(100 * float(bsec[recent_mask].mean()), 1),
        "pct_short_interval_pooled": round(100 * float(dummies["bi_short"].mean()), 1),
        "pct_short_interval_recent": round(100 * float(dummies["bi_short"][recent_mask].mean()), 1),
    }
    log(f"  %<secondary edu: pooled {prev['pct_below_secondary_pooled']} -> "
        f"recent {prev['pct_below_secondary_recent']}")
    log(f"  %short interval(<24mo): pooled {prev['pct_short_interval_pooled']} -> "
        f"recent {prev['pct_short_interval_recent']}")

    # Point estimates: pooled (all rows) vs recent-prevalence (recent rows), same beta
    Xr = X[recent_mask]
    rows = []
    for label, shift in PAF_FACTORS:
        sh = {k: v for k, v in shift.items() if k in colidx}
        paf_pooled = _paf_one(X, beta, colidx, sh) * 100          # reproduces canonical
        paf_recent = _paf_one(Xr, beta, colidx, sh) * 100
        rec = {"risk_factor": label,
               "PAF_pooled_recomputed": round(paf_pooled, 2),
               "PAF_pooled_canonical": POOLED_CANONICAL.get(label, ""),
               "PAF_recent_prevalence": round(paf_recent, 2),
               "delta_recent_minus_pooled": round(paf_recent - paf_pooled, 2),
               "recent_cohort": f"{RECENT_LO}-{RECENT_HI} births", "n_recent": n_rec}
        rec.update(prev)
        rows.append(rec)
        log(f"  {label}: pooled(recomputed)={paf_pooled:.2f}% "
            f"recent-prevalence={paf_recent:.2f}% (Δ={paf_recent-paf_pooled:+.2f})")
    pd.DataFrame(rows).to_csv(
        OUTDIR / "table_recentdecade_paf_fulldata.csv", index=False)
    log("  point estimates saved; starting cluster bootstrap...")

    del Xr; gc.collect()

    # Country cluster bootstrap: refit beta on each resample, recompute BOTH the
    # pooled and recent-prevalence PAF (paired draws -> CI for the difference).
    # Per-country contribution capped at CAP records (drawn once, seeded) to bound
    # peak memory under the machine's ~16 GB external RAM pressure; this matches the
    # joint-PAF capped cluster bootstrap and is marginally conservative and robust to
    # within-cluster subsampling (Davison & Hinkley, 1997).
    CAP = 20_000
    cc = df["country_code"].values
    uniq, groups = FD.cluster_row_index(cc)
    rng = np.random.default_rng(RNG_SEED)
    capped = {}
    for c in uniq:
        r = groups[c]
        if len(r) > CAP:
            r = rng.choice(r, size=CAP, replace=False)
        capped[c] = np.sort(r)
    k = len(uniq)
    bp = {lab: [] for lab, _ in PAF_FACTORS}   # pooled
    br = {lab: [] for lab, _ in PAF_FACTORS}   # recent-prevalence
    bd = {lab: [] for lab, _ in PAF_FACTORS}   # difference
    t0 = time.time()
    for b in range(N_BOOT_PAF):
        pick = rng.integers(0, k, size=k)
        idx = np.concatenate([capped[uniq[j]] for j in pick])
        Xb, yb = X[idx], y[idx]
        rmask_b = recent_mask[idx]
        try:
            bb = FD.irls_logit(Xb, yb, start=beta)     # warm-started MLE (fast)
            Xrb = Xb[rmask_b]
            for lab, shift in PAF_FACTORS:
                sh = {k: v for k, v in shift.items() if k in colidx}
                pp = _paf_one(Xb, bb, colidx, sh) * 100
                pr = _paf_one(Xrb, bb, colidx, sh) * 100
                bp[lab].append(pp); br[lab].append(pr); bd[lab].append(pr - pp)
            del Xrb
        except Exception:
            pass
        del Xb, yb, idx
        if (b + 1) % 25 == 0:
            el = time.time() - t0
            eta = (N_BOOT_PAF - b - 1) / ((b + 1) / el)
            log(f"    boot {b+1}/{N_BOOT_PAF} ({el:.0f}s, ~{eta:.0f}s left)")
            gc.collect()

    def _ci(a):
        a = np.asarray(a)
        return (round(float(np.percentile(a, 2.5)), 2),
                round(float(np.percentile(a, 97.5)), 2)) if len(a) >= 20 else ("", "")

    for rec, (lab, _) in zip(rows, PAF_FACTORS):
        lo_p, hi_p = _ci(bp[lab]); lo_r, hi_r = _ci(br[lab]); lo_d, hi_d = _ci(bd[lab])
        rec["PAF_pooled_CI"] = f"({lo_p}–{hi_p})"
        rec["PAF_recent_CI"] = f"({lo_r}–{hi_r})"
        rec["delta_CI"] = f"({lo_d}–{hi_d})"
        rec["PAF_recent_fmt"] = f"{rec['PAF_recent_prevalence']} ({lo_r}–{hi_r})"
        log(f"  {lab}: recent-prevalence PAF = {rec['PAF_recent_fmt']}; "
            f"Δ vs pooled {rec['delta_recent_minus_pooled']} ({lo_d}–{hi_d})")
    pd.DataFrame(rows).to_csv(
        OUTDIR / "table_recentdecade_paf_fulldata.csv", index=False)
    log("RECENT-PREVALENCE PAF COMPLETE")


if __name__ == "__main__":
    main()
