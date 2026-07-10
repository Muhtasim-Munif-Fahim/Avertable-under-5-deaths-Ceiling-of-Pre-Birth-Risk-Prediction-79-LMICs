"""
run_fulldata_joint_counterfactual.py — JOINT edu+spacing avertable deaths
==========================================================================

NEW analysis for the LGH repositioning (joint-counterfactual avertable-deaths
pillar). Adds a *joint* maternal-education + birth-spacing g-computation
counterfactual on FULL data (all 10.67M births) and expresses it as ABSOLUTE
avertable under-5 deaths — pooled, per-WHO-region, and per-country — plus the
country concentration of the avertable burden.

Why this is separate from run_fulldata_all.py::table3():
  * keeps the canonical single-lever PAF outputs (edu 37.5 / spacing 20.8)
    byte-stable — this script writes only NEW files;
  * reuses the EXACT Model-3 surface, dummy coding, and the unweighted
    whole-population g-computation primitive `_paf_one` (PAF provenance:
    unweighted GLM, whole-population shift, NO survey weighting, NO
    worse-only counterfactual — see memory s05c-paf-provenance).

The joint scenario is a SINGLE g-computation pass with the MERGED shift dict
{**EDU_SHIFT, **BI_SHIFT}: g-computation predicts under both shifts at once, so
non-additivity is captured directly. We never sum the single-lever PAFs.

Absolute avertable deaths are a decomposition of ONE global Model-3 surface:
for each birth i the model-implied death-probability reduction is
delta_i = p_factual_i - p_counterfactual_i; summed and scaled to the cohort's
recorded under-5 deaths (s = recorded_total / sum(p_factual)) so that the
per-country / per-region avertable counts sum EXACTLY to the pooled headline.
This anchors the absolute count to the 885,756 recorded cohort deaths.

Outputs:
  output/fulldata/table_joint_counterfactual_fulldata.csv
      pooled + regional joint PAF (point + cluster-bootstrap 95% CI) and
      absolute avertable deaths; also the single-lever joint comparison.
  output/tables/table_s_percountry_joint_paf.csv
      per-country joint PAF + absolute avertable deaths + cumulative
      concentration share, ranked by avertable burden.

Run (SOLO — never concurrently with another full-data job; OOM risk):
  python -m src.inferential.run_fulldata_joint_counterfactual
"""
from __future__ import annotations
import os, sys, gc, time
from pathlib import Path
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.paths import RNG_SEED, TABLES_DIR
from src.inferential import _fulldata as FD
from src.inferential.run_fulldata_all import (
    build_dummies, design_from, M3, _paf_one, log, OUTDIR, N_BOOT_PAF,
)

# Recorded under-5 deaths in the in-scope cohort (anchors absolute counts).
# Cross-checked against evidence_registry.csv; also recomputed below from data.
RECORDED_U5_DEATHS = 885_756

EDU_SHIFT = {"edu_primary": 0, "edu_secondary": 1, "edu_higher": 0}
BI_SHIFT = {"bi_firstmissing": 0, "bi_short": 0, "bi_long": 0}
JOINT_SHIFT = {**EDU_SHIFT, **BI_SHIFT}

POOLED_OUT = OUTDIR / "table_joint_counterfactual_fulldata.csv"
PERCOUNTRY_OUT = TABLES_DIR / "table_s_percountry_joint_paf.csv"

LOAD_COLS = [
    "country_code", "who_region", "survey_year", "died_under5",
    "child_male", "multiple_birth", "birth_order", "birth_interval_months",
    "maternal_age_at_birth", "edu_level", "wealth_quintile",
    "currently_married", "urban",
]


def _gcomp_delta(X, beta, colidx, shift):
    """Return (p_factual, delta) where delta_i = p_factual_i - p_cf_i.

    Memory-frugal: edits only the shifted columns of X in place, then restores
    them (mirrors _paf_one). delta and p_factual are ~85 MB each at full N.
    """
    pf = expit(X @ beta)
    saved = {}
    for cname, val in shift.items():
        ci = colidx[cname]
        saved[ci] = X[:, ci].copy()
        X[:, ci] = val
    pcf = expit(X @ beta)
    for ci, col in saved.items():
        X[:, ci] = col
    return pf, pf - pcf


def main():
    t_start = time.time()
    log("JOINT COUNTERFACTUAL: full-data avertable deaths (edu + birth spacing)")

    df = FD.load_full(LOAD_COLS, scope="primary")
    n = len(df)
    y = df["died_under5"].astype(np.float64).values
    recorded = float(y.sum())
    log(f"  loaded {n:,} births; recorded under-5 deaths = {recorded:,.0f} "
        f"(registry anchor {RECORDED_U5_DEATHS:,})")

    dummies = build_dummies(df)
    keep = [k for k in M3 if dummies[k].std() > 0]
    X = design_from(dummies, keep, n)
    colidx = {nm: i + 1 for i, nm in enumerate(keep)}   # +1 for intercept
    joint = {k: v for k, v in JOINT_SHIFT.items() if k in colidx}
    edu = {k: v for k, v in EDU_SHIFT.items() if k in colidx}
    bi = {k: v for k, v in BI_SHIFT.items() if k in colidx}

    log("  fitting global Model-3 logistic surface...")
    beta = FD.logit_coefs(X, y)

    # ----- point estimates: joint g-computation decomposition --------------- #
    pf, delta = _gcomp_delta(X, beta, colidx, joint)
    pf_total = float(pf.sum())
    scale = recorded / pf_total            # anchor expected -> recorded deaths
    paf_joint_all = float(delta.sum() / pf_total)
    log(f"  joint PAF (All) = {paf_joint_all*100:.2f}%  "
        f"(model calibration scale = {scale:.4f})")

    # single-lever joint comparison (same surface) for the non-additivity note
    paf_edu_all = _paf_one(X, beta, colidx, edu)
    paf_bi_all = _paf_one(X, beta, colidx, bi)
    naive_sum = paf_edu_all + paf_bi_all
    log(f"  single-lever PAFs: edu={paf_edu_all*100:.2f}%  "
        f"bi={paf_bi_all*100:.2f}%  naive-sum={naive_sum*100:.2f}%  "
        f"(joint {paf_joint_all*100:.2f}% must be < naive-sum, >= each lever)")

    region = df["who_region"].values
    cc = df["country_code"].values
    agg = pd.DataFrame({"who_region": region, "country_code": cc,
                        "pf": pf, "delta": delta, "died": y})

    # ----- regional avertable deaths (decomposition, sums to pooled) -------- #
    reg_g = agg.groupby("who_region", sort=True)
    reg = reg_g.agg(pf=("pf", "sum"), delta=("delta", "sum"),
                    n=("died", "size"), recorded=("died", "sum")).reset_index()
    reg["paf_joint"] = reg["delta"] / reg["pf"]
    reg["avertable_deaths"] = reg["delta"] * scale

    # ----- per-country avertable deaths + concentration --------------------- #
    cty_g = agg.groupby("country_code", sort=False)
    cty = cty_g.agg(pf=("pf", "sum"), delta=("delta", "sum"),
                    n_births=("died", "size"),
                    recorded_deaths=("died", "sum"),
                    who_region=("who_region", "first")).reset_index()
    cty["paf_joint"] = (cty["delta"] / cty["pf"] * 100).round(1)
    cty["avertable_deaths_joint"] = (cty["delta"] * scale).round(0).astype(int)
    cty["recorded_u5m_per1000"] = (cty["recorded_deaths"] / cty["n_births"]
                                   * 1000).round(1)
    cty["recorded_deaths"] = cty["recorded_deaths"].round(0).astype(int)
    cty = cty.sort_values("avertable_deaths_joint", ascending=False).reset_index(drop=True)
    total_avert = cty["avertable_deaths_joint"].sum()
    cty["cum_share_pct"] = (cty["avertable_deaths_joint"].cumsum()
                            / total_avert * 100).round(1)
    cty["rank"] = np.arange(1, len(cty) + 1)

    # country names (best-effort from survey inventory)
    try:
        inv = pd.read_csv(TABLES_DIR / "table_s1_survey_inventory.csv")
        names = inv.groupby("country_code")["country_name"].first()
        cty["country_name"] = cty["country_code"].map(names).fillna(cty["country_code"])
    except Exception:
        cty["country_name"] = cty["country_code"]

    cty_out = cty[["rank", "country_code", "country_name", "who_region",
                   "n_births", "recorded_deaths", "recorded_u5m_per1000",
                   "paf_joint", "avertable_deaths_joint", "cum_share_pct"]]
    cty_out.to_csv(PERCOUNTRY_OUT, index=False)
    top10_share = float(cty["cum_share_pct"].iloc[9]) if len(cty) >= 10 else np.nan
    top20_share = float(cty["cum_share_pct"].iloc[19]) if len(cty) >= 20 else np.nan
    log(f"  per-country: {len(cty)} countries; total avertable = {total_avert:,} "
        f"deaths; top-10 share = {top10_share}%  top-20 share = {top20_share}%")
    log(f"  wrote {PERCOUNTRY_OUT.name}")

    del pf, delta, agg; gc.collect()

    # ----- cluster bootstrap for joint PAF CIs (All + regions) -------------- #
    # Speed: cap each country's row pool (systematic thinning) so the 500
    # full-data refits are tractable. The POINT estimates above stay on the
    # full cohort; the cap only affects CI width (slightly conservative),
    # mirroring the capped cluster bootstrap used in s05c.
    BOOT_CAP = 20000
    log(f"  cluster bootstrap for joint-PAF CIs ({N_BOOT_PAF} reps, "
        f"cap {BOOT_CAP}/country)...")
    regions = ["All"] + sorted(pd.unique(region[pd.notna(region)]))
    uniq, groups = FD.cluster_row_index(cc)
    groups_cap = {}
    for _kc, _pos in groups.items():
        if len(_pos) > BOOT_CAP:
            sel = np.linspace(0, len(_pos) - 1, BOOT_CAP).astype(np.int64)
            groups_cap[_kc] = _pos[sel]
        else:
            groups_cap[_kc] = _pos
    rng = np.random.default_rng(RNG_SEED)
    boot = {rr: [] for rr in regions}
    t0 = time.time()
    for b, idx in FD.iter_cluster_bootstrap(uniq, groups_cap, rng, N_BOOT_PAF):
        Xb, yb, rb = X[idx], y[idx], region[idx]
        try:
            bb = FD.logit_coefs(Xb, yb)
            for rr in regions:
                mask = slice(None) if rr == "All" else (rb == rr)
                boot[rr].append(_paf_one(Xb[mask], bb, colidx, joint))
        except Exception:
            pass
        del Xb, yb, rb
        if (b + 1) % 25 == 0:
            el = time.time() - t0
            eta = (N_BOOT_PAF - b - 1) / ((b + 1) / el)
            log(f"    joint-PAF boot {b+1}/{N_BOOT_PAF} ({el:.0f}s, ~{eta:.0f}s left)")
            gc.collect()

    # checkpoint the raw bootstrap arrays BEFORE the (fragile) assembly, so a
    # failure in assembly never discards the expensive bootstrap again.
    try:
        np.savez(str(POOLED_OUT.with_suffix(".bootckpt.npz")),
                 **{rr: np.asarray(boot[rr], dtype=float) for rr in regions})
        log(f"  bootstrap checkpoint saved ({POOLED_OUT.stem}.bootckpt.npz)")
    except Exception as e:
        log(f"  checkpoint save failed (non-fatal): {e}")

    # ----- assemble pooled/regional output ---------------------------------- #
    reg_lookup = reg.set_index("who_region")
    rows = []
    for r in regions:
        if r == "All":
            paf_pt, n_r, rec_r, avert_r = (paf_joint_all, n, recorded,
                                           float(total_avert))
        else:
            rr = reg_lookup.loc[r]
            paf_pt = float(rr["paf_joint"]); n_r = int(rr["n"])
            rec_r = float(rr["recorded"]); avert_r = float(rr["avertable_deaths"])
        bv = np.array(boot[r]) * 100
        lo = round(float(np.percentile(bv, 2.5)), 2) if len(bv) >= 20 else np.nan
        hi = round(float(np.percentile(bv, 97.5)), 2) if len(bv) >= 20 else np.nan
        rows.append({
            "scenario": "joint_edu_spacing",
            "region": r,
            "n_births": int(n_r),
            "recorded_u5_deaths": int(round(rec_r)),
            "PAF_joint": round(paf_pt * 100, 2),
            "CI_lo": lo, "CI_hi": hi,
            "PAF_joint_fmt": f"{round(paf_pt*100,1)} ({lo}–{hi})",
            "avertable_deaths": int(round(avert_r)),
        })
    pooled = pd.DataFrame(rows)

    # reference rows: single-lever PAFs on the same surface + non-additivity
    pooled.loc[len(pooled)] = {
        "scenario": "edu_only_ref", "region": "All", "n_births": n,
        "recorded_u5_deaths": int(round(recorded)),
        "PAF_joint": round(paf_edu_all * 100, 2), "CI_lo": np.nan, "CI_hi": np.nan,
        "PAF_joint_fmt": f"{round(paf_edu_all*100,1)}", "avertable_deaths": int(round(paf_edu_all*recorded)),
    }
    pooled.loc[len(pooled)] = {
        "scenario": "spacing_only_ref", "region": "All", "n_births": n,
        "recorded_u5_deaths": int(round(recorded)),
        "PAF_joint": round(paf_bi_all * 100, 2), "CI_lo": np.nan, "CI_hi": np.nan,
        "PAF_joint_fmt": f"{round(paf_bi_all*100,1)}", "avertable_deaths": int(round(paf_bi_all*recorded)),
    }
    pooled.loc[len(pooled)] = {
        "scenario": "naive_sum_ref", "region": "All", "n_births": n,
        "recorded_u5_deaths": int(round(recorded)),
        "PAF_joint": round(naive_sum * 100, 2), "CI_lo": np.nan, "CI_hi": np.nan,
        "PAF_joint_fmt": f"{round(naive_sum*100,1)}", "avertable_deaths": int(round(naive_sum*recorded)),
    }
    pooled.to_csv(POOLED_OUT, index=False)
    log(f"  wrote {POOLED_OUT.name}")
    log(f"JOINT COUNTERFACTUAL done ({time.time()-t_start:.0f}s)")


if __name__ == "__main__":
    main()
