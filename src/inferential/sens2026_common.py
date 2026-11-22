"""Shared machinery for the 2026-09 reviewer-round sensitivity analyses.

Used by ``sens2026_survival.py`` and ``sens2026_binary.py``. Nothing here fits a model;
it standardises fitted risks to an exposure distribution, runs the checkpointed
shard bookkeeping, and summarises bootstrap draws.

Contemporary standardisation is the estimator of the published
``run_fulldata_recentdecade_paf*.py`` scripts, restated so it can be recomputed
inside a *refitting* bootstrap: per-country x per-exposure-cell sufficient statistics,
importance-reweighted so the joint education x spacing cell distribution matches that
of 2018-2024 births, with weight in cells the cohort does not support dropped rather
than extrapolated.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from config.paths import OUTPUT_DIR

OUT_ROOT = OUTPUT_DIR / "phase4" / "sensitivity_2026-09"
SCENARIOS = ("education", "spacing", "joint")
RECENT_LO, RECENT_HI = 2018, 2024


class Standardiser:
    """Long-window and contemporary PAFs from pattern-level risks."""

    def __init__(self, pattern_country, pattern_cell, pattern_weight,
                 recent_country, recent_cell, recent_weight):
        pattern_country = np.asarray(pattern_country, dtype=object)
        recent_country = np.asarray(recent_country, dtype=object)
        self.countries = sorted(set(pattern_country.tolist()))
        cells = sorted(set(np.asarray(pattern_cell).tolist())
                       | set(np.asarray(recent_cell).tolist()))
        coix = {c: i for i, c in enumerate(self.countries)}
        cix = {c: i for i, c in enumerate(cells)}
        self.C, self.K = len(self.countries), len(cells)
        self.gid = (np.fromiter((coix[c] for c in pattern_country), np.int64, len(pattern_country))
                    * self.K
                    + np.fromiter((cix[c] for c in pattern_cell), np.int64, len(pattern_country)))
        self.w = np.asarray(pattern_weight, dtype=float)
        self.S_w = self._sum(np.ones_like(self.w))
        keep = np.isin(recent_country, self.countries)
        rgid = (np.fromiter((coix[c] for c in recent_country[keep]), np.int64, int(keep.sum()))
                * self.K
                + np.fromiter((cix[c] for c in np.asarray(recent_cell)[keep]), np.int64,
                              int(keep.sum())))
        self.R_w = np.bincount(rgid, weights=np.asarray(recent_weight, float)[keep],
                               minlength=self.C * self.K).reshape(self.C, self.K)
        self.country_index = coix

    def _sum(self, values: np.ndarray) -> np.ndarray:
        return np.bincount(self.gid, weights=self.w * values,
                           minlength=self.C * self.K).reshape(self.C, self.K)

    def pafs(self, mult: np.ndarray, factual: np.ndarray,
             counterfactual: dict[str, np.ndarray]) -> dict[tuple[str, str], float]:
        """{(estimand, scenario): PAF %} for one country-multiplicity vector."""
        S_wf = self._sum(factual)
        cw, cwf, rw = mult @ self.S_w, mult @ S_wf, mult @ self.R_w
        supported = cw > 0
        rw_sup = np.where(supported, rw, 0.0)
        ratio = np.zeros(self.K)
        if rw_sup.sum() > 0 and cw.sum() > 0:
            ratio[supported] = (rw_sup[supported] / rw_sup.sum()) / (cw[supported] / cw.sum())
        out: dict[tuple[str, str], float] = {}
        for scenario, cf in counterfactual.items():
            cwcf = mult @ self._sum(cf)
            f_long, c_long = cwf.sum(), cwcf.sum()
            f_now, c_now = (ratio * cwf).sum(), (ratio * cwcf).sum()
            out[("long_window", scenario)] = 100.0 * (f_long - c_long) / f_long
            out[("contemporary", scenario)] = (100.0 * (f_now - c_now) / f_now
                                              if f_now > 0 else float("nan"))
        return out

    def multiplicity(self, draw: dict[str, int]) -> np.ndarray:
        mult = np.zeros(self.C)
        for country, count in draw.items():
            if country in self.country_index:
                mult[self.country_index[country]] = count
        return mult


def git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=OUTPUT_DIR.parent).stdout.strip()
    except Exception:
        return ""


def shard_paths(out_dir: Path, shard: int) -> tuple[Path, Path]:
    return out_dir / f"boot_shard{shard}.csv", out_dir / f"diag_shard{shard}.csv"


def completed_replicates(out_dir: Path, shard: int) -> set[int]:
    _, diag = shard_paths(out_dir, shard)
    if not diag.exists():
        return set()
    return set(pd.read_csv(diag)["replicate"].astype(int))


def append_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    frame = pd.DataFrame(rows)
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def summarise(out_dir: Path, point: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """Merge shard draws into percentile intervals; write summary + manifest."""
    boots = [pd.read_csv(p) for p in sorted(out_dir.glob("boot_shard*.csv"))]
    diags = [pd.read_csv(p) for p in sorted(out_dir.glob("diag_shard*.csv"))]
    draws = pd.concat(boots, ignore_index=True) if boots else pd.DataFrame(
        columns=["replicate", "estimand", "scenario", "paf_pct"])
    diag = pd.concat(diags, ignore_index=True) if diags else pd.DataFrame(
        columns=["replicate", "success"])
    draws = draws.drop_duplicates(["replicate", "estimand", "scenario"])
    diag = diag.drop_duplicates("replicate", keep="last")
    rows = []
    for r in point.itertuples(index=False):
        d = draws[(draws.estimand == r.estimand) & (draws.scenario == r.scenario)]["paf_pct"]
        d = pd.to_numeric(d, errors="coerce").dropna()
        rows.append({"estimand": r.estimand, "scenario": r.scenario,
                     "paf_pct": round(float(r.paf_pct), 2),
                     "ci_lo": round(float(d.quantile(0.025)), 2) if len(d) else np.nan,
                     "ci_hi": round(float(d.quantile(0.975)), 2) if len(d) else np.nan,
                     "n_success": int(len(d))})
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    manifest = {**meta,
                "summarised_at_utc": datetime.now(timezone.utc).isoformat(),
                "git_head": git_head(),
                "n_replicates_attempted": int(len(diag)),
                "n_replicates_failed": int((~diag["success"].astype(bool)).sum()) if len(diag) else 0,
                "summary": summary.to_dict("records")}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str),
                                           encoding="utf-8")
    return summary
