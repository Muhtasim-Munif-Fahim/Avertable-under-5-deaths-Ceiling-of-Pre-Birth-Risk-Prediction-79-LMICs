"""
run_fulldata_ml.py — run the ML pipeline on the FULL cohort (100%)
==================================================================

Re-runs the canonical leakage-screened ML chain (PROJECT_PROGRESS run order)
on all in-scope births instead of the 14.2% systematic sample, by setting
ML_FULL_SAMPLE=1 (read by s06_ml_data_prep). Hyperparameters are NOT re-tuned
(Optuna s07d is not in the canonical chain; s07 uses fixed params) and SHAP
(s08/s08b) already subsamples internally — so full-data is RAM-safe.

Safety: the existing output/ml/ (14.2% results) is backed up to
output/ml_backup_14pct/ before anything is overwritten, so the prior pipeline
remains fully recoverable.

Run (background, solo): python -m src.ml.run_fulldata_ml
"""
from __future__ import annotations
import os, sys, shutil, subprocess, time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.paths import OUTPUT_DIR

ML_DIR = OUTPUT_DIR / "ml"
BACKUP = OUTPUT_DIR / "ml_backup_14pct"
LOG = ML_DIR / "fulldata_run.log"

# Canonical chain (PROJECT_PROGRESS §run order) + s07e pre-birth-only sensitivity
# (eTable S33 number cited in the manuscript).
CHAIN = [
    "s06_ml_data_prep",
    "s07_xgboost_training",
    "s07b_model_comparison",
    "s07c_lr_baseline",
    "s07e_prebirth_only_sensitivity",
    "s06b_loro",
    "s08_shap_explainability",
    "s08b_regional_shap",
    "s09_calibration",
    "s09a_recalibration",
    "s09b_decision_curve",
    "s09c_fairness",
    "s10_risk_score",
    "s11_sensitivity",
    "s12_delong_cis",
]


def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def main():
    # 1) back up the existing 14.2% pipeline outputs (once)
    if not BACKUP.exists():
        shutil.copytree(ML_DIR, BACKUP)
        print(f"Backed up {ML_DIR} -> {BACKUP}")
    LOG.unlink(missing_ok=True)
    log("=" * 70)
    log("FULL-DATA ML PIPELINE START (ML_FULL_SAMPLE=1)")
    log(f"Backup of prior 14.2% outputs at: {BACKUP}")

    env = dict(os.environ)
    env["ML_FULL_SAMPLE"] = "1"
    env.setdefault("OPENBLAS_NUM_THREADS", "4")
    repo = str(Path(__file__).resolve().parents[2])

    for mod in CHAIN:
        t0 = time.time()
        log(f"--> running {mod}")
        r = subprocess.run(
            [sys.executable, "-m", f"src.ml.{mod}"],
            cwd=repo, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        tail = "\n".join(r.stdout.strip().splitlines()[-12:]) if r.stdout else ""
        if r.returncode == 0:
            log(f"    {mod} OK ({time.time()-t0:.0f}s)")
            log(f"    last output:\n{tail}")
        else:
            log(f"    {mod} FAILED (exit {r.returncode}) ({time.time()-t0:.0f}s)")
            log(f"    output tail:\n{tail}")
            log("ABORTING chain (downstream depends on this step).")
            return
    log("FULL-DATA ML PIPELINE COMPLETE")


if __name__ == "__main__":
    main()
