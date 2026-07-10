"""
run_fulldata_ml_continue.py — finish the full-data ML chain past SHAP
=====================================================================

The full-data chain (run_fulldata_ml.py) completed s06->s07->s07b->s07c->s07e
->s06b but aborted at s08 because SHAP cannot import in this environment
(numba requires NumPy <=2.3; the project is on NumPy 2.4). SHAP is an
environment problem, NOT a data problem — it fails identically at 14.2%.

s08/s08b (SHAP) are the ONLY scripts importing shap; every downstream metric
script (calibration, recalibration, DCA, fairness, risk score, sensitivity,
DeLong) depends only on the trained model + splits, which are already full-data.
This continues the chain past SHAP with CONTINUE-on-failure so one bad step
never blocks the others. The existing (backed-up) SHAP feature ranking is
retained — the top-feature ordering is stable and cannot be regenerated until
the numba/NumPy clash is resolved.

Run: python -m src.ml.run_fulldata_ml_continue
"""
from __future__ import annotations
import os, sys, subprocess, time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.paths import OUTPUT_DIR

LOG = OUTPUT_DIR / "ml" / "fulldata_run.log"

CHAIN = [
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
    log("=" * 70)
    log("FULL-DATA ML CONTINUATION (post-SHAP; continue-on-failure)")
    log("NOTE: s08/s08b SHAP skipped — numba needs NumPy<=2.3, env has 2.4. "
        "Backed-up SHAP ranking retained.")
    env = dict(os.environ)
    env["ML_FULL_SAMPLE"] = "1"
    env.setdefault("OPENBLAS_NUM_THREADS", "4")
    repo = str(Path(__file__).resolve().parents[2])
    failed = []
    for mod in CHAIN:
        t0 = time.time()
        log(f"--> running {mod}")
        r = subprocess.run(
            [sys.executable, "-m", f"src.ml.{mod}"],
            cwd=repo, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        tail = "\n".join(r.stdout.strip().splitlines()[-10:]) if r.stdout else ""
        if r.returncode == 0:
            log(f"    {mod} OK ({time.time()-t0:.0f}s)\n    {tail}")
        else:
            failed.append(mod)
            log(f"    {mod} FAILED (exit {r.returncode}) ({time.time()-t0:.0f}s)\n    {tail}")
    log(f"CONTINUATION COMPLETE. failed={failed or 'none'}")


if __name__ == "__main__":
    main()
