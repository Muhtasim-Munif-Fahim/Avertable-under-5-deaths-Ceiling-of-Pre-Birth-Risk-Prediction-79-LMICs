"""Command-line entry point for the frozen Phase 3 survival gate."""

from __future__ import annotations

import argparse

from config.paths import HARMONIZED_DIR, OUTPUT_DIR, RNG_SEED
from src.inferential.phase3_survival import run_survival_validation


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the Phase 3 discrete-time survival validation"
    )
    parser.add_argument(
        "--pilot-path",
        default=str(HARMONIZED_DIR / "global_analysis_v2_pilot_step20.parquet"),
    )
    parser.add_argument("--bootstrap-reps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    parser.add_argument(
        "--compute-backend",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    args = parser.parse_args()
    result = run_survival_validation(
        args.pilot_path,
        OUTPUT_DIR / "phase3",
        phase2_dir=OUTPUT_DIR / "phase2",
        require_decision_pilot_provenance=True,
        bootstrap_reps=args.bootstrap_reps,
        seed=args.seed,
        compute_backend=args.compute_backend,
    )
    print(f"Phase 3 decision: {result.decision}")
    print(f"Aggregate outputs: {(OUTPUT_DIR / 'phase3').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
