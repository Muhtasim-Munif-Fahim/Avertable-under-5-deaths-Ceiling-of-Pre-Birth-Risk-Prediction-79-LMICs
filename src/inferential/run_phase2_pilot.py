"""Command-line entry point for the frozen Phase 2 decision pilot."""

from __future__ import annotations

import argparse

from config.paths import OUTPUT_DIR, RNG_SEED
from src.inferential.phase2_analysis import run_analysis
from src.inferential.phase2_pilot import PilotConfig, build_phase2_dataset


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build and analyse the frozen exact-outcome Phase 2 pilot"
    )
    parser.add_argument("--sample-step", type=int, default=20)
    parser.add_argument("--bootstrap-reps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    parser.add_argument(
        "--rebuild-data",
        action="store_true",
        help="Rebuild the ignored individual-level pilot before analysis",
    )
    args = parser.parse_args()

    config = PilotConfig(
        sample_step=args.sample_step,
        bootstrap_reps=args.bootstrap_reps,
        seed=args.seed,
        rebuild_data=args.rebuild_data,
    )
    built = build_phase2_dataset(config)
    result = run_analysis(
        built.data_path,
        OUTPUT_DIR / "phase2",
        bootstrap_reps=config.bootstrap_reps,
        seed=config.seed,
    )
    print(f"Phase 2 decision: {result.decision}")
    print(f"Aggregate outputs: {(OUTPUT_DIR / 'phase2').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
