"""Run survival-only experiments (Condition A).

This script runs 100 simulations with survival-only architecture.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import argparse

from src.simulation.runner import SimulationRunner
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Run survival-only simulation")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/phase1/config_phase1.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--run_id", type=int, default=0, help="Run ID for this simulation"
    )
    parser.add_argument(
        "--num_runs", type=int, default=1, help="Number of runs to execute"
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Force survival-only architecture
    config["architecture"] = "single"

    print("=" * 60)
    print("PHASE 1 - CONDITION A: Survival-Only SNN")
    print("=" * 60)

    # Run simulations
    for run_id in range(args.run_id, args.run_id + args.num_runs):
        print(f"\n{'='*60}")
        print(f"Starting Run {run_id + 1}/{args.run_id + args.num_runs}")
        print(f"{'='*60}\n")

        # Create runner
        runner = SimulationRunner(config=config, run_id=run_id)

        # Run simulation
        results = runner.run()

        # Save summary
        runner.logger.log_summary(results)
        runner.logger.close()

        print(f"\nRun {run_id} completed!")
        print(f"Final alive: {results['architecture_stats']['alive_count']}")
        print(f"Log directory: {runner.logger.run_dir}")

    print("\n" + "=" * 60)
    print("All runs completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
