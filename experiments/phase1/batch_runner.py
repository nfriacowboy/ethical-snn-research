"""Batch runner for Phase 1 experiments.

Runs both conditions (A and B) with multiple runs each for statistical analysis.

Example:
    Run both conditions with 10 runs each:
    >>> python experiments/phase1/batch_runner.py --num_runs 10 --condition both

    Run only Condition A with 5 runs:
    >>> python experiments/phase1/batch_runner.py --num_runs 5 --condition A
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from src.simulation.runner import SimulationRunner
from src.utils.config import load_config


def run_condition(
    config: Dict[str, Any], condition: str, num_runs: int, start_seed: int = 42
) -> List[Dict[str, Any]]:
    """Run all simulations for one condition.

    Args:
        config: Configuration dictionary
        condition: 'A' (survival-only) or 'B' (dual-process)
        num_runs: Number of runs
        start_seed: Starting random seed (default: 42)

    Returns:
        List of statistics dictionaries from all runs
    """
    condition_name = {
        "A": "Condition A (Survival-Only)",
        "B": "Condition B (Dual-Process)",
    }[condition]

    print("\n" + "=" * 70)
    print(f"RUNNING {condition_name}")
    print(f"{num_runs} runs")
    print("=" * 70)

    start_time = time.time()
    all_stats = []

    # Update config for this condition
    run_config = config.copy()
    run_config["condition"] = condition

    for i in range(num_runs):
        run_seed = start_seed + i

        print(
            f"\n[{datetime.now().strftime('%H:%M:%S')}] Run {i+1}/{num_runs} (Seed: {run_seed})"
        )

        try:
            runner = SimulationRunner(config=run_config, seed=run_seed)
            results = runner.run()
            all_stats.append(results)

            alive = results["organisms"]["alive"]
            dead = results["organisms"]["dead"]
            print(f"✓ Completed - Alive: {alive}, Dead: {dead}")

        except Exception as e:
            print(f"✗ Error in run {i+1}: {e}")
            import traceback

            traceback.print_exc()
            continue

    elapsed = time.time() - start_time
    print(f"\n{condition_name} completed in {elapsed/60:.2f} minutes")
    print(f"Successful runs: {len(all_stats)}/{num_runs}")

    return all_stats


def aggregate_statistics(
    condition_stats: List[Dict[str, Any]], condition: str
) -> Dict[str, Any]:
    """Aggregate statistics across multiple runs.

    Args:
        condition_stats: List of statistics from individual runs
        condition: 'A' or 'B'

    Returns:
        Aggregated statistics dictionary
    """
    if not condition_stats:
        return {}

    # Extract key metrics
    survival_times = []
    final_alive_counts = []
    final_timesteps = []
    final_energy_avgs = []

    for stats in condition_stats:
        survival_times.extend(stats["organisms"]["survival_times"])
        final_alive_counts.append(stats["organisms"]["alive"])
        final_timesteps.append(stats["final_timestep"])
        final_energy_avgs.append(stats["energy"]["final_avg"])

    aggregated = {
        "condition": condition,
        "num_runs": len(condition_stats),
        "survival_time": {
            "mean": float(np.mean(survival_times)) if survival_times else 0,
            "std": float(np.std(survival_times)) if survival_times else 0,
            "median": float(np.median(survival_times)) if survival_times else 0,
            "min": float(np.min(survival_times)) if survival_times else 0,
            "max": float(np.max(survival_times)) if survival_times else 0,
        },
        "final_alive": {
            "mean": float(np.mean(final_alive_counts)),
            "std": float(np.std(final_alive_counts)),
            "median": float(np.median(final_alive_counts)),
        },
        "final_timestep": {
            "mean": float(np.mean(final_timesteps)),
            "std": float(np.std(final_timesteps)),
            "median": float(np.median(final_timesteps)),
        },
        "final_energy": {
            "mean": (
                float(np.mean([e for e in final_energy_avgs if e > 0]))
                if any(e > 0 for e in final_energy_avgs)
                else 0
            ),
            "std": (
                float(np.std([e for e in final_energy_avgs if e > 0]))
                if any(e > 0 for e in final_energy_avgs)
                else 0
            ),
        },
    }

    # Add dual-process specific stats if applicable
    if condition == "B":
        veto_rates = [
            stats.get("dual_process", {}).get("veto_rate", 0)
            for stats in condition_stats
        ]
        if veto_rates:
            aggregated["dual_process"] = {
                "veto_rate_mean": float(np.mean(veto_rates)),
                "veto_rate_std": float(np.std(veto_rates)),
                "veto_rate_median": float(np.median(veto_rates)),
            }

    return aggregated


def save_batch_results(
    condition_a_stats: List[Dict[str, Any]],
    condition_b_stats: List[Dict[str, Any]],
    aggregated_a: Dict[str, Any],
    aggregated_b: Dict[str, Any],
    output_dir: Path,
):
    """Save batch experiment results.

    Args:
        condition_a_stats: Individual run stats for Condition A
        condition_b_stats: Individual run stats for Condition B
        aggregated_a: Aggregated stats for Condition A
        aggregated_b: Aggregated stats for Condition B
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save aggregated results
    aggregated_file = output_dir / "batch_aggregated.json"
    with open(aggregated_file, "w") as f:
        json.dump(
            {
                "condition_a": aggregated_a,
                "condition_b": aggregated_b,
                "timestamp": datetime.now().isoformat(),
            },
            f,
            indent=2,
        )

    print(f"\n✓ Aggregated results saved to: {aggregated_file}")

    # Save individual run details
    if condition_a_stats:
        condition_a_file = output_dir / "condition_a_runs.json"
        with open(condition_a_file, "w") as f:
            json.dump(condition_a_stats, f, indent=2)
        print(f"✓ Condition A individual runs saved to: {condition_a_file}")

    if condition_b_stats:
        condition_b_file = output_dir / "condition_b_runs.json"
        with open(condition_b_file, "w") as f:
            json.dump(condition_b_stats, f, indent=2)
        print(f"✓ Condition B individual runs saved to: {condition_b_file}")


def print_comparison(aggregated_a: Dict[str, Any], aggregated_b: Dict[str, Any]):
    """Print comparison between conditions.

    Args:
        aggregated_a: Aggregated stats for Condition A
        aggregated_b: Aggregated stats for Condition B
    """
    print("\n" + "=" * 70)
    print("CONDITION COMPARISON")
    print("=" * 70)

    if aggregated_a and aggregated_b:
        print("\nSurvival Time:")
        print(
            f"  Condition A: {aggregated_a['survival_time']['mean']:.2f} ± {aggregated_a['survival_time']['std']:.2f}"
        )
        print(
            f"  Condition B: {aggregated_b['survival_time']['mean']:.2f} ± {aggregated_b['survival_time']['std']:.2f}"
        )

        print("\nFinal Alive Count:")
        print(
            f"  Condition A: {aggregated_a['final_alive']['mean']:.2f} ± {aggregated_a['final_alive']['std']:.2f}"
        )
        print(
            f"  Condition B: {aggregated_b['final_alive']['mean']:.2f} ± {aggregated_b['final_alive']['std']:.2f}"
        )

        if "dual_process" in aggregated_b:
            print("\nDual-Process Metrics (Condition B):")
            print(
                f"  Veto Rate: {aggregated_b['dual_process']['veto_rate_mean']:.2%} ± {aggregated_b['dual_process']['veto_rate_std']:.2%}"
            )

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Run full Phase 1 experiment batch")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/phase1/config_phase1.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=100,
        help="Number of runs per condition (default: 100)",
    )
    parser.add_argument(
        "--condition",
        type=str,
        choices=["A", "B", "both"],
        default="both",
        help="Which condition to run (default: both)",
    )
    parser.add_argument(
        "--start_seed", type=int, default=42, help="Starting random seed (default: 42)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/batch_experiments",
        help="Output directory for batch results",
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    print("=" * 70)
    print("PHASE 1 BATCH EXPERIMENT RUNNER")
    print("=" * 70)
    print(f"Configuration: {args.config}")
    print(f"Runs per condition: {args.num_runs}")
    print(f"Conditions: {args.condition}")
    print(f"Starting seed: {args.start_seed}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 70)

    start_time = time.time()

    # Run conditions
    condition_a_stats = []
    condition_b_stats = []

    if args.condition in ["A", "both"]:
        condition_a_stats = run_condition(
            config, condition="A", num_runs=args.num_runs, start_seed=args.start_seed
        )

    if args.condition in ["B", "both"]:
        condition_b_stats = run_condition(
            config,
            condition="B",
            num_runs=args.num_runs,
            start_seed=args.start_seed
            + (args.num_runs if args.condition == "both" else 0),
        )

    # Aggregate statistics
    aggregated_a = (
        aggregate_statistics(condition_a_stats, "A") if condition_a_stats else {}
    )
    aggregated_b = (
        aggregate_statistics(condition_b_stats, "B") if condition_b_stats else {}
    )

    # Save results
    output_dir = Path(args.output_dir)
    save_batch_results(
        condition_a_stats, condition_b_stats, aggregated_a, aggregated_b, output_dir
    )

    # Print comparison
    if args.condition == "both":
        print_comparison(aggregated_a, aggregated_b)

    # Final summary
    total_elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("BATCH EXPERIMENT COMPLETED")
    print("=" * 70)
    print(
        f"Total time: {total_elapsed/60:.2f} minutes ({total_elapsed/3600:.2f} hours)"
    )

    total_runs = len(condition_a_stats) + len(condition_b_stats)
    print(f"Total successful simulations: {total_runs}")
    if total_runs > 0:
        print(f"Average time per run: {total_elapsed/total_runs:.2f} seconds")

    print(f"\nResults saved in: {args.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
