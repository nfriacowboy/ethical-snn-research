"""Batch runner for Phase 1 experiments.

Runs both conditions (A and B) with 100 runs each.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.simulation.runner import SimulationRunner
from src.utils.config import load_config
import argparse
import time
from datetime import datetime


def run_condition(config: dict, architecture: str, num_runs: int, start_run_id: int = 0):
    """Run all simulations for one condition.
    
    Args:
        config: Configuration dictionary
        architecture: 'single' or 'dual'
        num_runs: Number of runs
        start_run_id: Starting run ID
    """
    config['architecture'] = architecture
    condition_name = "Condition A (Survival-Only)" if architecture == 'single' else "Condition B (Dual-Process)"
    
    print("\n" + "=" * 70)
    print(f"RUNNING {condition_name}")
    print(f"{num_runs} runs")
    print("=" * 70)
    
    start_time = time.time()
    
    for i in range(num_runs):
        run_id = start_run_id + i
        
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Run {i+1}/{num_runs} (ID: {run_id})")
        
        try:
            runner = SimulationRunner(config=config, run_id=run_id)
            results = runner.run()
            runner.logger.log_summary(results)
            runner.logger.close()
            
            print(f"✓ Completed - Alive: {results['architecture_stats']['alive_count']}")
            
        except Exception as e:
            print(f"✗ Error in run {run_id}: {e}")
            continue
    
    elapsed = time.time() - start_time
    print(f"\n{condition_name} completed in {elapsed/60:.2f} minutes")


def main():
    parser = argparse.ArgumentParser(description='Run full Phase 1 experiment batch')
    parser.add_argument('--config', type=str, default='experiments/phase1/config_phase1.yaml',
                       help='Path to config file')
    parser.add_argument('--num_runs', type=int, default=100,
                       help='Number of runs per condition')
    parser.add_argument('--condition', type=str, choices=['A', 'B', 'both'], default='both',
                       help='Which condition to run')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    print("=" * 70)
    print("PHASE 1 BATCH EXPERIMENT RUNNER")
    print("=" * 70)
    print(f"Configuration: {args.config}")
    print(f"Runs per condition: {args.num_runs}")
    print(f"Conditions: {args.condition}")
    print("=" * 70)
    
    start_time = time.time()
    
    # Run conditions
    if args.condition in ['A', 'both']:
        run_condition(config, 'single', args.num_runs, start_run_id=0)
    
    if args.condition in ['B', 'both']:
        run_condition(config, 'dual', args.num_runs, start_run_id=args.num_runs)
    
    # Final summary
    total_elapsed = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("BATCH EXPERIMENT COMPLETED")
    print("=" * 70)
    print(f"Total time: {total_elapsed/60:.2f} minutes ({total_elapsed/3600:.2f} hours)")
    
    if args.condition == 'both':
        print(f"Total simulations: {args.num_runs * 2}")
        print(f"Average time per run: {total_elapsed/(args.num_runs*2):.2f} seconds")
    else:
        print(f"Total simulations: {args.num_runs}")
        print(f"Average time per run: {total_elapsed/args.num_runs:.2f} seconds")
    
    print("\nResults saved in: results/")
    print("=" * 70)


if __name__ == '__main__':
    main()
