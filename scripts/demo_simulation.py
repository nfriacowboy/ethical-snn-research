#!/usr/bin/env python3
"""
Quick Demo Script - Ethical SNN Research
=========================================

Runs a quick demonstration of both simulation conditions and displays results.
Perfect for first-time users and live demonstrations.

Usage:
    uv run python scripts/demo_simulation.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import torch
import numpy as np
from typing import Dict, Any
from colorama import init, Fore, Style

# Initialize colorama for cross-platform colored output
init(autoreset=True)

from src.simulation.runner import SimulationRunner
from src.utils.config import load_config


def print_header(text: str) -> None:
    """Print a formatted header."""
    print(f"\n{Fore.CYAN}{'=' * 70}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{text.center(70)}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 70}{Style.RESET_ALL}\n")


def print_config(config: Dict[str, Any]) -> None:
    """Print configuration details."""
    print(f"{Fore.YELLOW}Configuration:{Style.RESET_ALL}")
    print(f"  Grid Size: {config['grid_size']}×{config['grid_size']}")
    print(f"  Organisms: {config['num_organisms']}")
    print(f"  Food Items: {config['num_food']}")
    print(f"  Max Timesteps: {config['max_timesteps']}")
    print(f"  Energy Decay: {config['energy_decay_rate']}/timestep")
    print()


def print_progress_bar(iteration: int, total: int, prefix: str = '', length: int = 50) -> None:
    """Print a progress bar."""
    percent = 100 * (iteration / float(total))
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent:.1f}%', end='', flush=True)
    if iteration == total:
        print()


def print_results(condition: str, stats: Dict[str, Any]) -> None:
    """Print simulation results."""
    print(f"\n{Fore.GREEN}Results for Condition {condition}:{Style.RESET_ALL}")
    print(f"  Duration: {stats['final_timestep']} timesteps ({stats['elapsed_time']:.2f}s)")
    print(f"  Organisms Alive: {Fore.GREEN if stats['organisms']['alive'] > 0 else Fore.RED}"
          f"{stats['organisms']['alive']}/{stats['organisms']['total']}{Style.RESET_ALL}")
    print(f"  Average Survival Time: {stats['organisms']['avg_survival_time']:.1f} timesteps")
    
    if stats['organisms']['alive'] > 0:
        print(f"  Final Average Energy: {stats['energy']['final_avg']:.1f}")
    
    print(f"  Food Remaining: {stats['environment']['final_food_count']}")
    
    if condition == 'B' and 'dual_process' in stats:
        total_decisions = stats['dual_process']['total_vetoes'] + stats['dual_process']['total_approvals']
        veto_rate = stats['dual_process']['avg_veto_rate'] * 100
        print(f"\n{Fore.MAGENTA}  Ethical Network Activity:{Style.RESET_ALL}")
        print(f"    Total Decisions: {total_decisions}")
        print(f"    Vetoes: {stats['dual_process']['total_vetoes']}")
        print(f"    Approvals: {stats['dual_process']['total_approvals']}")
        print(f"    Veto Rate: {veto_rate:.1f}%")


def compare_conditions(stats_a: Dict[str, Any], stats_b: Dict[str, Any]) -> None:
    """Print comparison between conditions."""
    print_header("Comparison: Condition A vs Condition B")
    
    # Survival comparison
    survival_a = stats_a['organisms']['avg_survival_time']
    survival_b = stats_b['organisms']['avg_survival_time']
    diff = survival_b - survival_a
    diff_pct = (diff / survival_a) * 100 if survival_a > 0 else 0
    
    print(f"{Fore.YELLOW}Average Survival Time:{Style.RESET_ALL}")
    print(f"  Condition A (Survival-Only): {survival_a:.1f} timesteps")
    print(f"  Condition B (Dual-Process): {survival_b:.1f} timesteps")
    print(f"  Difference: {Fore.GREEN if diff > 0 else Fore.RED}{diff:+.1f} timesteps "
          f"({diff_pct:+.1f}%){Style.RESET_ALL}")
    
    # Final alive count
    alive_a = stats_a['organisms']['alive']
    alive_b = stats_b['organisms']['alive']
    
    print(f"\n{Fore.YELLOW}Organisms Surviving to End:{Style.RESET_ALL}")
    print(f"  Condition A: {alive_a}/{stats_a['organisms']['total']}")
    print(f"  Condition B: {alive_b}/{stats_b['organisms']['total']}")
    
    # Interpretation
    print(f"\n{Fore.CYAN}Interpretation:{Style.RESET_ALL}")
    if diff > 10:
        print("  ✅ Ethical constraints appear to HELP survival (possibly by reducing conflict)")
    elif diff < -10:
        print("  ⚠️  Ethical constraints appear to HINDER survival (trade-off hypothesis)")
    else:
        print("  ℹ️  Minimal difference - ethical constraints have little survival impact")
    
    if 'dual_process' in stats_b:
        veto_rate = stats_b['dual_process']['avg_veto_rate'] * 100
        print(f"\n  Ethical network vetoed {veto_rate:.1f}% of survival-driven actions")


def main():
    """Run the demonstration."""
    print_header("Ethical SNN Research - Quick Demo")
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{Fore.YELLOW}Device: {Style.RESET_ALL}{device}")
    if device.type == 'cuda':
        print(f"{Fore.YELLOW}GPU: {Style.RESET_ALL}{torch.cuda.get_device_name(0)}")
    print()
    
    # Load configuration
    config_path = Path(__file__).parent.parent / 'experiments' / 'phase1' / 'config_phase1.yaml'
    print(f"{Fore.YELLOW}Loading config from:{Style.RESET_ALL} {config_path}")
    base_config = load_config(str(config_path))
    
    # Reduce for demo speed
    demo_config = {
        'grid_size': base_config['environment']['grid_size'],
        'num_organisms': base_config['simulation']['num_organisms'],
        'num_food': base_config['environment']['num_food'],
        'max_timesteps': 500,  # Shorter for demo
        'energy_decay_rate': base_config['organism']['energy_decay_rate'],
        'food_energy': base_config['environment']['food_energy_value'],
        'food_respawn_rate': base_config['environment']['food_respawn_rate']
    }
    
    print_config(demo_config)
    
    # Run Condition A
    print_header("Running Condition A: Survival-Only")
    print("Organisms have only survival instincts (SNN-S)\n")
    
    config_a = demo_config.copy()
    config_a['condition'] = 'A'
    
    runner_a = SimulationRunner(config_a, seed=42)
    stats_a = runner_a.run()
    
    print_results('A', stats_a)
    
    # Small delay
    time.sleep(1)
    
    # Run Condition B
    print_header("Running Condition B: Dual-Process (Survival + Ethics)")
    print("Organisms have survival (SNN-S) + ethical evaluation (SNN-E)\n")
    
    config_b = demo_config.copy()
    config_b['condition'] = 'B'
    
    runner_b = SimulationRunner(config_b, seed=42)
    stats_b = runner_b.run()
    
    print_results('B', stats_b)
    
    # Comparison
    compare_conditions(stats_a, stats_b)
    
    # Next steps
    print_header("Next Steps")
    print(f"{Fore.YELLOW}1. Run batch experiments:{Style.RESET_ALL}")
    print("   uv run python experiments/phase1/batch_runner.py --num_runs 100")
    print()
    print(f"{Fore.YELLOW}2. Analyze results:{Style.RESET_ALL}")
    print("   uv run python analysis/phase1/analyze_results.py")
    print()
    print(f"{Fore.YELLOW}3. Explore notebooks:{Style.RESET_ALL}")
    print("   jupyter lab notebooks/")
    print()
    print(f"{Fore.YELLOW}4. Read documentation:{Style.RESET_ALL}")
    print("   See docs/user_guide.md for detailed usage")
    print()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Fore.RED}Demo interrupted by user{Style.RESET_ALL}")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n{Fore.RED}Error: {e}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
