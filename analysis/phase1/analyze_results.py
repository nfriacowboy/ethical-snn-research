"""Analyze Phase 1 batch experiment results.

This script loads batch experiment results, performs statistical analysis,
and generates visualizations for the Phase 1 preregistered study.

Example:
    python analysis/phase1/analyze_results.py --input results/batch_experiments
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List

from analysis.phase1.statistical_tests import mann_whitney_test, cohens_d
from analysis.phase1.visualization import (
    plot_metric_comparison,
    plot_all_metrics
)


def load_batch_results(results_dir: Path) -> Dict[str, Any]:
    """Load batch experiment results from JSON files.
    
    Args:
        results_dir: Directory containing batch results
        
    Returns:
        Dictionary with condition_a, condition_b, and aggregated data
    """
    # Load aggregated results
    aggregated_file = results_dir / 'batch_aggregated.json'
    with open(aggregated_file, 'r') as f:
        aggregated = json.load(f)
    
    # Load individual runs
    condition_a_file = results_dir / 'condition_a_runs.json'
    condition_b_file = results_dir / 'condition_b_runs.json'
    
    condition_a_runs = []
    condition_b_runs = []
    
    if condition_a_file.exists():
        with open(condition_a_file, 'r') as f:
            condition_a_runs = json.load(f)
    
    if condition_b_file.exists():
        with open(condition_b_file, 'r') as f:
            condition_b_runs = json.load(f)
    
    return {
        'aggregated': aggregated,
        'condition_a': condition_a_runs,
        'condition_b': condition_b_runs
    }


def extract_metrics_dataframe(runs: List[Dict[str, Any]], condition: str) -> pd.DataFrame:
    """Extract metrics from runs into a pandas DataFrame.
    
    Args:
        runs: List of run statistics
        condition: 'A' or 'B'
        
    Returns:
        DataFrame with metrics per run
    """
    data = []
    
    for run in runs:
        row = {
            'condition': condition,
            'seed': run['seed'],
            'final_timestep': run['final_timestep'],
            'alive_count': run['organisms']['alive'],
            'dead_count': run['organisms']['dead'],
            'avg_survival_time': run['organisms']['avg_survival_time'],
            'final_energy_avg': run['energy']['final_avg'],
            'final_energy_std': run['energy']['final_std']
        }
        
        # Add dual-process metrics if available
        if 'dual_process' in run:
            row['total_vetoes'] = run['dual_process']['total_vetoes']
            row['total_approvals'] = run['dual_process']['total_approvals']
            row['avg_veto_rate'] = run['dual_process']['avg_veto_rate']
        
        data.append(row)
    
    return pd.DataFrame(data)


def perform_statistical_tests(df_a: pd.DataFrame, df_b: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Perform statistical tests comparing conditions.
    
    Args:
        df_a: DataFrame for Condition A
        df_b: DataFrame for Condition B
        
    Returns:
        Dictionary of test results for each metric
    """
    metrics_to_test = [
        'avg_survival_time',
        'alive_count',
        'final_energy_avg'
    ]
    
    results = {}
    
    for metric in metrics_to_test:
        if metric in df_a.columns and metric in df_b.columns:
            group_a = df_a[metric].values
            group_b = df_b[metric].values
            
            # Mann-Whitney U test
            test_result = mann_whitney_test(group_a, group_b, metric_name=metric)
            
            # Add effect size
            test_result['cohens_d'] = cohens_d(group_a, group_b)
            
            results[metric] = test_result
    
    return results


def print_summary(data: Dict[str, Any]):
    """Print summary of analysis results.
    
    Args:
        data: Dictionary with aggregated results
    """
    print("\n" + "=" * 70)
    print("PHASE 1 BATCH EXPERIMENT ANALYSIS")
    print("=" * 70)
    
    aggregated = data['aggregated']
    
    # Condition A summary
    if 'condition_a' in aggregated:
        print("\nCondition A (Survival-Only):")
        print(f"  Number of runs: {aggregated['condition_a']['num_runs']}")
        print(f"  Avg survival time: {aggregated['condition_a']['survival_time']['mean']:.2f} ± {aggregated['condition_a']['survival_time']['std']:.2f}")
        print(f"  Avg final alive: {aggregated['condition_a']['final_alive']['mean']:.2f} ± {aggregated['condition_a']['final_alive']['std']:.2f}")
        print(f"  Avg final energy: {aggregated['condition_a']['final_energy']['mean']:.2f} ± {aggregated['condition_a']['final_energy']['std']:.2f}")
    
    # Condition B summary
    if 'condition_b' in aggregated and aggregated['condition_b']:
        print("\nCondition B (Dual-Process):")
        print(f"  Number of runs: {aggregated['condition_b']['num_runs']}")
        print(f"  Avg survival time: {aggregated['condition_b']['survival_time']['mean']:.2f} ± {aggregated['condition_b']['survival_time']['std']:.2f}")
        print(f"  Avg final alive: {aggregated['condition_b']['final_alive']['mean']:.2f} ± {aggregated['condition_b']['final_alive']['std']:.2f}")
        print(f"  Avg final energy: {aggregated['condition_b']['final_energy']['mean']:.2f} ± {aggregated['condition_b']['final_energy']['std']:.2f}")
        
        if 'dual_process' in aggregated['condition_b']:
            print(f"  Avg veto rate: {aggregated['condition_b']['dual_process']['veto_rate_mean']:.2%} ± {aggregated['condition_b']['dual_process']['veto_rate_std']:.2%}")
    
    print("=" * 70)


def print_statistical_results(test_results: Dict[str, Dict[str, Any]]):
    """Print statistical test results.
    
    Args:
        test_results: Dictionary of test results
    """
    print("\n" + "=" * 70)
    print("STATISTICAL TESTS")
    print("=" * 70)
    
    for metric, result in test_results.items():
        print(f"\n{metric.replace('_', ' ').title()}:")
        print(f"  Test: {result['test']}")
        print(f"  Statistic: {result['statistic']:.4f}")
        print(f"  P-value: {result['p_value']:.4f}")
        print(f"  Significant: {'Yes' if result['significant'] else 'No'} (α=0.05)")
        print(f"  Cohen's d: {result['cohens_d']:.4f}")
        print(f"  Condition A: {result['group_a_mean']:.2f} ± {result['group_a_std']:.2f} (n={result['n_a']})")
        print(f"  Condition B: {result['group_b_mean']:.2f} ± {result['group_b_std']:.2f} (n={result['n_b']})")
    
    print("=" * 70)


def create_visualizations(df_a: pd.DataFrame, df_b: pd.DataFrame, output_dir: Path):
    """Create all visualizations.
    
    Args:
        df_a: DataFrame for Condition A
        df_b: DataFrame for Condition B
        output_dir: Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating visualizations...")
    
    # Survival time comparison
    fig = plot_metric_comparison(
        df_a, df_b, 
        'avg_survival_time',
        save_path=output_dir / 'survival_time_comparison.png'
    )
    plt.close(fig)
    print(f"  ✓ Saved: {output_dir / 'survival_time_comparison.png'}")
    
    # Alive count comparison
    fig = plot_metric_comparison(
        df_a, df_b,
        'alive_count',
        save_path=output_dir / 'alive_count_comparison.png'
    )
    plt.close(fig)
    print(f"  ✓ Saved: {output_dir / 'alive_count_comparison.png'}")
    
    # Energy comparison
    fig = plot_metric_comparison(
        df_a, df_b,
        'final_energy_avg',
        save_path=output_dir / 'energy_comparison.png'
    )
    plt.close(fig)
    print(f"  ✓ Saved: {output_dir / 'energy_comparison.png'}")
    
    # All metrics overview
    metrics = ['avg_survival_time', 'alive_count', 'final_energy_avg']
    fig = plot_all_metrics(
        df_a, df_b,
        metrics,
        save_path=output_dir / 'all_metrics_overview.png'
    )
    plt.close(fig)
    print(f"  ✓ Saved: {output_dir / 'all_metrics_overview.png'}")
    
    # Dual-process specific plot
    if 'avg_veto_rate' in df_b.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(df_b['avg_veto_rate'], bins=20, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Average Veto Rate')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Veto Rates (Condition B)')
        plt.tight_layout()
        plt.savefig(output_dir / 'veto_rate_distribution.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Saved: {output_dir / 'veto_rate_distribution.png'}")


def save_analysis_report(
    data: Dict[str, Any],
    test_results: Dict[str, Dict[str, Any]],
    output_dir: Path
):
    """Save analysis report to JSON.
    
    Args:
        data: Loaded batch results
        test_results: Statistical test results
        output_dir: Directory to save report
    """
    # Convert numpy bool to Python bool
    def convert_to_json_serializable(obj):
        """Convert numpy types to Python native types."""
        import numpy as np
        if isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(v) for v in obj]
        elif isinstance(obj, (np.bool_, np.bool)):
            return bool(obj)
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    report = {
        'aggregated_results': data['aggregated'],
        'statistical_tests': convert_to_json_serializable(test_results),
        'num_runs_a': len(data['condition_a']),
        'num_runs_b': len(data['condition_b'])
    }
    
    report_file = output_dir / 'analysis_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✓ Analysis report saved to: {report_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze Phase 1 batch experiment results'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Directory containing batch results'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/analysis_phase1',
        help='Output directory for analysis results'
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    # Load results
    print(f"Loading results from: {input_dir}")
    data = load_batch_results(input_dir)
    
    # Print summary
    print_summary(data)
    
    # Extract DataFrames
    df_a = extract_metrics_dataframe(data['condition_a'], 'A')
    df_b = extract_metrics_dataframe(data['condition_b'], 'B')
    
    # Perform statistical tests
    if len(df_a) > 0 and len(df_b) > 0:
        test_results = perform_statistical_tests(df_a, df_b)
        print_statistical_results(test_results)
        
        # Create visualizations
        create_visualizations(df_a, df_b, output_dir)
        
        # Save report
        save_analysis_report(data, test_results, output_dir)
    else:
        print("\n⚠ Insufficient data for statistical comparison")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print(f"Results saved in: {output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
