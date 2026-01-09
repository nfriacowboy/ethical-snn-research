"""Automated report generator for Phase 1."""

import os
from typing import Dict, Any
import pandas as pd
from datetime import datetime
from .preprocessing import load_condition_data, create_summary_dataframe
from .statistics import compare_conditions, survival_analysis, generate_statistical_report
from .visualization import (plot_all_metrics, plot_effect_sizes, 
                           create_summary_figure, plot_metric_comparison)


def generate_full_report(condition_a_runs: list, condition_b_runs: list,
                        output_dir: str = "analysis/phase1/reports"):
    """Generate complete analysis report for Phase 1.
    
    Args:
        condition_a_runs: List of run IDs for condition A
        condition_b_runs: List of run IDs for condition B
        output_dir: Directory to save report and figures
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("GENERATING PHASE 1 ANALYSIS REPORT")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    data_a = load_condition_data(condition_a_runs)
    data_b = load_condition_data(condition_b_runs)
    
    print(f"  Condition A: {len(data_a)} runs loaded")
    print(f"  Condition B: {len(data_b)} runs loaded")
    
    # Create dataframes
    df_a = create_summary_dataframe(data_a)
    df_b = create_summary_dataframe(data_b)
    
    # Define metrics to analyze
    metrics = ['final_alive', 'final_avg_energy', 'final_avg_age', 
               'total_food_consumed', 'final_timestep']
    
    # Statistical analysis
    print("\nPerforming statistical analysis...")
    comparison_df = compare_conditions(df_a, df_b, metrics)
    survival_results = survival_analysis(df_a, df_b)
    
    # Generate text report
    print("\nGenerating text report...")
    report_text = generate_statistical_report(df_a, df_b, metrics)
    
    # Add survival analysis to report
    report_text += "\n\nSURVIVAL ANALYSIS:\n"
    report_text += "-" * 70 + "\n"
    report_text += f"Condition A survival rate: {survival_results['survival_rate_a']:.2%}\n"
    report_text += f"Condition B survival rate: {survival_results['survival_rate_b']:.2%}\n"
    report_text += f"Chi-square test: χ² = {survival_results['chi2_statistic']:.3f}, "
    report_text += f"p = {survival_results['p_value']:.4f}"
    report_text += " *" if survival_results['significant'] else ""
    report_text += "\n"
    
    # Save text report
    report_path = os.path.join(output_dir, "statistical_report.txt")
    with open(report_path, 'w') as f:
        f.write(report_text)
    print(f"  Saved text report: {report_path}")
    
    # Generate figures
    print("\nGenerating figures...")
    
    # 1. All metrics comparison
    fig1_path = os.path.join(output_dir, "all_metrics_comparison.png")
    plot_all_metrics(df_a, df_b, metrics, save_path=fig1_path)
    print(f"  Saved: {fig1_path}")
    
    # 2. Effect sizes
    fig2_path = os.path.join(output_dir, "effect_sizes.png")
    plot_effect_sizes(comparison_df, save_path=fig2_path)
    print(f"  Saved: {fig2_path}")
    
    # 3. Summary figure
    fig3_path = os.path.join(output_dir, "summary_figure.png")
    create_summary_figure(df_a, df_b, comparison_df, save_path=fig3_path)
    print(f"  Saved: {fig3_path}")
    
    # 4. Individual metric plots
    for metric in ['final_alive', 'final_avg_energy']:
        metric_path = os.path.join(output_dir, f"{metric}_comparison.png")
        plot_metric_comparison(df_a, df_b, metric, save_path=metric_path)
        print(f"  Saved: {metric_path}")
    
    # Save processed data
    print("\nSaving processed data...")
    df_a.to_csv(os.path.join(output_dir, "condition_a_summary.csv"), index=False)
    df_b.to_csv(os.path.join(output_dir, "condition_b_summary.csv"), index=False)
    comparison_df.to_csv(os.path.join(output_dir, "statistical_comparisons.csv"), index=False)
    
    # Generate HTML report
    print("\nGenerating HTML report...")
    html_report = generate_html_report(comparison_df, survival_results, output_dir)
    html_path = os.path.join(output_dir, "report.html")
    with open(html_path, 'w') as f:
        f.write(html_report)
    print(f"  Saved: {html_path}")
    
    print("\n" + "=" * 70)
    print("REPORT GENERATION COMPLETE")
    print("=" * 70)
    print(f"\nAll outputs saved to: {output_dir}")
    print(f"View the full report at: {html_path}")


def generate_html_report(comparison_df: pd.DataFrame, 
                        survival_results: Dict[str, Any],
                        output_dir: str) -> str:
    """Generate HTML report.
    
    Args:
        comparison_df: DataFrame with comparison results
        survival_results: Survival analysis results
        output_dir: Output directory (for relative image paths)
        
    Returns:
        HTML string
    """
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Phase 1 Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #2c3e50; }}
            h2 {{ color: #34495e; margin-top: 30px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #3498db; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .significant {{ color: #27ae60; font-weight: bold; }}
            .not-significant {{ color: #e74c3c; }}
            img {{ max-width: 100%; margin: 20px 0; }}
            .summary {{ background-color: #ecf0f1; padding: 20px; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <h1>Phase 1 Analysis Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="summary">
            <h2>Summary</h2>
            <p>This report presents the statistical analysis of Phase 1 experiments comparing
            survival-only (Condition A) and dual-process (Condition B) architectures.</p>
        </div>
        
        <h2>Survival Analysis</h2>
        <p>
            <strong>Condition A survival rate:</strong> {survival_results['survival_rate_a']:.2%}<br>
            <strong>Condition B survival rate:</strong> {survival_results['survival_rate_b']:.2%}<br>
            <strong>Chi-square test:</strong> χ² = {survival_results['chi2_statistic']:.3f}, 
            p = {survival_results['p_value']:.4f}
            <span class="{'significant' if survival_results['significant'] else 'not-significant'}">
                {'(Significant)' if survival_results['significant'] else '(Not significant)'}
            </span>
        </p>
        
        <h2>Metric Comparisons</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Condition A Mean</th>
                <th>Condition B Mean</th>
                <th>p-value</th>
                <th>Cohen's d</th>
                <th>Significant</th>
            </tr>
    """
    
    for _, row in comparison_df.iterrows():
        sig_class = 'significant' if row['significant'] else 'not-significant'
        html += f"""
            <tr>
                <td>{row['metric']}</td>
                <td>{row['group_a_mean']:.3f}</td>
                <td>{row['group_b_mean']:.3f}</td>
                <td>{row['p_value']:.4f}</td>
                <td>{row['cohens_d']:.3f}</td>
                <td class="{sig_class}">{'Yes' if row['significant'] else 'No'}</td>
            </tr>
        """
    
    html += """
        </table>
        
        <h2>Visualizations</h2>
        <h3>All Metrics Comparison</h3>
        <img src="all_metrics_comparison.png" alt="All metrics comparison">
        
        <h3>Effect Sizes</h3>
        <img src="effect_sizes.png" alt="Effect sizes">
        
        <h3>Summary Figure</h3>
        <img src="summary_figure.png" alt="Summary figure">
        
    </body>
    </html>
    """
    
    return html
