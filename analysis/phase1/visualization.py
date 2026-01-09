"""Visualization for Phase 1 analysis."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Optional


sns.set_style("whitegrid")
sns.set_palette("husl")


def plot_metric_comparison(df_a: pd.DataFrame, df_b: pd.DataFrame, 
                          metric: str, save_path: Optional[str] = None):
    """Plot comparison of a metric between conditions.
    
    Args:
        df_a: DataFrame for condition A
        df_b: DataFrame for condition B
        metric: Metric column name
        save_path: Path to save figure (optional)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Box plot
    data = pd.DataFrame({
        'Condition': ['Survival-Only'] * len(df_a) + ['Dual-Process'] * len(df_b),
        metric: list(df_a[metric]) + list(df_b[metric])
    })
    
    sns.boxplot(data=data, x='Condition', y=metric, ax=axes[0])
    axes[0].set_title(f'{metric} by Condition')
    axes[0].set_ylabel(metric.replace('_', ' ').title())
    
    # Violin plot
    sns.violinplot(data=data, x='Condition', y=metric, ax=axes[1])
    axes[1].set_title(f'{metric} Distribution')
    axes[1].set_ylabel(metric.replace('_', ' ').title())
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_all_metrics(df_a: pd.DataFrame, df_b: pd.DataFrame,
                    metrics: List[str], save_path: Optional[str] = None):
    """Plot comparison of all metrics.
    
    Args:
        df_a: DataFrame for condition A
        df_b: DataFrame for condition B
        metrics: List of metric column names
        save_path: Path to save figure (optional)
    """
    n_metrics = len(metrics)
    n_cols = 2
    n_rows = (n_metrics + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        data = pd.DataFrame({
            'Condition': ['Survival-Only'] * len(df_a) + ['Dual-Process'] * len(df_b),
            metric: list(df_a[metric]) + list(df_b[metric])
        })
        
        sns.boxplot(data=data, x='Condition', y=metric, ax=axes[i])
        axes[i].set_title(metric.replace('_', ' ').title())
        axes[i].set_ylabel('')
    
    # Hide unused subplots
    for i in range(n_metrics, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_timeseries_comparison(timeseries_a: List[pd.DataFrame],
                               timeseries_b: List[pd.DataFrame],
                               metric: str = 'avg_energy',
                               save_path: Optional[str] = None):
    """Plot time series comparison between conditions.
    
    Args:
        timeseries_a: List of time series DataFrames for condition A
        timeseries_b: List of time series DataFrames for condition B
        metric: Metric to plot
        save_path: Path to save figure (optional)
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Calculate mean and std for condition A
    all_data_a = pd.concat(timeseries_a)
    mean_a = all_data_a.groupby('timestep')[metric].mean()
    std_a = all_data_a.groupby('timestep')[metric].std()
    
    # Calculate mean and std for condition B
    all_data_b = pd.concat(timeseries_b)
    mean_b = all_data_b.groupby('timestep')[metric].mean()
    std_b = all_data_b.groupby('timestep')[metric].std()
    
    # Plot
    ax.plot(mean_a.index, mean_a.values, label='Survival-Only', linewidth=2)
    ax.fill_between(mean_a.index, 
                    mean_a.values - std_a.values,
                    mean_a.values + std_a.values,
                    alpha=0.3)
    
    ax.plot(mean_b.index, mean_b.values, label='Dual-Process', linewidth=2)
    ax.fill_between(mean_b.index,
                    mean_b.values - std_b.values,
                    mean_b.values + std_b.values,
                    alpha=0.3)
    
    ax.set_xlabel('Timestep')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(f'{metric.replace("_", " ").title()} Over Time (Mean ± SD)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_effect_sizes(comparison_df: pd.DataFrame, save_path: Optional[str] = None):
    """Plot effect sizes for all comparisons.
    
    Args:
        comparison_df: DataFrame with comparison results
        save_path: Path to save figure (optional)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = comparison_df['metric'].values
    cohens_d = comparison_df['cohens_d'].values
    
    colors = ['green' if d > 0 else 'red' for d in cohens_d]
    
    ax.barh(metrics, cohens_d, color=colors, alpha=0.7)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax.axvline(x=0.2, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.axvline(x=-0.2, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.axvline(x=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.axvline(x=-0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.axvline(x=0.8, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.axvline(x=-0.8, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    ax.set_xlabel("Cohen's d (Effect Size)")
    ax.set_title("Effect Sizes: Dual-Process vs Survival-Only\n"
                "Positive = Dual-Process higher")
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_summary_figure(df_a: pd.DataFrame, df_b: pd.DataFrame,
                         comparison_df: pd.DataFrame,
                         save_path: Optional[str] = None):
    """Create comprehensive summary figure.
    
    Args:
        df_a: DataFrame for condition A
        df_b: DataFrame for condition B
        comparison_df: DataFrame with comparison results
        save_path: Path to save figure (optional)
    """
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Key metrics
    key_metrics = ['final_alive', 'final_avg_energy', 'final_avg_age']
    
    for i, metric in enumerate(key_metrics):
        ax = fig.add_subplot(gs[0, i])
        
        data = pd.DataFrame({
            'Condition': ['A'] * len(df_a) + ['B'] * len(df_b),
            metric: list(df_a[metric]) + list(df_b[metric])
        })
        
        sns.boxplot(data=data, x='Condition', y=metric, ax=ax)
        ax.set_title(metric.replace('_', ' ').title())
    
    # Effect sizes
    ax = fig.add_subplot(gs[1, :])
    metrics = comparison_df['metric'].values
    cohens_d = comparison_df['cohens_d'].values
    colors = ['green' if d > 0 else 'red' for d in cohens_d]
    
    ax.barh(metrics, cohens_d, color=colors, alpha=0.7)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel("Cohen's d")
    ax.set_title("Effect Sizes")
    ax.grid(True, alpha=0.3, axis='x')
    
    # Summary statistics table
    ax = fig.add_subplot(gs[2, :])
    ax.axis('off')
    
    table_data = []
    for _, row in comparison_df.iterrows():
        table_data.append([
            row['metric'],
            f"{row['group_a_mean']:.2f}",
            f"{row['group_b_mean']:.2f}",
            f"{row['p_value']:.4f}",
            "✓" if row['significant'] else "✗"
        ])
    
    table = ax.table(cellText=table_data,
                    colLabels=['Metric', 'Condition A', 'Condition B', 'p-value', 'Sig.'],
                    cellLoc='center',
                    loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
