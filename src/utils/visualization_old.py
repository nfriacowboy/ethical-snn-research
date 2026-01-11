"""Visualization utilities."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Any
import matplotlib.patches as patches


def plot_grid_world(grid: np.ndarray, organism_positions: List[tuple], 
                   food_positions: List[tuple], timestep: int = 0):
    """Plot grid world state.
    
    Args:
        grid: Grid array
        organism_positions: List of (x, y) organism positions
        food_positions: List of (x, y) food positions
        timestep: Current timestep
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Draw grid
    ax.imshow(grid.T, cmap='Greys', alpha=0.3, origin='lower')
    
    # Draw food
    if food_positions:
        food_x, food_y = zip(*food_positions)
        ax.scatter(food_x, food_y, c='green', s=100, marker='s', 
                  label='Food', alpha=0.7)
    
    # Draw organisms
    if organism_positions:
        org_x, org_y = zip(*organism_positions)
        ax.scatter(org_x, org_y, c='red', s=200, marker='o', 
                  label='Organisms', alpha=0.8)
    
    ax.set_xlim(-0.5, grid.shape[0] - 0.5)
    ax.set_ylim(-0.5, grid.shape[1] - 0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Grid World - Timestep {timestep}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_energy_over_time(organism_histories: List[List[Dict]], save_path: str = None):
    """Plot organism energy levels over time.
    
    Args:
        organism_histories: List of organism history lists
        save_path: Path to save figure (optional)
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, history in enumerate(organism_histories):
        if not history:
            continue
        
        timesteps = [h['age'] for h in history]
        energies = [h['energy'] for h in history]
        
        ax.plot(timesteps, energies, label=f'Organism {i}', alpha=0.7)
    
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Energy')
    ax.set_title('Organism Energy Over Time')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_population_statistics(stats_history: List[Dict], save_path: str = None):
    """Plot population-level statistics.
    
    Args:
        stats_history: List of statistics dictionaries
        save_path: Path to save figure (optional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    timesteps = list(range(len(stats_history)))
    
    # Alive count
    alive_counts = [s['alive_count'] for s in stats_history]
    axes[0, 0].plot(timesteps, alive_counts, linewidth=2)
    axes[0, 0].set_xlabel('Timestep')
    axes[0, 0].set_ylabel('Alive Count')
    axes[0, 0].set_title('Population Size Over Time')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Average energy
    avg_energies = [s['avg_energy'] for s in stats_history]
    axes[0, 1].plot(timesteps, avg_energies, linewidth=2, color='orange')
    axes[0, 1].set_xlabel('Timestep')
    axes[0, 1].set_ylabel('Average Energy')
    axes[0, 1].set_title('Average Energy Over Time')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Average age
    avg_ages = [s['avg_age'] for s in stats_history]
    axes[1, 0].plot(timesteps, avg_ages, linewidth=2, color='green')
    axes[1, 0].set_xlabel('Timestep')
    axes[1, 0].set_ylabel('Average Age')
    axes[1, 0].set_title('Average Age Over Time')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Summary text
    axes[1, 1].axis('off')
    summary_text = f"""
    Final Statistics:
    
    Alive: {alive_counts[-1]}/{stats_history[0]['total_organisms']}
    Avg Energy: {avg_energies[-1]:.2f}
    Avg Age: {avg_ages[-1]:.2f}
    Max Age: {max(avg_ages):.2f}
    """
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=14, 
                   verticalalignment='center', family='monospace')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_comparison(condition_a_data: Dict, condition_b_data: Dict, 
                   metric: str = 'avg_energy', save_path: str = None):
    """Plot comparison between two conditions.
    
    Args:
        condition_a_data: Data from condition A
        condition_b_data: Data from condition B
        metric: Metric to compare
        save_path: Path to save figure (optional)
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Extract metric over time
    timesteps_a = list(range(len(condition_a_data)))
    values_a = [d[metric] for d in condition_a_data]
    
    timesteps_b = list(range(len(condition_b_data)))
    values_b = [d[metric] for d in condition_b_data]
    
    ax.plot(timesteps_a, values_a, label='Survival Only', linewidth=2, alpha=0.8)
    ax.plot(timesteps_b, values_b, label='Dual Process', linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Timestep')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(f'Comparison: {metric.replace("_", " ").title()}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
