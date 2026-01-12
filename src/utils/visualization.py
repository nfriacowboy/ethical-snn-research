"""Visualization utilities for simulation analysis.

This module provides functions for visualizing simulation data including:
- Static grid plots showing organism and food positions
- Energy over time plots
- Animated visualizations from HDF5 log files

Example:
    >>> from src.utils.visualization import plot_grid, plot_energy_over_time
    >>> fig = plot_grid(environment, organisms)
    >>> fig.savefig('grid_snapshot.png')
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Circle, Rectangle

# Set seaborn style for better looking plots
sns.set_style("whitegrid")
sns.set_palette("husl")


def plot_grid(
    environment: Any,
    organisms: List[Any],
    title: str = "Grid World State",
    figsize: Tuple[int, int] = (10, 10),
    show_grid: bool = True,
) -> plt.Figure:
    """Plot static snapshot of grid world with organisms and food.

    Args:
        environment: GridWorld instance with food positions
        organisms: List of Organism instances with positions and energies
        title: Plot title
        figsize: Figure size (width, height)
        show_grid: Whether to show grid lines

    Returns:
        Matplotlib figure object

    Example:
        >>> fig = plot_grid(env, organisms)
        >>> fig.savefig('snapshot.png')
    """
    fig, ax = plt.subplots(figsize=figsize)

    grid_size = environment.grid_size

    # Set up axes
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)
    ax.set_aspect("equal")

    # Background
    ax.set_facecolor("#f0f0f0")

    # Draw grid lines
    if show_grid:
        for i in range(grid_size + 1):
            ax.axhline(i - 0.5, color="gray", linewidth=0.5, alpha=0.3)
            ax.axvline(i - 0.5, color="gray", linewidth=0.5, alpha=0.3)

    # Draw food as green squares
    food_positions = environment.food_positions
    if len(food_positions) > 0:
        for fx, fy in food_positions:
            rect = Rectangle(
                (fx - 0.4, fy - 0.4),
                0.8,
                0.8,
                facecolor="green",
                edgecolor="darkgreen",
                linewidth=2,
                alpha=0.7,
            )
            ax.add_patch(rect)

    # Draw organisms as colored circles
    colors = plt.cm.tab10(np.linspace(0, 1, len(organisms)))

    for idx, (organism, color) in enumerate(zip(organisms, colors)):
        if organism.is_alive():
            x, y = organism.position

            # Circle for organism
            circle = Circle(
                (x, y), 0.35, facecolor=color, edgecolor="black", linewidth=2, alpha=0.8
            )
            ax.add_patch(circle)

            # Energy text
            energy_pct = (organism.energy / organism.max_energy) * 100
            ax.text(
                x,
                y - 0.7,
                f"{energy_pct:.0f}%",
                ha="center",
                va="top",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
            )

            # Organism ID
            ax.text(
                x,
                y,
                str(idx),
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="white",
            )

    # Legend
    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            label="Food",
            markerfacecolor="green",
            markersize=10,
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Organism",
            markerfacecolor="red",
            markersize=10,
        ),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("X Position", fontsize=12)
    ax.set_ylabel("Y Position", fontsize=12)

    plt.tight_layout()
    return fig


def plot_energy_over_time(
    log_file: str,
    organism_ids: Optional[List[int]] = None,
    figsize: Tuple[int, int] = (12, 6),
) -> plt.Figure:
    """Plot energy levels over time from HDF5 log file.

    Args:
        log_file: Path to HDF5 simulation log file
        organism_ids: List of organism IDs to plot (None = all)
        figsize: Figure size (width, height)

    Returns:
        Matplotlib figure object

    Example:
        >>> fig = plot_energy_over_time('results/simulation_001.hdf5')
        >>> fig.savefig('energy_plot.png')
    """
    fig, ax = plt.subplots(figsize=figsize)

    with h5py.File(log_file, "r") as f:
        num_organisms = len(f["organisms"].keys())

        if organism_ids is None:
            organism_ids = list(range(num_organisms))

        colors = plt.cm.tab10(np.linspace(0, 1, len(organism_ids)))

        for idx, org_id in enumerate(organism_ids):
            org_key = f"organism_{org_id}"
            if org_key in f["organisms"]:
                energies = f["organisms"][org_key]["energies"][:]
                timesteps = np.arange(len(energies))
                alive = f["organisms"][org_key]["alive"][:]

                # Plot energy line
                ax.plot(
                    timesteps,
                    energies,
                    label=f"Organism {org_id}",
                    color=colors[idx],
                    linewidth=2,
                    alpha=0.8,
                )

                # Mark death point if applicable
                if not alive[-1]:
                    death_timestep = np.where(~alive)[0][0]
                    ax.scatter(
                        death_timestep,
                        energies[death_timestep],
                        color=colors[idx],
                        s=100,
                        marker="x",
                        linewidths=3,
                        zorder=5,
                    )
                    ax.annotate(
                        "Death",
                        (death_timestep, energies[death_timestep]),
                        xytext=(10, 10),
                        textcoords="offset points",
                        fontsize=9,
                        color=colors[idx],
                    )

    ax.set_xlabel("Timestep", fontsize=12)
    ax.set_ylabel("Energy", fontsize=12)
    ax.set_title("Organism Energy Over Time", fontsize=14, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def create_animation(
    log_file: str,
    output_file: str = "simulation.mp4",
    fps: int = 10,
    interval: int = 100,
    figsize: Tuple[int, int] = (10, 10),
) -> str:
    """Create animated video from HDF5 simulation log.

    Args:
        log_file: Path to HDF5 simulation log file
        output_file: Output video file path
        fps: Frames per second for video
        interval: Milliseconds between frames
        figsize: Figure size (width, height)

    Returns:
        Path to created animation file

    Example:
        >>> create_animation('results/simulation_001.hdf5', 'animation.mp4')
        'animation.mp4'
    """
    # Read all data from HDF5
    with h5py.File(log_file, "r") as f:
        # Get grid size from config or default
        config_str = f.attrs.get("config", "{}")
        if isinstance(config_str, bytes):
            config_str = config_str.decode("utf-8")

        import json

        try:
            config = json.loads(config_str)
            grid_size = config.get("environment", {}).get("grid_size", 20)
        except:
            grid_size = 20

        num_organisms = len(f["organisms"].keys())
        total_timesteps = len(f["organisms"]["organism_0"]["positions"])

        # Load all organism data
        organism_data = []
        for org_id in range(num_organisms):
            org_key = f"organism_{org_id}"
            organism_data.append(
                {
                    "positions": f["organisms"][org_key]["positions"][:],
                    "energies": f["organisms"][org_key]["energies"][:],
                    "alive": f["organisms"][org_key]["alive"][:],
                }
            )

        # Load food data (padded array)
        food_data = f["environment"]["food_positions"][:]

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.tab10(np.linspace(0, 1, num_organisms))

    def init():
        """Initialize animation."""
        ax.clear()
        ax.set_xlim(-0.5, grid_size - 0.5)
        ax.set_ylim(-0.5, grid_size - 0.5)
        ax.set_aspect("equal")
        ax.set_facecolor("#f0f0f0")
        return []

    def animate(frame):
        """Update function for animation."""
        ax.clear()
        ax.set_xlim(-0.5, grid_size - 0.5)
        ax.set_ylim(-0.5, grid_size - 0.5)
        ax.set_aspect("equal")
        ax.set_facecolor("#f0f0f0")

        # Draw grid
        for i in range(grid_size + 1):
            ax.axhline(i - 0.5, color="gray", linewidth=0.5, alpha=0.3)
            ax.axvline(i - 0.5, color="gray", linewidth=0.5, alpha=0.3)

        # Draw food
        frame_food = food_data[frame]
        for fx, fy in frame_food:
            if fx > 0 or fy > 0:  # Check for valid food (padding uses 0,0)
                rect = Rectangle(
                    (fx - 0.4, fy - 0.4),
                    0.8,
                    0.8,
                    facecolor="green",
                    edgecolor="darkgreen",
                    linewidth=2,
                    alpha=0.7,
                )
                ax.add_patch(rect)

        # Draw organisms
        for org_id, (org_data, color) in enumerate(zip(organism_data, colors)):
            if org_data["alive"][frame]:
                x, y = org_data["positions"][frame]
                energy = org_data["energies"][frame]

                # Circle for organism
                circle = Circle(
                    (x, y),
                    0.35,
                    facecolor=color,
                    edgecolor="black",
                    linewidth=2,
                    alpha=0.8,
                )
                ax.add_patch(circle)

                # Organism ID
                ax.text(
                    x,
                    y,
                    str(org_id),
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                    color="white",
                )

        ax.set_title(
            f"Simulation - Timestep {frame}/{total_timesteps-1}",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")

        return []

    # Create animation
    anim = animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=total_timesteps,
        interval=interval,
        blit=False,
        repeat=False,
    )

    # Save animation
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    Writer = animation.writers["ffmpeg"]
    writer = Writer(fps=fps, metadata=dict(artist="Ethical-SNN-Research"), bitrate=1800)
    anim.save(str(output_path), writer=writer)

    plt.close(fig)

    return str(output_path)


def plot_action_distribution(
    log_file: str, organism_id: int = 0, figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """Plot distribution of actions taken by an organism.

    Args:
        log_file: Path to HDF5 simulation log file
        organism_id: ID of organism to analyze
        figsize: Figure size (width, height)

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    with h5py.File(log_file, "r") as f:
        org_key = f"organism_{organism_id}"
        actions = f["organisms"][org_key]["actions"][:]

        # Decode byte strings if needed
        if actions.dtype.kind == "S":
            actions = [a.decode("utf-8") for a in actions]

        # Count actions
        unique_actions, counts = np.unique(actions, return_counts=True)

        # Create bar plot
        ax.bar(unique_actions, counts, color="steelblue", alpha=0.7)
        ax.set_xlabel("Action", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title(
            f"Action Distribution - Organism {organism_id}",
            fontsize=14,
            fontweight="bold",
        )
        ax.tick_params(axis="x", rotation=45)

        # Add count labels on bars
        for i, (action, count) in enumerate(zip(unique_actions, counts)):
            ax.text(i, count, str(count), ha="center", va="bottom")

    plt.tight_layout()
    return fig
