"""Metrics calculation utilities."""

from typing import Any, Dict, List

import numpy as np
from scipy import stats


def calculate_survival_metrics(
    organism_histories: List[List[Dict]],
) -> Dict[str, float]:
    """Calculate survival-related metrics.

    Args:
        organism_histories: List of organism history lists

    Returns:
        Dictionary with survival metrics
    """
    lifespans = []
    final_energies = []
    total_food_collected = []

    for history in organism_histories:
        if not history:
            continue

        lifespan = len(history)
        lifespans.append(lifespan)

        final_energy = history[-1]["energy"]
        final_energies.append(final_energy)

        # Calculate food collected (energy gains)
        food_collected = 0
        for i in range(1, len(history)):
            energy_delta = history[i]["energy"] - history[i - 1]["energy"]
            if energy_delta > 0:
                food_collected += energy_delta
        total_food_collected.append(food_collected)

    return {
        "mean_lifespan": np.mean(lifespans) if lifespans else 0,
        "std_lifespan": np.std(lifespans) if lifespans else 0,
        "mean_final_energy": np.mean(final_energies) if final_energies else 0,
        "mean_food_collected": (
            np.mean(total_food_collected) if total_food_collected else 0
        ),
        "survival_rate": (
            sum(1 for e in final_energies if e > 0) / len(final_energies)
            if final_energies
            else 0
        ),
    }


def calculate_behavioral_metrics(
    organism_histories: List[List[Dict]],
) -> Dict[str, float]:
    """Calculate behavioral metrics.

    Args:
        organism_histories: List of organism history lists

    Returns:
        Dictionary with behavioral metrics
    """
    total_movements = []
    movement_patterns = []

    for history in organism_histories:
        if len(history) < 2:
            continue

        movements = 0
        for i in range(1, len(history)):
            prev_pos = history[i - 1]["position"]
            curr_pos = history[i]["position"]

            if prev_pos != curr_pos:
                movements += 1

                # Calculate distance moved
                dx = curr_pos[0] - prev_pos[0]
                dy = curr_pos[1] - prev_pos[1]
                distance = abs(dx) + abs(dy)
                movement_patterns.append(distance)

        total_movements.append(movements)

    return {
        "mean_movements": np.mean(total_movements) if total_movements else 0,
        "mean_movement_distance": (
            np.mean(movement_patterns) if movement_patterns else 0
        ),
        "movement_variance": np.var(movement_patterns) if movement_patterns else 0,
    }


def calculate_population_metrics(stats_history: List[Dict]) -> Dict[str, float]:
    """Calculate population-level metrics.

    Args:
        stats_history: List of population statistics

    Returns:
        Dictionary with population metrics
    """
    alive_counts = [s["alive_count"] for s in stats_history]
    avg_energies = [s["avg_energy"] for s in stats_history]

    return {
        "final_population": alive_counts[-1] if alive_counts else 0,
        "max_population": max(alive_counts) if alive_counts else 0,
        "mean_population": np.mean(alive_counts) if alive_counts else 0,
        "extinction_timestep": (
            alive_counts.index(0) if 0 in alive_counts else len(alive_counts)
        ),
        "mean_energy_overall": np.mean(avg_energies) if avg_energies else 0,
    }


def compare_conditions(
    condition_a_metrics: Dict[str, float], condition_b_metrics: Dict[str, float]
) -> Dict[str, Any]:
    """Statistical comparison between two conditions.

    Args:
        condition_a_metrics: Metrics from condition A
        condition_b_metrics: Metrics from condition B

    Returns:
        Dictionary with comparison statistics
    """
    comparisons = {}

    for metric in condition_a_metrics.keys():
        if metric not in condition_b_metrics:
            continue

        val_a = condition_a_metrics[metric]
        val_b = condition_b_metrics[metric]

        # Calculate effect size (Cohen's d)
        # Note: This assumes metrics are means from multiple runs
        # For proper analysis, you'd need the raw data distributions
        pooled_std = 1.0  # Placeholder
        cohens_d = (val_b - val_a) / pooled_std if pooled_std > 0 else 0

        comparisons[metric] = {
            "condition_a": val_a,
            "condition_b": val_b,
            "difference": val_b - val_a,
            "percent_change": ((val_b - val_a) / val_a * 100) if val_a != 0 else 0,
            "cohens_d": cohens_d,
        }

    return comparisons


def mann_whitney_test(sample_a: List[float], sample_b: List[float]) -> Dict[str, float]:
    """Perform Mann-Whitney U test.

    Args:
        sample_a: Sample from condition A
        sample_b: Sample from condition B

    Returns:
        Dictionary with test results
    """
    if len(sample_a) < 2 or len(sample_b) < 2:
        return {"statistic": 0, "p_value": 1.0, "significant": False}

    statistic, p_value = stats.mannwhitneyu(sample_a, sample_b, alternative="two-sided")

    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
        "mean_a": np.mean(sample_a),
        "mean_b": np.mean(sample_b),
        "median_a": np.median(sample_a),
        "median_b": np.median(sample_b),
    }


def cohens_d(sample_a: List[float], sample_b: List[float]) -> float:
    """Calculate Cohen's d effect size.

    Args:
        sample_a: Sample from condition A
        sample_b: Sample from condition B

    Returns:
        Cohen's d value
    """
    mean_a = np.mean(sample_a)
    mean_b = np.mean(sample_b)

    std_a = np.std(sample_a, ddof=1)
    std_b = np.std(sample_b, ddof=1)

    n_a = len(sample_a)
    n_b = len(sample_b)

    # Pooled standard deviation
    pooled_std = np.sqrt(
        ((n_a - 1) * std_a**2 + (n_b - 1) * std_b**2) / (n_a + n_b - 2)
    )

    if pooled_std == 0:
        return 0.0

    d = (mean_b - mean_a) / pooled_std

    return float(d)
