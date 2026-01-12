"""Data preprocessing for Phase 1 analysis."""

import json
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def load_run_data(run_id: int, results_dir: str = "results") -> Dict[str, Any]:
    """Load data from a single run.

    Args:
        run_id: Run ID to load
        results_dir: Results directory

    Returns:
        Dictionary with run data
    """
    run_dir = os.path.join(results_dir, f"run_{run_id:04d}")

    # Load summary
    summary_path = os.path.join(run_dir, "summary.json")
    with open(summary_path, "r") as f:
        summary = json.load(f)

    # Load timestep data
    timesteps = []
    timestep_path = os.path.join(run_dir, "timesteps.jsonl")

    if os.path.exists(timestep_path):
        with open(timestep_path, "r") as f:
            for line in f:
                timesteps.append(json.loads(line))

    return {"run_id": run_id, "summary": summary, "timesteps": timesteps}


def load_condition_data(
    run_ids: List[int], results_dir: str = "results"
) -> List[Dict[str, Any]]:
    """Load data from multiple runs (one condition).

    Args:
        run_ids: List of run IDs
        results_dir: Results directory

    Returns:
        List of run data dictionaries
    """
    condition_data = []

    for run_id in run_ids:
        try:
            run_data = load_run_data(run_id, results_dir)
            condition_data.append(run_data)
        except Exception as e:
            print(f"Warning: Could not load run {run_id}: {e}")
            continue

    return condition_data


def extract_summary_metrics(run_data: Dict[str, Any]) -> Dict[str, float]:
    """Extract key metrics from run summary.

    Args:
        run_data: Run data dictionary

    Returns:
        Dictionary with extracted metrics
    """
    summary = run_data["summary"]
    arch_stats = summary.get("architecture_stats", {})
    env_stats = summary.get("environment_stats", {})

    return {
        "run_id": run_data["run_id"],
        "final_alive": arch_stats.get("alive_count", 0),
        "final_avg_energy": arch_stats.get("avg_energy", 0),
        "final_avg_age": arch_stats.get("avg_age", 0),
        "total_organisms": arch_stats.get("total_organisms", 0),
        "total_food_spawned": env_stats.get("total_food_spawned", 0),
        "total_food_consumed": env_stats.get("total_food_consumed", 0),
        "final_timestep": summary.get("final_timestep", 0),
        "elapsed_time": summary.get("elapsed_time", 0),
    }


def create_summary_dataframe(condition_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """Create pandas DataFrame from condition data.

    Args:
        condition_data: List of run data dictionaries

    Returns:
        DataFrame with summary metrics
    """
    metrics_list = []

    for run_data in condition_data:
        metrics = extract_summary_metrics(run_data)
        metrics_list.append(metrics)

    df = pd.DataFrame(metrics_list)
    return df


def extract_timeseries_data(run_data: Dict[str, Any]) -> pd.DataFrame:
    """Extract time series data from run.

    Args:
        run_data: Run data dictionary

    Returns:
        DataFrame with time series
    """
    timesteps = run_data["timesteps"]

    timeseries = []
    for ts_data in timesteps:
        timestep = ts_data["timestep"]
        organism_states = ts_data["organism_states"]

        # Aggregate organism data
        alive_count = sum(1 for org in organism_states if org["alive"])

        if alive_count > 0:
            avg_energy = (
                sum(org["energy"] for org in organism_states if org["alive"])
                / alive_count
            )
            avg_age = (
                sum(org["age"] for org in organism_states if org["alive"]) / alive_count
            )
        else:
            avg_energy = 0
            avg_age = 0

        timeseries.append(
            {
                "run_id": run_data["run_id"],
                "timestep": timestep,
                "alive_count": alive_count,
                "avg_energy": avg_energy,
                "avg_age": avg_age,
                "food_count": ts_data["environment_state"].get("food_count", 0),
            }
        )

    df = pd.DataFrame(timeseries)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate data.

    Args:
        df: Input DataFrame

    Returns:
        Cleaned DataFrame
    """
    # Remove rows with NaN values
    df = df.dropna()

    # Remove outliers (optional - based on IQR)
    # For now, just validate ranges
    if "final_alive" in df.columns:
        df = df[df["final_alive"] >= 0]

    if "final_avg_energy" in df.columns:
        df = df[df["final_avg_energy"] >= 0]

    return df


def save_processed_data(df: pd.DataFrame, output_path: str):
    """Save processed data to CSV.

    Args:
        df: DataFrame to save
        output_path: Output file path
    """
    df.to_csv(output_path, index=False)
    print(f"Saved processed data to {output_path}")
