"""Statistical tests for Phase 1 analysis."""

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def mann_whitney_test(
    group_a: np.ndarray, group_b: np.ndarray, metric_name: str = ""
) -> Dict[str, Any]:
    """Perform Mann-Whitney U test between two groups.

    Args:
        group_a: Data from condition A
        group_b: Data from condition B
        metric_name: Name of the metric being tested

    Returns:
        Dictionary with test results
    """
    # Perform test
    statistic, p_value = stats.mannwhitneyu(group_a, group_b, alternative="two-sided")

    # Calculate descriptive statistics
    result = {
        "metric": metric_name,
        "test": "Mann-Whitney U",
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
        "group_a_mean": float(np.mean(group_a)),
        "group_a_median": float(np.median(group_a)),
        "group_a_std": float(np.std(group_a)),
        "group_b_mean": float(np.mean(group_b)),
        "group_b_median": float(np.median(group_b)),
        "group_b_std": float(np.std(group_b)),
        "n_a": len(group_a),
        "n_b": len(group_b),
    }

    return result


def cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Calculate Cohen's d effect size.

    Args:
        group_a: Data from condition A
        group_b: Data from condition B

    Returns:
        Cohen's d value
    """
    mean_a = np.mean(group_a)
    mean_b = np.mean(group_b)

    std_a = np.std(group_a, ddof=1)
    std_b = np.std(group_b, ddof=1)

    n_a = len(group_a)
    n_b = len(group_b)

    # Pooled standard deviation
    pooled_std = np.sqrt(
        ((n_a - 1) * std_a**2 + (n_b - 1) * std_b**2) / (n_a + n_b - 2)
    )

    if pooled_std == 0:
        return 0.0

    d = (mean_b - mean_a) / pooled_std

    return float(d)


def bonferroni_correction(p_values: List[float]) -> Tuple[List[float], float]:
    """Apply Bonferroni correction for multiple comparisons.

    Args:
        p_values: List of p-values

    Returns:
        Tuple of (corrected p-values, corrected alpha)
    """
    n_tests = len(p_values)
    corrected_alpha = 0.05 / n_tests

    corrected_p_values = [p * n_tests for p in p_values]

    return corrected_p_values, corrected_alpha


def compare_conditions(
    df_a: pd.DataFrame, df_b: pd.DataFrame, metrics: List[str]
) -> pd.DataFrame:
    """Compare multiple metrics between two conditions.

    Args:
        df_a: DataFrame for condition A
        df_b: DataFrame for condition B
        metrics: List of metric column names to compare

    Returns:
        DataFrame with comparison results
    """
    results = []

    for metric in metrics:
        if metric not in df_a.columns or metric not in df_b.columns:
            print(f"Warning: Metric '{metric}' not found in both dataframes")
            continue

        group_a = df_a[metric].values
        group_b = df_b[metric].values

        # Mann-Whitney test
        mw_result = mann_whitney_test(group_a, group_b, metric_name=metric)

        # Effect size
        effect_size = cohens_d(group_a, group_b)
        mw_result["cohens_d"] = effect_size

        # Interpret effect size
        if abs(effect_size) < 0.2:
            mw_result["effect_size_interpretation"] = "negligible"
        elif abs(effect_size) < 0.5:
            mw_result["effect_size_interpretation"] = "small"
        elif abs(effect_size) < 0.8:
            mw_result["effect_size_interpretation"] = "medium"
        else:
            mw_result["effect_size_interpretation"] = "large"

        results.append(mw_result)

    results_df = pd.DataFrame(results)

    # Apply Bonferroni correction
    if len(results) > 1:
        corrected_p, corrected_alpha = bonferroni_correction(
            results_df["p_value"].tolist()
        )
        results_df["p_value_corrected"] = corrected_p
        results_df["corrected_alpha"] = corrected_alpha
        results_df["significant_corrected"] = results_df["p_value_corrected"] < 0.05

    return results_df


def survival_analysis(df_a: pd.DataFrame, df_b: pd.DataFrame) -> Dict[str, Any]:
    """Analyze survival rates between conditions.

    Args:
        df_a: DataFrame for condition A
        df_b: DataFrame for condition B

    Returns:
        Dictionary with survival analysis results
    """
    # Calculate survival rates
    survival_rate_a = (df_a["final_alive"] > 0).sum() / len(df_a)
    survival_rate_b = (df_b["final_alive"] > 0).sum() / len(df_b)

    # Chi-square test for survival
    contingency = [
        [(df_a["final_alive"] > 0).sum(), (df_a["final_alive"] == 0).sum()],
        [(df_b["final_alive"] > 0).sum(), (df_b["final_alive"] == 0).sum()],
    ]

    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

    return {
        "survival_rate_a": float(survival_rate_a),
        "survival_rate_b": float(survival_rate_b),
        "chi2_statistic": float(chi2),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
        "n_survived_a": int((df_a["final_alive"] > 0).sum()),
        "n_died_a": int((df_a["final_alive"] == 0).sum()),
        "n_survived_b": int((df_b["final_alive"] > 0).sum()),
        "n_died_b": int((df_b["final_alive"] == 0).sum()),
    }


def generate_statistical_report(
    df_a: pd.DataFrame, df_b: pd.DataFrame, metrics: List[str]
) -> str:
    """Generate text report of statistical analyses.

    Args:
        df_a: DataFrame for condition A
        df_b: DataFrame for condition B
        metrics: List of metrics to analyze

    Returns:
        Formatted report string
    """
    report = []
    report.append("=" * 70)
    report.append("PHASE 1 STATISTICAL ANALYSIS REPORT")
    report.append("=" * 70)
    report.append("")

    # Sample sizes
    report.append(f"Sample Sizes:")
    report.append(f"  Condition A (Survival-Only): n = {len(df_a)}")
    report.append(f"  Condition B (Dual-Process): n = {len(df_b)}")
    report.append("")

    # Comparisons
    comparison_results = compare_conditions(df_a, df_b, metrics)

    report.append("Metric Comparisons:")
    report.append("-" * 70)

    for _, row in comparison_results.iterrows():
        report.append(f"\n{row['metric'].upper()}:")
        report.append(
            f"  Condition A: M = {row['group_a_mean']:.3f}, "
            f"Mdn = {row['group_a_median']:.3f}, SD = {row['group_a_std']:.3f}"
        )
        report.append(
            f"  Condition B: M = {row['group_b_mean']:.3f}, "
            f"Mdn = {row['group_b_median']:.3f}, SD = {row['group_b_std']:.3f}"
        )
        report.append(
            f"  Mann-Whitney U = {row['statistic']:.3f}, p = {row['p_value']:.4f} "
            f"{'*' if row['significant'] else ''}"
        )
        report.append(
            f"  Cohen's d = {row['cohens_d']:.3f} ({row['effect_size_interpretation']})"
        )

        if "p_value_corrected" in row:
            report.append(
                f"  Corrected p = {row['p_value_corrected']:.4f} "
                f"{'*' if row['significant_corrected'] else ''}"
            )

    report.append("")
    report.append("* = significant at α = 0.05")
    report.append("=" * 70)

    return "\n".join(report)
