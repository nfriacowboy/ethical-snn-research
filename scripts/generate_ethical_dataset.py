#!/usr/bin/env python3
"""Generate synthetic ethical dataset for SNN-E pre-training."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import json

import numpy as np

from src.training.ethical_dataset import EthicalDataset


def main():
    parser = argparse.ArgumentParser(description="Generate ethical training dataset")
    parser.add_argument(
        "--num_scenarios",
        type=int,
        default=1000,
        help="Number of scenarios to generate",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/ethical_dataset.json",
        help="Output file path",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--test_ratio", type=float, default=0.2, help="Ratio of data for testing"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Ethical Dataset Generator")
    print("=" * 60)
    print(f"Number of scenarios: {args.num_scenarios}")
    print(f"Random seed: {args.seed}")
    print(f"Test ratio: {args.test_ratio}")
    print()

    # Generate dataset
    print("Generating scenarios...")
    dataset = EthicalDataset(num_scenarios=args.num_scenarios, seed=args.seed)
    scenarios, labels = dataset.generate()

    # Statistics
    print(f"\nGenerated {len(scenarios)} scenarios")
    print("\nLabel distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    label_names = ["Positive (ethical)", "Neutral", "Negative (unethical)"]
    for label, count in zip(unique, counts):
        print(f"  {label_names[label]}: {count} ({count/len(labels)*100:.1f}%)")

    # Split into train/test
    print("\nSplitting into train/test sets...")
    train_scenarios, test_scenarios, train_labels, test_labels = (
        dataset.get_train_test_split(test_ratio=args.test_ratio)
    )

    print(f"  Train: {len(train_scenarios)} scenarios")
    print(f"  Test: {len(test_scenarios)} scenarios")

    # Save dataset
    print(f"\nSaving to {args.output}...")

    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    output_data = {
        "metadata": {
            "num_scenarios": args.num_scenarios,
            "seed": args.seed,
            "test_ratio": args.test_ratio,
        },
        "train": {"scenarios": train_scenarios, "labels": train_labels},
        "test": {"scenarios": test_scenarios, "labels": test_labels},
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print("✓ Dataset saved successfully")

    # Show example scenarios
    print("\nExample scenarios:")
    print("-" * 60)

    for i in range(min(3, len(scenarios))):
        scenario = scenarios[i]
        label = labels[i]

        print(f"\nScenario {i+1}:")
        print(f"  Type: {scenario['type']}")
        print(f"  Label: {label_names[label]}")
        print(f"  Details: {scenario}")

    print("\n" + "=" * 60)
    print("Dataset generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
