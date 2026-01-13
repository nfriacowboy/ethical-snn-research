#!/usr/bin/env python3
"""Generate synthetic ethical dataset for SNN-E pre-training using MACHIAVELLI taxonomy."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
from pathlib import Path

from src.training.ethical_dataset import EthicalDatasetGenerator


def main():
    parser = argparse.ArgumentParser(
        description="Generate ethical training dataset using MACHIAVELLI taxonomy"
    )
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

    args = parser.parse_args()

    print("=" * 60)
    print("MACHIAVELLI-Inspired Ethical Dataset Generator")
    print("=" * 60)
    print(f"Number of scenarios: {args.num_scenarios}")
    print(f"Random seed: {args.seed}")
    print()

    # Generate dataset
    print("Generating scenarios...")
    generator = EthicalDatasetGenerator(num_scenarios=args.num_scenarios, seed=args.seed)
    scenarios = generator.generate()

    # Statistics
    stats = generator.get_statistics()
    
    print(f"\nGenerated {stats['total_scenarios']} scenarios")
    print(f"\nEthical balance:")
    print(f"  Ethical: {stats['ethical_count']} ({stats['ethical_ratio']*100:.1f}%)")
    print(f"  Unethical: {stats['unethical_count']} ({(1-stats['ethical_ratio'])*100:.1f}%)")
    
    print(f"\nViolation distribution:")
    for violation, count in stats["violation_distribution"].items():
        percentage = (count / stats['total_scenarios']) * 100
        print(f"  {violation}: {count} ({percentage:.1f}%)")
    
    print(f"\nPrinciple distribution:")
    for principle, count in stats["principle_distribution"].items():
        percentage = (count / stats['total_scenarios']) * 100
        print(f"  {principle}: {count} ({percentage:.1f}%)")
    
    print(f"\nAction distribution:")
    for action, count in stats["action_distribution"].items():
        percentage = (count / stats['total_scenarios']) * 100
        print(f"  {action}: {count} ({percentage:.1f}%)")

    # Save dataset
    print(f"\nSaving to {args.output}...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    generator.save(str(output_path))

    print("✓ Dataset saved successfully")

    # Show example scenarios
    print("\nExample scenarios:")
    print("-" * 60)

    for i in range(min(5, len(scenarios))):
        scenario = scenarios[i]
        print(f"\nScenario {i+1}:")
        print(f"  Action: {scenario.action}")
        print(f"  Ethical: {scenario.is_ethical}")
        if scenario.violation:
            print(f"  Violation: {scenario.violation}")
            print(f"  Disutility: {scenario.disutility:.2f}")
        if scenario.principle:
            print(f"  Principle: {scenario.principle}")
        print(f"  Energies: self={scenario.self_energy:.1f}, other={scenario.other_energy:.1f}")
        print(f"  Food available: {scenario.food_available}")
        print(f"  Reasoning: {scenario.reasoning}")

    print("\n" + "=" * 60)
    print("Dataset generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
