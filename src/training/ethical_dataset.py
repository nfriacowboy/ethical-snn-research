"""Ethical training dataset generation for SNN-E pre-training.

Generates 1000 synthetic scenarios with ethical labels based on:
- Resource scarcity (self vs other energy levels)
- Harm potential (attack when other is vulnerable)
- Food availability context
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


@dataclass
class EthicalScenario:
    """Single ethical scenario with state and label.

    Attributes:
        self_energy: Organism's own energy level [0, 100]
        other_energy: Other organism's energy level [0, 100]
        food_available: Whether food is present in environment
        distance_to_other: Distance to other organism [0, 20]
        action: Proposed action ('ATTACK', 'EAT', 'MOVE', 'WAIT')
        is_ethical: Ground truth label (True=ethical, False=unethical)
        scenario_type: Category for analysis
    """

    self_energy: float
    other_energy: float
    food_available: bool
    distance_to_other: float
    action: str
    is_ethical: bool
    scenario_type: str

    def to_feature_vector(self) -> np.ndarray:
        """Convert scenario to 8-dimensional feature vector.

        Features:
            [0]: self_energy normalized [0, 1]
            [1]: other_energy normalized [0, 1]
            [2]: food_available (0 or 1)
            [3]: distance_to_other normalized [0, 1]
            [4-7]: action one-hot encoding (ATTACK, EAT, MOVE, WAIT)

        Returns:
            8-dimensional numpy array
        """
        features = np.zeros(8, dtype=np.float32)
        features[0] = self.self_energy / 100.0
        features[1] = self.other_energy / 100.0
        features[2] = 1.0 if self.food_available else 0.0
        features[3] = self.distance_to_other / 20.0  # Max grid diagonal ~28, use 20

        # Action one-hot encoding
        action_map = {"ATTACK": 4, "EAT": 5, "MOVE": 6, "WAIT": 7}
        if self.action in action_map:
            features[action_map[self.action]] = 1.0

        return features

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "self_energy": float(self.self_energy),
            "other_energy": float(self.other_energy),
            "food_available": bool(self.food_available),
            "distance_to_other": float(self.distance_to_other),
            "action": self.action,
            "is_ethical": bool(self.is_ethical),
            "scenario_type": self.scenario_type,
        }


class EthicalDatasetGenerator:
    """Generate synthetic ethical scenarios for SNN-E training.

    Implements ethical rules from preregistration:
    1. Don't attack when self has high energy and other has low energy
    2. Don't attack when food is readily available (non-zero-sum)
    3. Cooperation when both can benefit (implicit in scenarios)

    Generates balanced dataset with ~33% unethical, ~33% neutral, ~33% ethical.
    """

    def __init__(self, num_scenarios: int = 1000, seed: int = 42):
        """Initialize dataset generator.

        Args:
            num_scenarios: Total scenarios to generate (default: 1000)
            seed: Random seed for reproducibility
        """
        self.num_scenarios = num_scenarios
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.scenarios: List[EthicalScenario] = []

    def generate(self) -> List[EthicalScenario]:
        """Generate complete dataset of ethical scenarios.

        Returns:
            List of EthicalScenario objects (length: num_scenarios)
        """
        self.scenarios = []

        # Generate equal proportions of each scenario type
        scenarios_per_type = self.num_scenarios // 3

        # Type 1: Attack scenarios (ethical vs unethical attacks)
        for _ in range(scenarios_per_type):
            self.scenarios.append(self._generate_attack_scenario())

        # Type 2: Resource competition (with food available)
        for _ in range(scenarios_per_type):
            self.scenarios.append(self._generate_resource_scenario())

        # Type 3: Neutral actions (movement, eating when alone)
        for _ in range(self.num_scenarios - 2 * scenarios_per_type):
            self.scenarios.append(self._generate_neutral_scenario())

        # Shuffle scenarios
        self.rng.shuffle(self.scenarios)

        return self.scenarios

    def _generate_attack_scenario(self) -> EthicalScenario:
        """Generate attack scenario with ethical judgment.

        Ethical rules:
        - Unethical: Attack when other_energy < 30 and self_energy > 60
        - Ethical: Don't attack (or attack only when self_energy < 30)
        - Neutral: Attack when both have moderate energy

        Returns:
            EthicalScenario with action='ATTACK'
        """
        self_energy = self.rng.uniform(10, 100)
        other_energy = self.rng.uniform(10, 100)
        food_available = self.rng.choice([True, False])
        distance = self.rng.uniform(1, 3)  # Close enough to attack

        # Determine ethicality based on energy asymmetry
        if self_energy > 60 and other_energy < 30:
            # Strong attacking weak = unethical
            is_ethical = False
            scenario_type = "unethical_attack"
        elif food_available and self_energy > 50:
            # Attack when food available = unethical (non-zero-sum)
            is_ethical = False
            scenario_type = "unnecessary_attack"
        elif self_energy < 30 and other_energy > 60:
            # Weak attacking strong (desperate) = ethically neutral
            is_ethical = True
            scenario_type = "desperate_attack"
        else:
            # Comparable energy levels = neutral
            is_ethical = True
            scenario_type = "fair_conflict"

        return EthicalScenario(
            self_energy=self_energy,
            other_energy=other_energy,
            food_available=food_available,
            distance_to_other=distance,
            action="ATTACK",
            is_ethical=is_ethical,
            scenario_type=scenario_type,
        )

    def _generate_resource_scenario(self) -> EthicalScenario:
        """Generate resource-related scenario (eating behavior).

        Ethical rules:
        - Ethical: Eat when food available (benefits self without harming others)
        - Unethical: Eating when others are starving nearby (context-dependent)

        Returns:
            EthicalScenario with action='EAT'
        """
        self_energy = self.rng.uniform(20, 90)
        other_energy = self.rng.uniform(10, 100)
        food_available = True  # Always true for eat scenarios
        distance = self.rng.uniform(1, 20)

        # Eating is generally ethical (positive-sum)
        if other_energy < 20 and distance < 5 and self_energy > 70:
            # Eating in front of starving other = questionable
            is_ethical = self.rng.choice([True, False], p=[0.3, 0.7])
            scenario_type = "selfish_eating"
        else:
            is_ethical = True
            scenario_type = "normal_eating"

        return EthicalScenario(
            self_energy=self_energy,
            other_energy=other_energy,
            food_available=food_available,
            distance_to_other=distance,
            action="EAT",
            is_ethical=is_ethical,
            scenario_type=scenario_type,
        )

    def _generate_neutral_scenario(self) -> EthicalScenario:
        """Generate neutral scenario (movement, waiting).

        These actions have no direct ethical implications.

        Returns:
            EthicalScenario with action='MOVE' or 'WAIT'
        """
        self_energy = self.rng.uniform(20, 100)
        other_energy = self.rng.uniform(20, 100)
        food_available = self.rng.choice([True, False])
        distance = self.rng.uniform(5, 20)  # Far from others
        action = self.rng.choice(["MOVE", "WAIT"])

        # Neutral actions are always ethical (no harm)
        return EthicalScenario(
            self_energy=self_energy,
            other_energy=other_energy,
            food_available=food_available,
            distance_to_other=distance,
            action=action,
            is_ethical=True,
            scenario_type="neutral_action",
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics for validation.

        Returns:
            Dictionary with counts and proportions
        """
        if not self.scenarios:
            return {}

        total = len(self.scenarios)
        ethical_count = sum(1 for s in self.scenarios if s.is_ethical)
        unethical_count = total - ethical_count

        action_counts = {}
        for action in ["ATTACK", "EAT", "MOVE", "WAIT"]:
            action_counts[action] = sum(1 for s in self.scenarios if s.action == action)

        scenario_type_counts = {}
        for s in self.scenarios:
            scenario_type_counts[s.scenario_type] = (
                scenario_type_counts.get(s.scenario_type, 0) + 1
            )

        return {
            "total_scenarios": total,
            "ethical_count": ethical_count,
            "unethical_count": unethical_count,
            "ethical_ratio": ethical_count / total if total > 0 else 0,
            "action_distribution": action_counts,
            "scenario_types": scenario_type_counts,
        }

    def to_tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert scenarios to PyTorch tensors.

        Returns:
            Tuple of (features, labels):
                features: [N, 8] tensor of scenario features
                labels: [N] tensor of binary labels (1=ethical, 0=unethical)
        """
        if not self.scenarios:
            raise ValueError("No scenarios generated. Call generate() first.")

        features = np.stack([s.to_feature_vector() for s in self.scenarios])
        labels = np.array([1.0 if s.is_ethical else 0.0 for s in self.scenarios])

        return torch.tensor(features, dtype=torch.float32), torch.tensor(
            labels, dtype=torch.float32
        )

    def split_train_test(
        self, test_ratio: float = 0.2
    ) -> Tuple[List[EthicalScenario], List[EthicalScenario]]:
        """Split dataset into train and test sets.

        Args:
            test_ratio: Proportion for test set (default: 0.2)

        Returns:
            Tuple of (train_scenarios, test_scenarios)
        """
        if not self.scenarios:
            raise ValueError("No scenarios generated. Call generate() first.")

        n_test = int(len(self.scenarios) * test_ratio)
        indices = self.rng.permutation(len(self.scenarios))

        test_indices = indices[:n_test]
        train_indices = indices[n_test:]

        train_scenarios = [self.scenarios[i] for i in train_indices]
        test_scenarios = [self.scenarios[i] for i in test_indices]

        return train_scenarios, test_scenarios

    def save(self, filepath: str) -> None:
        """Save dataset to JSON file.

        Args:
            filepath: Path to output JSON file
        """
        import json

        data = {
            "metadata": {
                "num_scenarios": self.num_scenarios,
                "seed": self.seed,
                "statistics": self.get_statistics(),
            },
            "scenarios": [s.to_dict() for s in self.scenarios],
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load(filepath: str) -> "EthicalDatasetGenerator":
        """Load dataset from JSON file.

        Args:
            filepath: Path to input JSON file

        Returns:
            EthicalDatasetGenerator with loaded scenarios
        """
        import json

        with open(filepath, "r") as f:
            data = json.load(f)

        generator = EthicalDatasetGenerator(
            num_scenarios=data["metadata"]["num_scenarios"],
            seed=data["metadata"]["seed"],
        )

        generator.scenarios = [EthicalScenario(**s) for s in data["scenarios"]]

        return generator
