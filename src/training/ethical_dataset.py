"""Ethical training dataset generation for SNN-E pre-training.

Generates 1000 synthetic scenarios with ethical labels based on MACHIAVELLI taxonomy:
- UNNECESSARY_HARM: Attacks without strategic necessity (20%)
- RESOURCE_HOARDING: Monopolizing resources (15%)
- UNFAIR_COMPETITION: Exploiting power imbalances (20%)
- DECEPTIVE_ACTION: Misleading behavior (10%)
- WASTEFUL_AGGRESSION: Violence without benefit (10%)
- ETHICAL_POSITIVE: Pro-social behaviors (25%)

Adapted from: Pan et al. (2023) MACHIAVELLI Benchmark, ICML.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.utils.ethical_categories import (
    ActionType,
    EthicalContext,
    EthicalEvaluation,
    EthicalEvaluator,
    EthicalPrinciple,
    EthicalViolation,
)


@dataclass
class EthicalScenario:
    """Single ethical scenario with state and label using MACHIAVELLI taxonomy.

    Attributes:
        self_energy: Organism's own energy level [0, 100]
        other_energy: Other organism's energy level [0, 100]
        food_available: Whether food is present in environment
        distance_to_other: Distance to other organism [0, 28]
        distance_to_food: Distance to nearest food [0, 28] (optional)
        action: Proposed action ('ATTACK', 'EAT', 'MOVE', 'WAIT')
        is_ethical: Ground truth label (True=ethical, False=unethical)
        violation: Type of ethical violation (None if ethical)
        principle: Positive ethical principle (None if unethical)
        disutility: Harm caused by action [0, ∞)
        reasoning: Explanation of ethical judgment
    """

    self_energy: float
    other_energy: float
    food_available: bool
    distance_to_other: float
    action: str
    is_ethical: bool
    violation: Optional[str] = None  # EthicalViolation enum value
    principle: Optional[str] = None  # EthicalPrinciple enum value
    distance_to_food: Optional[float] = None
    disutility: float = 0.0
    reasoning: str = ""

    # Legacy field for backward compatibility
    @property
    def scenario_type(self) -> str:
        """Legacy property mapping to violation or principle."""
        if self.violation:
            return self.violation
        elif self.principle:
            return self.principle
        else:
            return "neutral_action"

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
            "distance_to_food": float(self.distance_to_food)
            if self.distance_to_food is not None
            else None,
            "action": self.action,
            "is_ethical": bool(self.is_ethical),
            "violation": self.violation,
            "principle": self.principle,
            "disutility": float(self.disutility),
            "reasoning": self.reasoning,
            # Legacy field
            "scenario_type": self.scenario_type,
        }


class EthicalDatasetGenerator:
    """Generate synthetic ethical scenarios for SNN-E training.

    Implements MACHIAVELLI-inspired taxonomy adapted for spatial organisms:
    - UNNECESSARY_HARM (20%): Attacks without necessity
    - RESOURCE_HOARDING (15%): Monopolizing resources
    - UNFAIR_COMPETITION (20%): Exploiting power imbalances
    - DECEPTIVE_ACTION (10%): Misleading behavior
    - WASTEFUL_AGGRESSION (10%): Violence without benefit
    - ETHICAL_POSITIVE (25%): Pro-social behaviors

    Reference: Pan et al. (2023) MACHIAVELLI Benchmark, ICML.
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
        self.evaluator = EthicalEvaluator()

    def generate(self) -> List[EthicalScenario]:
        """Generate complete dataset of ethical scenarios.

        Distribution (MACHIAVELLI-inspired):
        - UNNECESSARY_HARM: 200 scenarios (20%)
        - UNFAIR_COMPETITION: 200 scenarios (20%)
        - RESOURCE_HOARDING: 150 scenarios (15%)
        - DECEPTIVE_ACTION: 100 scenarios (10%)
        - WASTEFUL_AGGRESSION: 100 scenarios (10%)
        - ETHICAL_POSITIVE: 250 scenarios (25%)

        Returns:
            List of EthicalScenario objects (length: num_scenarios)
        """
        self.scenarios = []

        # Calculate scenario counts per category
        unnecessary_harm_count = int(self.num_scenarios * 0.20)
        unfair_competition_count = int(self.num_scenarios * 0.20)
        resource_hoarding_count = int(self.num_scenarios * 0.15)
        deceptive_action_count = int(self.num_scenarios * 0.10)
        wasteful_aggression_count = int(self.num_scenarios * 0.10)
        ethical_positive_count = self.num_scenarios - (
            unnecessary_harm_count
            + unfair_competition_count
            + resource_hoarding_count
            + deceptive_action_count
            + wasteful_aggression_count
        )

        # Generate each category
        for _ in range(unnecessary_harm_count):
            self.scenarios.append(
                self._generate_violation_scenario(EthicalViolation.UNNECESSARY_HARM)
            )

        for _ in range(unfair_competition_count):
            self.scenarios.append(
                self._generate_violation_scenario(EthicalViolation.UNFAIR_COMPETITION)
            )

        for _ in range(resource_hoarding_count):
            self.scenarios.append(
                self._generate_violation_scenario(EthicalViolation.RESOURCE_HOARDING)
            )

        for _ in range(deceptive_action_count):
            self.scenarios.append(
                self._generate_violation_scenario(EthicalViolation.DECEPTIVE_ACTION)
            )

        for _ in range(wasteful_aggression_count):
            self.scenarios.append(
                self._generate_violation_scenario(EthicalViolation.WASTEFUL_AGGRESSION)
            )

        for _ in range(ethical_positive_count):
            self.scenarios.append(self._generate_ethical_scenario())

        # Shuffle scenarios
        self.rng.shuffle(self.scenarios)

        return self.scenarios

    def _generate_violation_scenario(
        self, violation_type: EthicalViolation
    ) -> EthicalScenario:
        """Generate scenario with specific ethical violation.

        Args:
            violation_type: Type of violation to generate

        Returns:
            EthicalScenario with violation
        """
        if violation_type == EthicalViolation.UNNECESSARY_HARM:
            return self._generate_unnecessary_harm()
        elif violation_type == EthicalViolation.UNFAIR_COMPETITION:
            return self._generate_unfair_competition()
        elif violation_type == EthicalViolation.RESOURCE_HOARDING:
            return self._generate_resource_hoarding()
        elif violation_type == EthicalViolation.DECEPTIVE_ACTION:
            return self._generate_deceptive_action()
        elif violation_type == EthicalViolation.WASTEFUL_AGGRESSION:
            return self._generate_wasteful_aggression()
        else:
            raise ValueError(f"Unknown violation type: {violation_type}")

    def _generate_unnecessary_harm(self) -> EthicalScenario:
        """Generate UNNECESSARY_HARM scenario: attacking when unnecessary.
        
        Note: Must avoid triggering higher-priority violations:
        - No recent_movement="toward_food" (would be deceptive_action)
        - No self_energy < 20 (would be wasteful_aggression)
        - energy_diff must be <= 40 (else unfair_competition)
        
        Two valid cases:
        1. Strong vs weak but energy_diff <= 40
        2. Attack when food available with energy_diff < 20
        """
        if self.rng.rand() < 0.5:
            # Case 1: Strong vs weak (70+ vs <30) but keep diff <= 40
            self_energy = self.rng.uniform(71, 80)  # Above 70
            other_energy = self.rng.uniform(30, 39)  # Below 40 to ensure diff <= 50
            # Actually, let's be safer: self - other <= 40
            # If self=80, other must be >= 40 (but rule says <30 for unnecessary_harm)
            # Let me reconsider: the rule says "self > 70 AND other < 30"
            # If self=71 and other=29, diff=42 > 40 → unfair_competition
            # This is tricky! Let's focus on Case 2 instead.
            pass
        
        # Focus on Case 2: Attack when food available
        self_energy = self.rng.uniform(55, 95)  # Above 50 (rule requirement)
        other_energy = self.rng.uniform(35, 95)  # Keep diff < 20 for fair competition exception
        
        # Ensure energy_diff < 40 to avoid unfair_competition
        while (self_energy - other_energy) >= 40:
            self_energy = self.rng.uniform(55, 95)
            other_energy = self.rng.uniform(35, 95)
        
        # Ensure energy_diff >= 20 to trigger unnecessary_harm (avoid fair competition exception)
        while abs(self_energy - other_energy) < 20:
            self_energy = self.rng.uniform(55, 95)
            other_energy = self.rng.uniform(35, 95)
        
        food_available = True
        distance_to_other = self.rng.uniform(1, 3)

        context = EthicalContext(
            self_energy=self_energy,
            other_energy=other_energy,
            food_available=food_available,
            distance_to_other=distance_to_other,
            action=ActionType.ATTACK,
            # No recent_movement to avoid deceptive_action
        )

        eval = self.evaluator.evaluate(context)

        return EthicalScenario(
            self_energy=self_energy,
            other_energy=other_energy,
            food_available=food_available,
            distance_to_other=distance_to_other,
            action="ATTACK",
            is_ethical=False,
            violation=eval.violation.value if eval.violation else None,
            disutility=eval.disutility,
            reasoning=eval.reasoning,
        )

    def _generate_unfair_competition(self) -> EthicalScenario:
        """Generate UNFAIR_COMPETITION scenario: exploiting power imbalance."""
        # Large energy difference (>40)
        self_energy = self.rng.uniform(70, 100)
        other_energy = self.rng.uniform(10, 30)  # Ensure diff > 40
        food_available = self.rng.choice([True, False])
        distance_to_other = self.rng.uniform(1, 3)

        context = EthicalContext(
            self_energy=self_energy,
            other_energy=other_energy,
            food_available=food_available,
            distance_to_other=distance_to_other,
            action=ActionType.ATTACK,
        )

        eval = self.evaluator.evaluate(context)

        return EthicalScenario(
            self_energy=self_energy,
            other_energy=other_energy,
            food_available=food_available,
            distance_to_other=distance_to_other,
            action="ATTACK",
            is_ethical=False,
            violation=eval.violation.value if eval.violation else None,
            disutility=eval.disutility,
            reasoning=eval.reasoning,
        )

    def _generate_resource_hoarding(self) -> EthicalScenario:
        """Generate RESOURCE_HOARDING scenario: eating when satiated while other starves."""
        self_energy = self.rng.uniform(80, 95)
        other_energy = self.rng.uniform(10, 20)
        food_available = True
        distance_to_other = self.rng.uniform(1, 5)

        context = EthicalContext(
            self_energy=self_energy,
            other_energy=other_energy,
            food_available=food_available,
            distance_to_other=distance_to_other,
            action=ActionType.EAT,
        )

        eval = self.evaluator.evaluate(context)

        return EthicalScenario(
            self_energy=self_energy,
            other_energy=other_energy,
            food_available=food_available,
            distance_to_other=distance_to_other,
            action="EAT",
            is_ethical=False,
            violation=eval.violation.value if eval.violation else None,
            disutility=eval.disutility,
            reasoning=eval.reasoning,
        )

    def _generate_deceptive_action(self) -> EthicalScenario:
        """Generate DECEPTIVE_ACTION scenario: feinting toward food then attacking."""
        self_energy = self.rng.uniform(50, 80)
        other_energy = self.rng.uniform(40, 70)
        food_available = True
        distance_to_other = self.rng.uniform(1, 3)

        context = EthicalContext(
            self_energy=self_energy,
            other_energy=other_energy,
            food_available=food_available,
            distance_to_other=distance_to_other,
            action=ActionType.ATTACK,
            recent_movement="toward_food",
        )

        eval = self.evaluator.evaluate(context)

        return EthicalScenario(
            self_energy=self_energy,
            other_energy=other_energy,
            food_available=food_available,
            distance_to_other=distance_to_other,
            action="ATTACK",
            is_ethical=False,
            violation=eval.violation.value if eval.violation else None,
            disutility=eval.disutility,
            reasoning=eval.reasoning,
        )

    def _generate_wasteful_aggression(self) -> EthicalScenario:
        """Generate WASTEFUL_AGGRESSION scenario: attacking with critically low energy."""
        self_energy = self.rng.uniform(5, 20)
        other_energy = self.rng.uniform(40, 80)
        food_available = True
        distance_to_other = self.rng.uniform(1, 3)

        context = EthicalContext(
            self_energy=self_energy,
            other_energy=other_energy,
            food_available=food_available,
            distance_to_other=distance_to_other,
            action=ActionType.ATTACK,
        )

        eval = self.evaluator.evaluate(context)

        return EthicalScenario(
            self_energy=self_energy,
            other_energy=other_energy,
            food_available=food_available,
            distance_to_other=distance_to_other,
            action="ATTACK",
            is_ethical=False,
            violation=eval.violation.value if eval.violation else None,
            disutility=eval.disutility,
            reasoning=eval.reasoning,
        )

    def _generate_ethical_scenario(self) -> EthicalScenario:
        """Generate ethical positive scenario with pro-social behavior."""
        scenario_type = self.rng.choice(
            ["harm_avoidance", "resource_sharing", "fair_competition", "cooperation"]
        )

        if scenario_type == "harm_avoidance":
            # Moving away from vulnerable organism
            self_energy = self.rng.uniform(70, 95)
            other_energy = self.rng.uniform(10, 30)
            food_available = self.rng.choice([True, False])
            distance_to_other = self.rng.uniform(2, 6)
            action = "MOVE"

        elif scenario_type == "resource_sharing":
            # Waiting to let starving organism eat
            self_energy = self.rng.uniform(70, 95)
            other_energy = self.rng.uniform(10, 30)
            food_available = True
            distance_to_other = self.rng.uniform(1, 5)
            action = "WAIT"

        elif scenario_type == "fair_competition":
            # Fair fight between equals
            self_energy = self.rng.uniform(40, 60)
            other_energy = self.rng.uniform(40, 60)
            food_available = self.rng.choice([True, False])
            distance_to_other = self.rng.uniform(1, 3)
            action = "ATTACK"

        else:  # cooperation
            # Eating when both healthy
            self_energy = self.rng.uniform(30, 70)
            other_energy = self.rng.uniform(50, 80)
            food_available = True
            distance_to_other = self.rng.uniform(5, 15)
            action = "EAT"

        context = EthicalContext(
            self_energy=self_energy,
            other_energy=other_energy,
            food_available=food_available,
            distance_to_other=distance_to_other,
            action=ActionType[action],
        )

        eval = self.evaluator.evaluate(context)

        return EthicalScenario(
            self_energy=self_energy,
            other_energy=other_energy,
            food_available=food_available,
            distance_to_other=distance_to_other,
            action=action,
            is_ethical=True,
            principle=eval.principle.value if eval.principle else None,
            disutility=0.0,
            reasoning=eval.reasoning,
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

        # Count violations and principles
        violation_counts = {}
        principle_counts = {}
        for s in self.scenarios:
            if s.violation:
                violation_counts[s.violation] = violation_counts.get(s.violation, 0) + 1
            if s.principle:
                principle_counts[s.principle] = principle_counts.get(s.principle, 0) + 1

        return {
            "total_scenarios": total,
            "ethical_count": ethical_count,
            "unethical_count": unethical_count,
            "ethical_ratio": ethical_count / total if total > 0 else 0,
            "action_distribution": action_counts,
            "violation_distribution": violation_counts,
            "principle_distribution": principle_counts,
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

        # Load scenarios, filtering out 'scenario_type' if present (backward compatibility)
        scenarios = []
        for s in data["scenarios"]:
            # Remove scenario_type if present (it's now a property)
            s_copy = s.copy()
            s_copy.pop("scenario_type", None)
            scenarios.append(EthicalScenario(**s_copy))
        
        generator.scenarios = scenarios

        return generator
