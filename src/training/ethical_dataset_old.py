"""Synthetic ethical dataset generation."""

import numpy as np
from typing import List, Dict, Any, Tuple


class EthicalDataset:
    """Generate synthetic ethical scenarios for SNN-E pre-training.
    
    Creates labeled scenarios with ethical valence (positive, neutral, negative).
    """
    
    def __init__(self, num_scenarios: int = 1000, seed: int = 42):
        """Initialize ethical dataset generator.
        
        Args:
            num_scenarios: Number of scenarios to generate
            seed: Random seed for reproducibility
        """
        self.num_scenarios = num_scenarios
        self.seed = seed
        np.random.seed(seed)
        
        self.scenarios: List[Dict[str, Any]] = []
        self.labels: List[int] = []  # 0=positive, 1=neutral, 2=negative
    
    def generate(self) -> Tuple[List[Dict[str, Any]], List[int]]:
        """Generate synthetic ethical scenarios.
        
        Returns:
            Tuple of (scenarios, labels)
        """
        for i in range(self.num_scenarios):
            scenario, label = self._generate_scenario()
            self.scenarios.append(scenario)
            self.labels.append(label)
        
        return self.scenarios, self.labels
    
    def _generate_scenario(self) -> Tuple[Dict[str, Any], int]:
        """Generate a single ethical scenario.
        
        Returns:
            Tuple of (scenario dict, label)
        """
        scenario_type = np.random.choice(['resource_sharing', 'harm_avoidance', 'neutral_action'])
        
        if scenario_type == 'resource_sharing':
            return self._resource_sharing_scenario()
        elif scenario_type == 'harm_avoidance':
            return self._harm_avoidance_scenario()
        else:
            return self._neutral_scenario()
    
    def _resource_sharing_scenario(self) -> Tuple[Dict[str, Any], int]:
        """Generate resource sharing scenario.
        
        Returns:
            Tuple of (scenario, label)
        """
        other_energy = np.random.uniform(0, 50)  # Low energy = need
        self_energy = np.random.uniform(50, 100)
        resource_available = True
        
        # Positive valence: sharing when others need and self has enough
        if other_energy < 30 and self_energy > 60:
            label = 0  # Positive (ethical to share)
        elif other_energy > 60:
            label = 1  # Neutral (no need to share)
        else:
            label = 1  # Neutral
        
        scenario = {
            'type': 'resource_sharing',
            'other_energy': other_energy,
            'self_energy': self_energy,
            'resource_available': resource_available,
            'proximity': np.random.uniform(1, 10)
        }
        
        return scenario, label
    
    def _harm_avoidance_scenario(self) -> Tuple[Dict[str, Any], int]:
        """Generate harm avoidance scenario.
        
        Returns:
            Tuple of (scenario, label)
        """
        collision_imminent = np.random.choice([True, False])
        other_vulnerable = np.random.choice([True, False])
        
        # Negative valence: causing harm
        if collision_imminent and other_vulnerable:
            label = 2  # Negative (unethical to collide)
        else:
            label = 1  # Neutral
        
        scenario = {
            'type': 'harm_avoidance',
            'collision_imminent': collision_imminent,
            'other_vulnerable': other_vulnerable,
            'distance': np.random.uniform(1, 10)
        }
        
        return scenario, label
    
    def _neutral_scenario(self) -> Tuple[Dict[str, Any], int]:
        """Generate neutral scenario (no ethical implications).
        
        Returns:
            Tuple of (scenario, label)
        """
        scenario = {
            'type': 'neutral_action',
            'empty_space': True,
            'no_organisms_nearby': True,
            'movement': np.random.choice(['N', 'S', 'E', 'W'])
        }
        
        return scenario, 1  # Neutral
    
    def encode_scenario(self, scenario: Dict[str, Any]) -> np.ndarray:
        """Encode scenario as feature vector for SNN input.
        
        Args:
            scenario: Scenario dictionary
            
        Returns:
            Feature vector (64-dimensional)
        """
        features = np.zeros(64)
        
        if scenario['type'] == 'resource_sharing':
            features[0] = scenario['other_energy'] / 100.0
            features[1] = scenario['self_energy'] / 100.0
            features[2] = 1.0 if scenario['resource_available'] else 0.0
            features[3] = scenario['proximity'] / 10.0
        
        elif scenario['type'] == 'harm_avoidance':
            features[4] = 1.0 if scenario['collision_imminent'] else 0.0
            features[5] = 1.0 if scenario['other_vulnerable'] else 0.0
            features[6] = scenario['distance'] / 10.0
        
        elif scenario['type'] == 'neutral_action':
            features[7] = 1.0
        
        return features
    
    def get_train_test_split(self, test_ratio: float = 0.2) -> Tuple[List, List, List, List]:
        """Split dataset into train and test sets.
        
        Args:
            test_ratio: Proportion of data for testing
            
        Returns:
            Tuple of (train_scenarios, test_scenarios, train_labels, test_labels)
        """
        n_test = int(len(self.scenarios) * test_ratio)
        n_train = len(self.scenarios) - n_test
        
        # Shuffle
        indices = np.random.permutation(len(self.scenarios))
        
        train_scenarios = [self.scenarios[i] for i in indices[:n_train]]
        test_scenarios = [self.scenarios[i] for i in indices[n_train:]]
        train_labels = [self.labels[i] for i in indices[:n_train]]
        test_labels = [self.labels[i] for i in indices[n_train:]]
        
        return train_scenarios, test_scenarios, train_labels, test_labels
