"""Tests for ethical training dataset generation."""

import pytest
import numpy as np
import torch
import json
import tempfile
from pathlib import Path

from src.training.ethical_dataset import (
    EthicalScenario,
    EthicalDatasetGenerator
)


class TestEthicalScenario:
    """Test EthicalScenario dataclass."""
    
    def test_scenario_creation(self):
        """Test creating an ethical scenario."""
        scenario = EthicalScenario(
            self_energy=80.0,
            other_energy=20.0,
            food_available=True,
            distance_to_other=5.0,
            action='ATTACK',
            is_ethical=False,
            scenario_type='unethical_attack'
        )
        
        assert scenario.self_energy == 80.0
        assert scenario.other_energy == 20.0
        assert scenario.food_available is True
        assert scenario.action == 'ATTACK'
        assert scenario.is_ethical is False
    
    def test_to_feature_vector(self):
        """Test converting scenario to feature vector."""
        scenario = EthicalScenario(
            self_energy=50.0,
            other_energy=75.0,
            food_available=True,
            distance_to_other=10.0,
            action='EAT',
            is_ethical=True,
            scenario_type='normal_eating'
        )
        
        features = scenario.to_feature_vector()
        
        assert features.shape == (8,)
        assert features.dtype == np.float32
        assert features[0] == pytest.approx(0.5)  # self_energy / 100
        assert features[1] == pytest.approx(0.75)  # other_energy / 100
        assert features[2] == 1.0  # food_available
        assert features[3] == pytest.approx(0.5)  # distance / 20
        assert features[5] == 1.0  # EAT action (index 5)
    
    def test_action_encoding(self):
        """Test one-hot encoding of actions."""
        actions = ['ATTACK', 'EAT', 'MOVE', 'WAIT']
        expected_indices = [4, 5, 6, 7]
        
        for action, expected_idx in zip(actions, expected_indices):
            scenario = EthicalScenario(
                self_energy=50.0,
                other_energy=50.0,
                food_available=False,
                distance_to_other=10.0,
                action=action,
                is_ethical=True,
                scenario_type='test'
            )
            
            features = scenario.to_feature_vector()
            assert features[expected_idx] == 1.0
            # All other action indices should be 0
            other_indices = [i for i in range(4, 8) if i != expected_idx]
            assert all(features[i] == 0.0 for i in other_indices)
    
    def test_to_dict(self):
        """Test converting scenario to dictionary."""
        scenario = EthicalScenario(
            self_energy=60.0,
            other_energy=40.0,
            food_available=False,
            distance_to_other=8.0,
            action='MOVE',
            is_ethical=True,
            scenario_type='neutral_action'
        )
        
        d = scenario.to_dict()
        
        assert isinstance(d, dict)
        assert d['self_energy'] == 60.0
        assert d['other_energy'] == 40.0
        assert d['food_available'] is False
        assert d['action'] == 'MOVE'
        assert d['is_ethical'] is True


class TestEthicalDatasetGenerator:
    """Test EthicalDatasetGenerator."""
    
    def test_initialization(self):
        """Test generator initialization."""
        generator = EthicalDatasetGenerator(num_scenarios=500, seed=123)
        
        assert generator.num_scenarios == 500
        assert generator.seed == 123
        assert len(generator.scenarios) == 0
    
    def test_generate_creates_scenarios(self):
        """Test that generate() creates correct number of scenarios."""
        generator = EthicalDatasetGenerator(num_scenarios=100, seed=42)
        scenarios = generator.generate()
        
        assert len(scenarios) == 100
        assert len(generator.scenarios) == 100
    
    def test_scenario_types_distribution(self):
        """Test that scenarios are distributed across types."""
        generator = EthicalDatasetGenerator(num_scenarios=900, seed=42)
        scenarios = generator.generate()
        
        attack_count = sum(1 for s in scenarios if s.action == 'ATTACK')
        eat_count = sum(1 for s in scenarios if s.action == 'EAT')
        neutral_count = sum(1 for s in scenarios if s.action in ['MOVE', 'WAIT'])
        
        # Should have roughly equal distribution (300 each)
        assert 250 <= attack_count <= 350
        assert 250 <= eat_count <= 350
        assert 250 <= neutral_count <= 350
    
    def test_ethical_unethical_balance(self):
        """Test that dataset has both ethical and unethical scenarios."""
        generator = EthicalDatasetGenerator(num_scenarios=1000, seed=42)
        scenarios = generator.generate()
        
        ethical_count = sum(1 for s in scenarios if s.is_ethical)
        unethical_count = sum(1 for s in scenarios if not s.is_ethical)
        
        # Should have significant representation of both
        assert ethical_count > 100
        assert unethical_count > 100
        assert ethical_count + unethical_count == 1000
    
    def test_attack_scenario_rules(self):
        """Test ethical rules for attack scenarios."""
        generator = EthicalDatasetGenerator(num_scenarios=1000, seed=42)
        scenarios = generator.generate()
        
        attack_scenarios = [s for s in scenarios if s.action == 'ATTACK']
        
        # Check unethical attack rule: strong attacking weak
        for scenario in attack_scenarios:
            if scenario.self_energy > 60 and scenario.other_energy < 30:
                assert scenario.is_ethical is False
                assert 'attack' in scenario.scenario_type.lower()
    
    def test_feature_vector_range(self):
        """Test that feature vectors are properly normalized."""
        generator = EthicalDatasetGenerator(num_scenarios=100, seed=42)
        scenarios = generator.generate()
        
        for scenario in scenarios:
            features = scenario.to_feature_vector()
            # All features should be in [0, 1]
            assert np.all(features >= 0.0)
            assert np.all(features <= 1.0)
    
    def test_get_statistics(self):
        """Test dataset statistics calculation."""
        generator = EthicalDatasetGenerator(num_scenarios=100, seed=42)
        generator.generate()
        
        stats = generator.get_statistics()
        
        assert 'total_scenarios' in stats
        assert 'ethical_count' in stats
        assert 'unethical_count' in stats
        assert 'ethical_ratio' in stats
        assert 'action_distribution' in stats
        assert 'scenario_types' in stats
        
        assert stats['total_scenarios'] == 100
        assert stats['ethical_count'] + stats['unethical_count'] == 100
        assert 0.0 <= stats['ethical_ratio'] <= 1.0
    
    def test_statistics_empty_dataset(self):
        """Test statistics on empty dataset."""
        generator = EthicalDatasetGenerator(num_scenarios=100, seed=42)
        stats = generator.get_statistics()
        
        assert stats == {}
    
    def test_to_tensors(self):
        """Test converting dataset to PyTorch tensors."""
        generator = EthicalDatasetGenerator(num_scenarios=50, seed=42)
        generator.generate()
        
        features, labels = generator.to_tensors()
        
        assert isinstance(features, torch.Tensor)
        assert isinstance(labels, torch.Tensor)
        assert features.shape == (50, 8)
        assert labels.shape == (50,)
        assert features.dtype == torch.float32
        assert labels.dtype == torch.float32
    
    def test_to_tensors_empty_raises(self):
        """Test that to_tensors raises on empty dataset."""
        generator = EthicalDatasetGenerator(num_scenarios=100, seed=42)
        
        with pytest.raises(ValueError, match="No scenarios generated"):
            generator.to_tensors()
    
    def test_split_train_test(self):
        """Test train/test split."""
        generator = EthicalDatasetGenerator(num_scenarios=100, seed=42)
        generator.generate()
        
        train, test = generator.split_train_test(test_ratio=0.2)
        
        assert len(train) == 80
        assert len(test) == 20
        assert all(isinstance(s, EthicalScenario) for s in train)
        assert all(isinstance(s, EthicalScenario) for s in test)
    
    def test_split_different_ratios(self):
        """Test different train/test split ratios."""
        generator = EthicalDatasetGenerator(num_scenarios=100, seed=42)
        generator.generate()
        
        train, test = generator.split_train_test(test_ratio=0.3)
        assert len(test) == 30
        assert len(train) == 70
    
    def test_split_empty_raises(self):
        """Test that split raises on empty dataset."""
        generator = EthicalDatasetGenerator(num_scenarios=100, seed=42)
        
        with pytest.raises(ValueError, match="No scenarios generated"):
            generator.split_train_test()
    
    def test_save_and_load(self):
        """Test saving and loading dataset."""
        generator = EthicalDatasetGenerator(num_scenarios=50, seed=42)
        generator.generate()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name
        
        try:
            # Save
            generator.save(filepath)
            assert Path(filepath).exists()
            
            # Load
            loaded_generator = EthicalDatasetGenerator.load(filepath)
            
            assert loaded_generator.num_scenarios == 50
            assert loaded_generator.seed == 42
            assert len(loaded_generator.scenarios) == 50
            
            # Compare first scenario
            original = generator.scenarios[0]
            loaded = loaded_generator.scenarios[0]
            assert original.self_energy == loaded.self_energy
            assert original.other_energy == loaded.other_energy
            assert original.action == loaded.action
            assert original.is_ethical == loaded.is_ethical
        
        finally:
            Path(filepath).unlink()
    
    def test_reproducibility(self):
        """Test that same seed produces same dataset."""
        gen1 = EthicalDatasetGenerator(num_scenarios=100, seed=42)
        scenarios1 = gen1.generate()
        
        gen2 = EthicalDatasetGenerator(num_scenarios=100, seed=42)
        scenarios2 = gen2.generate()
        
        for s1, s2 in zip(scenarios1, scenarios2):
            assert s1.self_energy == pytest.approx(s2.self_energy)
            assert s1.other_energy == pytest.approx(s2.other_energy)
            assert s1.action == s2.action
            assert s1.is_ethical == s2.is_ethical
    
    def test_different_seeds_different_data(self):
        """Test that different seeds produce different datasets."""
        gen1 = EthicalDatasetGenerator(num_scenarios=100, seed=42)
        scenarios1 = gen1.generate()
        
        gen2 = EthicalDatasetGenerator(num_scenarios=100, seed=123)
        scenarios2 = gen2.generate()
        
        # Should have at least some different values
        energy_diffs = [
            abs(s1.self_energy - s2.self_energy) 
            for s1, s2 in zip(scenarios1, scenarios2)
        ]
        assert sum(energy_diffs) > 0  # Not all identical


class TestEthicalRules:
    """Test specific ethical rules implementation."""
    
    def test_strong_attacking_weak_is_unethical(self):
        """Test that strong attacking weak is marked unethical."""
        generator = EthicalDatasetGenerator(num_scenarios=1000, seed=42)
        scenarios = generator.generate()
        
        # Find scenarios matching this pattern
        strong_vs_weak = [
            s for s in scenarios 
            if s.action == 'ATTACK' 
            and s.self_energy > 60 
            and s.other_energy < 30
        ]
        
        # All should be unethical
        for scenario in strong_vs_weak:
            assert scenario.is_ethical is False
    
    def test_attack_with_food_available_is_unethical(self):
        """Test that attacking when food is available is unethical."""
        generator = EthicalDatasetGenerator(num_scenarios=1000, seed=42)
        scenarios = generator.generate()
        
        # Find attack scenarios with food and high self energy
        unnecessary_attacks = [
            s for s in scenarios 
            if s.action == 'ATTACK' 
            and s.food_available 
            and s.self_energy > 50
        ]
        
        # Most should be unethical
        unethical_count = sum(1 for s in unnecessary_attacks if not s.is_ethical)
        assert unethical_count > 0
    
    def test_neutral_actions_are_ethical(self):
        """Test that MOVE and WAIT are always ethical."""
        generator = EthicalDatasetGenerator(num_scenarios=1000, seed=42)
        scenarios = generator.generate()
        
        neutral_actions = [
            s for s in scenarios 
            if s.action in ['MOVE', 'WAIT']
        ]
        
        # All neutral actions should be ethical
        for scenario in neutral_actions:
            assert scenario.is_ethical is True
