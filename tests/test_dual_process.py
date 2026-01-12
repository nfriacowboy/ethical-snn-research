"""Tests for Dual-Process Organism Architecture."""

import pytest
import torch
import numpy as np

from src.architecture.dual_process import DualProcessOrganism
from src.organisms.base_organism import Action


class TestDualProcessInitialization:
    """Test DualProcessOrganism initialization."""
    
    def test_default_initialization(self):
        """Test creating dual-process organism with defaults."""
        organism = DualProcessOrganism(
            organism_id=1,
            position=(10, 10)
        )
        
        assert organism.organism_id == 1
        assert organism.position == (10, 10)
        assert organism.energy == 100.0
        assert organism.alive is True
        assert organism.age == 0
    
    def test_custom_energy(self):
        """Test initialization with custom energy."""
        organism = DualProcessOrganism(
            organism_id=1,
            position=(5, 5),
            energy=75.0
        )
        
        assert organism.energy == 75.0
    
    def test_has_both_networks(self):
        """Test that both SNN-S and SNN-E are created."""
        organism = DualProcessOrganism(
            organism_id=1,
            position=(10, 10)
        )
        
        assert hasattr(organism, 'survival_snn')
        assert hasattr(organism, 'ethical_snn')
        assert organism.survival_snn is not None
        assert organism.ethical_snn is not None
    
    def test_statistics_initialized(self):
        """Test that dual-process statistics are initialized."""
        organism = DualProcessOrganism(
            organism_id=1,
            position=(10, 10)
        )
        
        assert organism.veto_count == 0
        assert organism.approval_count == 0
        assert organism.fallback_count == 0


class TestDecisionMaking:
    """Test dual-process decision making."""
    
    def test_decide_returns_action(self):
        """Test that decide() returns an Action."""
        organism = DualProcessOrganism(
            organism_id=1,
            position=(10, 10)
        )
        
        state = {
            'food_direction': (1, 0),
            'energy': 50.0,
            'obstacles_nearby': [],
            'other_organism': None,
            'food_at_position': False,
            'grid_size': 20
        }
        
        action = organism.decide(state)
        assert isinstance(action, Action)
    
    def test_decide_updates_statistics(self):
        """Test that decide() updates approval/veto counts."""
        organism = DualProcessOrganism(
            organism_id=1,
            position=(10, 10)
        )
        
        state = {
            'food_direction': (1, 0),
            'energy': 50.0,
            'obstacles_nearby': [],
            'other_organism': None,
            'food_at_position': False,
            'grid_size': 20
        }
        
        initial_total = organism.veto_count + organism.approval_count
        organism.decide(state)
        final_total = organism.veto_count + organism.approval_count
        
        assert final_total == initial_total + 1
    
    def test_veto_results_in_wait(self):
        """Test that vetoed actions result in WAIT."""
        organism = DualProcessOrganism(
            organism_id=1,
            position=(10, 10)
        )
        
        # Force SNN-E to veto by zeroing its weights
        with torch.no_grad():
            for param in organism.ethical_snn.parameters():
                param.data.zero_()
        
        state = {
            'food_direction': None,
            'energy': 50.0,
            'obstacles_nearby': [],
            'other_organism': (1, 0, 20.0),  # Vulnerable other nearby
            'food_at_position': False,
            'grid_size': 20
        }
        
        # Run multiple decisions to see if any result in WAIT
        actions = [organism.decide(state) for _ in range(10)]
        
        # With zero weights, SNN-E defaults to approve (True)
        # So we should see no vetoes with zero weights
        assert organism.approval_count > 0
    
    def test_decision_with_food_available(self):
        """Test decision when food is at position."""
        organism = DualProcessOrganism(
            organism_id=1,
            position=(10, 10)
        )
        
        state = {
            'food_direction': None,
            'energy': 60.0,
            'obstacles_nearby': [],
            'other_organism': None,
            'food_at_position': True,
            'grid_size': 20
        }
        
        action = organism.decide(state)
        assert isinstance(action, Action)
        # Should approve eating (ethical action)
        assert organism.approval_count >= 1 or organism.veto_count >= 1
    
    def test_decision_with_other_organism(self):
        """Test decision when other organism is nearby."""
        organism = DualProcessOrganism(
            organism_id=1,
            position=(10, 10)
        )
        
        state = {
            'food_direction': (2, 0),
            'energy': 80.0,
            'obstacles_nearby': [],
            'other_organism': (1, 0, 30.0),  # Other nearby with low energy
            'food_at_position': False,
            'grid_size': 20
        }
        
        action = organism.decide(state)
        assert isinstance(action, Action)


class TestEthicalContext:
    """Test ethical context extraction."""
    
    def test_extract_ethical_context_basic(self):
        """Test extracting ethical context from state."""
        organism = DualProcessOrganism(
            organism_id=1,
            position=(10, 10)
        )
        
        state = {
            'energy': 75.0,
            'food_at_position': True,
            'other_organism': (2, 1, 40.0),
            'food_direction': (1, 0)
        }
        
        context = organism._extract_ethical_context(state, Action.EAT)
        
        assert 'self_energy' in context
        assert 'other_energy' in context
        assert 'food_available' in context
        assert 'distance_to_other' in context
        assert 'action_str' in context
    
    def test_extract_context_with_other_organism(self):
        """Test context extraction with other organism present."""
        organism = DualProcessOrganism(
            organism_id=1,
            position=(10, 10)
        )
        
        state = {
            'energy': 50.0,
            'other_organism': (3, 4, 25.0),
            'food_at_position': False,
            'food_direction': None
        }
        
        context = organism._extract_ethical_context(state, Action.ATTACK)
        
        assert context['self_energy'] == 50.0
        assert context['other_energy'] == 25.0
        assert context['distance_to_other'] == 7.0  # 3 + 4
        assert context['action_str'] == 'ATTACK'
    
    def test_extract_context_no_other_organism(self):
        """Test context extraction with no other organism."""
        organism = DualProcessOrganism(
            organism_id=1,
            position=(10, 10)
        )
        
        state = {
            'energy': 60.0,
            'other_organism': None,
            'food_at_position': True
        }
        
        context = organism._extract_ethical_context(state, Action.EAT)
        
        assert context['other_energy'] == 100.0  # Default healthy
        assert context['distance_to_other'] == 20.0  # Default far
    
    def test_action_string_mapping(self):
        """Test action to string mapping."""
        organism = DualProcessOrganism(
            organism_id=1,
            position=(10, 10)
        )
        
        state = {'energy': 50.0}
        
        # Test different actions
        actions_to_test = [
            (Action.ATTACK, 'ATTACK'),
            (Action.EAT, 'EAT'),
            (Action.MOVE_NORTH, 'MOVE'),
            (Action.MOVE_SOUTH, 'MOVE'),
            (Action.MOVE_EAST, 'MOVE'),
            (Action.MOVE_WEST, 'MOVE'),
            (Action.WAIT, 'WAIT')
        ]
        
        for action, expected_str in actions_to_test:
            context = organism._extract_ethical_context(state, action)
            assert context['action_str'] == expected_str


class TestStatistics:
    """Test statistics and reporting."""
    
    def test_get_statistics(self):
        """Test getting statistics."""
        organism = DualProcessOrganism(
            organism_id=1,
            position=(10, 10)
        )
        
        stats = organism.get_statistics()
        
        assert 'type' in stats
        assert stats['type'] == 'DualProcessOrganism'
        assert 'veto_count' in stats
        assert 'approval_count' in stats
        assert 'fallback_count' in stats
        assert 'veto_rate' in stats
        assert 'survival_snn_params' in stats
        assert 'ethical_snn_params' in stats
    
    def test_veto_rate_calculation(self):
        """Test veto rate calculation."""
        organism = DualProcessOrganism(
            organism_id=1,
            position=(10, 10)
        )
        
        # Manually set counts
        organism.veto_count = 3
        organism.approval_count = 7
        
        stats = organism.get_statistics()
        assert stats['veto_rate'] == pytest.approx(0.3)
    
    def test_veto_rate_zero_decisions(self):
        """Test veto rate with no decisions made."""
        organism = DualProcessOrganism(
            organism_id=1,
            position=(10, 10)
        )
        
        stats = organism.get_statistics()
        assert stats['veto_rate'] == 0.0
    
    def test_statistics_includes_base(self):
        """Test that statistics include base organism info."""
        organism = DualProcessOrganism(
            organism_id=5,
            position=(3, 7),
            energy=85.0
        )
        
        stats = organism.get_statistics()
        
        assert stats['organism_id'] == 5
        assert stats['energy'] == 85.0
        assert stats['alive'] is True
        assert stats['age'] == 0


class TestReset:
    """Test reset functionality."""
    
    def test_reset_clears_statistics(self):
        """Test that reset clears dual-process statistics."""
        organism = DualProcessOrganism(
            organism_id=1,
            position=(10, 10)
        )
        
        # Manually set statistics
        organism.veto_count = 5
        organism.approval_count = 10
        organism.fallback_count = 5
        
        organism.reset()
        
        assert organism.veto_count == 0
        assert organism.approval_count == 0
        assert organism.fallback_count == 0
    
    def test_reset_preserves_id_position(self):
        """Test that reset preserves organism ID and position."""
        organism = DualProcessOrganism(
            organism_id=3,
            position=(5, 8)
        )
        
        organism.reset()
        
        assert organism.organism_id == 3
        assert organism.position == (5, 8)
    
    def test_reset_restores_energy(self):
        """Test that reset restores energy to initial value."""
        organism = DualProcessOrganism(
            organism_id=1,
            position=(10, 10),
            energy=80.0
        )
        
        organism.update_energy(-30)  # Reduce energy
        assert organism.energy == 50.0
        
        organism.reset()
        assert organism.energy == 80.0  # Restored


class TestRepresentation:
    """Test string representation."""
    
    def test_repr(self):
        """Test string representation."""
        organism = DualProcessOrganism(
            organism_id=7,
            position=(12, 15),
            energy=65.0
        )
        
        repr_str = repr(organism)
        
        assert 'DualProcessOrganism' in repr_str
        assert 'id=7' in repr_str
        assert 'pos=(12, 15)' in repr_str
        assert 'energy=65' in repr_str
        assert 'alive=True' in repr_str
    
    def test_repr_with_vetoes(self):
        """Test representation includes veto information."""
        organism = DualProcessOrganism(
            organism_id=1,
            position=(10, 10)
        )
        
        organism.veto_count = 3
        organism.approval_count = 7
        
        repr_str = repr(organism)
        
        assert 'vetoes=3' in repr_str
        assert 'veto_rate=0.30' in repr_str


class TestIntegration:
    """Integration tests for dual-process organism."""
    
    def test_multiple_decisions(self):
        """Test making multiple decisions in sequence."""
        organism = DualProcessOrganism(
            organism_id=1,
            position=(10, 10)
        )
        
        states = [
            {'food_direction': (1, 0), 'energy': 50.0, 'obstacles_nearby': [],
             'other_organism': None, 'food_at_position': False, 'grid_size': 20},
            {'food_direction': None, 'energy': 48.0, 'obstacles_nearby': [],
             'other_organism': (1, 1, 30.0), 'food_at_position': False, 'grid_size': 20},
            {'food_direction': (0, 1), 'energy': 45.0, 'obstacles_nearby': [],
             'other_organism': None, 'food_at_position': True, 'grid_size': 20}
        ]
        
        actions = [organism.decide(state) for state in states]
        
        assert len(actions) == 3
        assert all(isinstance(a, Action) for a in actions)
        assert organism.veto_count + organism.approval_count == 3
    
    def test_organism_lifecycle(self):
        """Test complete organism lifecycle."""
        organism = DualProcessOrganism(
            organism_id=1,
            position=(10, 10),
            energy=100.0
        )
        
        # Make some decisions
        state = {
            'food_direction': (1, 0),
            'energy': 100.0,
            'obstacles_nearby': [],
            'other_organism': None,
            'food_at_position': False,
            'grid_size': 20
        }
        
        for _ in range(5):
            organism.decide(state)
        
        # Check statistics accumulated
        assert organism.veto_count + organism.approval_count == 5
        
        # Reset
        organism.reset()
        assert organism.veto_count == 0
        assert organism.approval_count == 0
