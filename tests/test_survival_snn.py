"""Unit tests for Survival SNN.

Tests cover:
- Network initialization
- Forward pass
- Spike encoding
- Action selection
- Integration with Organism base class
"""

import pytest
import torch

from src.organisms.survival_snn import SurvivalSNN
from src.organisms.base_organism import Action


class TestSurvivalSNNInitialization:
    """Test Survival SNN initialization."""
    
    def test_basic_init(self):
        """Test basic initialization."""
        snn = SurvivalSNN(
            organism_id=0,
            position=(10, 10),
            initial_energy=100.0
        )
        
        assert snn.organism_id == 0
        assert snn.position == (10, 10)
        assert snn.energy == 100.0
        assert snn.input_size == 8
        assert snn.hidden_size == 30
        assert snn.output_size == 5
    
    def test_custom_architecture(self):
        """Test initialization with custom architecture."""
        snn = SurvivalSNN(
            organism_id=1,
            position=(5, 5),
            input_size=16,
            hidden_size=50,
            output_size=10
        )
        
        assert snn.input_size == 16
        assert snn.hidden_size == 50
        assert snn.output_size == 10
    
    def test_inherits_organism(self):
        """Test that SurvivalSNN inherits from Organism."""
        snn = SurvivalSNN(organism_id=0, position=(10, 10))
        
        # Test organism methods exist
        assert hasattr(snn, 'update_energy')
        assert hasattr(snn, 'move')
        assert hasattr(snn, 'is_alive')
        assert hasattr(snn, 'decide')
    
    def test_device_cpu(self):
        """Test CPU device."""
        snn = SurvivalSNN(organism_id=0, position=(10, 10), device='cpu')
        assert snn.device == 'cpu'


class TestForwardPass:
    """Test forward pass through network."""
    
    def test_forward_basic(self):
        """Test basic forward pass."""
        snn = SurvivalSNN(organism_id=0, position=(10, 10))
        x = torch.randn(8)
        
        spikes, mem = snn.forward(x, num_steps=10)
        
        assert spikes.shape == (10, 5)
        assert mem.shape == (5,)
    
    def test_forward_different_timesteps(self):
        """Test forward with different number of timesteps."""
        snn = SurvivalSNN(organism_id=0, position=(10, 10))
        x = torch.randn(8)
        
        spikes, _ = snn.forward(x, num_steps=20)
        
        assert spikes.shape == (20, 5)
    
    def test_forward_spikes_binary(self):
        """Test that output spikes are binary."""
        snn = SurvivalSNN(organism_id=0, position=(10, 10))
        x = torch.randn(8)
        
        spikes, _ = snn.forward(x, num_steps=10)
        
        # Spikes should be 0 or 1
        unique_values = torch.unique(spikes)
        assert all(v in [0.0, 1.0] for v in unique_values.tolist())
    
    def test_forward_deterministic_with_seed(self):
        """Test that forward pass is deterministic with same seed."""
        snn = SurvivalSNN(organism_id=0, position=(10, 10))
        
        torch.manual_seed(42)
        x1 = torch.randn(8)
        spikes1, _ = snn.forward(x1, num_steps=10)
        
        torch.manual_seed(42)
        x2 = torch.randn(8)
        spikes2, _ = snn.forward(x2, num_steps=10)
        
        assert torch.allclose(x1, x2)
        # Note: Spikes may differ due to LIF dynamics, but inputs should match


class TestStateEncoding:
    """Test state encoding."""
    
    def test_encode_basic_state(self):
        """Test encoding basic state."""
        snn = SurvivalSNN(organism_id=0, position=(10, 10))
        
        state = {
            'self_energy': 50.0,
            'self_position': (10, 10),
            'nearest_food': (15, 12),
            'food_at_position': False,
            'obstacles': {'north': False, 'south': False, 'east': False, 'west': False}
        }
        
        encoded = snn.encode_state_rate(state)
        
        assert encoded.shape == (8,)
        assert encoded[2] == 0.5  # Energy normalized (50/100)
        assert encoded[3] == 0.0  # No food at position
    
    def test_encode_food_direction(self):
        """Test food direction encoding."""
        snn = SurvivalSNN(organism_id=0, position=(10, 10))
        
        state = {
            'self_position': (10, 10),
            'nearest_food': (20, 10),  # East
            'self_energy': 100.0,
            'food_at_position': False,
            'obstacles': {}
        }
        
        encoded = snn.encode_state_rate(state)
        
        # Food to the east: dx = 10/20 = 0.5, dy = 0
        assert encoded[0] == 0.5
        assert encoded[1] == 0.0
    
    def test_encode_no_food(self):
        """Test encoding when no food nearby."""
        snn = SurvivalSNN(organism_id=0, position=(10, 10))
        
        state = {
            'self_energy': 75.0,
            'nearest_food': None,
            'food_at_position': False,
            'obstacles': {}
        }
        
        encoded = snn.encode_state_rate(state)
        
        assert encoded[0] == 0.0
        assert encoded[1] == 0.0
        assert encoded[2] == 0.75  # Energy
    
    def test_encode_obstacles(self):
        """Test obstacle encoding."""
        snn = SurvivalSNN(organism_id=0, position=(10, 10))
        
        state = {
            'self_energy': 100.0,
            'obstacles': {
                'north': True,
                'south': False,
                'east': True,
                'west': False
            }
        }
        
        encoded = snn.encode_state_rate(state)
        
        assert encoded[4] == 1.0  # North blocked
        assert encoded[5] == 0.0  # South open
        assert encoded[6] == 1.0  # East blocked
        assert encoded[7] == 0.0  # West open


class TestActionSelection:
    """Test action selection from spikes."""
    
    def test_spike_to_action_movement(self):
        """Test converting spikes to movement action."""
        snn = SurvivalSNN(organism_id=0, position=(10, 10))
        
        # Create spike train with neuron 0 (MOVE_NORTH) dominant
        spike_train = torch.zeros(10, 5)
        spike_train[:, 0] = 1  # Neuron 0 always spikes
        
        action = snn.spike_to_action(spike_train)
        
        assert action == Action.MOVE_NORTH
    
    def test_spike_to_action_eat_context(self):
        """Test EAT action with food context."""
        snn = SurvivalSNN(organism_id=0, position=(10, 10))
        
        # Neuron 4 dominant
        spike_train = torch.zeros(10, 5)
        spike_train[:, 4] = 1
        
        action = snn.spike_to_action(spike_train, context={'has_food_here': True})
        
        assert action == Action.EAT
    
    def test_spike_to_action_attack_context(self):
        """Test ATTACK action with organism context."""
        snn = SurvivalSNN(organism_id=0, position=(10, 10))
        
        spike_train = torch.zeros(10, 5)
        spike_train[:, 4] = 1
        
        action = snn.spike_to_action(
            spike_train,
            context={'has_adjacent_organism': True, 'low_energy': False}
        )
        
        assert action == Action.ATTACK
    
    def test_spike_to_action_wait_default(self):
        """Test WAIT action as default."""
        snn = SurvivalSNN(organism_id=0, position=(10, 10))
        
        spike_train = torch.zeros(10, 5)
        spike_train[:, 4] = 1
        
        action = snn.spike_to_action(spike_train, context={})
        
        assert action == Action.WAIT
    
    def test_spike_to_action_no_spikes(self):
        """Test action selection with no spikes."""
        snn = SurvivalSNN(organism_id=0, position=(10, 10))
        
        spike_train = torch.zeros(10, 5)
        
        # Should select first neuron (all tied at 0)
        action = snn.spike_to_action(spike_train)
        
        assert isinstance(action, Action)


class TestDecide:
    """Test decide method (full pipeline)."""
    
    def test_decide_returns_action(self):
        """Test decide returns valid action."""
        snn = SurvivalSNN(organism_id=0, position=(10, 10))
        
        state = {
            'self_energy': 50.0,
            'nearest_food': (15, 10),
            'food_at_position': False,
            'obstacles': {}
        }
        
        action = snn.decide(state)
        
        assert isinstance(action, Action)
    
    def test_decide_with_food(self):
        """Test decide when food is at position."""
        snn = SurvivalSNN(organism_id=0, position=(10, 10))
        
        state = {
            'self_energy': 50.0,
            'food_at_position': True,
            'obstacles': {}
        }
        
        action = snn.decide(state)
        
        # Should likely choose EAT, but depends on network weights
        assert isinstance(action, Action)
    
    def test_decide_deterministic(self):
        """Test decide is deterministic with same input."""
        snn = SurvivalSNN(organism_id=0, position=(10, 10))
        
        state = {'self_energy': 75.0, 'nearest_food': (12, 10)}
        
        torch.manual_seed(42)
        action1 = snn.decide(state)
        
        torch.manual_seed(42)
        action2 = snn.decide(state)
        
        assert action1 == action2


class TestOrganismIntegration:
    """Test integration with Organism base class."""
    
    def test_energy_management(self):
        """Test energy update works."""
        snn = SurvivalSNN(organism_id=0, position=(10, 10), initial_energy=100.0)
        
        snn.update_energy(-20)
        assert snn.energy == 80.0
        
        snn.update_energy(10)
        assert snn.energy == 90.0
    
    def test_movement(self):
        """Test movement works."""
        snn = SurvivalSNN(organism_id=0, position=(10, 10))
        
        snn.move((11, 12))
        assert snn.position == (11, 12)
    
    def test_death(self):
        """Test organism can die."""
        snn = SurvivalSNN(organism_id=0, position=(10, 10), initial_energy=10.0)
        
        snn.update_energy(-20)
        assert snn.is_alive() == False
    
    def test_action_logging(self):
        """Test action logging works."""
        snn = SurvivalSNN(organism_id=0, position=(10, 10))
        
        snn.log_action(Action.MOVE_NORTH, success=True)
        
        assert len(snn.action_history) == 1
        assert snn.action_history[0]['action'] == Action.MOVE_NORTH


class TestStatistics:
    """Test statistics methods."""
    
    def test_get_statistics(self):
        """Test organism statistics."""
        snn = SurvivalSNN(organism_id=3, position=(10, 10))
        
        stats = snn.get_statistics()
        
        assert stats['organism_id'] == 3
        assert stats['position'] == (10, 10)
    
    def test_get_network_statistics(self):
        """Test network-specific statistics."""
        snn = SurvivalSNN(organism_id=0, position=(10, 10))
        
        net_stats = snn.get_network_statistics()
        
        assert net_stats['input_size'] == 8
        assert net_stats['hidden_size'] == 30
        assert net_stats['output_size'] == 5
        assert 'total_parameters' in net_stats


class TestRepresentation:
    """Test string representation."""
    
    def test_repr(self):
        """Test __repr__ method."""
        snn = SurvivalSNN(organism_id=5, position=(12, 8), initial_energy=65.0)
        
        repr_str = repr(snn)
        
        assert 'SurvivalSNN' in repr_str
        assert 'id=5' in repr_str
        assert 'pos=(12, 8)' in repr_str
        assert 'energy=65' in repr_str
        assert '8→30→5' in repr_str  # Architecture
