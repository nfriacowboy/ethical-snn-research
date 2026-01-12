"""Tests for Ethical SNN (SNN-E)."""

import pytest
import torch
import numpy as np

from src.organisms.ethical_snn import EthicalSNN


class TestEthicalSNNInitialization:
    """Test EthicalSNN initialization."""
    
    def test_default_initialization(self):
        """Test creating EthicalSNN with default parameters."""
        snn = EthicalSNN()
        
        assert snn.input_size == 8
        assert snn.hidden_size == 20
        assert snn.output_size == 2
        assert snn.beta == 0.9
        assert snn.veto_threshold == 0.5
    
    def test_custom_initialization(self):
        """Test creating EthicalSNN with custom parameters."""
        snn = EthicalSNN(
            input_size=12,
            hidden_size=30,
            output_size=3,
            beta=0.85
        )
        
        assert snn.input_size == 12
        assert snn.hidden_size == 30
        assert snn.output_size == 3
        assert snn.beta == 0.85
    
    def test_layers_exist(self):
        """Test that all network layers are created."""
        snn = EthicalSNN()
        
        assert hasattr(snn, 'fc1')
        assert hasattr(snn, 'lif1')
        assert hasattr(snn, 'fc2')
        assert hasattr(snn, 'lif2')
    
    def test_layer_shapes(self):
        """Test that layers have correct shapes."""
        snn = EthicalSNN(input_size=8, hidden_size=20, output_size=2)
        
        assert snn.fc1.in_features == 8
        assert snn.fc1.out_features == 20
        assert snn.fc2.in_features == 20
        assert snn.fc2.out_features == 2


class TestEthicalSNNForward:
    """Test forward pass through EthicalSNN."""
    
    def test_forward_single_sample(self):
        """Test forward pass with single sample."""
        snn = EthicalSNN()
        x = torch.randn(8)
        
        spike_train, mem = snn.forward(x, num_steps=10)
        
        assert spike_train.shape == (10, 2)  # [num_steps, output_size]
        assert mem.shape == (1, 2)  # [batch_size=1, output_size]
    
    def test_forward_batch(self):
        """Test forward pass with batch."""
        snn = EthicalSNN()
        x = torch.randn(5, 8)  # batch_size=5
        
        spike_train, mem = snn.forward(x, num_steps=10)
        
        assert spike_train.shape == (10, 5, 2)  # [num_steps, batch, output]
        assert mem.shape == (5, 2)
    
    def test_forward_different_timesteps(self):
        """Test forward pass with different number of timesteps."""
        snn = EthicalSNN()
        x = torch.randn(8)
        
        spike_train_5, _ = snn.forward(x, num_steps=5)
        spike_train_20, _ = snn.forward(x, num_steps=20)
        
        assert spike_train_5.shape[0] == 5
        assert spike_train_20.shape[0] == 20
    
    def test_forward_produces_binary_spikes(self):
        """Test that forward pass produces binary spikes (0 or 1)."""
        snn = EthicalSNN()
        x = torch.randn(8)
        
        spike_train, _ = snn.forward(x, num_steps=10)
        
        # All values should be 0 or 1
        assert torch.all((spike_train == 0) | (spike_train == 1))


class TestStateEncoding:
    """Test state encoding for ethical evaluation."""
    
    def test_encode_state_shape(self):
        """Test that encode_state produces correct shape."""
        snn = EthicalSNN()
        
        state = snn.encode_state(
            self_energy=50.0,
            other_energy=75.0,
            food_available=True,
            distance_to_other=10.0,
            proposed_action='ATTACK'
        )
        
        assert state.shape == (8,)
        assert state.dtype == torch.float32
    
    def test_encode_state_normalization(self):
        """Test that continuous values are normalized to [0, 1]."""
        snn = EthicalSNN()
        
        state = snn.encode_state(
            self_energy=50.0,
            other_energy=100.0,
            food_available=False,
            distance_to_other=20.0,
            proposed_action='EAT'
        )
        
        assert state[0] == pytest.approx(0.5)  # self_energy / 100
        assert state[1] == pytest.approx(1.0)  # other_energy / 100
        assert state[2] == 0.0  # food_available
        assert state[3] == pytest.approx(1.0)  # distance / 20
    
    def test_encode_state_action_encoding(self):
        """Test one-hot encoding of actions."""
        snn = EthicalSNN()
        
        actions = ['ATTACK', 'EAT', 'MOVE', 'WAIT']
        expected_indices = [4, 5, 6, 7]
        
        for action, expected_idx in zip(actions, expected_indices):
            state = snn.encode_state(
                self_energy=50.0,
                other_energy=50.0,
                food_available=False,
                distance_to_other=10.0,
                proposed_action=action
            )
            
            assert state[expected_idx] == 1.0
            # Other action indices should be 0
            other_indices = [i for i in range(4, 8) if i != expected_idx]
            assert all(state[i] == 0.0 for i in other_indices)
    
    def test_encode_state_food_available(self):
        """Test encoding of food_available flag."""
        snn = EthicalSNN()
        
        state_with_food = snn.encode_state(
            self_energy=50.0,
            other_energy=50.0,
            food_available=True,
            distance_to_other=10.0,
            proposed_action='EAT'
        )
        
        state_no_food = snn.encode_state(
            self_energy=50.0,
            other_energy=50.0,
            food_available=False,
            distance_to_other=10.0,
            proposed_action='EAT'
        )
        
        assert state_with_food[2] == 1.0
        assert state_no_food[2] == 0.0


class TestActionEvaluation:
    """Test ethical action evaluation."""
    
    def test_evaluate_action_returns_bool(self):
        """Test that evaluate_action returns boolean."""
        snn = EthicalSNN()
        
        result = snn.evaluate_action(
            self_energy=50.0,
            other_energy=50.0,
            food_available=True,
            distance_to_other=10.0,
            proposed_action='EAT'
        )
        
        assert isinstance(result, bool)
    
    def test_evaluate_different_actions(self):
        """Test evaluating different action types."""
        snn = EthicalSNN()
        
        actions = ['ATTACK', 'EAT', 'MOVE', 'WAIT']
        
        for action in actions:
            result = snn.evaluate_action(
                self_energy=50.0,
                other_energy=50.0,
                food_available=True,
                distance_to_other=10.0,
                proposed_action=action,
                num_steps=10
            )
            assert isinstance(result, bool)
    
    def test_evaluate_action_deterministic(self):
        """Test that evaluation is deterministic (no randomness)."""
        snn = EthicalSNN()
        
        # Same input should give same output
        results = []
        for _ in range(5):
            result = snn.evaluate_action(
                self_energy=80.0,
                other_energy=20.0,
                food_available=False,
                distance_to_other=2.0,
                proposed_action='ATTACK',
                num_steps=10
            )
            results.append(result)
        
        # All results should be identical
        assert all(r == results[0] for r in results)
    
    def test_no_spikes_defaults_to_approve(self):
        """Test that no spikes defaults to approval (conservative)."""
        snn = EthicalSNN()
        
        # Zero input should produce no spikes
        with torch.no_grad():
            # Set all weights to zero to ensure no spikes
            for param in snn.parameters():
                param.data.zero_()
        
        result = snn.evaluate_action(
            self_energy=50.0,
            other_energy=50.0,
            food_available=True,
            distance_to_other=10.0,
            proposed_action='WAIT',
            num_steps=10
        )
        
        # Should default to True when uncertain
        assert result is True


class TestEthicalScore:
    """Test continuous ethical score computation."""
    
    def test_get_ethical_score_range(self):
        """Test that ethical score is in [0, 1]."""
        snn = EthicalSNN()
        
        score = snn.get_ethical_score(
            self_energy=50.0,
            other_energy=50.0,
            food_available=True,
            distance_to_other=10.0,
            proposed_action='EAT',
            num_steps=10
        )
        
        assert 0.0 <= score <= 1.0
    
    def test_ethical_score_no_spikes(self):
        """Test ethical score when no spikes occur."""
        snn = EthicalSNN()
        
        # Zero weights = no spikes
        with torch.no_grad():
            for param in snn.parameters():
                param.data.zero_()
        
        score = snn.get_ethical_score(
            self_energy=50.0,
            other_energy=50.0,
            food_available=True,
            distance_to_other=10.0,
            proposed_action='WAIT',
            num_steps=10
        )
        
        # Should return 0.5 (neutral) when uncertain
        assert score == 0.5
    
    def test_ethical_score_deterministic(self):
        """Test that ethical score is deterministic."""
        snn = EthicalSNN()
        
        scores = []
        for _ in range(5):
            score = snn.get_ethical_score(
                self_energy=80.0,
                other_energy=20.0,
                food_available=False,
                distance_to_other=2.0,
                proposed_action='ATTACK',
                num_steps=10
            )
            scores.append(score)
        
        # All scores should be identical
        assert all(s == pytest.approx(scores[0]) for s in scores)


class TestStatistics:
    """Test network statistics and information."""
    
    def test_get_statistics(self):
        """Test getting network statistics."""
        snn = EthicalSNN()
        stats = snn.get_statistics()
        
        assert 'type' in stats
        assert 'input_size' in stats
        assert 'hidden_size' in stats
        assert 'output_size' in stats
        assert 'beta' in stats
        assert 'veto_threshold' in stats
        assert 'total_parameters' in stats
        assert 'trainable_parameters' in stats
    
    def test_statistics_values(self):
        """Test that statistics contain correct values."""
        snn = EthicalSNN(input_size=8, hidden_size=20, output_size=2)
        stats = snn.get_statistics()
        
        assert stats['type'] == 'EthicalSNN'
        assert stats['input_size'] == 8
        assert stats['hidden_size'] == 20
        assert stats['output_size'] == 2
        assert stats['beta'] == 0.9
    
    def test_parameter_count(self):
        """Test that parameter count is reasonable."""
        snn = EthicalSNN(input_size=8, hidden_size=20, output_size=2)
        stats = snn.get_statistics()
        
        # fc1: 8*20 + 20 = 180, fc2: 20*2 + 2 = 42, total = 222
        expected_params = (8 * 20 + 20) + (20 * 2 + 2)
        assert stats['total_parameters'] == expected_params


class TestMiscellaneous:
    """Test miscellaneous functionality."""
    
    def test_repr(self):
        """Test string representation."""
        snn = EthicalSNN()
        repr_str = repr(snn)
        
        assert 'EthicalSNN' in repr_str
        assert 'input=' in repr_str
        assert 'hidden=' in repr_str
        assert 'output=' in repr_str
        assert 'params=' in repr_str
    
    def test_reset_membrane_potentials(self):
        """Test reset method (should not raise errors)."""
        snn = EthicalSNN()
        snn.reset_membrane_potentials()  # Should not raise
    
    def test_multiple_forward_passes(self):
        """Test multiple forward passes don't interfere."""
        snn = EthicalSNN()
        x = torch.randn(8)
        
        spike_train1, _ = snn.forward(x, num_steps=10)
        spike_train2, _ = snn.forward(x, num_steps=10)
        
        # Should produce identical results (deterministic)
        assert torch.allclose(spike_train1, spike_train2)
    
    def test_gradient_computation(self):
        """Test that gradients can be computed (for training)."""
        snn = EthicalSNN()
        x = torch.randn(8, requires_grad=True)
        
        spike_train, mem = snn.forward(x, num_steps=5)
        loss = spike_train.sum()  # Dummy loss
        loss.backward()
        
        # Check that gradients exist
        assert any(p.grad is not None for p in snn.parameters())
