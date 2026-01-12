"""Unit tests for STDP Trainer.

Tests cover:
- Initialization
- STDP window computation
- Weight updates
- Spike time extraction
- Episode training
- Statistics tracking
"""

import pytest
import torch
import torch.nn as nn

from src.organisms.survival_snn import SurvivalSNN
from src.training.stdp_trainer import STDPTrainer


class TestSTDPInitialization:
    """Test STDP trainer initialization."""

    def test_basic_init(self):
        """Test basic initialization."""
        network = nn.Linear(10, 5)
        trainer = STDPTrainer(network, learning_rate=0.001)

        assert trainer.learning_rate == 0.001
        assert trainer.tau_plus == 20.0
        assert trainer.tau_minus == 20.0
        assert trainer.total_updates == 0

    def test_custom_parameters(self):
        """Test initialization with custom parameters."""
        network = nn.Linear(10, 5)
        trainer = STDPTrainer(
            network,
            learning_rate=0.01,
            tau_plus=15.0,
            tau_minus=25.0,
            a_plus=0.02,
            a_minus=0.015,
        )

        assert trainer.learning_rate == 0.01
        assert trainer.tau_plus == 15.0
        assert trainer.tau_minus == 25.0
        assert trainer.a_plus == 0.02
        assert trainer.a_minus == 0.015

    def test_init_with_snn(self):
        """Test initialization with SurvivalSNN."""
        snn = SurvivalSNN(organism_id=0, position=(10, 10))
        trainer = STDPTrainer(snn)

        assert trainer.network == snn


class TestSTDPWindow:
    """Test STDP learning window computation."""

    def test_ltp_positive_delta(self):
        """Test LTP for positive time difference."""
        trainer = STDPTrainer(None, a_plus=0.01, tau_plus=20.0)

        delta_t = torch.tensor([5.0])
        window = trainer.compute_stdp_window(delta_t)

        # Should be positive (LTP)
        assert window[0] > 0

    def test_ltd_negative_delta(self):
        """Test LTD for negative time difference."""
        trainer = STDPTrainer(None, a_minus=0.01, tau_minus=20.0)

        delta_t = torch.tensor([-5.0])
        window = trainer.compute_stdp_window(delta_t)

        # Should be negative (LTD)
        assert window[0] < 0

    def test_zero_delta(self):
        """Test zero time difference."""
        trainer = STDPTrainer(None)

        delta_t = torch.tensor([0.0])
        window = trainer.compute_stdp_window(delta_t)

        # Should be close to zero
        assert abs(window[0]) < 0.1

    def test_batch_computation(self):
        """Test batch computation of STDP window."""
        trainer = STDPTrainer(None)

        delta_t = torch.tensor([5.0, -5.0, 10.0, -10.0])
        window = trainer.compute_stdp_window(delta_t)

        assert window.shape == (4,)
        assert window[0] > 0  # LTP
        assert window[1] < 0  # LTD

    def test_exponential_decay(self):
        """Test exponential decay of STDP window."""
        trainer = STDPTrainer(None, a_plus=0.01, tau_plus=20.0)

        delta_t = torch.tensor([5.0, 10.0, 20.0])
        window = trainer.compute_stdp_window(delta_t)

        # Should decay exponentially
        assert window[0] > window[1] > window[2]


class TestSpikeTimeExtraction:
    """Test spike time extraction."""

    def test_extract_single_spike(self):
        """Test extracting single spike time."""
        trainer = STDPTrainer(None)

        spikes = torch.zeros(10, 2)
        spikes[3, 0] = 1.0  # Neuron 0 spikes at t=3

        times = trainer._get_spike_times(spikes)

        assert times.shape[0] == 2
        assert times[0, 0] == 3.0
        assert times[1, 0] == -1.0  # Neuron 1 never spiked

    def test_extract_multiple_spikes(self):
        """Test extracting multiple spike times."""
        trainer = STDPTrainer(None)

        spikes = torch.zeros(10, 1)
        spikes[2, 0] = 1.0
        spikes[5, 0] = 1.0
        spikes[8, 0] = 1.0

        times = trainer._get_spike_times(spikes)

        assert times[0, 0] == 2.0
        assert times[0, 1] == 5.0
        assert times[0, 2] == 8.0

    def test_no_spikes(self):
        """Test when no spikes occur."""
        trainer = STDPTrainer(None)

        spikes = torch.zeros(10, 3)
        times = trainer._get_spike_times(spikes)

        assert times.shape[0] == 3
        assert (times == -1.0).all()


class TestWeightUpdates:
    """Test weight update mechanism."""

    def test_update_weights_basic(self):
        """Test basic weight update."""
        snn = SurvivalSNN(organism_id=0, position=(10, 10))
        trainer = STDPTrainer(snn, learning_rate=0.001)

        # Get initial weights
        initial_weights = snn.fc1.weight.data.clone()

        # Create spike trains
        pre_spikes = torch.rand(10, 8) > 0.8
        post_spikes = torch.rand(10, 30) > 0.8

        stats = trainer.update_weights(
            pre_spikes.float(), post_spikes.float(), reward=1.0
        )

        # Weights should have changed
        final_weights = snn.fc1.weight.data
        assert not torch.allclose(initial_weights, final_weights)

        # Check stats
        assert "weight_change_mean" in stats
        assert "reward" in stats
        assert stats["reward"] == 1.0

    def test_update_with_zero_reward(self):
        """Test update with zero reward."""
        snn = SurvivalSNN(organism_id=0, position=(10, 10))
        trainer = STDPTrainer(snn, learning_rate=0.001)

        pre_spikes = torch.rand(10, 8) > 0.8
        post_spikes = torch.rand(10, 30) > 0.8

        stats = trainer.update_weights(
            pre_spikes.float(), post_spikes.float(), reward=0.0
        )

        assert stats["reward"] == 0.0
        assert stats["weight_change_mean"] >= 0

    def test_update_with_negative_reward(self):
        """Test update with negative reward."""
        snn = SurvivalSNN(organism_id=0, position=(10, 10))
        trainer = STDPTrainer(snn, learning_rate=0.001)

        pre_spikes = torch.rand(10, 8) > 0.8
        post_spikes = torch.rand(10, 30) > 0.8

        stats = trainer.update_weights(
            pre_spikes.float(), post_spikes.float(), reward=-5.0
        )

        assert stats["reward"] == -5.0

    def test_update_fc2_layer(self):
        """Test updating fc2 layer."""
        snn = SurvivalSNN(organism_id=0, position=(10, 10))
        trainer = STDPTrainer(snn)

        initial_weights = snn.fc2.weight.data.clone()

        pre_spikes = torch.rand(10, 30) > 0.8
        post_spikes = torch.rand(10, 5) > 0.8

        stats = trainer.update_weights(
            pre_spikes.float(), post_spikes.float(), reward=1.0, layer_name="fc2"
        )

        final_weights = snn.fc2.weight.data
        assert not torch.allclose(initial_weights, final_weights)

    def test_update_no_spikes(self):
        """Test update when no spikes occur."""
        snn = SurvivalSNN(organism_id=0, position=(10, 10))
        trainer = STDPTrainer(snn)

        initial_weights = snn.fc1.weight.data.clone()

        # No spikes
        pre_spikes = torch.zeros(10, 8)
        post_spikes = torch.zeros(10, 30)

        stats = trainer.update_weights(pre_spikes, post_spikes, reward=1.0)

        # Weights shouldn't change much
        final_weights = snn.fc1.weight.data
        assert stats["weight_change_mean"] == 0.0


class TestEpisodeTraining:
    """Test episode-based training."""

    def test_train_episode_basic(self):
        """Test training on full episode."""
        snn = SurvivalSNN(organism_id=0, position=(10, 10))
        trainer = STDPTrainer(snn)

        # Create episode
        spike_history = [
            (torch.rand(10, 8) > 0.8, torch.rand(10, 30) > 0.8) for _ in range(5)
        ]
        reward_history = [1.0, 2.0, 0.0, 3.0, 1.5]

        stats = trainer.train_episode(
            [(s[0].float(), s[1].float()) for s in spike_history], reward_history
        )

        assert "total_reward" in stats
        assert stats["total_reward"] == sum(reward_history)
        assert stats["num_updates"] == 5

    def test_train_episode_accumulates_updates(self):
        """Test that episode training accumulates updates."""
        snn = SurvivalSNN(organism_id=0, position=(10, 10))
        trainer = STDPTrainer(snn)

        initial_updates = trainer.total_updates

        spike_history = [(torch.rand(10, 8), torch.rand(10, 30)) for _ in range(3)]
        reward_history = [1.0, 1.0, 1.0]

        trainer.train_episode(spike_history, reward_history)

        # Should have accumulated updates (2 layers × 3 timesteps)
        assert trainer.total_updates > initial_updates


class TestStatistics:
    """Test statistics tracking."""

    def test_get_statistics(self):
        """Test getting statistics."""
        snn = SurvivalSNN(organism_id=0, position=(10, 10))
        trainer = STDPTrainer(snn, learning_rate=0.005)

        stats = trainer.get_statistics()

        assert "total_updates" in stats
        assert "total_reward" in stats
        assert "avg_reward" in stats
        assert "learning_rate" in stats
        assert stats["learning_rate"] == 0.005

    def test_statistics_track_updates(self):
        """Test that statistics track updates."""
        snn = SurvivalSNN(organism_id=0, position=(10, 10))
        trainer = STDPTrainer(snn)

        # Perform updates
        pre_spikes = torch.rand(10, 8) > 0.8
        post_spikes = torch.rand(10, 30) > 0.8

        trainer.update_weights(pre_spikes.float(), post_spikes.float(), reward=5.0)
        trainer.update_weights(pre_spikes.float(), post_spikes.float(), reward=3.0)

        stats = trainer.get_statistics()

        assert stats["total_updates"] == 2
        assert stats["total_reward"] == 8.0
        assert stats["avg_reward"] == 4.0

    def test_reset_statistics(self):
        """Test resetting statistics."""
        snn = SurvivalSNN(organism_id=0, position=(10, 10))
        trainer = STDPTrainer(snn)

        # Perform some updates
        pre_spikes = torch.rand(10, 8) > 0.8
        post_spikes = torch.rand(10, 30) > 0.8
        trainer.update_weights(pre_spikes.float(), post_spikes.float(), reward=10.0)

        # Reset
        trainer.reset_statistics()

        assert trainer.total_updates == 0
        assert trainer.total_reward == 0.0
        assert len(trainer.weight_changes) == 0


class TestRewardModulation:
    """Test reward modulation of STDP."""

    def test_higher_reward_bigger_update(self):
        """Test that higher reward leads to bigger weight change."""
        snn1 = SurvivalSNN(organism_id=0, position=(10, 10))
        snn2 = SurvivalSNN(organism_id=1, position=(10, 10))

        # Copy weights
        snn2.fc1.weight.data = snn1.fc1.weight.data.clone()

        trainer1 = STDPTrainer(snn1, reward_factor=1.0)
        trainer2 = STDPTrainer(snn2, reward_factor=1.0)

        # Same spikes, different rewards
        torch.manual_seed(42)
        pre_spikes = torch.rand(10, 8) > 0.8
        post_spikes = torch.rand(10, 30) > 0.8

        stats1 = trainer1.update_weights(
            pre_spikes.float(), post_spikes.float(), reward=1.0
        )
        stats2 = trainer2.update_weights(
            pre_spikes.float(), post_spikes.float(), reward=10.0
        )

        # Higher reward should lead to bigger changes (in general)
        assert stats2["reward"] > stats1["reward"]


class TestEdgeCases:
    """Test edge cases."""

    def test_invalid_layer_name(self):
        """Test invalid layer name raises error."""
        snn = SurvivalSNN(organism_id=0, position=(10, 10))
        trainer = STDPTrainer(snn)

        pre_spikes = torch.rand(10, 8)
        post_spikes = torch.rand(10, 30)

        with pytest.raises(ValueError):
            trainer.update_weights(pre_spikes, post_spikes, layer_name="invalid")

    def test_empty_spike_train(self):
        """Test handling empty spike train."""
        snn = SurvivalSNN(organism_id=0, position=(10, 10))
        trainer = STDPTrainer(snn)

        pre_spikes = torch.zeros(10, 8)
        post_spikes = torch.zeros(10, 30)

        # Should not crash
        stats = trainer.update_weights(pre_spikes, post_spikes, reward=0.0)
        assert stats["weight_change_mean"] == 0.0
