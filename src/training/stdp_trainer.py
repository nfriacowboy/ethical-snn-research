"""Spike-Timing-Dependent Plasticity (STDP) trainer for Survival SNN.

This module implements reward-modulated STDP for unsupervised learning
of survival behaviors. STDP adjusts synaptic weights based on the relative
timing of pre- and post-synaptic spikes, modulated by reward signals.

Learning rule:
- If pre-synaptic spike occurs before post-synaptic: LTP (potentiation)
- If post-synaptic spike occurs before pre-synaptic: LTD (depression)
- Weight changes are modulated by reward signal
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class STDPTrainer:
    """STDP trainer for Survival SNN with reward modulation.

    Implements reward-modulated Spike-Timing-Dependent Plasticity for
    unsupervised learning of survival behaviors.

    Attributes:
        network: SNN to train
        learning_rate: Global learning rate multiplier
        tau_plus: Time constant for LTP (ms)
        tau_minus: Time constant for LTD (ms)
        a_plus: LTP amplitude
        a_minus: LTD amplitude
        reward_factor: Reward modulation strength
    """

    def __init__(
        self,
        network: nn.Module,
        config: Optional[Dict[str, Any]] = None,
        learning_rate: float = 0.001,
        tau_plus: float = 20.0,
        tau_minus: float = 20.0,
        a_plus: float = 0.01,
        a_minus: float = 0.01,
        reward_factor: float = 1.0,
    ):
        """Initialize STDP trainer.

        Args:
            network: SNN network to train
            config: Configuration dictionary (optional)
            learning_rate: Global learning rate
            tau_plus: LTP time constant (ms)
            tau_minus: LTD time constant (ms)
            a_plus: LTP amplitude
            a_minus: LTD amplitude
            reward_factor: Reward modulation strength

        Example:
            >>> from src.organisms.survival_snn import SurvivalSNN
            >>> snn = SurvivalSNN(organism_id=0, position=(10, 10))
            >>> trainer = STDPTrainer(snn, learning_rate=0.001)
            >>> trainer.learning_rate
            0.001
        """
        self.network = network

        # Load from config if provided
        if config is not None:
            stdp_config = config.get("stdp", {})
            self.learning_rate = stdp_config.get("learning_rate", learning_rate)
            self.tau_plus = stdp_config.get("tau_plus", tau_plus)
            self.tau_minus = stdp_config.get("tau_minus", tau_minus)
            self.a_plus = stdp_config.get("a_plus", a_plus)
            self.a_minus = stdp_config.get("a_minus", a_minus)
            self.reward_factor = stdp_config.get("reward_factor", reward_factor)
        else:
            self.learning_rate = learning_rate
            self.tau_plus = tau_plus
            self.tau_minus = tau_minus
            self.a_plus = a_plus
            self.a_minus = a_minus
            self.reward_factor = reward_factor

        # Statistics tracking
        self.total_updates = 0
        self.total_reward = 0.0
        self.weight_changes = []

    def compute_stdp_window(self, delta_t: torch.Tensor) -> torch.Tensor:
        """Compute STDP learning window.

        Args:
            delta_t: Time difference (post_spike_time - pre_spike_time)

        Returns:
            Weight change based on spike timing

        Example:
            >>> trainer = STDPTrainer(None, learning_rate=0.001)
            >>> delta_t = torch.tensor([5.0, -5.0, 0.0])
            >>> window = trainer.compute_stdp_window(delta_t)
            >>> window.shape
            torch.Size([3])
        """
        # LTP: delta_t > 0 (post after pre)
        ltp = self.a_plus * torch.exp(-delta_t / self.tau_plus)
        ltp = torch.where(delta_t > 0, ltp, torch.zeros_like(delta_t))

        # LTD: delta_t < 0 (pre after post)
        ltd = -self.a_minus * torch.exp(delta_t / self.tau_minus)
        ltd = torch.where(delta_t < 0, ltd, torch.zeros_like(delta_t))

        return ltp + ltd

    def update_weights(
        self,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        reward: float = 0.0,
        layer_name: str = "fc1",
    ) -> Dict[str, float]:
        """Update network weights using reward-modulated STDP.

        Args:
            pre_spikes: Pre-synaptic spike train [num_timesteps, num_pre]
            post_spikes: Post-synaptic spike train [num_timesteps, num_post]
            reward: Reward signal for modulation
            layer_name: Name of layer to update ('fc1' or 'fc2')

        Returns:
            Dictionary with update statistics

        Example:
            >>> from src.organisms.survival_snn import SurvivalSNN
            >>> snn = SurvivalSNN(organism_id=0, position=(10, 10))
            >>> trainer = STDPTrainer(snn)
            >>> pre = torch.rand(10, 8) > 0.8  # Random spikes
            >>> post = torch.rand(10, 30) > 0.8
            >>> stats = trainer.update_weights(pre.float(), post.float(), reward=10.0)
            >>> 'weight_change_mean' in stats
            True
        """
        # Get the layer
        if layer_name == "fc1":
            layer = self.network.fc1
        elif layer_name == "fc2":
            layer = self.network.fc2
        else:
            raise ValueError(f"Unknown layer: {layer_name}")

        num_timesteps = pre_spikes.shape[0]
        num_pre = pre_spikes.shape[1]
        num_post = post_spikes.shape[1]

        # Initialize weight changes
        weight_delta = torch.zeros_like(layer.weight.data)

        # Find spike times
        pre_spike_times = self._get_spike_times(pre_spikes)  # [num_pre, max_spikes]
        post_spike_times = self._get_spike_times(post_spikes)  # [num_post, max_spikes]

        # Compute pairwise STDP updates
        for post_idx in range(num_post):
            for pre_idx in range(num_pre):
                # Get spike times for this pair
                pre_times = pre_spike_times[pre_idx]
                post_times = post_spike_times[post_idx]

                # Remove invalid times (-1)
                pre_times = pre_times[pre_times >= 0]
                post_times = post_times[post_times >= 0]

                if len(pre_times) == 0 or len(post_times) == 0:
                    continue

                # Compute all pairwise time differences
                for post_t in post_times:
                    for pre_t in pre_times:
                        delta_t = post_t - pre_t

                        # Compute STDP weight change
                        dw = self.compute_stdp_window(delta_t)

                        # Accumulate (reward-modulated)
                        weight_delta[post_idx, pre_idx] += dw * (
                            1.0 + self.reward_factor * reward
                        )

        # Apply weight update
        with torch.no_grad():
            layer.weight.data += self.learning_rate * weight_delta

            # Optional: Weight normalization to prevent runaway
            # layer.weight.data = torch.clamp(layer.weight.data, -10, 10)

        # Track statistics
        self.total_updates += 1
        self.total_reward += reward

        stats = {
            "weight_change_mean": weight_delta.abs().mean().item(),
            "weight_change_max": weight_delta.abs().max().item(),
            "reward": reward,
            "total_updates": self.total_updates,
        }

        self.weight_changes.append(stats["weight_change_mean"])

        return stats

    def _get_spike_times(self, spike_train: torch.Tensor) -> torch.Tensor:
        """Extract spike times from spike train.

        Args:
            spike_train: Binary spike train [num_timesteps, num_neurons]

        Returns:
            Spike times tensor [num_neurons, max_spikes]
            Padded with -1 for neurons with fewer spikes

        Example:
            >>> trainer = STDPTrainer(None)
            >>> spikes = torch.tensor([[1, 0], [0, 1], [1, 0], [0, 0]], dtype=torch.float32)
            >>> times = trainer._get_spike_times(spikes)
            >>> times.shape[0]
            2
        """
        num_timesteps, num_neurons = spike_train.shape

        spike_times = []
        for neuron_idx in range(num_neurons):
            # Find timesteps where neuron spiked
            times = torch.where(spike_train[:, neuron_idx] > 0)[0].float()
            spike_times.append(times)

        # Pad to same length
        max_spikes = max(len(times) for times in spike_times) if spike_times else 1
        max_spikes = max(1, max_spikes)  # At least 1

        padded_times = torch.full((num_neurons, max_spikes), -1.0)
        for idx, times in enumerate(spike_times):
            if len(times) > 0:
                padded_times[idx, : len(times)] = times

        return padded_times

    def train_episode(
        self, spike_history: list, reward_history: list
    ) -> Dict[str, Any]:
        """Train on full episode of spikes and rewards.

        Args:
            spike_history: List of (pre_spikes, post_spikes) tuples
            reward_history: List of reward values for each timestep

        Returns:
            Dictionary with episode training statistics

        Example:
            >>> from src.organisms.survival_snn import SurvivalSNN
            >>> snn = SurvivalSNN(organism_id=0, position=(10, 10))
            >>> trainer = STDPTrainer(snn)
            >>> spike_hist = [(torch.rand(10, 8), torch.rand(10, 30)) for _ in range(5)]
            >>> reward_hist = [1.0, 2.0, 0.0, 5.0, 3.0]
            >>> stats = trainer.train_episode(spike_hist, reward_hist)
            >>> 'total_reward' in stats
            True
        """
        total_reward = sum(reward_history)
        updates = []

        for (pre_spikes, post_spikes), reward in zip(spike_history, reward_history):
            # Update both layers
            stats1 = self.update_weights(
                pre_spikes, post_spikes, reward, layer_name="fc1"
            )
            stats2 = self.update_weights(
                post_spikes[:, : post_spikes.shape[1]],
                torch.rand(post_spikes.shape[0], 5),
                reward,
                layer_name="fc2",
            )
            updates.append((stats1, stats2))

        return {
            "total_reward": total_reward,
            "num_updates": len(updates),
            "avg_weight_change": (
                np.mean(self.weight_changes[-len(updates) :])
                if self.weight_changes
                else 0.0
            ),
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get trainer statistics.

        Returns:
            Dictionary with training stats

        Example:
            >>> trainer = STDPTrainer(None)
            >>> stats = trainer.get_statistics()
            >>> 'total_updates' in stats
            True
        """
        return {
            "total_updates": self.total_updates,
            "total_reward": self.total_reward,
            "avg_reward": self.total_reward / max(1, self.total_updates),
            "learning_rate": self.learning_rate,
            "tau_plus": self.tau_plus,
            "tau_minus": self.tau_minus,
            "weight_change_history": self.weight_changes,
        }

    def reset_statistics(self) -> None:
        """Reset training statistics.

        Example:
            >>> trainer = STDPTrainer(None)
            >>> trainer.total_updates = 100
            >>> trainer.reset_statistics()
            >>> trainer.total_updates
            0
        """
        self.total_updates = 0
        self.total_reward = 0.0
        self.weight_changes = []
