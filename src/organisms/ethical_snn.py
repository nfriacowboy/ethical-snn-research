"""Ethical SNN (SNN-E) for ethical behavior evaluation.

This module implements the second neural network in the dual-process architecture.
SNN-E evaluates proposed actions and can veto unethical behaviors.
"""

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from typing import Dict, Any, Optional, Tuple
import numpy as np


class EthicalSNN(nn.Module):
    """Ethical Spiking Neural Network for action evaluation.
    
    Architecture:
        Input (8) → fc1 → LIF1 (20) → fc2 → LIF2 (2)
    
    Input encoding (8 neurons):
        [0]: self_energy normalized [0, 1]
        [1]: other_energy normalized [0, 1]
        [2]: food_available (0 or 1)
        [3]: distance_to_other normalized [0, 1]
        [4-7]: proposed_action one-hot (ATTACK, EAT, MOVE, WAIT)
    
    Output (2 neurons):
        [0]: Ethical (approve action)
        [1]: Unethical (veto action)
    
    Training:
        - Supervised learning on synthetic ethical dataset
        - Binary classification: ethical vs unethical
        - Uses pre-generated scenarios from EthicalDatasetGenerator
    """
    
    def __init__(
        self,
        input_size: int = 8,
        hidden_size: int = 20,
        output_size: int = 2,
        beta: float = 0.9,
        spike_grad: Optional[Any] = None
    ):
        """Initialize Ethical SNN.
        
        Args:
            input_size: Number of input neurons (default: 8)
            hidden_size: Number of hidden layer neurons (default: 20)
            output_size: Number of output neurons (default: 2)
            beta: LIF neuron decay rate (default: 0.9)
            spike_grad: Surrogate gradient function (default: fast_sigmoid)
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.beta = beta
        
        if spike_grad is None:
            spike_grad = surrogate.fast_sigmoid()
        
        # Network layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        # Veto threshold: if unethical spikes > ethical spikes, veto
        self.veto_threshold = 0.5
    
    def forward(
        self, 
        x: torch.Tensor, 
        num_steps: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the ethical SNN.
        
        Args:
            x: Input tensor [batch_size, input_size] or [input_size]
            num_steps: Number of simulation timesteps (default: 10)
        
        Returns:
            Tuple of:
                - spike_train: [num_steps, batch_size, output_size] or [num_steps, output_size]
                - membrane_potential: Final membrane potential of output layer
        """
        # Handle single sample (no batch dimension)
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension
            single_sample = True
        else:
            single_sample = False
        
        batch_size = x.size(0)
        device = x.device
        
        # Initialize membrane potentials
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        
        # Record output spikes
        spike_train = []
        
        # Run simulation
        for t in range(num_steps):
            # Layer 1
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            
            # Layer 2
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            
            spike_train.append(spk2)
        
        # Stack spike train: [num_steps, batch_size, output_size]
        spike_train = torch.stack(spike_train, dim=0)
        
        # Remove batch dimension if single sample
        if single_sample:
            spike_train = spike_train.squeeze(1)  # [num_steps, output_size]
        
        return spike_train, mem2
    
    def encode_state(
        self,
        self_energy: float,
        other_energy: float,
        food_available: bool,
        distance_to_other: float,
        proposed_action: str
    ) -> torch.Tensor:
        """Encode ethical evaluation context as rate-coded input.
        
        Args:
            self_energy: Own energy level [0, 100]
            other_energy: Other organism's energy [0, 100]
            food_available: Whether food is present
            distance_to_other: Distance to other organism [0, 20]
            proposed_action: Action to evaluate ('ATTACK', 'EAT', 'MOVE', 'WAIT')
        
        Returns:
            Input tensor [8] for SNN
        """
        state = np.zeros(8, dtype=np.float32)
        
        # Normalize continuous values to [0, 1]
        state[0] = self_energy / 100.0
        state[1] = other_energy / 100.0
        state[2] = 1.0 if food_available else 0.0
        state[3] = min(distance_to_other / 20.0, 1.0)  # Cap at 1.0
        
        # One-hot encode action
        action_map = {'ATTACK': 4, 'EAT': 5, 'MOVE': 6, 'WAIT': 7}
        if proposed_action in action_map:
            state[action_map[proposed_action]] = 1.0
        
        return torch.tensor(state, dtype=torch.float32)
    
    def evaluate_action(
        self,
        self_energy: float,
        other_energy: float,
        food_available: bool,
        distance_to_other: float,
        proposed_action: str,
        num_steps: int = 10
    ) -> bool:
        """Evaluate whether a proposed action is ethical.
        
        Args:
            self_energy: Own energy level [0, 100]
            other_energy: Other organism's energy [0, 100]
            food_available: Whether food is present
            distance_to_other: Distance to other organism [0, 20]
            proposed_action: Action to evaluate
            num_steps: Simulation timesteps (default: 10)
        
        Returns:
            True if action is ethical (approved), False if unethical (vetoed)
        """
        # Encode state
        state = self.encode_state(
            self_energy, other_energy, food_available,
            distance_to_other, proposed_action
        )
        
        # Forward pass
        with torch.no_grad():
            spike_train, _ = self.forward(state, num_steps=num_steps)
        
        # spike_train: [num_steps, 2] where [:, 0] = ethical, [:, 1] = unethical
        ethical_spikes = spike_train[:, 0].sum().item()
        unethical_spikes = spike_train[:, 1].sum().item()
        
        total_spikes = ethical_spikes + unethical_spikes
        
        if total_spikes == 0:
            # No spikes = uncertain, default to approve (conservative)
            return True
        
        # Compute ethical ratio
        ethical_ratio = ethical_spikes / total_spikes
        
        # Approve if ethical spikes dominate
        return ethical_ratio >= self.veto_threshold
    
    def get_ethical_score(
        self,
        self_energy: float,
        other_energy: float,
        food_available: bool,
        distance_to_other: float,
        proposed_action: str,
        num_steps: int = 10
    ) -> float:
        """Get continuous ethical score for an action.
        
        Args:
            self_energy: Own energy level [0, 100]
            other_energy: Other organism's energy [0, 100]
            food_available: Whether food is present
            distance_to_other: Distance to other organism
            proposed_action: Action to evaluate
            num_steps: Simulation timesteps
        
        Returns:
            Ethical score in [0, 1] where 1 = fully ethical, 0 = fully unethical
        """
        state = self.encode_state(
            self_energy, other_energy, food_available,
            distance_to_other, proposed_action
        )
        
        with torch.no_grad():
            spike_train, _ = self.forward(state, num_steps=num_steps)
        
        ethical_spikes = spike_train[:, 0].sum().item()
        unethical_spikes = spike_train[:, 1].sum().item()
        
        total_spikes = ethical_spikes + unethical_spikes
        
        if total_spikes == 0:
            return 0.5  # Neutral if uncertain
        
        return ethical_spikes / total_spikes
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get network statistics.
        
        Returns:
            Dictionary with network parameters and info
        """
        return {
            'type': 'EthicalSNN',
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'beta': self.beta,
            'veto_threshold': self.veto_threshold,
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(
                p.numel() for p in self.parameters() if p.requires_grad
            )
        }
    
    def reset_membrane_potentials(self) -> None:
        """Reset membrane potentials (for new episode)."""
        # Membrane potentials are reset automatically in forward pass
        pass
    
    def __repr__(self) -> str:
        """String representation."""
        stats = self.get_statistics()
        return (
            f"EthicalSNN("
            f"input={stats['input_size']}, "
            f"hidden={stats['hidden_size']}, "
            f"output={stats['output_size']}, "
            f"params={stats['total_parameters']})"
        )
