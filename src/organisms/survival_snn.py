"""Survival-only Spiking Neural Network (SNN-S).

This module implements the survival SNN that processes sensory inputs
and outputs survival actions using Leaky Integrate-and-Fire (LIF) neurons.

Architecture:
- Input layer: 8 neurons (rate-coded sensory inputs)
- Hidden layer: 30 LIF neurons
- Output layer: 5 LIF neurons (maps to 7 actions via winner-take-all)
"""

from typing import Any, Dict, Tuple

import snntorch as snn
import torch
import torch.nn as nn
from snntorch import surrogate

from src.organisms.base_organism import Action, Organism


class SurvivalSNN(nn.Module, Organism):
    """Survival-only organism with Spiking Neural Network.

    Uses rate coding for inputs and winner-take-all for action selection.
    Trained with STDP (Spike-Timing-Dependent Plasticity).

    Attributes:
        input_size: Number of input neurons (8)
        hidden_size: Number of hidden LIF neurons (30)
        output_size: Number of output LIF neurons (5)
        fc1: First fully-connected layer
        lif1: First LIF layer
        fc2: Second fully-connected layer
        lif2: Output LIF layer
    """

    def __init__(
        self,
        organism_id: int,
        position: Tuple[int, int],
        initial_energy: float = 100.0,
        max_energy: float = 100.0,
        input_size: int = 8,
        hidden_size: int = 30,
        output_size: int = 5,
        beta: float = 0.9,
        device: str = "cpu",
    ):
        """Initialize Survival SNN.

        Args:
            organism_id: Unique organism identifier
            position: Initial (x, y) position
            initial_energy: Starting energy level
            max_energy: Maximum energy capacity
            input_size: Input dimension (default: 8)
            hidden_size: Hidden layer size (default: 30)
            output_size: Output layer size (default: 5)
            beta: LIF decay rate (default: 0.9)
            device: 'cpu' or 'cuda'

        Example:
            >>> snn = SurvivalSNN(organism_id=0, position=(10, 10))
            >>> snn.input_size
            8
            >>> snn.hidden_size
            30
        """
        # Initialize both parent classes
        nn.Module.__init__(self)
        Organism.__init__(
            self,
            organism_id=organism_id,
            position=position,
            initial_energy=initial_energy,
            max_energy=max_energy,
        )

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.beta = beta
        self.device = device

        # Network layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

        self.fc2 = nn.Linear(hidden_size, output_size)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

        # Move to device
        self.to(device)

        # Spike counters for winner-take-all
        self.spike_counts = torch.zeros(output_size, device=device)

        # Action mapping (5 neurons -> 7 actions)
        # Neuron 0 -> MOVE_NORTH
        # Neuron 1 -> MOVE_SOUTH
        # Neuron 2 -> MOVE_EAST
        # Neuron 3 -> MOVE_WEST
        # Neuron 4 -> EAT (shared with ATTACK, WAIT based on context)
        self._action_map = [
            Action.MOVE_NORTH,
            Action.MOVE_SOUTH,
            Action.MOVE_EAST,
            Action.MOVE_WEST,
            Action.EAT,  # Will be context-dependent
        ]

    def forward(
        self, x: torch.Tensor, num_steps: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the SNN.

        Args:
            x: Input tensor [input_size]
            num_steps: Number of time steps to simulate

        Returns:
            Tuple of (spike_output, membrane_potential)
                - spike_output: Output spike train [num_steps, output_size]
                - membrane_potential: Final membrane potential [output_size]

        Example:
            >>> snn = SurvivalSNN(organism_id=0, position=(10, 10))
            >>> x = torch.randn(8)
            >>> spikes, mem = snn.forward(x, num_steps=10)
            >>> spikes.shape
            torch.Size([10, 5])
        """
        # Initialize membrane potentials
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # Record output spikes
        spike_output = []

        # Simulate for num_steps timesteps
        for step in range(num_steps):
            # Layer 1
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)

            # Layer 2
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            spike_output.append(spk2)

        spike_output = torch.stack(spike_output)

        return spike_output, mem2

    def spike_to_action(
        self, spike_train: torch.Tensor, context: Dict[str, Any] = None
    ) -> Action:
        """Convert spike train to action using winner-take-all.

        Args:
            spike_train: Spike output tensor [num_steps, output_size]
            context: Environmental context for action selection
                - 'has_food_here': bool (if True, neuron 4 -> EAT)
                - 'has_adjacent_organism': bool (if True, neuron 4 -> ATTACK)
                - 'low_energy': bool (if True, prefer WAIT)

        Returns:
            Selected action

        Example:
            >>> snn = SurvivalSNN(organism_id=0, position=(10, 10))
            >>> x = torch.randn(8)
            >>> spikes, _ = snn.forward(x)
            >>> action = snn.spike_to_action(spikes, context={'has_food_here': True})
            >>> isinstance(action, Action)
            True
        """
        # Count total spikes per output neuron
        spike_counts = spike_train.sum(dim=0)

        # Winner-take-all: neuron with most spikes wins
        winner_idx = torch.argmax(spike_counts).item()

        # Map to action
        if winner_idx < 4:
            # Movement actions (neurons 0-3)
            return self._action_map[winner_idx]
        else:
            # Context-dependent action (neuron 4)
            if context is None:
                context = {}

            if context.get("has_food_here", False):
                return Action.EAT
            elif context.get("has_adjacent_organism", False) and not context.get(
                "low_energy", False
            ):
                return Action.ATTACK
            else:
                return Action.WAIT

    def encode_state_rate(self, state: Dict[str, Any]) -> torch.Tensor:
        """Encode state into rate-coded input vector.

        Input encoding (8 neurons):
        [0-1]: Food direction (x, y) normalized
        [2]: Self energy normalized [0, 1]
        [3]: Has food at current position (binary)
        [4-7]: Obstacle/organism in 4 cardinal directions (binary)

        Args:
            state: Environment state dictionary
                - 'self_energy': float
                - 'self_position': Tuple[int, int]
                - 'nearest_food': Optional[Tuple[int, int]]
                - 'food_at_position': bool
                - 'obstacles': Dict[str, bool] (north, south, east, west)

        Returns:
            Rate-coded input tensor [input_size]

        Example:
            >>> snn = SurvivalSNN(organism_id=0, position=(10, 10))
            >>> state = {
            ...     'self_energy': 50.0,
            ...     'self_position': (10, 10),
            ...     'nearest_food': (15, 12),
            ...     'food_at_position': False,
            ...     'obstacles': {'north': False, 'south': False, 'east': False, 'west': False}
            ... }
            >>> encoded = snn.encode_state_rate(state)
            >>> encoded.shape
            torch.Size([8])
        """
        encoded = torch.zeros(self.input_size, device=self.device)

        # Food direction (normalized)
        if state.get("nearest_food") is not None:
            food_pos = state["nearest_food"]
            self_pos = state.get("self_position", self.position)

            dx = (food_pos[0] - self_pos[0]) / 20.0  # Normalize by grid size
            dy = (food_pos[1] - self_pos[1]) / 20.0

            encoded[0] = torch.clamp(torch.tensor(dx, device=self.device), -1, 1)
            encoded[1] = torch.clamp(torch.tensor(dy, device=self.device), -1, 1)

        # Self energy (normalized)
        energy = state.get("self_energy", self.energy)
        encoded[2] = energy / 100.0

        # Food at position
        encoded[3] = float(state.get("food_at_position", False))

        # Obstacles in cardinal directions
        obstacles = state.get("obstacles", {})
        encoded[4] = float(obstacles.get("north", False))
        encoded[5] = float(obstacles.get("south", False))
        encoded[6] = float(obstacles.get("east", False))
        encoded[7] = float(obstacles.get("west", False))

        return encoded

    def decide(self, state: Dict[str, Any]) -> Action:
        """Make decision based on state (implements Organism.decide).

        Args:
            state: Environment state dictionary

        Returns:
            Selected action

        Example:
            >>> snn = SurvivalSNN(organism_id=0, position=(10, 10))
            >>> state = {'self_energy': 50.0, 'nearest_food': (15, 10)}
            >>> action = snn.decide(state)
            >>> isinstance(action, Action)
            True
        """
        # Encode state
        encoded_state = self.encode_state_rate(state)

        # Forward pass through SNN
        with torch.no_grad():
            spike_train, _ = self.forward(encoded_state, num_steps=10)

        # Build context for action selection
        context = {
            "has_food_here": state.get("food_at_position", False),
            "has_adjacent_organism": any(state.get("obstacles", {}).values()),
            "low_energy": state.get("self_energy", self.energy) < 30,
        }

        # Convert spikes to action
        action = self.spike_to_action(spike_train, context)

        return action

    def get_network_statistics(self) -> Dict[str, Any]:
        """Get network statistics for analysis.

        Returns:
            Dictionary with network stats
        """
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "beta": self.beta,
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "device": str(self.device),
        }

    def __repr__(self) -> str:
        """String representation."""
        status = "alive" if self.alive else "dead"
        return (
            f"SurvivalSNN(id={self.organism_id}, pos={self.position}, "
            f"energy={self.energy:.1f}, {status}, arch={self.input_size}→{self.hidden_size}→{self.output_size})"
        )
