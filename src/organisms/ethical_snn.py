"""Dual-process SNN organism with ethical processing (SNN-E)."""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Any
from .survival_snn import SurvivalSNN


class EthicalSNN(SurvivalSNN):
    """Organism with dual-process architecture (survival + ethics).
    
    Extends SurvivalSNN with an additional ethical processing network
    that modulates survival-driven decisions.
    """
    
    def __init__(self, organism_id: int, position: Tuple[int, int],
                 input_size: int = 128, hidden_size: int = 256, output_size: int = 8,
                 ethical_input_size: int = 64, ethical_hidden_size: int = 128,
                 energy: float = 100.0, ethical_weight: float = 0.5):
        """Initialize ethical SNN organism.
        
        Args:
            organism_id: Unique identifier
            position: Initial position
            input_size: Survival network input size
            hidden_size: Survival network hidden size
            output_size: Number of actions
            ethical_input_size: Ethical network input size
            ethical_hidden_size: Ethical network hidden size
            energy: Initial energy
            ethical_weight: Weight of ethical modulation (0-1)
        """
        super().__init__(organism_id, position, input_size, hidden_size, output_size, energy)
        
        self.ethical_input_size = ethical_input_size
        self.ethical_hidden_size = ethical_hidden_size
        self.ethical_weight = ethical_weight
        
        # Ethical processing network (SNN-E)
        # TODO: Implement actual SNN with pre-training
        self.ethical_network = nn.Sequential(
            nn.Linear(ethical_input_size, ethical_hidden_size),
            nn.ReLU(),
            nn.Linear(ethical_hidden_size, 3)  # positive, neutral, negative valence
        )
        
        # Modulation layer: combines survival and ethical signals
        self.modulation = nn.Linear(output_size + 3, output_size)
    
    def encode_ethical_context(self, environment_state: Dict[str, Any]) -> torch.Tensor:
        """Encode ethical context from environment.
        
        Args:
            environment_state: Contains ethical cues (e.g., other organisms' states)
            
        Returns:
            Ethical context tensor
        """
        # TODO: Implement proper ethical context encoding
        # Placeholder: simple features
        features = torch.zeros(self.ethical_input_size)
        
        # Example: proximity to other organisms
        if 'nearby_organisms' in environment_state:
            nearby = environment_state['nearby_organisms']
            features[0] = len(nearby) / 10.0  # Normalized count
            
            # Average energy of nearby organisms (empathy signal?)
            if nearby:
                avg_energy = sum(org['energy'] for org in nearby) / len(nearby)
                features[1] = avg_energy / 100.0
        
        return features
    
    def ethical_evaluation(self, ethical_context: torch.Tensor) -> torch.Tensor:
        """Evaluate ethical valence of potential actions.
        
        Args:
            ethical_context: Encoded ethical context
            
        Returns:
            Ethical valence scores (positive, neutral, negative)
        """
        with torch.no_grad():
            valence = self.ethical_network(ethical_context)
            valence = torch.softmax(valence, dim=0)
        
        return valence
    
    def decide(self, sensory_input: torch.Tensor, ethical_context: torch.Tensor = None) -> int:
        """Decide action using dual-process integration.
        
        Args:
            sensory_input: Encoded sensory information
            ethical_context: Encoded ethical context (optional)
            
        Returns:
            Action index (0-7)
        """
        # Survival-driven action preferences
        survival_output = self.network(sensory_input)
        
        if ethical_context is None:
            # Fallback to survival-only
            with torch.no_grad():
                action = torch.argmax(survival_output).item()
            return action
        
        # Ethical evaluation
        ethical_valence = self.ethical_evaluation(ethical_context)
        
        # Combine survival and ethical signals
        combined = torch.cat([survival_output, ethical_valence])
        modulated_output = self.modulation(combined)
        
        # Weighted combination
        final_output = (1 - self.ethical_weight) * survival_output + \
                      self.ethical_weight * modulated_output
        
        with torch.no_grad():
            action = torch.argmax(final_output).item()
        
        return action
    
    def step(self, environment_state: Dict[str, Any]) -> Tuple[int, int]:
        """Complete perception-decision-action cycle with ethical processing.
        
        Args:
            environment_state: Current environmental state
            
        Returns:
            New position after action
        """
        if not self.alive:
            return self.position
        
        sensory_input = self.perceive(environment_state)
        ethical_context = self.encode_ethical_context(environment_state)
        
        action = self.decide(sensory_input, ethical_context)
        new_position = self.act(action)
        
        self.log_state()
        
        return new_position
