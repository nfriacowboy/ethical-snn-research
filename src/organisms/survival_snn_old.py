"""Survival-only SNN organism (SNN-S)."""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Any
from .base_organism import BaseOrganism


class SurvivalSNN(BaseOrganism):
    """Organism with survival-only spiking neural network.
    
    Uses a 3-layer recurrent SNN trained with STDP for survival behaviors.
    """
    
    def __init__(self, organism_id: int, position: Tuple[int, int], 
                 input_size: int = 128, hidden_size: int = 256, output_size: int = 8,
                 energy: float = 100.0):
        """Initialize survival SNN organism.
        
        Args:
            organism_id: Unique identifier
            position: Initial position
            input_size: Number of input neurons
            hidden_size: Number of hidden neurons
            output_size: Number of output neurons (actions)
            energy: Initial energy
        """
        super().__init__(organism_id, position, energy)
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # TODO: Implement actual SNN layers using snntorch
        # Placeholder for now
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        
        # Movement directions: N, NE, E, SE, S, SW, W, NW
        self.action_vectors = [
            (0, -1), (1, -1), (1, 0), (1, 1),
            (0, 1), (-1, 1), (-1, 0), (-1, -1)
        ]
    
    def perceive(self, environment_state: Dict[str, Any]) -> torch.Tensor:
        """Encode environmental state into spike train.
        
        Args:
            environment_state: Contains 'food_locations', 'organism_locations', etc.
            
        Returns:
            Encoded sensory tensor
        """
        # TODO: Implement proper spike encoding (rate coding or temporal coding)
        # Placeholder: simple feature vector
        features = torch.zeros(self.input_size)
        
        # Encode relative food locations (simplified)
        if 'nearest_food' in environment_state:
            food_pos = environment_state['nearest_food']
            if food_pos is not None:
                dx = food_pos[0] - self.position[0]
                dy = food_pos[1] - self.position[1]
                features[0] = dx / 50.0  # Normalized
                features[1] = dy / 50.0
        
        # Encode energy level
        features[2] = self.energy / 100.0
        
        return features
    
    def decide(self, sensory_input: torch.Tensor) -> int:
        """Decide action based on SNN output.
        
        Args:
            sensory_input: Encoded sensory information
            
        Returns:
            Action index (0-7)
        """
        with torch.no_grad():
            output = self.network(sensory_input)
            action = torch.argmax(output).item()
        
        return action
    
    def act(self, action: int) -> Tuple[int, int]:
        """Move in the selected direction.
        
        Args:
            action: Action index (0-7)
            
        Returns:
            New position
        """
        dx, dy = self.action_vectors[action]
        new_x = max(0, min(49, self.position[0] + dx))  # Clamp to grid
        new_y = max(0, min(49, self.position[1] + dy))
        
        self.position = (new_x, new_y)
        return self.position
    
    def step(self, environment_state: Dict[str, Any]) -> Tuple[int, int]:
        """Complete perception-decision-action cycle.
        
        Args:
            environment_state: Current environmental state
            
        Returns:
            New position after action
        """
        if not self.alive:
            return self.position
        
        sensory_input = self.perceive(environment_state)
        action = self.decide(sensory_input)
        new_position = self.act(action)
        
        self.log_state()
        
        return new_position
