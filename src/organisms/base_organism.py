"""Base organism abstract class."""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
import torch


class BaseOrganism(ABC):
    """Abstract base class for all organisms.
    
    Defines the interface that all organism implementations must follow.
    """
    
    def __init__(self, organism_id: int, position: Tuple[int, int], energy: float = 100.0):
        """Initialize base organism.
        
        Args:
            organism_id: Unique identifier for this organism
            position: Initial (x, y) position in the grid
            energy: Initial energy level
        """
        self.organism_id = organism_id
        self.position = position
        self.energy = energy
        self.age = 0
        self.alive = True
        self.history = []
    
    @abstractmethod
    def perceive(self, environment_state: Dict[str, Any]) -> torch.Tensor:
        """Process environmental inputs into sensory encoding.
        
        Args:
            environment_state: Dictionary containing environmental information
            
        Returns:
            Encoded sensory tensor
        """
        pass
    
    @abstractmethod
    def decide(self, sensory_input: torch.Tensor) -> int:
        """Make a decision based on sensory input.
        
        Args:
            sensory_input: Encoded sensory information
            
        Returns:
            Action index (0-7 for movement directions)
        """
        pass
    
    @abstractmethod
    def act(self, action: int) -> Tuple[int, int]:
        """Execute an action.
        
        Args:
            action: Action index to execute
            
        Returns:
            New (x, y) position
        """
        pass
    
    def update(self, energy_delta: float):
        """Update organism state.
        
        Args:
            energy_delta: Change in energy (can be negative)
        """
        self.energy += energy_delta
        self.age += 1
        
        if self.energy <= 0:
            self.alive = False
    
    def log_state(self):
        """Log current state to history."""
        self.history.append({
            'age': self.age,
            'position': self.position,
            'energy': self.energy,
            'alive': self.alive
        })
    
    def get_state(self) -> Dict[str, Any]:
        """Get current organism state.
        
        Returns:
            Dictionary containing current state
        """
        return {
            'organism_id': self.organism_id,
            'position': self.position,
            'energy': self.energy,
            'age': self.age,
            'alive': self.alive
        }
