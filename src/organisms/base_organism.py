"""Base organism abstract class and action definitions.

This module defines:
- Action enum for all possible organism actions
- Organism abstract base class
- Energy management and lifecycle methods
"""

from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Dict, Any, Tuple, Optional
import numpy as np


class Action(IntEnum):
    """Enumeration of all possible organism actions.
    
    Movement actions:
        MOVE_NORTH (0): Move up (y-1)
        MOVE_SOUTH (1): Move down (y+1)
        MOVE_EAST (2): Move right (x+1)
        MOVE_WEST (3): Move left (x-1)
    
    Interaction actions:
        EAT (4): Consume food at current position
        ATTACK (5): Attack adjacent organism
        WAIT (6): Do nothing (conserve energy)
    """
    MOVE_NORTH = 0
    MOVE_SOUTH = 1
    MOVE_EAST = 2
    MOVE_WEST = 3
    EAT = 4
    ATTACK = 5
    WAIT = 6
    
    @classmethod
    def is_movement(cls, action: 'Action') -> bool:
        """Check if action is a movement action."""
        return action in (cls.MOVE_NORTH, cls.MOVE_SOUTH, cls.MOVE_EAST, cls.MOVE_WEST)
    
    @classmethod
    def get_direction_vector(cls, action: 'Action') -> Tuple[int, int]:
        """Get (dx, dy) vector for movement action.
        
        Args:
            action: Movement action
            
        Returns:
            (dx, dy) tuple representing direction
            
        Raises:
            ValueError: If action is not a movement action
            
        Example:
            >>> Action.get_direction_vector(Action.MOVE_NORTH)
            (0, -1)
            >>> Action.get_direction_vector(Action.MOVE_EAST)
            (1, 0)
        """
        if action == cls.MOVE_NORTH:
            return (0, -1)
        elif action == cls.MOVE_SOUTH:
            return (0, 1)
        elif action == cls.MOVE_EAST:
            return (1, 0)
        elif action == cls.MOVE_WEST:
            return (-1, 0)
        else:
            raise ValueError(f"Action {action} is not a movement action")


class Organism(ABC):
    """Abstract base class for all organisms in the simulation.
    
    Organisms have:
    - Unique ID and position on grid
    - Energy level (dies when energy <= 0)
    - Age counter (increments each timestep)
    - Action history for logging
    
    Subclasses must implement:
    - decide(state) -> Action: Decision-making logic
    
    Attributes:
        organism_id: Unique identifier
        position: Current (x, y) position
        energy: Current energy level [0, max_energy]
        max_energy: Maximum energy capacity
        alive: Whether organism is alive
        age: Number of timesteps alive
        action_history: List of actions taken
    """
    
    def __init__(
        self,
        organism_id: int,
        position: Tuple[int, int],
        initial_energy: float = 100.0,
        max_energy: float = 100.0
    ):
        """Initialize organism.
        
        Args:
            organism_id: Unique identifier
            position: Initial (x, y) position
            initial_energy: Starting energy level
            max_energy: Maximum energy capacity
            
        Example:
            >>> org = TestOrganism(organism_id=0, position=(10, 10), initial_energy=80)
            >>> org.energy
            80.0
            >>> org.is_alive()
            True
        """
        self.organism_id = organism_id
        self.position = position
        self.energy = initial_energy
        self.max_energy = max_energy
        self.alive = True
        self.age = 0
        self.action_history: list = []
        
        # Statistics
        self.total_food_consumed = 0
        self.total_attacks_attempted = 0
        self.total_distance_moved = 0
    
    @abstractmethod
    def decide(self, state: Dict[str, Any]) -> Action:
        """Make a decision based on current state.
        
        This is the core decision-making method that subclasses must implement.
        
        Args:
            state: Dictionary containing environmental information:
                - 'local_view': numpy array of local grid (organisms, food)
                - 'self_energy': current energy level
                - 'self_position': current (x, y) position
                - 'food_positions': list of nearby food positions
                - 'organism_positions': dict of nearby organism positions
                
        Returns:
            Action to take
            
        Example:
            >>> # Subclass implementation
            >>> def decide(self, state):
            ...     if state['self_energy'] < 50:
            ...         return Action.EAT
            ...     return Action.WAIT
        """
        pass
    
    def update_energy(self, delta: float) -> None:
        """Update energy level by delta amount.
        
        Energy is clamped to [0, max_energy].
        If energy drops to 0, organism dies.
        
        Args:
            delta: Energy change (positive or negative)
            
        Example:
            >>> org = TestOrganism(organism_id=0, position=(10, 10), initial_energy=50)
            >>> org.update_energy(20)
            >>> org.energy
            70.0
            >>> org.update_energy(-100)
            >>> org.energy
            0.0
            >>> org.is_alive()
            False
        """
        self.energy += delta
        
        # Clamp to valid range
        if self.energy > self.max_energy:
            self.energy = self.max_energy
        elif self.energy <= 0:
            self.energy = 0.0
            self.alive = False
    
    def move(self, new_position: Tuple[int, int]) -> None:
        """Move to a new position.
        
        Updates position and calculates distance traveled for statistics.
        
        Args:
            new_position: Target (x, y) position
            
        Example:
            >>> org = TestOrganism(organism_id=0, position=(5, 5))
            >>> org.move((6, 6))
            >>> org.position
            (6, 6)
            >>> org.total_distance_moved
            2
        """
        if not self.alive:
            return
        
        # Calculate Manhattan distance
        old_x, old_y = self.position
        new_x, new_y = new_position
        distance = abs(new_x - old_x) + abs(new_y - old_y)
        
        self.position = new_position
        self.total_distance_moved += distance
    
    def is_alive(self) -> bool:
        """Check if organism is alive.
        
        Returns:
            True if alive, False otherwise
            
        Example:
            >>> org = TestOrganism(organism_id=0, position=(10, 10))
            >>> org.is_alive()
            True
            >>> org.update_energy(-100)
            >>> org.is_alive()
            False
        """
        return self.alive
    
    def increment_age(self) -> None:
        """Increment age counter.
        
        Called once per simulation timestep.
        
        Example:
            >>> org = TestOrganism(organism_id=0, position=(10, 10))
            >>> org.age
            0
            >>> org.increment_age()
            >>> org.age
            1
        """
        self.age += 1
    
    def log_action(self, action: Action, success: bool = True) -> None:
        """Log an action to history.
        
        Args:
            action: Action that was taken
            success: Whether action succeeded
            
        Example:
            >>> org = TestOrganism(organism_id=0, position=(10, 10))
            >>> org.log_action(Action.MOVE_NORTH, success=True)
            >>> len(org.action_history)
            1
            >>> org.action_history[0]['action']
            <Action.MOVE_NORTH: 0>
        """
        self.action_history.append({
            'timestep': self.age,
            'action': action,
            'success': success,
            'energy': self.energy,
            'position': self.position
        })
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get organism statistics.
        
        Returns:
            Dictionary with statistics:
                - organism_id
                - age
                - alive
                - energy
                - position
                - total_food_consumed
                - total_attacks_attempted
                - total_distance_moved
                - action_count
        """
        return {
            'organism_id': self.organism_id,
            'age': self.age,
            'alive': self.alive,
            'energy': self.energy,
            'position': self.position,
            'total_food_consumed': self.total_food_consumed,
            'total_attacks_attempted': self.total_attacks_attempted,
            'total_distance_moved': self.total_distance_moved,
            'action_count': len(self.action_history)
        }
    
    def reset(self, position: Tuple[int, int], initial_energy: Optional[float] = None) -> None:
        """Reset organism to initial state.
        
        Args:
            position: New initial position
            initial_energy: New initial energy (if None, uses max_energy)
            
        Example:
            >>> org = TestOrganism(organism_id=0, position=(10, 10))
            >>> org.update_energy(-50)
            >>> org.reset(position=(5, 5))
            >>> org.energy
            100.0
            >>> org.position
            (5, 5)
        """
        self.position = position
        self.energy = initial_energy if initial_energy is not None else self.max_energy
        self.alive = True
        self.age = 0
        self.action_history = []
        self.total_food_consumed = 0
        self.total_attacks_attempted = 0
        self.total_distance_moved = 0
    
    def __repr__(self) -> str:
        """String representation of organism."""
        status = "alive" if self.alive else "dead"
        return (
            f"{self.__class__.__name__}(id={self.organism_id}, "
            f"pos={self.position}, energy={self.energy:.1f}, {status})"
        )
