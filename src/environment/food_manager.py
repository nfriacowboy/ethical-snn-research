"""Food resource management."""

from typing import List, Tuple, Dict, Any
import numpy as np


class FoodManager:
    """Manages food resources in the environment.
    
    Handles food spawning, depletion, and scarcity scenarios.
    """
    
    def __init__(self, grid_size: Tuple[int, int],
                 spawn_rate: float = 0.02,
                 energy_value: float = 20.0,
                 max_food: int = 100):
        """Initialize food manager.
        
        Args:
            grid_size: Size of the grid
            spawn_rate: Probability of spawning per cell per timestep
            energy_value: Energy provided by each food item
            max_food: Maximum food items allowed simultaneously
        """
        self.grid_size = grid_size
        self.spawn_rate = spawn_rate
        self.energy_value = energy_value
        self.max_food = max_food
        
        self.food_items: List[Dict[str, Any]] = []
        self.spawn_history: List[int] = []
    
    def spawn_food(self, current_count: int) -> List[Tuple[int, int]]:
        """Spawn new food items.
        
        Args:
            current_count: Current number of food items
            
        Returns:
            List of new food positions
        """
        new_positions = []
        
        if current_count >= self.max_food:
            return new_positions
        
        # Calculate how many to spawn
        available_slots = self.max_food - current_count
        width, height = self.grid_size
        
        for _ in range(width * height):
            if len(new_positions) >= available_slots:
                break
            
            if np.random.random() < self.spawn_rate:
                x = np.random.randint(0, width)
                y = np.random.randint(0, height)
                new_positions.append((x, y))
        
        self.spawn_history.append(len(new_positions))
        return new_positions
    
    def create_scarcity_scenario(self, severity: float = 0.5) -> float:
        """Create a scarcity scenario by reducing spawn rate.
        
        Args:
            severity: Scarcity level (0 = normal, 1 = extreme)
            
        Returns:
            New spawn rate
        """
        self.spawn_rate *= (1 - severity)
        return self.spawn_rate
    
    def create_abundance_scenario(self, multiplier: float = 2.0) -> float:
        """Create an abundance scenario by increasing spawn rate.
        
        Args:
            multiplier: How much to multiply spawn rate
            
        Returns:
            New spawn rate
        """
        self.spawn_rate *= multiplier
        return self.spawn_rate
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get food spawning statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            'total_spawned': sum(self.spawn_history),
            'avg_spawn_per_step': np.mean(self.spawn_history) if self.spawn_history else 0,
            'current_spawn_rate': self.spawn_rate,
            'max_food': self.max_food
        }
