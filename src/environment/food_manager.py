"""Food management system for the simulation environment.

The FoodManager handles:
- Food spawning at initialization and during simulation
- Food consumption by organisms
- Food respawning according to configured rates
- Spatial queries (nearest food)
"""

from typing import Dict, List, Optional, Tuple
import numpy as np


class FoodManager:
    """Manages food items in the simulation environment.
    
    Attributes:
        grid_size: Size of the grid (square)
        max_food: Maximum number of food items allowed
        respawn_rate: Probability of spawning food each timestep [0, 1]
        active_food: List of (x, y) positions with food
        total_spawned: Total food items spawned (for statistics)
        total_consumed: Total food items consumed (for statistics)
    """
    
    def __init__(self, config: Dict, grid_size: int):
        """Initialize food manager.
        
        Args:
            config: Configuration dictionary with 'environment' section
            grid_size: Size of the grid (square)
        """
        self.grid_size = grid_size
        
        # Get parameters from config
        env_config = config.get('environment', {})
        self.max_food = env_config.get('max_food', 10)
        self.respawn_rate = env_config.get('food_respawn_rate', 0.1)
        self.initial_food = env_config.get('num_food', 5)
        
        # State
        self.active_food: List[Tuple[int, int]] = []
        self.total_spawned = 0
        self.total_consumed = 0
        
        # Spawn initial food
        self.spawn_food(self.initial_food)
    
    def spawn_food(self, num: int) -> List[Tuple[int, int]]:
        """Spawn multiple food items at random unoccupied positions.
        
        Args:
            num: Number of food items to spawn
            
        Returns:
            List of (x, y) positions where food was spawned
            
        Example:
            >>> manager = FoodManager(config, grid_size=20)
            >>> positions = manager.spawn_food(5)
            >>> len(positions) <= 5  # May spawn less if grid is full
            True
        """
        spawned = []
        
        # Don't exceed max_food
        available_slots = self.max_food - len(self.active_food)
        num_to_spawn = min(num, available_slots)
        
        # Track occupied positions for fast lookup
        occupied = set(self.active_food)
        
        attempts = 0
        max_attempts = num_to_spawn * 10  # Avoid infinite loop
        
        while len(spawned) < num_to_spawn and attempts < max_attempts:
            x = np.random.randint(0, self.grid_size)
            y = np.random.randint(0, self.grid_size)
            pos = (x, y)
            
            if pos not in occupied:
                self.active_food.append(pos)
                occupied.add(pos)
                spawned.append(pos)
                self.total_spawned += 1
            
            attempts += 1
        
        return spawned
    
    def consume_food(self, position: Tuple[int, int]) -> bool:
        """Consume food at a position if present.
        
        Args:
            position: (x, y) position tuple
            
        Returns:
            True if food was consumed, False if no food at position
            
        Example:
            >>> manager.spawn_food(1)  # Spawn at some position
            >>> pos = manager.active_food[0]
            >>> manager.consume_food(pos)
            True
            >>> manager.consume_food(pos)  # Already consumed
            False
        """
        if position in self.active_food:
            self.active_food.remove(position)
            self.total_consumed += 1
            return True
        
        return False
    
    def get_nearest_food(
        self, 
        position: Tuple[int, int], 
        toroidal: bool = True
    ) -> Optional[Tuple[int, int]]:
        """Find the nearest food to a position.
        
        Args:
            position: (x, y) position to search from
            toroidal: Whether to use toroidal distance (wrap-around)
            
        Returns:
            (x, y) position of nearest food, or None if no food exists
            
        Example:
            >>> manager = FoodManager(config, grid_size=20)
            >>> manager.spawn_food(3)
            >>> nearest = manager.get_nearest_food((10, 10))
            >>> nearest is not None
            True
        """
        if not self.active_food:
            return None
        
        min_distance = float('inf')
        nearest = None
        
        for food_pos in self.active_food:
            if toroidal:
                dist = self._toroidal_distance(position, food_pos)
            else:
                dist = self._euclidean_distance(position, food_pos)
            
            if dist < min_distance:
                min_distance = dist
                nearest = food_pos
        
        return nearest
    
    def update(self, timestep: int) -> List[Tuple[int, int]]:
        """Update food state (spawn new food with probability).
        
        Called each simulation timestep to handle food respawning.
        
        Args:
            timestep: Current simulation timestep (for logging)
            
        Returns:
            List of positions where food was spawned this update
            
        Example:
            >>> manager = FoodManager(config, grid_size=20)
            >>> manager.respawn_rate = 1.0  # Always spawn
            >>> spawned = manager.update(timestep=10)
            >>> len(spawned) > 0  # Should spawn if under max_food
            True
        """
        # Spawn with probability
        if np.random.random() < self.respawn_rate:
            # Spawn 1 food item per update (configurable if needed)
            spawned = self.spawn_food(1)
            return spawned
        
        return []
    
    def get_statistics(self) -> Dict:
        """Get current statistics about food.
        
        Returns:
            Dictionary with food counts and totals
        """
        return {
            'active_food_count': len(self.active_food),
            'total_spawned': self.total_spawned,
            'total_consumed': self.total_consumed,
            'max_food': self.max_food,
            'respawn_rate': self.respawn_rate
        }
    
    def reset(self) -> None:
        """Reset food manager to initial state."""
        self.active_food = []
        self.total_spawned = 0
        self.total_consumed = 0
        self.spawn_food(self.initial_food)
    
    def _toroidal_distance(
        self, 
        pos1: Tuple[int, int], 
        pos2: Tuple[int, int]
    ) -> float:
        """Calculate distance with toroidal topology (wrap-around).
        
        Args:
            pos1: First position (x, y)
            pos2: Second position (x, y)
            
        Returns:
            Minimum distance considering wrapping
        """
        x1, y1 = pos1
        x2, y2 = pos2
        
        # Calculate shortest distance in each dimension
        dx = min(abs(x2 - x1), self.grid_size - abs(x2 - x1))
        dy = min(abs(y2 - y1), self.grid_size - abs(y2 - y1))
        
        return np.sqrt(dx**2 + dy**2)
    
    def _euclidean_distance(
        self, 
        pos1: Tuple[int, int], 
        pos2: Tuple[int, int]
    ) -> float:
        """Calculate standard Euclidean distance.
        
        Args:
            pos1: First position (x, y)
            pos2: Second position (x, y)
            
        Returns:
            Euclidean distance
        """
        x1, y1 = pos1
        x2, y2 = pos2
        
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
