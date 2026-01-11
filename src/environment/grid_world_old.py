"""2D Grid world environment."""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional


class GridWorld:
    """2D grid environment for organism simulations.
    
    Manages food resources, organism positions, and environmental dynamics.
    """
    
    def __init__(self, size: Tuple[int, int] = (50, 50), 
                 food_spawn_rate: float = 0.02,
                 energy_decay: float = 1.0,
                 initial_food: int = 20):
        """Initialize grid world.
        
        Args:
            size: Grid dimensions (width, height)
            food_spawn_rate: Probability of food spawning per cell per timestep
            energy_decay: Energy lost per timestep per organism
            initial_food: Number of food items at start
        """
        self.size = size
        self.width, self.height = size
        self.food_spawn_rate = food_spawn_rate
        self.energy_decay = energy_decay
        
        # Grid state: 0 = empty, 1 = food
        self.grid = np.zeros(size, dtype=int)
        
        # Food tracking
        self.food_positions: List[Tuple[int, int]] = []
        self.food_energy_value = 20.0
        
        # Statistics
        self.timestep = 0
        self.total_food_spawned = 0
        self.total_food_consumed = 0
        
        # Initialize food
        self._spawn_initial_food(initial_food)
    
    def _spawn_initial_food(self, count: int):
        """Spawn initial food items.
        
        Args:
            count: Number of food items to spawn
        """
        for _ in range(count):
            x = np.random.randint(0, self.width)
            y = np.random.randint(0, self.height)
            
            if self.grid[x, y] == 0:  # Empty cell
                self.grid[x, y] = 1
                self.food_positions.append((x, y))
                self.total_food_spawned += 1
    
    def step(self, organism_positions: List[Tuple[int, int]]) -> Dict[str, Any]:
        """Update environment for one timestep.
        
        Args:
            organism_positions: Current positions of all organisms
            
        Returns:
            Dictionary with environment state and events
        """
        self.timestep += 1
        events = []
        
        # Spawn new food
        spawned = self._spawn_food()
        if spawned:
            events.append({'type': 'food_spawned', 'count': spawned})
        
        # Check for food consumption
        consumed = self._check_food_consumption(organism_positions)
        if consumed:
            events.append({'type': 'food_consumed', 'positions': consumed})
        
        return {
            'timestep': self.timestep,
            'food_count': len(self.food_positions),
            'events': events
        }
    
    def _spawn_food(self) -> int:
        """Randomly spawn food based on spawn rate.
        
        Returns:
            Number of food items spawned
        """
        spawned = 0
        
        # Probabilistic spawning across grid
        for x in range(self.width):
            for y in range(self.height):
                if self.grid[x, y] == 0 and np.random.random() < self.food_spawn_rate:
                    self.grid[x, y] = 1
                    self.food_positions.append((x, y))
                    self.total_food_spawned += 1
                    spawned += 1
        
        return spawned
    
    def _check_food_consumption(self, organism_positions: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Check if organisms consumed food at their positions.
        
        Args:
            organism_positions: Current organism positions
            
        Returns:
            List of positions where food was consumed
        """
        consumed_positions = []
        
        for pos in organism_positions:
            x, y = pos
            if self.grid[x, y] == 1:  # Food present
                self.grid[x, y] = 0
                self.food_positions.remove((x, y))
                consumed_positions.append((x, y))
                self.total_food_consumed += 1
        
        return consumed_positions
    
    def get_food_at_position(self, position: Tuple[int, int]) -> bool:
        """Check if food exists at position.
        
        Args:
            position: (x, y) coordinates
            
        Returns:
            True if food present
        """
        x, y = position
        return self.grid[x, y] == 1
    
    def consume_food(self, position: Tuple[int, int]) -> float:
        """Consume food at position and return energy.
        
        Args:
            position: (x, y) coordinates
            
        Returns:
            Energy gained (0 if no food)
        """
        x, y = position
        if self.grid[x, y] == 1:
            self.grid[x, y] = 0
            self.food_positions.remove((x, y))
            self.total_food_consumed += 1
            return self.food_energy_value
        return 0.0
    
    def find_nearest_food(self, position: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Find nearest food to given position.
        
        Args:
            position: (x, y) coordinates
            
        Returns:
            Position of nearest food, or None if no food exists
        """
        if not self.food_positions:
            return None
        
        x, y = position
        min_dist = float('inf')
        nearest = None
        
        for fx, fy in self.food_positions:
            dist = abs(fx - x) + abs(fy - y)  # Manhattan distance
            if dist < min_dist:
                min_dist = dist
                nearest = (fx, fy)
        
        return nearest
    
    def get_state(self) -> Dict[str, Any]:
        """Get current environment state.
        
        Returns:
            Dictionary with full environment state
        """
        return {
            'grid': self.grid.copy(),
            'food_positions': self.food_positions.copy(),
            'timestep': self.timestep,
            'total_food_spawned': self.total_food_spawned,
            'total_food_consumed': self.total_food_consumed
        }
    
    def reset(self):
        """Reset environment to initial state."""
        self.grid = np.zeros(self.size, dtype=int)
        self.food_positions = []
        self.timestep = 0
        self.total_food_spawned = 0
        self.total_food_consumed = 0
        self._spawn_initial_food(20)
