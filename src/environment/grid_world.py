"""Grid World environment with toroidal topology.

This module implements a 2D discrete grid world where organisms can move,
consume food, and interact. The grid uses toroidal (wrap-around) topology.

Example:
    >>> config = load_config("config.yaml")
    >>> grid = GridWorld(config)
    >>> state = grid.get_state()
    >>> local_view = grid.get_local_state((10, 10), radius=2)
"""

from typing import Tuple, List, Dict, Any, Optional
import numpy as np


class GridWorld:
    """2D Grid environment with toroidal topology.
    
    The grid world is a discrete square grid where:
    - Organisms can occupy cells
    - Food items spawn and can be consumed
    - Boundaries wrap around (toroidal topology)
    
    Attributes:
        grid_size: Size of the square grid
        grid: 2D numpy array representing grid state
        food_positions: List of (x, y) tuples for food locations
        organism_positions: Dict mapping organism_id to (x, y) position
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize grid world from configuration.
        
        Args:
            config: Configuration dictionary with 'environment' key
                    containing grid_size, num_food, etc.
        """
        env_config = config['environment']
        
        self.grid_size = env_config['grid_size']
        self.num_food = env_config['num_food']
        self.food_respawn_rate = env_config['food_respawn_rate']
        self.food_energy_value = env_config['food_energy_value']
        
        # Grid state: 0 = empty, 1 = food, 2 = organism
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        
        # Food tracking
        self.food_positions: List[Tuple[int, int]] = []
        self.max_food = self.num_food * 2  # Allow some dynamic range
        
        # Organism tracking
        self.organism_positions: Dict[int, Tuple[int, int]] = {}
        
        # Statistics
        self.timestep = 0
        self.total_food_spawned = 0
        self.total_food_consumed = 0
        
        # Initialize food
        self._spawn_initial_food()
    
    def _spawn_initial_food(self) -> None:
        """Spawn initial food items randomly across the grid."""
        for _ in range(self.num_food):
            # Find empty position
            attempts = 0
            while attempts < 100:
                x = np.random.randint(0, self.grid_size)
                y = np.random.randint(0, self.grid_size)
                
                if self.grid[x, y] == 0:  # Empty cell
                    self.food_positions.append((x, y))
                    self.grid[x, y] = 1
                    self.total_food_spawned += 1
                    break
                
                attempts += 1
    
    def wrap_position(self, position: Tuple[int, int]) -> Tuple[int, int]:
        """Wrap position to stay within grid bounds (toroidal topology).
        
        Args:
            position: (x, y) position tuple
            
        Returns:
            Wrapped (x, y) position within [0, grid_size)
            
        Example:
            >>> grid = GridWorld(config)
            >>> grid.wrap_position((21, 5))  # grid_size=20
            (1, 5)
            >>> grid.wrap_position((-1, 10))
            (19, 10)
        """
        x, y = position
        return (x % self.grid_size, y % self.grid_size)
    
    def is_valid_position(self, position: Tuple[int, int]) -> bool:
        """Check if position is valid (within bounds).
        
        Note: With toroidal topology, all positions can be wrapped,
        but this checks if position is already in valid range.
        
        Args:
            position: (x, y) position tuple
            
        Returns:
            True if position is within [0, grid_size), False otherwise
        """
        x, y = position
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size
    
    def get_state(self) -> np.ndarray:
        """Get complete grid state as 2D array.
        
        Returns:
            2D numpy array where:
                0 = empty
                1 = food
                2 = organism
                
        Example:
            >>> grid = GridWorld(config)
            >>> state = grid.get_state()
            >>> state.shape
            (20, 20)
        """
        return self.grid.copy()
    
    def get_local_state(self, position: Tuple[int, int], radius: int = 2) -> np.ndarray:
        """Get local view of grid centered at position.
        
        Uses toroidal wrapping for edges. Returns a square window of
        size (2*radius + 1, 2*radius + 1) centered on position.
        
        Args:
            position: Center (x, y) position
            radius: Radius of local view (default: 2)
            
        Returns:
            2D numpy array of size (2*radius+1, 2*radius+1)
            
        Example:
            >>> grid = GridWorld(config)
            >>> local = grid.get_local_state((10, 10), radius=2)
            >>> local.shape
            (5, 5)
        """
        x, y = self.wrap_position(position)
        window_size = 2 * radius + 1
        local_view = np.zeros((window_size, window_size), dtype=np.int8)
        
        for i in range(window_size):
            for j in range(window_size):
                # Calculate offset from center
                dx = i - radius
                dy = j - radius
                
                # Get wrapped position
                sample_x, sample_y = self.wrap_position((x + dx, y + dy))
                local_view[i, j] = self.grid[sample_x, sample_y]
        
        return local_view
    
    def place_organism(self, organism_id: int, position: Tuple[int, int]) -> bool:
        """Place an organism at a position.
        
        Args:
            organism_id: Unique organism identifier
            position: (x, y) position tuple
            
        Returns:
            True if placement successful, False if position occupied
        """
        x, y = self.wrap_position(position)
        
        if self.grid[x, y] == 0:  # Empty cell
            self.grid[x, y] = 2
            self.organism_positions[organism_id] = (x, y)
            return True
        
        return False
    
    def move_organism(self, organism_id: int, new_position: Tuple[int, int]) -> bool:
        """Move an organism to a new position.
        
        Args:
            organism_id: Organism identifier
            new_position: New (x, y) position
            
        Returns:
            True if move successful, False if position occupied or invalid
        """
        if organism_id not in self.organism_positions:
            return False
        
        old_x, old_y = self.organism_positions[organism_id]
        new_x, new_y = self.wrap_position(new_position)
        
        # Check if new position is empty or has food
        if self.grid[new_x, new_y] in [0, 1]:
            # Clear old position
            self.grid[old_x, old_y] = 0
            
            # Update new position
            self.grid[new_x, new_y] = 2
            self.organism_positions[organism_id] = (new_x, new_y)
            return True
        
        return False
    
    def remove_organism(self, organism_id: int) -> None:
        """Remove an organism from the grid.
        
        Args:
            organism_id: Organism identifier
        """
        if organism_id in self.organism_positions:
            x, y = self.organism_positions[organism_id]
            self.grid[x, y] = 0
            del self.organism_positions[organism_id]
    
    def has_food_at(self, position: Tuple[int, int]) -> bool:
        """Check if there is food at a position.
        
        Args:
            position: (x, y) position tuple
            
        Returns:
            True if food present, False otherwise
        """
        x, y = self.wrap_position(position)
        return self.grid[x, y] == 1
    
    def consume_food(self, position: Tuple[int, int]) -> bool:
        """Consume food at a position.
        
        Args:
            position: (x, y) position tuple
            
        Returns:
            True if food was consumed, False if no food present
        """
        x, y = self.wrap_position(position)
        
        if self.grid[x, y] == 1:
            self.grid[x, y] = 0
            if (x, y) in self.food_positions:
                self.food_positions.remove((x, y))
            self.total_food_consumed += 1
            return True
        
        return False
    
    def spawn_food(self) -> None:
        """Spawn new food according to respawn rate."""
        if len(self.food_positions) >= self.max_food:
            return
        
        # Spawn with probability
        if np.random.random() < self.food_respawn_rate:
            # Find empty position
            attempts = 0
            while attempts < 10:
                x = np.random.randint(0, self.grid_size)
                y = np.random.randint(0, self.grid_size)
                
                if self.grid[x, y] == 0:
                    self.food_positions.append((x, y))
                    self.grid[x, y] = 1
                    self.total_food_spawned += 1
                    break
                
                attempts += 1
    
    def check_collision(self, position: Tuple[int, int]) -> Optional[int]:
        """Check if there is an organism at a position.
        
        Args:
            position: (x, y) position tuple
            
        Returns:
            Organism ID if present, None otherwise
        """
        x, y = self.wrap_position(position)
        
        if self.grid[x, y] == 2:
            # Find which organism
            for org_id, org_pos in self.organism_positions.items():
                if org_pos == (x, y):
                    return org_id
        
        return None
    
    def step(self) -> None:
        """Advance environment by one timestep.
        
        Handles food spawning and other time-based updates.
        """
        self.timestep += 1
        self.spawn_food()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current environment statistics.
        
        Returns:
            Dictionary with stats like food count, organism count, etc.
        """
        return {
            'timestep': self.timestep,
            'food_count': len(self.food_positions),
            'organism_count': len(self.organism_positions),
            'total_food_spawned': self.total_food_spawned,
            'total_food_consumed': self.total_food_consumed
        }
    
    def reset(self) -> None:
        """Reset environment to initial state."""
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        self.food_positions = []
        self.organism_positions = {}
        self.timestep = 0
        self.total_food_spawned = 0
        self.total_food_consumed = 0
        self._spawn_initial_food()
