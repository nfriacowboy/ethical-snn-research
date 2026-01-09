"""Single-process (survival-only) architecture."""

from typing import List, Dict, Any
from ..organisms.survival_snn import SurvivalSNN


class SingleProcessArchitecture:
    """Manages simulation with survival-only organisms.
    
    This is Condition A in Phase 1 experiments.
    """
    
    def __init__(self, num_organisms: int, grid_size: tuple = (50, 50)):
        """Initialize single-process architecture.
        
        Args:
            num_organisms: Number of organisms to create
            grid_size: Size of the environment grid
        """
        self.num_organisms = num_organisms
        self.grid_size = grid_size
        self.organisms: List[SurvivalSNN] = []
        
        self._initialize_organisms()
    
    def _initialize_organisms(self):
        """Create and initialize all organisms."""
        import random
        
        for i in range(self.num_organisms):
            # Random starting position
            x = random.randint(0, self.grid_size[0] - 1)
            y = random.randint(0, self.grid_size[1] - 1)
            
            organism = SurvivalSNN(
                organism_id=i,
                position=(x, y),
                energy=100.0
            )
            self.organisms.append(organism)
    
    def step(self, environment_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute one timestep for all organisms.
        
        Args:
            environment_state: Current state of the environment
            
        Returns:
            List of organism states after stepping
        """
        states = []
        
        for organism in self.organisms:
            if organism.alive:
                organism.step(environment_state)
            states.append(organism.get_state())
        
        return states
    
    def get_alive_count(self) -> int:
        """Get number of living organisms.
        
        Returns:
            Count of alive organisms
        """
        return sum(1 for org in self.organisms if org.alive)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get population statistics.
        
        Returns:
            Dictionary with population metrics
        """
        alive_organisms = [org for org in self.organisms if org.alive]
        
        if not alive_organisms:
            return {
                'alive_count': 0,
                'avg_energy': 0.0,
                'avg_age': 0.0,
                'total_organisms': self.num_organisms
            }
        
        return {
            'alive_count': len(alive_organisms),
            'avg_energy': sum(org.energy for org in alive_organisms) / len(alive_organisms),
            'avg_age': sum(org.age for org in alive_organisms) / len(alive_organisms),
            'total_organisms': self.num_organisms
        }
