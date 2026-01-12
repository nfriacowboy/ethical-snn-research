"""Dual-process (survival + ethics) architecture."""

from typing import List, Dict, Any
from ..organisms.ethical_snn import EthicalSNN


class DualProcessArchitecture:
    """Manages simulation with dual-process organisms.
    
    This is Condition B in Phase 1 experiments.
    """
    
    def __init__(self, num_organisms: int, grid_size: tuple = (50, 50), 
                 ethical_weight: float = 0.5):
        """Initialize dual-process architecture.
        
        Args:
            num_organisms: Number of organisms to create
            grid_size: Size of the environment grid
            ethical_weight: Weight of ethical modulation (0-1)
        """
        self.num_organisms = num_organisms
        self.grid_size = grid_size
        self.ethical_weight = ethical_weight
        self.organisms: List[EthicalSNN] = []
        
        self._initialize_organisms()
    
    def _initialize_organisms(self):
        """Create and initialize all organisms with ethical networks."""
        import random
        
        for i in range(self.num_organisms):
            # Random starting position
            x = random.randint(0, self.grid_size[0] - 1)
            y = random.randint(0, self.grid_size[1] - 1)
            
            organism = EthicalSNN(
                organism_id=i,
                position=(x, y),
                energy=100.0,
                ethical_weight=self.ethical_weight
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
        
        # Update environment state with nearby organisms for each
        for organism in self.organisms:
            if organism.alive:
                # Find nearby organisms for ethical context
                nearby = self._get_nearby_organisms(organism)
                env_with_context = environment_state.copy()
                env_with_context['nearby_organisms'] = nearby
                
                organism.step(env_with_context)
            
            states.append(organism.get_state())
        
        return states
    
    def _get_nearby_organisms(self, target_organism: EthicalSNN, radius: int = 5) -> List[Dict[str, Any]]:
        """Find organisms near the target.
        
        Args:
            target_organism: Organism to find neighbors for
            radius: Search radius
            
        Returns:
            List of nearby organism states
        """
        nearby = []
        tx, ty = target_organism.position
        
        for organism in self.organisms:
            if organism.organism_id == target_organism.organism_id or not organism.alive:
                continue
            
            ox, oy = organism.position
            distance = abs(tx - ox) + abs(ty - oy)  # Manhattan distance
            
            if distance <= radius:
                nearby.append(organism.get_state())
        
        return nearby
    
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
                'total_organisms': self.num_organisms,
                'ethical_weight': self.ethical_weight
            }
        
        return {
            'alive_count': len(alive_organisms),
            'avg_energy': sum(org.energy for org in alive_organisms) / len(alive_organisms),
            'avg_age': sum(org.age for org in alive_organisms) / len(alive_organisms),
            'total_organisms': self.num_organisms,
            'ethical_weight': self.ethical_weight
        }
