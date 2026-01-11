"""Collision and interaction handling."""

from typing import List, Tuple, Dict, Any


class CollisionHandler:
    """Handles collisions and interactions between organisms.
    
    Manages spatial conflicts, resource competition, and interaction events.
    """
    
    def __init__(self, collision_penalty: float = 10.0):
        """Initialize collision handler.
        
        Args:
            collision_penalty: Energy penalty for collisions
        """
        self.collision_penalty = collision_penalty
        self.collision_history: List[Dict[str, Any]] = []
    
    def detect_collisions(self, organism_states: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect collisions between organisms.
        
        Args:
            organism_states: List of organism state dictionaries
            
        Returns:
            List of collision events
        """
        collisions = []
        position_map: Dict[Tuple[int, int], List[int]] = {}
        
        # Map positions to organism IDs
        for state in organism_states:
            if not state['alive']:
                continue
            
            pos = state['position']
            organism_id = state['organism_id']
            
            if pos not in position_map:
                position_map[pos] = []
            position_map[pos].append(organism_id)
        
        # Find positions with multiple organisms (collisions)
        for pos, organism_ids in position_map.items():
            if len(organism_ids) > 1:
                collision = {
                    'position': pos,
                    'organism_ids': organism_ids,
                    'count': len(organism_ids),
                    'penalty': self.collision_penalty
                }
                collisions.append(collision)
                self.collision_history.append(collision)
        
        return collisions
    
    def resolve_food_competition(self, position: Tuple[int, int], 
                                 competing_organisms: List[int]) -> int:
        """Resolve which organism gets food when multiple compete.
        
        Args:
            position: Position of the food
            competing_organisms: List of organism IDs at that position
            
        Returns:
            ID of organism that gets the food
        """
        # Simple resolution: random winner
        import random
        winner = random.choice(competing_organisms)
        
        return winner
    
    def calculate_interaction_matrix(self, organism_states: List[Dict[str, Any]], 
                                    radius: int = 5) -> Dict[int, List[int]]:
        """Calculate which organisms are near each other.
        
        Args:
            organism_states: List of organism states
            radius: Interaction radius (Manhattan distance)
            
        Returns:
            Dictionary mapping organism_id to list of nearby organism_ids
        """
        interactions = {}
        
        for state in organism_states:
            if not state['alive']:
                continue
            
            organism_id = state['organism_id']
            pos = state['position']
            nearby = []
            
            for other_state in organism_states:
                if not other_state['alive'] or other_state['organism_id'] == organism_id:
                    continue
                
                other_pos = other_state['position']
                distance = abs(pos[0] - other_pos[0]) + abs(pos[1] - other_pos[1])
                
                if distance <= radius:
                    nearby.append(other_state['organism_id'])
            
            interactions[organism_id] = nearby
        
        return interactions
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get collision statistics.
        
        Returns:
            Dictionary with statistics
        """
        if not self.collision_history:
            return {
                'total_collisions': 0,
                'avg_organisms_per_collision': 0.0,
                'total_penalty': 0.0
            }
        
        return {
            'total_collisions': len(self.collision_history),
            'avg_organisms_per_collision': sum(c['count'] for c in self.collision_history) / len(self.collision_history),
            'total_penalty': len(self.collision_history) * self.collision_penalty
        }
