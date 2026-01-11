"""Collision detection and interaction resolution.

Handles:
- Collision detection between organisms
- Attack resolution with energy transfer
- Movement conflict resolution
- Simultaneous action handling
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class AttackResult:
    """Result of an attack interaction.
    
    Attributes:
        success: Whether attack succeeded
        attacker_id: ID of attacking organism
        target_id: ID of target organism
        energy_stolen: Energy transferred from target to attacker
        target_killed: Whether target died from attack
    """
    success: bool
    attacker_id: int
    target_id: int
    energy_stolen: float
    target_killed: bool


def check_collision(
    pos1: Tuple[int, int], 
    pos2: Tuple[int, int],
    collision_radius: int = 0
) -> bool:
    """Check if two positions collide.
    
    Collision occurs when positions are within collision_radius.
    Default radius=0 means only exact same position counts as collision.
    
    Args:
        pos1: First position (x, y)
        pos2: Second position (x, y)
        collision_radius: Maximum distance for collision (default: 0)
        
    Returns:
        True if positions collide, False otherwise
        
    Example:
        >>> check_collision((5, 5), (5, 5))  # Same position
        True
        >>> check_collision((5, 5), (5, 6))  # Adjacent
        False
        >>> check_collision((5, 5), (5, 6), collision_radius=1)  # Adjacent with radius
        True
        >>> check_collision((5, 5), (7, 7))  # Far apart
        False
    """
    x1, y1 = pos1
    x2, y2 = pos2
    
    distance = abs(x2 - x1) + abs(y2 - y1)  # Manhattan distance
    
    return distance <= collision_radius


def resolve_attack(
    attacker_id: int,
    target_id: int,
    attacker_energy: float,
    target_energy: float,
    attack_cost: float = 5.0,
    steal_percentage: float = 0.5
) -> AttackResult:
    """Resolve an attack between two organisms.
    
    Attack mechanics:
    - Attacker pays attack_cost energy
    - If target has energy, attacker steals steal_percentage of it
    - If target's energy drops to 0, target dies
    
    Args:
        attacker_id: ID of attacking organism
        target_id: ID of target organism
        attacker_energy: Current energy of attacker
        target_energy: Current energy of target
        attack_cost: Energy cost for attacker to perform attack
        steal_percentage: Fraction of target energy stolen (0-1)
        
    Returns:
        AttackResult with details of the attack
        
    Example:
        >>> result = resolve_attack(
        ...     attacker_id=0, target_id=1,
        ...     attacker_energy=50, target_energy=40,
        ...     attack_cost=5, steal_percentage=0.5
        ... )
        >>> result.success
        True
        >>> result.energy_stolen
        20.0
    """
    # Check if attacker has enough energy to attack
    if attacker_energy < attack_cost:
        return AttackResult(
            success=False,
            attacker_id=attacker_id,
            target_id=target_id,
            energy_stolen=0.0,
            target_killed=False
        )
    
    # Calculate stolen energy
    energy_stolen = target_energy * steal_percentage
    
    # Check if target dies
    remaining_energy = target_energy - energy_stolen
    target_killed = remaining_energy <= 0
    
    return AttackResult(
        success=True,
        attacker_id=attacker_id,
        target_id=target_id,
        energy_stolen=energy_stolen,
        target_killed=target_killed
    )


def resolve_movement(
    proposed_moves: Dict[int, Tuple[int, int]],
    current_positions: Dict[int, Tuple[int, int]],
    priority_rule: str = "random"
) -> Dict[int, Tuple[int, int]]:
    """Resolve movement conflicts when multiple organisms try to move to same position.
    
    Handles simultaneous movement requests and prevents multiple organisms
    from occupying the same cell.
    
    Args:
        proposed_moves: Dict mapping organism_id -> desired new position
        current_positions: Dict mapping organism_id -> current position
        priority_rule: How to resolve conflicts ("random", "first", "energy")
            - "random": Random organism gets priority
            - "first": Lower ID gets priority
            - "energy": Would require energy values (not implemented)
            
    Returns:
        Dict mapping organism_id -> final position (may be current if blocked)
        
    Example:
        >>> proposed = {0: (5, 5), 1: (5, 5)}  # Both want same position
        >>> current = {0: (4, 5), 1: (6, 5)}
        >>> result = resolve_movement(proposed, current, priority_rule="first")
        >>> result[0]  # ID 0 gets the position
        (5, 5)
        >>> result[1]  # ID 1 stays in place
        (6, 5)
    """
    final_positions: Dict[int, Tuple[int, int]] = {}
    occupied: set = set(current_positions.values())
    
    # Group organisms by target position
    target_groups: Dict[Tuple[int, int], List[int]] = {}
    for org_id, target_pos in proposed_moves.items():
        if target_pos not in target_groups:
            target_groups[target_pos] = []
        target_groups[target_pos].append(org_id)
    
    # Resolve conflicts
    for target_pos, org_ids in target_groups.items():
        if len(org_ids) == 1:
            # No conflict, move succeeds if position not occupied by others
            org_id = org_ids[0]
            if target_pos not in occupied or target_pos == current_positions[org_id]:
                final_positions[org_id] = target_pos
                occupied.add(target_pos)
            else:
                # Position occupied, stay in place
                final_positions[org_id] = current_positions[org_id]
        else:
            # Multiple organisms want same position - resolve by priority
            if priority_rule == "random":
                winner_id = np.random.choice(org_ids)
            elif priority_rule == "first":
                winner_id = min(org_ids)
            else:
                # Default to first
                winner_id = min(org_ids)
            
            # Winner moves, others stay
            final_positions[winner_id] = target_pos
            occupied.add(target_pos)
            
            for org_id in org_ids:
                if org_id != winner_id:
                    final_positions[org_id] = current_positions[org_id]
    
    # Add organisms that didn't request movement
    for org_id, current_pos in current_positions.items():
        if org_id not in final_positions:
            final_positions[org_id] = current_pos
    
    return final_positions


def resolve_food_competition(
    food_position: Tuple[int, int],
    organism_positions: Dict[int, Tuple[int, int]],
    priority_rule: str = "random"
) -> Optional[int]:
    """Resolve competition when multiple organisms try to eat the same food.
    
    Args:
        food_position: Position of food item
        organism_positions: Dict mapping organism_id -> position
        priority_rule: How to choose winner ("random" or "first")
        
    Returns:
        ID of organism that gets the food, or None if no organism at position
        
    Example:
        >>> positions = {0: (5, 5), 1: (5, 5), 2: (6, 6)}
        >>> winner = resolve_food_competition((5, 5), positions, "first")
        >>> winner
        0
    """
    # Find all organisms at food position
    candidates = [
        org_id for org_id, pos in organism_positions.items()
        if pos == food_position
    ]
    
    if not candidates:
        return None
    
    if len(candidates) == 1:
        return candidates[0]
    
    # Multiple organisms - resolve by priority
    if priority_rule == "random":
        return int(np.random.choice(candidates))
    elif priority_rule == "first":
        return min(candidates)
    else:
        return min(candidates)


def detect_adjacent_organisms(
    position: Tuple[int, int],
    organism_positions: Dict[int, Tuple[int, int]],
    grid_size: int,
    toroidal: bool = True
) -> List[int]:
    """Detect organisms adjacent to a position (for attack range).
    
    Adjacent means Manhattan distance = 1 (4-connectivity).
    
    Args:
        position: Center position to check around
        organism_positions: Dict mapping organism_id -> position
        grid_size: Size of grid (for wrapping)
        toroidal: Whether to wrap around edges
        
    Returns:
        List of organism IDs adjacent to position
        
    Example:
        >>> positions = {0: (5, 5), 1: (5, 6), 2: (7, 7)}
        >>> adjacent = detect_adjacent_organisms((5, 5), positions, 20)
        >>> adjacent
        [1]
    """
    x, y = position
    adjacent_ids = []
    
    # Check 4 cardinal directions
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        
        if toroidal:
            nx = nx % grid_size
            ny = ny % grid_size
        else:
            if nx < 0 or nx >= grid_size or ny < 0 or ny >= grid_size:
                continue
        
        neighbor_pos = (nx, ny)
        
        # Check if any organism is at this position
        for org_id, org_pos in organism_positions.items():
            if org_pos == neighbor_pos:
                adjacent_ids.append(org_id)
    
    return adjacent_ids
