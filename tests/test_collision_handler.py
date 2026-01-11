"""Unit tests for collision_handler.

Tests cover:
- Collision detection
- Attack resolution
- Movement conflict resolution
- Food competition resolution
- Adjacent organism detection
"""

import pytest
import numpy as np

from src.environment.collision_handler import (
    check_collision,
    resolve_attack,
    resolve_movement,
    resolve_food_competition,
    detect_adjacent_organisms,
    AttackResult
)


class TestCollisionDetection:
    """Test collision detection function."""
    
    def test_same_position(self):
        """Test collision at same position."""
        assert check_collision((5, 5), (5, 5)) == True
    
    def test_adjacent_no_radius(self):
        """Test adjacent positions without collision radius."""
        assert check_collision((5, 5), (5, 6)) == False
        assert check_collision((5, 5), (6, 5)) == False
    
    def test_adjacent_with_radius(self):
        """Test adjacent positions with collision radius."""
        assert check_collision((5, 5), (5, 6), collision_radius=1) == True
        assert check_collision((5, 5), (6, 5), collision_radius=1) == True
    
    def test_far_apart(self):
        """Test positions far apart."""
        assert check_collision((5, 5), (10, 10)) == False
        assert check_collision((5, 5), (10, 10), collision_radius=5) == False
    
    def test_diagonal_distance(self):
        """Test diagonal positions."""
        # Manhattan distance (1,1) from (5,5) is 2
        assert check_collision((5, 5), (6, 6), collision_radius=1) == False
        assert check_collision((5, 5), (6, 6), collision_radius=2) == True
    
    def test_negative_coordinates(self):
        """Test with negative coordinates."""
        assert check_collision((-1, -1), (-1, -1)) == True
        assert check_collision((-1, -1), (0, 0), collision_radius=2) == True


class TestAttackResolution:
    """Test attack resolution."""
    
    def test_successful_attack(self):
        """Test successful attack."""
        result = resolve_attack(
            attacker_id=0,
            target_id=1,
            attacker_energy=50.0,
            target_energy=40.0,
            attack_cost=5.0,
            steal_percentage=0.5
        )
        
        assert result.success == True
        assert result.attacker_id == 0
        assert result.target_id == 1
        assert result.energy_stolen == 20.0
        assert result.target_killed == False
    
    def test_attack_kills_target(self):
        """Test attack that kills target."""
        result = resolve_attack(
            attacker_id=0,
            target_id=1,
            attacker_energy=50.0,
            target_energy=10.0,
            attack_cost=5.0,
            steal_percentage=1.0
        )
        
        assert result.success == True
        assert result.energy_stolen == 10.0
        assert result.target_killed == True
    
    def test_attack_insufficient_energy(self):
        """Test attack fails with insufficient energy."""
        result = resolve_attack(
            attacker_id=0,
            target_id=1,
            attacker_energy=3.0,
            target_energy=40.0,
            attack_cost=5.0,
            steal_percentage=0.5
        )
        
        assert result.success == False
        assert result.energy_stolen == 0.0
        assert result.target_killed == False
    
    def test_attack_zero_target_energy(self):
        """Test attack on target with no energy."""
        result = resolve_attack(
            attacker_id=0,
            target_id=1,
            attacker_energy=50.0,
            target_energy=0.0,
            attack_cost=5.0,
            steal_percentage=0.5
        )
        
        assert result.success == True
        assert result.energy_stolen == 0.0
        assert result.target_killed == True
    
    def test_attack_custom_parameters(self):
        """Test attack with custom parameters."""
        result = resolve_attack(
            attacker_id=5,
            target_id=10,
            attacker_energy=100.0,
            target_energy=60.0,
            attack_cost=10.0,
            steal_percentage=0.3
        )
        
        assert result.success == True
        assert result.attacker_id == 5
        assert result.target_id == 10
        assert result.energy_stolen == 18.0  # 60 * 0.3


class TestMovementResolution:
    """Test movement conflict resolution."""
    
    def test_no_conflict(self):
        """Test movement without conflicts."""
        proposed = {0: (5, 5), 1: (6, 6)}
        current = {0: (4, 5), 1: (5, 6)}
        
        result = resolve_movement(proposed, current)
        
        assert result[0] == (5, 5)
        assert result[1] == (6, 6)
    
    def test_conflict_first_priority(self):
        """Test conflict with 'first' priority rule."""
        proposed = {0: (5, 5), 1: (5, 5)}
        current = {0: (4, 5), 1: (6, 5)}
        
        result = resolve_movement(proposed, current, priority_rule="first")
        
        assert result[0] == (5, 5)  # Lower ID wins
        assert result[1] == (6, 5)  # Stays in place
    
    def test_conflict_random_priority(self):
        """Test conflict with 'random' priority rule."""
        np.random.seed(42)
        
        proposed = {0: (5, 5), 1: (5, 5)}
        current = {0: (4, 5), 1: (6, 5)}
        
        result = resolve_movement(proposed, current, priority_rule="random")
        
        # One should move, one should stay
        assert result[0] in [(5, 5), (4, 5)]
        assert result[1] in [(5, 5), (6, 5)]
        # Exactly one should be at target
        assert (result[0] == (5, 5)) != (result[1] == (5, 5))
    
    def test_three_way_conflict(self):
        """Test conflict with three organisms."""
        proposed = {0: (5, 5), 1: (5, 5), 2: (5, 5)}
        current = {0: (4, 5), 1: (6, 5), 2: (5, 4)}
        
        result = resolve_movement(proposed, current, priority_rule="first")
        
        assert result[0] == (5, 5)
        assert result[1] == (6, 5)
        assert result[2] == (5, 4)
    
    def test_no_movement_requested(self):
        """Test organism that doesn't request movement."""
        proposed = {0: (5, 5)}
        current = {0: (4, 5), 1: (6, 6)}
        
        result = resolve_movement(proposed, current)
        
        assert result[0] == (5, 5)
        assert result[1] == (6, 6)  # Stays in current
    
    def test_move_to_occupied_position(self):
        """Test moving to position occupied by another organism."""
        proposed = {0: (6, 6)}  # Try to move where 1 is
        current = {0: (5, 5), 1: (6, 6)}
        
        result = resolve_movement(proposed, current)
        
        assert result[0] == (5, 5)  # Blocked, stays in place
        assert result[1] == (6, 6)  # Stays


class TestFoodCompetition:
    """Test food competition resolution."""
    
    def test_single_organism(self):
        """Test single organism at food position."""
        positions = {0: (5, 5), 1: (6, 6)}
        winner = resolve_food_competition((5, 5), positions)
        
        assert winner == 0
    
    def test_no_organism_at_food(self):
        """Test no organism at food position."""
        positions = {0: (5, 5), 1: (6, 6)}
        winner = resolve_food_competition((10, 10), positions)
        
        assert winner is None
    
    def test_multiple_organisms_first_priority(self):
        """Test multiple organisms with 'first' priority."""
        positions = {0: (5, 5), 1: (5, 5), 2: (6, 6)}
        winner = resolve_food_competition((5, 5), positions, priority_rule="first")
        
        assert winner == 0
    
    def test_multiple_organisms_random_priority(self):
        """Test multiple organisms with 'random' priority."""
        np.random.seed(42)
        positions = {0: (5, 5), 1: (5, 5), 2: (5, 5)}
        winner = resolve_food_competition((5, 5), positions, priority_rule="random")
        
        assert winner in [0, 1, 2]
    
    def test_empty_positions(self):
        """Test with no organisms."""
        positions = {}
        winner = resolve_food_competition((5, 5), positions)
        
        assert winner is None


class TestAdjacentDetection:
    """Test adjacent organism detection."""
    
    def test_detect_adjacent_basic(self):
        """Test detecting adjacent organisms."""
        positions = {0: (5, 5), 1: (5, 6), 2: (6, 5), 3: (7, 7)}
        adjacent = detect_adjacent_organisms((5, 5), positions, grid_size=20)
        
        assert set(adjacent) == {1, 2}
    
    def test_no_adjacent(self):
        """Test no adjacent organisms."""
        positions = {0: (5, 5), 1: (10, 10)}
        adjacent = detect_adjacent_organisms((5, 5), positions, grid_size=20)
        
        assert adjacent == []
    
    def test_all_four_directions(self):
        """Test organism with neighbors in all 4 directions."""
        positions = {
            0: (5, 5),
            1: (4, 5),  # West
            2: (6, 5),  # East
            3: (5, 4),  # North
            4: (5, 6)   # South
        }
        adjacent = detect_adjacent_organisms((5, 5), positions, grid_size=20)
        
        assert set(adjacent) == {1, 2, 3, 4}
    
    def test_diagonal_not_adjacent(self):
        """Test that diagonal positions are not considered adjacent."""
        positions = {0: (5, 5), 1: (6, 6)}
        adjacent = detect_adjacent_organisms((5, 5), positions, grid_size=20)
        
        assert adjacent == []
    
    def test_toroidal_wrapping(self):
        """Test adjacent detection with toroidal wrapping."""
        positions = {0: (0, 0), 1: (19, 0), 2: (0, 19)}
        
        # From (0,0), (19,0) and (0,19) are adjacent with wrapping
        adjacent = detect_adjacent_organisms((0, 0), positions, grid_size=20, toroidal=True)
        
        assert set(adjacent) == {1, 2}
    
    def test_non_toroidal_edges(self):
        """Test adjacent detection without wrapping at edges."""
        positions = {0: (0, 0), 1: (19, 0)}
        
        # Without wrapping, (19,0) is not adjacent to (0,0)
        adjacent = detect_adjacent_organisms((0, 0), positions, grid_size=20, toroidal=False)
        
        assert adjacent == []
    
    def test_multiple_at_same_position(self):
        """Test multiple organisms at adjacent position."""
        # Note: This shouldn't happen in practice, but test behavior
        positions = {0: (5, 5), 1: (5, 6), 2: (5, 6)}
        adjacent = detect_adjacent_organisms((5, 5), positions, grid_size=20)
        
        assert set(adjacent) == {1, 2}


class TestAttackResultDataclass:
    """Test AttackResult dataclass."""
    
    def test_create_attack_result(self):
        """Test creating AttackResult."""
        result = AttackResult(
            success=True,
            attacker_id=1,
            target_id=2,
            energy_stolen=15.0,
            target_killed=False
        )
        
        assert result.success == True
        assert result.attacker_id == 1
        assert result.target_id == 2
        assert result.energy_stolen == 15.0
        assert result.target_killed == False
