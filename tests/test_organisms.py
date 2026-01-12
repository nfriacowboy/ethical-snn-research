"""Unit tests for base organism class and Action enum.

Tests cover:
- Action enum functionality
- Organism initialization
- Energy management
- Movement mechanics
- Lifecycle (alive/dead states)
- Statistics tracking
- Action logging
"""

import pytest
from typing import Dict, Any

from src.organisms.base_organism import Action, Organism


class TestOrganism(Organism):
    """Concrete test implementation of abstract Organism."""
    
    def decide(self, state: Dict[str, Any]) -> Action:
        """Simple test implementation: always wait."""
        return Action.WAIT


class TestAction:
    """Test Action enum."""
    
    def test_action_values(self):
        """Test action enum values."""
        assert Action.MOVE_NORTH == 0
        assert Action.MOVE_SOUTH == 1
        assert Action.MOVE_EAST == 2
        assert Action.MOVE_WEST == 3
        assert Action.EAT == 4
        assert Action.ATTACK == 5
        assert Action.WAIT == 6
    
    def test_is_movement(self):
        """Test is_movement classification."""
        assert Action.is_movement(Action.MOVE_NORTH) == True
        assert Action.is_movement(Action.MOVE_SOUTH) == True
        assert Action.is_movement(Action.MOVE_EAST) == True
        assert Action.is_movement(Action.MOVE_WEST) == True
        
        assert Action.is_movement(Action.EAT) == False
        assert Action.is_movement(Action.ATTACK) == False
        assert Action.is_movement(Action.WAIT) == False
    
    def test_get_direction_vector_north(self):
        """Test direction vector for north."""
        assert Action.get_direction_vector(Action.MOVE_NORTH) == (0, -1)
    
    def test_get_direction_vector_south(self):
        """Test direction vector for south."""
        assert Action.get_direction_vector(Action.MOVE_SOUTH) == (0, 1)
    
    def test_get_direction_vector_east(self):
        """Test direction vector for east."""
        assert Action.get_direction_vector(Action.MOVE_EAST) == (1, 0)
    
    def test_get_direction_vector_west(self):
        """Test direction vector for west."""
        assert Action.get_direction_vector(Action.MOVE_WEST) == (-1, 0)
    
    def test_get_direction_vector_invalid(self):
        """Test direction vector for non-movement action."""
        with pytest.raises(ValueError):
            Action.get_direction_vector(Action.EAT)


class TestOrganismInitialization:
    """Test organism initialization."""
    
    def test_basic_init(self):
        """Test basic initialization."""
        org = TestOrganism(
            organism_id=0,
            position=(10, 10),
            initial_energy=100.0
        )
        
        assert org.organism_id == 0
        assert org.position == (10, 10)
        assert org.energy == 100.0
        assert org.max_energy == 100.0
        assert org.alive == True
        assert org.age == 0
    
    def test_custom_energy(self):
        """Test initialization with custom energy."""
        org = TestOrganism(
            organism_id=5,
            position=(5, 5),
            initial_energy=80.0,
            max_energy=150.0
        )
        
        assert org.energy == 80.0
        assert org.max_energy == 150.0
    
    def test_initial_statistics(self):
        """Test initial statistics are zero."""
        org = TestOrganism(organism_id=0, position=(10, 10))
        
        assert org.total_food_consumed == 0
        assert org.total_attacks_attempted == 0
        assert org.total_distance_moved == 0
        assert len(org.action_history) == 0


class TestEnergyManagement:
    """Test energy update mechanics."""
    
    def test_positive_energy_update(self):
        """Test adding energy."""
        org = TestOrganism(
            organism_id=0,
            position=(10, 10),
            initial_energy=50.0
        )
        
        org.update_energy(20.0)
        assert org.energy == 70.0
        assert org.is_alive() == True
    
    def test_negative_energy_update(self):
        """Test removing energy."""
        org = TestOrganism(
            organism_id=0,
            position=(10, 10),
            initial_energy=100.0
        )
        
        org.update_energy(-30.0)
        assert org.energy == 70.0
        assert org.is_alive() == True
    
    def test_energy_clamped_to_max(self):
        """Test energy doesn't exceed max."""
        org = TestOrganism(
            organism_id=0,
            position=(10, 10),
            initial_energy=90.0,
            max_energy=100.0
        )
        
        org.update_energy(50.0)
        assert org.energy == 100.0  # Clamped to max
    
    def test_death_on_zero_energy(self):
        """Test organism dies when energy reaches 0."""
        org = TestOrganism(
            organism_id=0,
            position=(10, 10),
            initial_energy=50.0
        )
        
        org.update_energy(-50.0)
        assert org.energy == 0.0
        assert org.is_alive() == False
        assert org.alive == False
    
    def test_death_on_negative_energy(self):
        """Test organism dies when energy goes negative."""
        org = TestOrganism(
            organism_id=0,
            position=(10, 10),
            initial_energy=30.0
        )
        
        org.update_energy(-50.0)
        assert org.energy == 0.0  # Clamped to 0
        assert org.is_alive() == False
    
    def test_stays_dead(self):
        """Test organism stays dead after energy depletion."""
        org = TestOrganism(
            organism_id=0,
            position=(10, 10),
            initial_energy=10.0
        )
        
        org.update_energy(-20.0)
        assert org.is_alive() == False
        
        # Adding energy doesn't revive
        org.update_energy(50.0)
        assert org.is_alive() == False


class TestMovement:
    """Test movement mechanics."""
    
    def test_simple_move(self):
        """Test basic movement."""
        org = TestOrganism(organism_id=0, position=(10, 10))
        
        org.move((11, 10))
        assert org.position == (11, 10)
    
    def test_move_updates_distance(self):
        """Test distance tracking."""
        org = TestOrganism(organism_id=0, position=(10, 10))
        
        org.move((11, 11))
        # Manhattan distance: |11-10| + |11-10| = 2
        assert org.total_distance_moved == 2
    
    def test_multiple_moves(self):
        """Test multiple movements."""
        org = TestOrganism(organism_id=0, position=(10, 10))
        
        org.move((12, 10))  # Distance: 2
        org.move((12, 15))  # Distance: 5
        
        assert org.position == (12, 15)
        assert org.total_distance_moved == 7
    
    def test_dead_organism_cant_move(self):
        """Test dead organism doesn't move."""
        org = TestOrganism(organism_id=0, position=(10, 10))
        
        org.update_energy(-100)  # Kill it
        org.move((15, 15))
        
        assert org.position == (10, 10)  # Didn't move


class TestLifecycle:
    """Test organism lifecycle."""
    
    def test_initial_alive(self):
        """Test organism starts alive."""
        org = TestOrganism(organism_id=0, position=(10, 10))
        assert org.is_alive() == True
    
    def test_increment_age(self):
        """Test age increment."""
        org = TestOrganism(organism_id=0, position=(10, 10))
        
        assert org.age == 0
        org.increment_age()
        assert org.age == 1
        org.increment_age()
        assert org.age == 2
    
    def test_age_increments_while_alive(self):
        """Test age continues incrementing."""
        org = TestOrganism(organism_id=0, position=(10, 10))
        
        for i in range(100):
            org.increment_age()
        
        assert org.age == 100


class TestActionLogging:
    """Test action logging."""
    
    def test_log_single_action(self):
        """Test logging a single action."""
        org = TestOrganism(organism_id=0, position=(10, 10))
        
        org.log_action(Action.MOVE_NORTH, success=True)
        
        assert len(org.action_history) == 1
        assert org.action_history[0]['action'] == Action.MOVE_NORTH
        assert org.action_history[0]['success'] == True
    
    def test_log_multiple_actions(self):
        """Test logging multiple actions."""
        org = TestOrganism(organism_id=0, position=(10, 10))
        
        org.log_action(Action.MOVE_NORTH)
        org.log_action(Action.EAT)
        org.log_action(Action.WAIT)
        
        assert len(org.action_history) == 3
    
    def test_log_includes_state(self):
        """Test logged action includes state."""
        org = TestOrganism(organism_id=0, position=(10, 10), initial_energy=80.0)
        org.increment_age()
        
        org.log_action(Action.WAIT)
        
        log_entry = org.action_history[0]
        assert log_entry['timestep'] == 1
        assert log_entry['energy'] == 80.0
        assert log_entry['position'] == (10, 10)
    
    def test_log_failed_action(self):
        """Test logging failed action."""
        org = TestOrganism(organism_id=0, position=(10, 10))
        
        org.log_action(Action.ATTACK, success=False)
        
        assert org.action_history[0]['success'] == False


class TestStatistics:
    """Test statistics gathering."""
    
    def test_get_statistics(self):
        """Test getting statistics."""
        org = TestOrganism(organism_id=5, position=(10, 10), initial_energy=80.0)
        
        stats = org.get_statistics()
        
        assert stats['organism_id'] == 5
        assert stats['age'] == 0
        assert stats['alive'] == True
        assert stats['energy'] == 80.0
        assert stats['position'] == (10, 10)
    
    def test_statistics_track_food(self):
        """Test food consumption tracking."""
        org = TestOrganism(organism_id=0, position=(10, 10))
        
        org.total_food_consumed = 5
        stats = org.get_statistics()
        
        assert stats['total_food_consumed'] == 5
    
    def test_statistics_track_attacks(self):
        """Test attack tracking."""
        org = TestOrganism(organism_id=0, position=(10, 10))
        
        org.total_attacks_attempted = 3
        stats = org.get_statistics()
        
        assert stats['total_attacks_attempted'] == 3


class TestReset:
    """Test reset functionality."""
    
    def test_reset_basic(self):
        """Test basic reset."""
        org = TestOrganism(organism_id=0, position=(10, 10), initial_energy=50.0)
        
        # Modify state
        org.update_energy(-20)
        org.increment_age()
        org.move((15, 15))
        
        # Reset
        org.reset(position=(5, 5))
        
        assert org.position == (5, 5)
        assert org.energy == 100.0  # Back to max_energy
        assert org.alive == True
        assert org.age == 0
    
    def test_reset_custom_energy(self):
        """Test reset with custom energy."""
        org = TestOrganism(organism_id=0, position=(10, 10))
        
        org.reset(position=(8, 8), initial_energy=60.0)
        
        assert org.energy == 60.0
    
    def test_reset_clears_history(self):
        """Test reset clears action history."""
        org = TestOrganism(organism_id=0, position=(10, 10))
        
        org.log_action(Action.MOVE_NORTH)
        org.log_action(Action.EAT)
        
        org.reset(position=(5, 5))
        
        assert len(org.action_history) == 0
    
    def test_reset_clears_statistics(self):
        """Test reset clears statistics."""
        org = TestOrganism(organism_id=0, position=(10, 10))
        
        org.total_food_consumed = 10
        org.total_attacks_attempted = 5
        org.total_distance_moved = 100
        
        org.reset(position=(5, 5))
        
        assert org.total_food_consumed == 0
        assert org.total_attacks_attempted == 0
        assert org.total_distance_moved == 0


class TestDecide:
    """Test abstract decide method."""
    
    def test_decide_called(self):
        """Test decide method can be called."""
        org = TestOrganism(organism_id=0, position=(10, 10))
        
        state = {'self_energy': 50}
        action = org.decide(state)
        
        assert action == Action.WAIT  # TestOrganism always waits


class TestRepresentation:
    """Test string representation."""
    
    def test_repr_alive(self):
        """Test repr for alive organism."""
        org = TestOrganism(organism_id=3, position=(10, 15), initial_energy=75.0)
        
        repr_str = repr(org)
        
        assert 'TestOrganism' in repr_str
        assert 'id=3' in repr_str
        assert 'pos=(10, 15)' in repr_str
        assert 'energy=75' in repr_str
        assert 'alive' in repr_str
    
    def test_repr_dead(self):
        """Test repr for dead organism."""
        org = TestOrganism(organism_id=1, position=(5, 5))
        org.update_energy(-100)
        
        repr_str = repr(org)
        
        assert 'dead' in repr_str
