"""Unit tests for GridWorld environment.

Tests cover:
- Grid initialization
- Toroidal topology (wrapping)
- Local state extraction
- Organism placement and movement
- Food spawning and consumption
- Collision detection
"""

import numpy as np
import pytest

from src.environment.grid_world import GridWorld
from src.utils.config import get_default_config


@pytest.fixture
def grid_config():
    """Get default configuration for testing."""
    config = get_default_config()
    config["environment"]["grid_size"] = 20
    config["environment"]["num_food"] = 5
    return config


@pytest.fixture
def empty_grid(grid_config):
    """Create grid with no initial food for testing."""
    config = grid_config.copy()
    config["environment"]["num_food"] = 0
    return GridWorld(config)


class TestGridInitialization:
    """Test grid world initialization."""

    def test_grid_init(self, grid_config):
        """Test basic grid initialization."""
        grid = GridWorld(grid_config)

        assert grid.grid_size == 20
        assert grid.grid.shape == (20, 20)
        assert len(grid.food_positions) == 5
        assert grid.timestep == 0

    def test_grid_state_shape(self, grid_config):
        """Test grid state has correct shape."""
        grid = GridWorld(grid_config)
        state = grid.get_state()

        assert state.shape == (20, 20)
        assert state.dtype == np.int8

    def test_initial_food_placement(self, grid_config):
        """Test that initial food is placed on grid."""
        grid = GridWorld(grid_config)

        food_count = np.sum(grid.grid == 1)
        assert food_count == 5
        assert len(grid.food_positions) == 5


class TestToroidalTopology:
    """Test wrapping behavior for toroidal topology."""

    def test_wrap_position_positive_overflow(self, grid_config):
        """Test wrapping when position exceeds grid size."""
        grid = GridWorld(grid_config)

        wrapped = grid.wrap_position((21, 5))
        assert wrapped == (1, 5)

        wrapped = grid.wrap_position((20, 20))
        assert wrapped == (0, 0)

        wrapped = grid.wrap_position((25, 25))
        assert wrapped == (5, 5)

    def test_wrap_position_negative(self, grid_config):
        """Test wrapping with negative coordinates."""
        grid = GridWorld(grid_config)

        wrapped = grid.wrap_position((-1, 10))
        assert wrapped == (19, 10)

        wrapped = grid.wrap_position((10, -1))
        assert wrapped == (10, 19)

        wrapped = grid.wrap_position((-1, -1))
        assert wrapped == (19, 19)

    def test_wrap_position_no_change(self, grid_config):
        """Test that valid positions don't change."""
        grid = GridWorld(grid_config)

        wrapped = grid.wrap_position((10, 10))
        assert wrapped == (10, 10)

        wrapped = grid.wrap_position((0, 0))
        assert wrapped == (0, 0)

        wrapped = grid.wrap_position((19, 19))
        assert wrapped == (19, 19)

    def test_is_valid_position(self, grid_config):
        """Test position validity checking."""
        grid = GridWorld(grid_config)

        assert grid.is_valid_position((10, 10)) is True
        assert grid.is_valid_position((0, 0)) is True
        assert grid.is_valid_position((19, 19)) is True

        assert grid.is_valid_position((20, 10)) is False
        assert grid.is_valid_position((10, 20)) is False
        assert grid.is_valid_position((-1, 10)) is False


class TestLocalState:
    """Test local state extraction."""

    def test_get_local_state_center(self, empty_grid):
        """Test local state extraction from center of grid."""
        local = empty_grid.get_local_state((10, 10), radius=2)

        assert local.shape == (5, 5)
        assert local.dtype == np.int8

    def test_get_local_state_edge(self, empty_grid):
        """Test local state extraction at grid edge (wrapping)."""
        # Place marker at corner
        empty_grid.grid[0, 0] = 1

        # Sample from opposite corner (should wrap to include 0,0)
        local = empty_grid.get_local_state((19, 19), radius=1)

        assert local.shape == (3, 3)
        # The (0,0) position should appear in wrapped view
        assert np.any(local == 1)

    def test_get_local_state_radius_0(self, empty_grid):
        """Test local state with radius 0 (single cell)."""
        empty_grid.grid[10, 10] = 1

        local = empty_grid.get_local_state((10, 10), radius=0)

        assert local.shape == (1, 1)
        assert local[0, 0] == 1

    def test_get_local_state_radius_3(self, empty_grid):
        """Test local state with larger radius."""
        local = empty_grid.get_local_state((10, 10), radius=3)

        assert local.shape == (7, 7)


class TestOrganismManagement:
    """Test organism placement and movement."""

    def test_place_organism(self, empty_grid):
        """Test placing organism on grid."""
        success = empty_grid.place_organism(0, (10, 10))

        assert success is True
        assert 0 in empty_grid.organism_positions
        assert empty_grid.organism_positions[0] == (10, 10)
        assert empty_grid.grid[10, 10] == 2

    def test_place_organism_occupied(self, empty_grid):
        """Test that placement fails on occupied cell."""
        empty_grid.place_organism(0, (10, 10))
        success = empty_grid.place_organism(1, (10, 10))

        assert success is False
        assert 1 not in empty_grid.organism_positions

    def test_move_organism(self, empty_grid):
        """Test moving organism to new position."""
        empty_grid.place_organism(0, (10, 10))
        success = empty_grid.move_organism(0, (11, 11))

        assert success is True
        assert empty_grid.organism_positions[0] == (11, 11)
        assert empty_grid.grid[10, 10] == 0  # Old position cleared
        assert empty_grid.grid[11, 11] == 2  # New position set

    def test_move_organism_wrapping(self, empty_grid):
        """Test organism movement with position wrapping."""
        empty_grid.place_organism(0, (19, 19))
        success = empty_grid.move_organism(0, (20, 20))  # Should wrap to (0,0)

        assert success is True
        assert empty_grid.organism_positions[0] == (0, 0)

    def test_move_organism_blocked(self, empty_grid):
        """Test that movement fails when blocked by another organism."""
        empty_grid.place_organism(0, (10, 10))
        empty_grid.place_organism(1, (11, 11))

        success = empty_grid.move_organism(0, (11, 11))

        assert success is False
        assert empty_grid.organism_positions[0] == (10, 10)

    def test_remove_organism(self, empty_grid):
        """Test removing organism from grid."""
        empty_grid.place_organism(0, (10, 10))
        empty_grid.remove_organism(0)

        assert 0 not in empty_grid.organism_positions
        assert empty_grid.grid[10, 10] == 0


class TestFoodManagement:
    """Test food spawning and consumption."""

    def test_has_food_at(self, empty_grid):
        """Test checking for food at position."""
        empty_grid.grid[10, 10] = 1
        empty_grid.food_positions.append((10, 10))

        assert empty_grid.has_food_at((10, 10)) == True
        assert empty_grid.has_food_at((11, 11)) == False

    def test_consume_food(self, empty_grid):
        """Test consuming food from grid."""
        empty_grid.grid[10, 10] = 1
        empty_grid.food_positions.append((10, 10))

        success = empty_grid.consume_food((10, 10))

        assert success is True
        assert empty_grid.grid[10, 10] == 0
        assert (10, 10) not in empty_grid.food_positions
        assert empty_grid.total_food_consumed == 1

    def test_consume_food_not_present(self, empty_grid):
        """Test consuming food when none present."""
        success = empty_grid.consume_food((10, 10))

        assert success is False
        assert empty_grid.total_food_consumed == 0

    def test_spawn_food(self, grid_config):
        """Test food spawning."""
        grid = GridWorld(grid_config)
        initial_food = len(grid.food_positions)

        # Force spawn by setting high respawn rate
        grid.food_respawn_rate = 1.0
        grid.spawn_food()

        # Should have spawned at least initial food (might spawn more)
        assert len(grid.food_positions) >= initial_food


class TestCollisionDetection:
    """Test collision detection."""

    def test_check_collision_present(self, empty_grid):
        """Test detecting organism at position."""
        empty_grid.place_organism(0, (10, 10))

        org_id = empty_grid.check_collision((10, 10))
        assert org_id == 0

    def test_check_collision_none(self, empty_grid):
        """Test no collision when cell empty."""
        org_id = empty_grid.check_collision((10, 10))
        assert org_id is None

    def test_check_collision_food(self, empty_grid):
        """Test that food doesn't register as collision."""
        empty_grid.grid[10, 10] = 1

        org_id = empty_grid.check_collision((10, 10))
        assert org_id is None


class TestEnvironmentStep:
    """Test environment timestep advancement."""

    def test_step_increments_timestep(self, grid_config):
        """Test that step increments timestep."""
        grid = GridWorld(grid_config)
        initial_timestep = grid.timestep

        grid.step()

        assert grid.timestep == initial_timestep + 1

    def test_step_spawns_food(self, empty_grid):
        """Test that step can spawn food."""
        empty_grid.food_respawn_rate = 1.0  # Guarantee spawn
        empty_grid.max_food = 10  # Allow food to spawn

        empty_grid.step()

        # Should have spawned at least one food
        assert len(empty_grid.food_positions) > 0


class TestStatistics:
    """Test statistics gathering."""

    def test_get_statistics(self, grid_config):
        """Test getting environment statistics."""
        grid = GridWorld(grid_config)
        stats = grid.get_statistics()

        assert "timestep" in stats
        assert "food_count" in stats
        assert "organism_count" in stats
        assert "total_food_spawned" in stats
        assert "total_food_consumed" in stats

        assert stats["food_count"] == 5
        assert stats["organism_count"] == 0


class TestReset:
    """Test environment reset."""

    def test_reset(self, grid_config):
        """Test resetting environment to initial state."""
        grid = GridWorld(grid_config)

        # Make changes
        grid.place_organism(0, (10, 10))
        grid.consume_food(grid.food_positions[0])
        grid.step()
        grid.step()

        # Reset
        grid.reset()

        # Check reset state
        assert grid.timestep == 0
        assert len(grid.organism_positions) == 0
        assert len(grid.food_positions) == 5  # Back to initial
        assert grid.total_food_consumed == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
