"""Unit tests for FoodManager.

Tests cover:
- Initialization and configuration
- Food spawning
- Food consumption
- Nearest food queries
- Update/respawning logic
- Statistics and reset
"""

import numpy as np
import pytest

from src.environment.food_manager import FoodManager
from src.utils.config import get_default_config


@pytest.fixture
def food_config():
    """Get configuration for food manager testing."""
    config = get_default_config()
    config["environment"]["grid_size"] = 20
    config["environment"]["num_food"] = 5
    config["environment"]["max_food"] = 10
    config["environment"]["food_respawn_rate"] = 0.1
    return config


@pytest.fixture
def food_manager(food_config):
    """Create food manager for testing."""
    return FoodManager(food_config, grid_size=20)


class TestInitialization:
    """Test food manager initialization."""

    def test_init_basic(self, food_config):
        """Test basic initialization."""
        manager = FoodManager(food_config, grid_size=20)

        assert manager.grid_size == 20
        assert manager.max_food == 10
        assert manager.respawn_rate == 0.1
        assert manager.initial_food == 5

    def test_init_spawns_initial_food(self, food_manager):
        """Test that initial food is spawned."""
        assert len(food_manager.active_food) == 5
        assert food_manager.total_spawned == 5
        assert food_manager.total_consumed == 0

    def test_init_positions_valid(self, food_manager):
        """Test that spawned positions are within grid."""
        for x, y in food_manager.active_food:
            assert 0 <= x < 20
            assert 0 <= y < 20

    def test_init_positions_unique(self, food_manager):
        """Test that initial positions are unique."""
        positions = set(food_manager.active_food)
        assert len(positions) == len(food_manager.active_food)


class TestSpawning:
    """Test food spawning functionality."""

    def test_spawn_food_basic(self, food_config):
        """Test basic food spawning."""
        config = food_config.copy()
        config["environment"]["num_food"] = 0  # Start with no food
        manager = FoodManager(config, grid_size=20)

        spawned = manager.spawn_food(3)

        assert len(spawned) == 3
        assert len(manager.active_food) == 3
        assert manager.total_spawned == 3

    def test_spawn_respects_max_food(self, food_manager):
        """Test that spawning respects max_food limit."""
        # Try to spawn more than max allows
        food_manager.spawn_food(20)

        assert len(food_manager.active_food) <= food_manager.max_food

    def test_spawn_returns_actual_spawned(self, food_manager):
        """Test that spawn returns actual positions spawned."""
        current_count = len(food_manager.active_food)

        spawned = food_manager.spawn_food(2)

        assert len(spawned) <= 2
        assert len(food_manager.active_food) == current_count + len(spawned)

    def test_spawn_no_duplicates(self, food_manager):
        """Test that spawning doesn't create duplicates."""
        food_manager.spawn_food(5)

        positions = set(food_manager.active_food)
        assert len(positions) == len(food_manager.active_food)


class TestConsumption:
    """Test food consumption."""

    def test_consume_existing_food(self, food_manager):
        """Test consuming food that exists."""
        food_pos = food_manager.active_food[0]
        initial_count = len(food_manager.active_food)

        result = food_manager.consume_food(food_pos)

        assert result == True
        assert len(food_manager.active_food) == initial_count - 1
        assert food_pos not in food_manager.active_food
        assert food_manager.total_consumed == 1

    def test_consume_nonexistent_food(self, food_manager):
        """Test consuming food at empty position."""
        result = food_manager.consume_food((100, 100))

        assert result == False
        assert food_manager.total_consumed == 0

    def test_consume_multiple_times(self, food_manager):
        """Test consuming same position twice."""
        food_pos = food_manager.active_food[0]

        result1 = food_manager.consume_food(food_pos)
        result2 = food_manager.consume_food(food_pos)

        assert result1 == True
        assert result2 == False
        assert food_manager.total_consumed == 1

    def test_consume_all_food(self, food_manager):
        """Test consuming all food."""
        positions = food_manager.active_food.copy()

        for pos in positions:
            food_manager.consume_food(pos)

        assert len(food_manager.active_food) == 0
        assert food_manager.total_consumed == len(positions)


class TestNearestFood:
    """Test nearest food queries."""

    def test_get_nearest_basic(self, food_config):
        """Test finding nearest food."""
        config = food_config.copy()
        config["environment"]["num_food"] = 0
        manager = FoodManager(config, grid_size=20)

        # Manually place food
        manager.active_food = [(5, 5), (15, 15)]

        nearest = manager.get_nearest_food((6, 6), toroidal=False)
        assert nearest == (5, 5)

    def test_get_nearest_no_food(self, food_config):
        """Test getting nearest when no food exists."""
        config = food_config.copy()
        config["environment"]["num_food"] = 0
        manager = FoodManager(config, grid_size=20)

        nearest = manager.get_nearest_food((10, 10))
        assert nearest is None

    def test_get_nearest_toroidal(self, food_config):
        """Test nearest food with toroidal distance."""
        config = food_config.copy()
        config["environment"]["num_food"] = 0
        manager = FoodManager(config, grid_size=20)

        # Place food at opposite corners
        manager.active_food = [(1, 1), (19, 19)]

        # From (0, 0), both are equidistant with wrapping (sqrt(2))
        # So either could be nearest
        nearest = manager.get_nearest_food((0, 0), toroidal=True)
        assert nearest in [(1, 1), (19, 19)]

    def test_get_nearest_euclidean(self, food_config):
        """Test nearest food with Euclidean distance."""
        config = food_config.copy()
        config["environment"]["num_food"] = 0
        manager = FoodManager(config, grid_size=20)

        manager.active_food = [(1, 1), (19, 19)]

        # Without wrapping, (1, 1) is closer
        nearest = manager.get_nearest_food((0, 0), toroidal=False)
        assert nearest == (1, 1)


class TestUpdate:
    """Test food update/respawn logic."""

    def test_update_increments_timestep(self, food_manager):
        """Test that update handles timestep parameter."""
        # Should not crash with timestep parameter
        spawned = food_manager.update(timestep=10)
        assert isinstance(spawned, list)

    def test_update_respawn_probability(self, food_manager):
        """Test that respawn follows probability."""
        np.random.seed(42)

        food_manager.respawn_rate = 1.0  # Always spawn
        food_manager.active_food = []  # Clear food

        spawned = food_manager.update(timestep=1)
        assert len(spawned) > 0

    def test_update_no_respawn_when_full(self, food_manager):
        """Test that no food spawns when at max."""
        food_manager.respawn_rate = 1.0
        food_manager.spawn_food(100)  # Fill to max

        initial_count = len(food_manager.active_food)
        spawned = food_manager.update(timestep=1)

        assert len(spawned) == 0
        assert len(food_manager.active_food) == initial_count

    def test_update_multiple_timesteps(self, food_manager):
        """Test update over multiple timesteps."""
        np.random.seed(42)
        food_manager.respawn_rate = 0.5
        food_manager.active_food = []

        total_spawned = 0
        for t in range(100):
            spawned = food_manager.update(timestep=t)
            total_spawned += len(spawned)

        # With 0.5 probability and max_food=10, should spawn until limit
        # Then stop, so expect around 10 spawned (hitting max_food)
        assert 5 < total_spawned <= 10


class TestStatistics:
    """Test statistics tracking."""

    def test_get_statistics(self, food_manager):
        """Test getting statistics."""
        stats = food_manager.get_statistics()

        assert "active_food_count" in stats
        assert "total_spawned" in stats
        assert "total_consumed" in stats
        assert "max_food" in stats
        assert "respawn_rate" in stats

    def test_statistics_track_consumption(self, food_manager):
        """Test that statistics track consumption."""
        food_pos = food_manager.active_food[0]
        food_manager.consume_food(food_pos)

        stats = food_manager.get_statistics()
        assert stats["total_consumed"] == 1

    def test_statistics_track_spawning(self, food_manager):
        """Test that statistics track spawning."""
        initial_spawned = food_manager.total_spawned

        food_manager.spawn_food(2)

        stats = food_manager.get_statistics()
        assert stats["total_spawned"] >= initial_spawned + 2


class TestReset:
    """Test reset functionality."""

    def test_reset_clears_state(self, food_manager):
        """Test that reset clears state."""
        # Modify state
        food_manager.consume_food(food_manager.active_food[0])
        food_manager.spawn_food(2)

        initial_food = food_manager.initial_food

        # Reset
        food_manager.reset()

        assert len(food_manager.active_food) == initial_food
        assert food_manager.total_spawned == initial_food
        assert food_manager.total_consumed == 0

    def test_reset_respawns_initial(self, food_manager):
        """Test that reset respawns initial food."""
        food_manager.active_food = []
        food_manager.reset()

        assert len(food_manager.active_food) == food_manager.initial_food


class TestDistanceCalculations:
    """Test distance calculation methods."""

    def test_toroidal_distance(self, food_manager):
        """Test toroidal distance calculation."""
        # Adjacent positions
        dist1 = food_manager._toroidal_distance((0, 0), (1, 1))
        assert abs(dist1 - np.sqrt(2)) < 0.01

        # Wrapping distance
        dist2 = food_manager._toroidal_distance((0, 0), (19, 19))
        # With wrapping: min(19, 1) in each dim = 1
        expected = np.sqrt(1**2 + 1**2)
        assert abs(dist2 - expected) < 0.01

    def test_euclidean_distance(self, food_manager):
        """Test Euclidean distance calculation."""
        dist = food_manager._euclidean_distance((0, 0), (3, 4))
        assert abs(dist - 5.0) < 0.01

    def test_distance_symmetry(self, food_manager):
        """Test that distance is symmetric."""
        pos1 = (5, 10)
        pos2 = (15, 3)

        dist1 = food_manager._toroidal_distance(pos1, pos2)
        dist2 = food_manager._toroidal_distance(pos2, pos1)

        assert abs(dist1 - dist2) < 0.01
