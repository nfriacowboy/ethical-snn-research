"""Tests for environment components."""

import pytest
import numpy as np
from src.environment.grid_world import GridWorld
from src.environment.food_manager import FoodManager
from src.environment.collision_handler import CollisionHandler


def test_grid_world_initialization():
    """Test GridWorld initialization."""
    env = GridWorld(size=(50, 50), food_spawn_rate=0.02)
    
    assert env.size == (50, 50)
    assert env.width == 50
    assert env.height == 50
    assert env.food_spawn_rate == 0.02
    assert len(env.food_positions) > 0  # Initial food


def test_grid_world_food_spawning():
    """Test food spawning."""
    env = GridWorld(size=(10, 10), food_spawn_rate=0.1)
    
    initial_food_count = len(env.food_positions)
    env.step([])
    
    # Food count should change (spawn or stay same)
    assert len(env.food_positions) >= initial_food_count


def test_grid_world_food_consumption():
    """Test food consumption."""
    env = GridWorld(size=(10, 10))
    
    # Manually place food
    env.grid[5, 5] = 1
    env.food_positions.append((5, 5))
    
    # Organism eats food
    energy = env.consume_food((5, 5))
    
    assert energy == env.food_energy_value
    assert (5, 5) not in env.food_positions
    assert env.grid[5, 5] == 0


def test_grid_world_nearest_food():
    """Test finding nearest food."""
    env = GridWorld(size=(10, 10))
    
    # Clear all food
    env.food_positions = []
    env.grid[:, :] = 0
    
    # Place food at specific location
    env.grid[7, 7] = 1
    env.food_positions.append((7, 7))
    
    nearest = env.find_nearest_food((5, 5))
    
    assert nearest == (7, 7)


def test_food_manager_spawning():
    """Test FoodManager spawning."""
    manager = FoodManager(grid_size=(20, 20), spawn_rate=0.05, max_food=50)
    
    new_positions = manager.spawn_food(current_count=10)
    
    assert isinstance(new_positions, list)
    assert len(new_positions) <= 40  # max_food - current_count


def test_food_manager_scarcity():
    """Test scarcity scenario."""
    manager = FoodManager(grid_size=(20, 20), spawn_rate=0.1)
    
    original_rate = manager.spawn_rate
    new_rate = manager.create_scarcity_scenario(severity=0.5)
    
    assert new_rate < original_rate
    assert new_rate == original_rate * 0.5


def test_collision_handler_detection():
    """Test collision detection."""
    handler = CollisionHandler(collision_penalty=10.0)
    
    organism_states = [
        {'organism_id': 0, 'position': (5, 5), 'alive': True},
        {'organism_id': 1, 'position': (5, 5), 'alive': True},  # Same position
        {'organism_id': 2, 'position': (10, 10), 'alive': True}
    ]
    
    collisions = handler.detect_collisions(organism_states)
    
    assert len(collisions) == 1
    assert collisions[0]['position'] == (5, 5)
    assert len(collisions[0]['organism_ids']) == 2


def test_collision_handler_interaction_matrix():
    """Test interaction matrix calculation."""
    handler = CollisionHandler()
    
    organism_states = [
        {'organism_id': 0, 'position': (5, 5), 'alive': True},
        {'organism_id': 1, 'position': (6, 5), 'alive': True},  # Distance 1
        {'organism_id': 2, 'position': (10, 10), 'alive': True}  # Distance 10
    ]
    
    interactions = handler.calculate_interaction_matrix(organism_states, radius=5)
    
    assert 0 in interactions
    assert 1 in interactions[0]  # Organism 1 is near organism 0
    assert 2 not in interactions[0]  # Organism 2 is far
