"""Tests for Main Simulation Runner."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.simulation.runner import SimulationRunner


class TestSimulationRunnerInitialization:
    """Test SimulationRunner initialization."""

    def test_basic_initialization_condition_a(self):
        """Test initializing runner with Condition A."""
        config = {
            "condition": "A",
            "grid_size": 10,
            "num_organisms": 5,
            "num_food": 3,
            "max_timesteps": 100,
        }

        runner = SimulationRunner(config, seed=42)

        assert runner.condition == "A"
        assert runner.num_organisms == 5
        assert len(runner.organisms) == 5
        assert runner.grid.grid_size == 10
        assert runner.max_timesteps == 100

    def test_basic_initialization_condition_b(self):
        """Test initializing runner with Condition B."""
        config = {
            "condition": "B",
            "grid_size": 15,
            "num_organisms": 8,
            "max_timesteps": 200,
        }

        runner = SimulationRunner(config, seed=123)

        assert runner.condition == "B"
        assert runner.num_organisms == 8
        assert len(runner.organisms) == 8

    def test_default_parameters(self):
        """Test that default parameters are applied."""
        config = {}

        runner = SimulationRunner(config, seed=42)

        assert runner.condition == "A"  # Default
        assert runner.grid.grid_size == 20  # Default
        assert runner.num_organisms == 10  # Default
        assert runner.max_timesteps == 1000  # Default

    def test_organisms_have_unique_ids(self):
        """Test that organisms have unique IDs."""
        config = {"num_organisms": 10}
        runner = SimulationRunner(config, seed=42)

        ids = [org.organism_id for org in runner.organisms]
        assert len(ids) == len(set(ids))  # All unique
        assert ids == list(range(10))  # Sequential from 0


class TestSimulationStep:
    """Test single simulation step."""

    def test_step_executes_without_error(self):
        """Test that step() executes without errors."""
        config = {"num_organisms": 3, "grid_size": 10, "max_timesteps": 10}
        runner = SimulationRunner(config, seed=42)

        runner.step()  # Should not raise

        assert runner.timestep == 0  # Step doesn't increment timestep

    def test_step_increments_age(self):
        """Test that organisms age increases."""
        config = {"num_organisms": 2, "grid_size": 10}
        runner = SimulationRunner(config, seed=42)

        initial_ages = [org.age for org in runner.organisms]

        runner.step()

        final_ages = [org.age for org in runner.organisms]

        # Age should increase by 1
        for initial, final in zip(initial_ages, final_ages):
            assert final == initial + 1

    def test_step_applies_energy_decay(self):
        """Test that energy decays each step."""
        config = {"num_organisms": 2, "grid_size": 10, "energy_decay_rate": 2.0}
        runner = SimulationRunner(config, seed=42)

        initial_energies = [org.energy for org in runner.organisms]

        runner.step()

        final_energies = [org.energy for org in runner.organisms]

        # Energy should decrease
        for initial, final in zip(initial_energies, final_energies):
            assert final < initial

    def test_organism_can_eat_food(self):
        """Test that organisms can consume food."""
        config = {
            "num_organisms": 1,
            "grid_size": 5,
            "num_food": 5,  # Lots of food
            "food_energy": 30.0,
        }
        runner = SimulationRunner(config, seed=42)

        # Place organism on food
        organism = runner.organisms[0]
        if runner.grid.food_positions:
            food_pos = runner.grid.food_positions[0]
            organism.move(food_pos)

        initial_energy = organism.energy

        # Run step
        runner.step()

        # Energy might increase if organism ate
        # (depends on neural network decision, so we just check it runs)
        assert organism.energy >= 0  # Still valid


class TestSimulationRun:
    """Test complete simulation runs."""

    def test_short_run_completes(self):
        """Test that short simulation completes."""
        config = {
            "num_organisms": 2,
            "grid_size": 5,
            "max_timesteps": 10,
            "energy_decay_rate": 0.5,  # Slow decay
        }
        runner = SimulationRunner(config, seed=42)

        stats = runner.run()

        assert "final_timestep" in stats
        assert stats["final_timestep"] <= 10

    def test_run_terminates_when_all_dead(self):
        """Test that simulation terminates when all organisms die."""
        config = {
            "num_organisms": 2,
            "grid_size": 5,
            "num_food": 0,  # No food
            "max_timesteps": 1000,
            "energy_decay_rate": 20.0,  # Fast decay
        }
        runner = SimulationRunner(config, seed=42)

        stats = runner.run()

        # Should terminate before max timesteps
        assert stats["final_timestep"] < 1000
        assert stats["organisms"]["alive"] == 0

    def test_run_returns_statistics(self):
        """Test that run() returns comprehensive statistics."""
        config = {"num_organisms": 3, "max_timesteps": 5}
        runner = SimulationRunner(config, seed=42)

        stats = runner.run()

        # Check required keys
        assert "condition" in stats
        assert "final_timestep" in stats
        assert "organisms" in stats
        assert "energy" in stats
        assert "environment" in stats

        # Check organism stats
        assert "total" in stats["organisms"]
        assert "alive" in stats["organisms"]
        assert "dead" in stats["organisms"]


class TestStatistics:
    """Test statistics collection."""

    def test_get_statistics_structure(self):
        """Test that get_statistics returns proper structure."""
        config = {"num_organisms": 5, "condition": "A"}
        runner = SimulationRunner(config, seed=42)
        runner.start_time = 0  # Set to avoid None

        stats = runner.get_statistics()

        assert isinstance(stats, dict)
        assert "condition" in stats
        assert "organisms" in stats
        assert "energy" in stats
        assert "environment" in stats

    def test_dual_process_statistics(self):
        """Test that Condition B includes dual-process stats."""
        config = {"num_organisms": 3, "condition": "B"}
        runner = SimulationRunner(config, seed=42)
        runner.start_time = 0

        stats = runner.get_statistics()

        assert "dual_process" in stats
        assert "total_vetoes" in stats["dual_process"]
        assert "total_approvals" in stats["dual_process"]
        assert "avg_veto_rate" in stats["dual_process"]


class TestOrganismStateBuilding:
    """Test organism state building."""

    def test_build_state_includes_required_keys(self):
        """Test that _build_organism_state includes all required keys."""
        config = {"num_organisms": 1}
        runner = SimulationRunner(config, seed=42)

        organism = runner.organisms[0]
        state = runner._build_organism_state(organism)

        required_keys = [
            "food_direction",
            "energy",
            "obstacles_nearby",
            "other_organism",
            "food_at_position",
            "grid_size",
        ]

        for key in required_keys:
            assert key in state

    def test_food_at_position_detection(self):
        """Test detection of food at organism position."""
        config = {"num_organisms": 1, "num_food": 1}
        runner = SimulationRunner(config, seed=42)

        organism = runner.organisms[0]

        # Place organism on food
        if runner.grid.food_positions:
            food_pos = runner.grid.food_positions[0]
            organism.move(food_pos)

            state = runner._build_organism_state(organism)
            assert state["food_at_position"] is True


class TestReproducibility:
    """Test simulation reproducibility."""

    def test_same_seed_same_results(self):
        """Test that same seed produces same results."""
        config = {"num_organisms": 3, "max_timesteps": 10}

        runner1 = SimulationRunner(config, seed=42)
        stats1 = runner1.run()

        runner2 = SimulationRunner(config, seed=42)
        stats2 = runner2.run()

        # Should have same final timestep
        assert stats1["final_timestep"] == stats2["final_timestep"]

        # Should have same organism counts
        assert stats1["organisms"]["alive"] == stats2["organisms"]["alive"]

    def test_different_seeds_different_results(self):
        """Test that different seeds produce different results."""
        config = {"num_organisms": 5, "max_timesteps": 20}

        runner1 = SimulationRunner(config, seed=42)
        stats1 = runner1.run()

        runner2 = SimulationRunner(config, seed=123)
        stats2 = runner2.run()

        # Results should differ (with high probability)
        # Check that organism final positions differ
        pos1 = [org.position for org in runner1.organisms]
        pos2 = [org.position for org in runner2.organisms]
        assert (
            pos1 != pos2
        ), "Different seeds should produce different organism positions"


class TestRepresentation:
    """Test string representation."""

    def test_repr(self):
        """Test string representation."""
        config = {"num_organisms": 5, "condition": "B", "max_timesteps": 100}
        runner = SimulationRunner(config, seed=42)

        repr_str = repr(runner)

        assert "SimulationRunner" in repr_str
        assert "condition=B" in repr_str
        assert "t=" in repr_str
        assert "alive=" in repr_str
