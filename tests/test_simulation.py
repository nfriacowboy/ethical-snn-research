"""Tests for simulation components."""

import os
import tempfile

import pytest

from src.simulation.checkpointer import Checkpointer
from src.simulation.logger import SimulationLogger
from src.simulation.runner import SimulationRunner
from src.utils.config import Config


def test_simulation_runner_initialization():
    """Test SimulationRunner initialization."""
    config = Config().to_dict()
    config["max_timesteps"] = 10  # Short simulation for testing

    runner = SimulationRunner(config=config, run_id=0)

    assert runner.run_id == 0
    assert runner.timestep == 0
    assert runner.max_timesteps == 10
    assert hasattr(runner, "environment")
    assert hasattr(runner, "architecture")


def test_simulation_logger():
    """Test SimulationLogger."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = SimulationLogger(run_id=0, log_dir=tmpdir)

        # Log some timesteps
        for t in range(5):
            logger.log_timestep(
                timestep=t,
                organism_states=[{"organism_id": 0, "alive": True}],
                environment_state={"food_count": 10},
                collisions=[],
            )

        logger.flush()

        # Check log file exists
        assert os.path.exists(logger.timestep_log_path)

        # Check content
        with open(logger.timestep_log_path, "r") as f:
            lines = f.readlines()
            assert len(lines) == 5


def test_checkpointer_save_load():
    """Test Checkpointer save and load."""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpointer = Checkpointer(run_id=0, checkpoint_dir=tmpdir)

        # Save checkpoint
        checkpoint_data = {"timestep": 100, "test_value": 42}
        checkpointer.save(checkpoint_data, timestep=100)

        # Load checkpoint
        loaded_data = checkpointer.load(timestep=100)

        assert loaded_data["timestep"] == 100
        assert loaded_data["test_value"] == 42


def test_checkpointer_list():
    """Test listing checkpoints."""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpointer = Checkpointer(run_id=0, checkpoint_dir=tmpdir)

        # Save multiple checkpoints
        for t in [100, 200, 300]:
            checkpointer.save({"timestep": t}, timestep=t)

        # List checkpoints
        checkpoints = checkpointer.list_checkpoints()

        assert len(checkpoints) == 3
        assert 100 in checkpoints
        assert 200 in checkpoints
        assert 300 in checkpoints
