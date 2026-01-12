"""Unit tests for configuration management system.

Tests cover:
- Loading valid configurations from YAML
- Validation of configuration parameters
- Error handling for invalid configs
- Schema validation with Pydantic
"""

import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest
from pydantic import ValidationError

from src.utils.config import (EnvironmentConfig, ExperimentConfig,
                              OrganismConfig, SimulationConfig, SNNConfig,
                              STDPConfig, get_default_config, load_config,
                              save_config, validate_config)


class TestConfigLoading:
    """Test configuration loading from YAML files."""

    def test_load_valid_config(self):
        """Test loading a valid configuration file."""
        config = load_config("experiments/phase1/config_phase1.yaml")

        assert config is not None
        assert "environment" in config
        assert "organism" in config
        assert "snn_survival" in config
        assert "simulation" in config

    def test_load_missing_file(self):
        """Test that loading missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent/config.yaml")

    def test_load_invalid_yaml(self):
        """Test that malformed YAML raises ValueError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [\n")
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Invalid YAML format"):
                load_config(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_config_structure(self):
        """Test that loaded config has correct structure."""
        config = load_config("experiments/phase1/config_phase1.yaml")

        # Check environment params
        assert config["environment"]["grid_size"] == 20
        assert config["environment"]["num_food"] == 5

        # Check organism params
        assert config["organism"]["initial_energy"] == 100
        assert config["organism"]["max_energy"] == 150

        # Check simulation params
        assert config["simulation"]["max_timesteps"] == 1000
        assert config["simulation"]["random_seed"] == 42


class TestConfigValidation:
    """Test configuration validation logic."""

    def test_validate_valid_config(self):
        """Test that valid config passes validation."""
        config = get_default_config()
        assert validate_config(config) is True

    def test_validate_invalid_grid_size(self):
        """Test that grid_size < 10 fails validation."""
        config = get_default_config()
        config["environment"]["grid_size"] = 5

        assert validate_config(config) is False

    def test_validate_invalid_energy(self):
        """Test that max_energy < initial_energy fails."""
        config = get_default_config()
        config["organism"]["max_energy"] = 50
        config["organism"]["initial_energy"] = 100

        assert validate_config(config) is False

    def test_validate_missing_required_field(self):
        """Test that missing required fields fail validation."""
        config = get_default_config()
        del config["experiment_name"]

        assert validate_config(config) is False

    def test_validate_invalid_beta_range(self):
        """Test that beta outside [0.1, 0.99] fails."""
        config = get_default_config()
        config["snn_survival"]["beta"] = 1.5

        assert validate_config(config) is False


class TestEnvironmentConfig:
    """Test EnvironmentConfig Pydantic model."""

    def test_valid_environment_config(self):
        """Test creating valid environment config."""
        env = EnvironmentConfig(
            grid_size=20, num_food=5, food_respawn_rate=0.1, food_energy_value=20
        )

        assert env.grid_size == 20
        assert env.num_food == 5

    def test_grid_size_too_small(self):
        """Test that grid_size < 10 raises ValueError."""
        with pytest.raises(ValidationError):
            EnvironmentConfig(
                grid_size=5, num_food=5, food_respawn_rate=0.1, food_energy_value=20
            )

    def test_negative_food(self):
        """Test that negative num_food raises ValidationError."""
        with pytest.raises(ValidationError):
            EnvironmentConfig(
                grid_size=20, num_food=-1, food_respawn_rate=0.1, food_energy_value=20
            )


class TestOrganismConfig:
    """Test OrganismConfig Pydantic model."""

    def test_valid_organism_config(self):
        """Test creating valid organism config."""
        org = OrganismConfig(
            initial_energy=100,
            energy_decay_rate=1.0,
            max_energy=150,
            attack_cost=10,
            attack_damage=30,
        )

        assert org.initial_energy == 100
        assert org.max_energy == 150

    def test_max_energy_less_than_initial(self):
        """Test that max_energy < initial_energy fails."""
        with pytest.raises(ValidationError):
            OrganismConfig(
                initial_energy=100,
                energy_decay_rate=1.0,
                max_energy=50,
                attack_cost=10,
                attack_damage=30,
            )


class TestSNNConfig:
    """Test SNNConfig Pydantic model."""

    def test_valid_snn_config(self):
        """Test creating valid SNN config."""
        snn = SNNConfig(
            input_size=25,
            hidden_size=50,
            output_size=5,
            learning_rate=0.001,
            beta=0.9,
            num_steps=10,
        )

        assert snn.input_size == 25
        assert snn.beta == 0.9

    def test_beta_out_of_range(self):
        """Test that beta outside [0.1, 0.99] fails."""
        with pytest.raises(ValidationError):
            SNNConfig(
                input_size=25,
                hidden_size=50,
                output_size=5,
                learning_rate=0.001,
                beta=1.5,
                num_steps=10,
            )

    def test_negative_learning_rate(self):
        """Test that negative learning rate fails."""
        with pytest.raises(ValidationError):
            SNNConfig(
                input_size=25,
                hidden_size=50,
                output_size=5,
                learning_rate=-0.001,
                beta=0.9,
                num_steps=10,
            )


class TestSimulationConfig:
    """Test SimulationConfig Pydantic model."""

    def test_valid_simulation_config(self):
        """Test creating valid simulation config."""
        sim = SimulationConfig(
            max_timesteps=1000,
            random_seed=42,
            device="cpu",
            num_organisms=2,
            log_frequency=1,
        )

        assert sim.max_timesteps == 1000
        assert sim.device == "cpu"

    def test_invalid_device(self):
        """Test that invalid device string fails."""
        with pytest.raises(ValidationError):
            SimulationConfig(
                max_timesteps=1000,
                random_seed=42,
                device="gpu",  # Should be "cpu" or "cuda"
                num_organisms=2,
                log_frequency=1,
            )


class TestConfigSaveLoad:
    """Test saving and loading configurations."""

    def test_save_and_load_config(self):
        """Test that saved config can be loaded back."""
        config = get_default_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test_config.yaml"
            save_config(config, str(config_path))

            loaded_config = load_config(str(config_path))

            assert loaded_config["experiment_name"] == config["experiment_name"]
            assert (
                loaded_config["environment"]["grid_size"]
                == config["environment"]["grid_size"]
            )

    def test_save_invalid_config(self):
        """Test that saving invalid config raises ValueError."""
        invalid_config = {"invalid": "structure"}

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "invalid.yaml"
            with pytest.raises(ValueError, match="Cannot save invalid configuration"):
                save_config(invalid_config, str(config_path))


class TestDefaultConfig:
    """Test default configuration generation."""

    def test_get_default_config(self):
        """Test that default config is valid and complete."""
        config = get_default_config()

        assert validate_config(config) is True
        assert config["experiment_name"] == "phase1_baseline"
        assert config["environment"]["grid_size"] == 20
        assert config["simulation"]["random_seed"] == 42

    def test_default_config_structure(self):
        """Test that default config has all required fields."""
        config = get_default_config()

        required_keys = [
            "experiment_name",
            "environment",
            "organism",
            "snn_survival",
            "stdp",
            "simulation",
        ]

        for key in required_keys:
            assert key in config, f"Missing required key: {key}"


class TestExperimentConfig:
    """Test ExperimentConfig top-level model."""

    def test_dual_process_with_ethical_snn(self):
        """Test config with optional ethical SNN."""
        config_dict = get_default_config()
        config_dict["snn_ethical"] = config_dict["snn_survival"].copy()
        config_dict["snn_ethical"]["input_size"] = 30

        config = ExperimentConfig(**config_dict)

        assert config.snn_ethical is not None
        assert config.snn_ethical.input_size == 30

    def test_single_process_without_ethical_snn(self):
        """Test config without ethical SNN (single-process)."""
        config_dict = get_default_config()

        config = ExperimentConfig(**config_dict)

        assert config.snn_ethical is None
        assert config.snn_survival is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
