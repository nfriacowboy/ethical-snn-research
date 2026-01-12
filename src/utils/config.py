"""Configuration management for the ethical SNN research project.

This module provides configuration loading, validation, and schema definition
using Pydantic for type safety and YAML for human-readable configuration files.

Example:
    >>> config = load_config("experiments/phase1/config_phase1.yaml")
    >>> config['environment']['grid_size']
    20
"""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator


class EnvironmentConfig(BaseModel):
    """Environment parameters for the grid world simulation."""

    grid_size: int = Field(ge=10, le=100, description="Size of the square grid")
    num_food: int = Field(
        ge=1, le=50, description="Number of food items in environment"
    )
    food_respawn_rate: float = Field(
        ge=0.0, le=1.0, description="Probability of food respawn per timestep"
    )
    food_energy_value: int = Field(
        ge=1, le=100, description="Energy gained from eating food"
    )

    @field_validator("grid_size")
    @classmethod
    def validate_grid_size(cls, v: int) -> int:
        """Ensure grid size is reasonable for simulation."""
        if v < 10:
            raise ValueError("Grid size must be at least 10 for meaningful simulation")
        return v


class OrganismConfig(BaseModel):
    """Organism parameters for energy management."""

    initial_energy: int = Field(ge=10, le=200, description="Starting energy level")
    energy_decay_rate: float = Field(
        ge=0.0, le=10.0, description="Energy lost per timestep"
    )
    max_energy: int = Field(ge=50, le=200, description="Maximum energy capacity")
    attack_cost: int = Field(ge=1, le=50, description="Energy cost to attempt attack")
    attack_damage: int = Field(
        ge=1, le=100, description="Damage dealt by successful attack"
    )

    @field_validator("max_energy")
    @classmethod
    def validate_max_energy(cls, v: int, info) -> int:
        """Ensure max energy is greater than initial energy."""
        initial = info.data.get("initial_energy", 0)
        if v < initial:
            raise ValueError(f"max_energy ({v}) must be >= initial_energy ({initial})")
        return v


class SNNConfig(BaseModel):
    """Spiking Neural Network architecture parameters."""

    input_size: int = Field(ge=1, description="Number of input neurons")
    hidden_size: int = Field(ge=5, le=500, description="Number of hidden layer neurons")
    output_size: int = Field(ge=1, description="Number of output neurons")
    learning_rate: float = Field(
        ge=1e-5, le=1.0, description="Learning rate for training"
    )
    beta: float = Field(ge=0.1, le=0.99, description="Decay rate for LIF neurons")
    num_steps: int = Field(
        ge=1, le=100, description="Number of timesteps per forward pass"
    )

    @field_validator("beta")
    @classmethod
    def validate_beta(cls, v: float) -> float:
        """Ensure beta is in valid range for LIF neurons."""
        if not 0.1 <= v <= 0.99:
            raise ValueError(f"beta must be in [0.1, 0.99], got {v}")
        return v


class STDPConfig(BaseModel):
    """STDP (Spike-Timing-Dependent Plasticity) learning parameters."""

    a_plus: float = Field(ge=0.0, le=1.0, description="LTP learning rate")
    a_minus: float = Field(ge=0.0, le=1.0, description="LTD learning rate")
    tau_plus: float = Field(ge=1.0, le=100.0, description="LTP time constant")
    tau_minus: float = Field(ge=1.0, le=100.0, description="LTD time constant")


class SimulationConfig(BaseModel):
    """Simulation execution parameters."""

    max_timesteps: int = Field(
        ge=100, le=10000, description="Maximum simulation timesteps"
    )
    random_seed: int = Field(ge=0, description="Random seed for reproducibility")
    device: str = Field(
        pattern=r"^(cpu|cuda)$", description="Device for PyTorch (cpu or cuda)"
    )
    num_organisms: int = Field(
        ge=1, le=20, description="Number of organisms in simulation"
    )
    log_frequency: int = Field(ge=1, le=100, description="Log state every N timesteps")

    @field_validator("num_organisms")
    @classmethod
    def validate_num_organisms(cls, v: int, info) -> int:
        """Ensure number of organisms fits in grid."""
        if v < 1:
            raise ValueError("Must have at least 1 organism")
        return v


class ExperimentConfig(BaseModel):
    """Top-level configuration container."""

    experiment_name: str = Field(description="Name of the experiment")
    environment: EnvironmentConfig
    organism: OrganismConfig
    snn_survival: SNNConfig
    snn_ethical: Optional[SNNConfig] = None  # Only for dual-process organisms
    stdp: STDPConfig
    simulation: SimulationConfig


# Legacy Config class for backwards compatibility
class Config:
    """Configuration manager for simulations.

    Handles loading and accessing configuration parameters.
    """

    DEFAULT_CONFIG = {
        "architecture": "single",  # 'single' or 'dual'
        "num_organisms": 10,
        "grid_size": [50, 50],
        "max_timesteps": 1000,
        "food_spawn_rate": 0.02,
        "energy_decay": 1.0,
        "collision_penalty": 10.0,
        "ethical_weight": 0.5,
        "checkpoint_frequency": 100,
        "seed": None,  # If None, uses run_id
        # Network parameters
        "snn_input_size": 128,
        "snn_hidden_size": 256,
        "snn_output_size": 8,
        "ethical_input_size": 64,
        "ethical_hidden_size": 128,
        # Training parameters
        "stdp_learning_rate": 0.01,
        "supervised_learning_rate": 0.001,
        "batch_size": 32,
        "num_epochs": 50,
    }

    def __init__(self, config_path: str = None):
        """Initialize configuration.

        Args:
            config_path: Path to YAML config file (optional)
        """
        self.config = self.DEFAULT_CONFIG.copy()

        if config_path:
            self.load(config_path)

    def load(self, config_path: str):
        """Load configuration from YAML file.

        Args:
            config_path: Path to YAML file
        """
        with open(config_path, "r") as f:
            user_config = yaml.safe_load(f)

        # Update with user config
        self.config.update(user_config)

    def save(self, config_path: str):
        """Save configuration to YAML file.

        Args:
            config_path: Path to save YAML file
        """
        with open(config_path, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value.

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value
        """
        return self.config.get(key, default)

    def set(self, key: str, value: Any):
        """Set configuration value.

        Args:
            key: Configuration key
            value: Value to set
        """
        self.config[key] = value

    def to_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary.

        Returns:
            Configuration dictionary
        """
        return self.config.copy()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and validate configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Validated configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValidationError: If config doesn't match schema
        yaml.YAMLError: If YAML is malformed

    Example:
        >>> config = load_config("experiments/phase1/config_phase1.yaml")
        >>> config['environment']['grid_size']
        20
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Load YAML
    with open(path, "r") as f:
        try:
            raw_config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format in {config_path}: {e}") from e

    # Validate with Pydantic
    try:
        validated_config = ExperimentConfig(**raw_config)
    except ValidationError as e:
        raise ValueError(f"Configuration validation failed:\n{e}") from e

    # Convert back to dict for easier access
    return validated_config.model_dump()


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration dictionary against schema.

    Args:
        config: Configuration dictionary to validate

    Returns:
        True if valid, False otherwise

    Example:
        >>> config = load_config("config.yaml")
        >>> validate_config(config)
        True
    """
    try:
        ExperimentConfig(**config)
        return True
    except (ValidationError, TypeError, KeyError):
        return False


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """Save configuration dictionary to YAML file.

    Args:
        config: Configuration dictionary
        output_path: Path where to save YAML file

    Raises:
        ValueError: If config is invalid
    """
    # Validate before saving
    if not validate_config(config):
        raise ValueError("Cannot save invalid configuration")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def get_default_config() -> Dict[str, Any]:
    """Get default configuration for Phase 1 experiments.

    Returns:
        Default configuration dictionary
    """
    default = ExperimentConfig(
        experiment_name="phase1_baseline",
        environment=EnvironmentConfig(
            grid_size=20, num_food=5, food_respawn_rate=0.1, food_energy_value=20
        ),
        organism=OrganismConfig(
            initial_energy=100,
            energy_decay_rate=1.0,
            max_energy=150,
            attack_cost=10,
            attack_damage=30,
        ),
        snn_survival=SNNConfig(
            input_size=25,  # 5x5 local view (grid cells)
            hidden_size=50,
            output_size=5,  # 4 directions + eat/attack
            learning_rate=1e-3,
            beta=0.9,
            num_steps=10,
        ),
        stdp=STDPConfig(a_plus=0.01, a_minus=0.01, tau_plus=20.0, tau_minus=20.0),
        simulation=SimulationConfig(
            max_timesteps=1000,
            random_seed=42,
            device="cpu",
            num_organisms=2,
            log_frequency=1,
        ),
    )

    return default.model_dump()
