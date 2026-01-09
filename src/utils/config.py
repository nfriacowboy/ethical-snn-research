"""Global configuration management."""

import yaml
from typing import Dict, Any


class Config:
    """Configuration manager for simulations.
    
    Handles loading and accessing configuration parameters.
    """
    
    DEFAULT_CONFIG = {
        'architecture': 'single',  # 'single' or 'dual'
        'num_organisms': 10,
        'grid_size': [50, 50],
        'max_timesteps': 1000,
        'food_spawn_rate': 0.02,
        'energy_decay': 1.0,
        'collision_penalty': 10.0,
        'ethical_weight': 0.5,
        'checkpoint_frequency': 100,
        'seed': None,  # If None, uses run_id
        
        # Network parameters
        'snn_input_size': 128,
        'snn_hidden_size': 256,
        'snn_output_size': 8,
        'ethical_input_size': 64,
        'ethical_hidden_size': 128,
        
        # Training parameters
        'stdp_learning_rate': 0.01,
        'supervised_learning_rate': 0.001,
        'batch_size': 32,
        'num_epochs': 50
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
        with open(config_path, 'r') as f:
            user_config = yaml.safe_load(f)
        
        # Update with user config
        self.config.update(user_config)
    
    def save(self, config_path: str):
        """Save configuration to YAML file.
        
        Args:
            config_path: Path to save YAML file
        """
        with open(config_path, 'w') as f:
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
    """Load configuration from file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Configuration dictionary
    """
    config = Config(config_path)
    return config.to_dict()
