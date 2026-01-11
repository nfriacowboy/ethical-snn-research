"""Simulation logging system with HDF5 storage and metadata tracking.

This module provides comprehensive logging for simulation runs, including:
- Timestep-by-timestep organism states (positions, energies, actions)
- Environment state (food positions)
- Event tracking (attacks, food consumption, deaths)
- Metadata (git hash, config, timestamps)

HDF5 Structure:
    simulation_001.hdf5
    ├── metadata (attrs: seed, config_hash, git_hash, timestamp)
    ├── organisms/
    │   ├── organism_0/
    │   │   ├── positions [T, 2]
    │   │   ├── energies [T]
    │   │   ├── actions [T]
    │   │   └── alive [T]
    │   └── organism_1/ ...
    ├── environment/
    │   └── food_positions [T, N, 2]
    └── events/
        ├── food_consumed [E, 3]  # (timestep, organism_id, position)
        ├── attacks [E, 3]         # (timestep, attacker, target)
        └── deaths [D, 2]          # (timestep, organism_id)

Example:
    >>> logger = SimulationLogger(config, run_id=1, output_dir="results")
    >>> logger.log_action(t=0, organism_id=0, action="MOVE_NORTH", success=True)
    >>> logger.save_to_hdf5("results/simulation_001.hdf5")
"""

import json
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from collections import defaultdict
import hashlib

import h5py
import numpy as np
import yaml


def get_git_hash() -> str:
    """Get current git commit hash for auditability.
    
    Returns:
        Git commit hash (short), or 'unknown' if not in git repo
    """
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True,
            text=True,
            check=True,
            timeout=5
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return "unknown"


def hash_config(config: Dict[str, Any]) -> str:
    """Generate hash of configuration for reproducibility tracking.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        SHA256 hash of config (first 16 chars)
    """
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


class SimulationLogger:
    """Logs simulation data with HDF5 storage and comprehensive metadata tracking.
    
    Tracks all organism states, environment changes, and events for full
    reproducibility and analysis.
    
    Attributes:
        config: Simulation configuration
        run_id: Unique run identifier
        output_dir: Directory for output files
        metadata: Dict containing git hash, timestamps, etc.
    """
    
    def __init__(self, config: Dict[str, Any], run_id: int = 0, output_dir: str = "results"):
        """Initialize simulation logger.
        
        Args:
            config: Validated configuration dictionary
            run_id: Unique identifier for this run
            output_dir: Directory where to save log files
        """
        self.config = config
        self.run_id = run_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Metadata
        self.metadata = {
            'run_id': run_id,
            'git_hash': get_git_hash(),
            'config_hash': hash_config(config),
            'timestamp_start': datetime.now().isoformat(),
            'timestamp_end': None,
            'random_seed': config['simulation']['random_seed'],
            'experiment_name': config['experiment_name']
        }
        
        # Data buffers (timestep-indexed)
        self.num_organisms = config['simulation']['num_organisms']
        self.organism_data = defaultdict(lambda: {
            'positions': [],
            'energies': [],
            'actions': [],
            'alive': []
        })
        
        self.environment_data = {
            'food_positions': []
        }
        
        # Event buffers (timestep, data tuples)
        self.events = {
            'food_consumed': [],  # (timestep, organism_id, position)
            'attacks': [],         # (timestep, attacker_id, target_id)
            'deaths': []           # (timestep, organism_id)
        }
        
        self.current_timestep = 0
    
    def log_timestep(self, timestep: int, state: Dict[str, Any]) -> None:
        """Log complete state for one timestep.
        
        Args:
            timestep: Current simulation timestep
            state: Dictionary containing all state information:
                - organism_states: List of organism state dicts
                - environment_state: Environment state dict
        """
        self.current_timestep = timestep
        
        # Log organism states
        for org_id, org_state in enumerate(state.get('organism_states', [])):
            self.organism_data[org_id]['positions'].append(org_state.get('position', (-1, -1)))
            self.organism_data[org_id]['energies'].append(org_state.get('energy', 0))
            self.organism_data[org_id]['actions'].append(org_state.get('action', 'NONE'))
            self.organism_data[org_id]['alive'].append(org_state.get('alive', False))
        
        # Log environment state
        food_pos = state.get('environment_state', {}).get('food_positions', [])
        self.environment_data['food_positions'].append(food_pos)
    
    def log_action(self, timestep: int, organism_id: int, action: str, success: bool = True) -> None:
        """Log an action taken by an organism.
        
        Args:
            timestep: Current timestep
            organism_id: ID of organism performing action
            action: Action name (e.g., "MOVE_NORTH", "EAT", "ATTACK")
            success: Whether action succeeded
        """
        # Actions are stored in organism_data during log_timestep
        # This method is for explicit action logging if needed
        pass
    
    def log_event(self, event_type: str, data: Tuple) -> None:
        """Log a simulation event.
        
        Args:
            event_type: Type of event ('food_consumed', 'attacks', 'deaths')
            data: Event data tuple (format depends on event type)
            
        Example:
            >>> logger.log_event('attacks', (timestep, attacker_id, target_id))
        """
        if event_type in self.events:
            self.events[event_type].append(data)
    
    def save_to_hdf5(self, filepath: Optional[str] = None) -> Path:
        """Save all logged data to HDF5 file.
        
        Args:
            filepath: Optional custom filepath. If None, uses default naming.
            
        Returns:
            Path to saved HDF5 file
            
        Raises:
            ValueError: If no data has been logged
        """
        if filepath is None:
            filepath = self.output_dir / f"simulation_{self.run_id:03d}.hdf5"
        else:
            filepath = Path(filepath)
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Update metadata
        self.metadata['timestamp_end'] = datetime.now().isoformat()
        self.metadata['total_timesteps'] = self.current_timestep + 1
        
        with h5py.File(filepath, 'w') as f:
            # Write metadata as attributes
            for key, value in self.metadata.items():
                f.attrs[key] = value if value is not None else "None"
            
            # Write configuration as JSON string
            f.attrs['config'] = json.dumps(self.config)
            
            # Write organism data
            organisms_group = f.create_group('organisms')
            for org_id in range(self.num_organisms):
                org_group = organisms_group.create_group(f'organism_{org_id}')
                
                # Positions as [T, 2] array
                positions = np.array(self.organism_data[org_id]['positions'], dtype=np.int32)
                org_group.create_dataset('positions', data=positions, compression='gzip')
                
                # Energies as [T] array
                energies = np.array(self.organism_data[org_id]['energies'], dtype=np.float32)
                org_group.create_dataset('energies', data=energies, compression='gzip')
                
                # Actions as string array [T]
                actions = self.organism_data[org_id]['actions']
                dt = h5py.string_dtype(encoding='utf-8')
                org_group.create_dataset('actions', data=actions, dtype=dt, compression='gzip')
                
                # Alive as boolean array [T]
                alive = np.array(self.organism_data[org_id]['alive'], dtype=bool)
                org_group.create_dataset('alive', data=alive, compression='gzip')
            
            # Write environment data
            env_group = f.create_group('environment')
            
            # Food positions as ragged array - store as variable-length
            # For simplicity, we'll pad and store as [T, max_food, 2]
            food_data = self.environment_data['food_positions']
            if food_data:
                max_food = max(len(fp) if fp else 0 for fp in food_data)
                padded_food = np.zeros((len(food_data), max_food, 2), dtype=np.int32)
                for t, food_pos in enumerate(food_data):
                    if food_pos:
                        padded_food[t, :len(food_pos), :] = food_pos
                env_group.create_dataset('food_positions', data=padded_food, compression='gzip')
            
            # Write events
            events_group = f.create_group('events')
            
            for event_name, event_list in self.events.items():
                if event_list:
                    # Convert events to flat arrays
                    # food_consumed: (t, org_id, (x, y)) -> flatten to (t, org_id, x, y)
                    if event_name == 'food_consumed':
                        flat_events = [(t, org_id, pos[0], pos[1]) for t, org_id, pos in event_list]
                        event_array = np.array(flat_events, dtype=np.int32)
                    else:
                        event_array = np.array(event_list, dtype=np.int32)
                    events_group.create_dataset(event_name, data=event_array, compression='gzip')
        
        return filepath
    
    def save_config_yaml(self, filepath: Optional[str] = None) -> Path:
        """Save configuration to YAML file alongside HDF5 data.
        
        Args:
            filepath: Optional custom filepath
            
        Returns:
            Path to saved YAML file
        """
        if filepath is None:
            filepath = self.output_dir / f"config_{self.run_id:03d}.yaml"
        else:
            filepath = Path(filepath)
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
        
        return filepath
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for the simulation.
        
        Returns:
            Dictionary with summary metrics
        """
        summary = {
            'run_id': self.run_id,
            'total_timesteps': self.current_timestep + 1,  # current_timestep is last logged (0-indexed)
            'num_organisms': self.num_organisms,
            'git_hash': self.metadata['git_hash'],
            'config_hash': self.metadata['config_hash'],
        }
        
        # Per-organism statistics
        for org_id in range(self.num_organisms):
            org_data = self.organism_data[org_id]
            if org_data['energies']:
                summary[f'organism_{org_id}'] = {
                    'final_energy': org_data['energies'][-1],
                    'max_energy': max(org_data['energies']),
                    'min_energy': min(org_data['energies']),
                    'avg_energy': np.mean(org_data['energies']),
                    'survival_time': sum(org_data['alive'])
                }
        
        # Event counts
        summary['events'] = {
            'food_consumed': len(self.events['food_consumed']),
            'attacks': len(self.events['attacks']),
            'deaths': len(self.events['deaths'])
        }
        
        return summary
    
    def close(self) -> None:
        """Close logger and finalize all data."""
        # Update end timestamp
        self.metadata['timestamp_end'] = datetime.now().isoformat()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False

