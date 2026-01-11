"""Unit tests for simulation logging system.

Tests cover:
- HDF5 file creation and structure
- Metadata tracking (git hash, config, timestamps)
- Organism state logging
- Environment state logging
- Event logging
- Summary statistics generation
"""

import pytest
import tempfile
from pathlib import Path
import h5py
import numpy as np

from src.simulation.logger import SimulationLogger, get_git_hash, hash_config
from src.utils.config import get_default_config


class TestGitAndHashing:
    """Test utility functions for metadata."""
    
    def test_get_git_hash(self):
        """Test git hash retrieval."""
        git_hash = get_git_hash()
        assert isinstance(git_hash, str)
        # Should either be a valid hash or 'unknown'
        assert len(git_hash) >= 7 or git_hash == "unknown"
    
    def test_hash_config(self):
        """Test configuration hashing."""
        config = get_default_config()
        hash1 = hash_config(config)
        hash2 = hash_config(config)
        
        assert hash1 == hash2  # Same config = same hash
        assert len(hash1) == 16  # Hash should be 16 chars
        
        # Different config = different hash
        config['simulation']['random_seed'] = 999
        hash3 = hash_config(config)
        assert hash1 != hash3


class TestSimulationLoggerInit:
    """Test logger initialization."""
    
    def test_logger_init(self):
        """Test basic logger initialization."""
        config = get_default_config()
        logger = SimulationLogger(config, run_id=1, output_dir="test_results")
        
        assert logger.run_id == 1
        assert logger.output_dir.name == "test_results"
        assert logger.num_organisms == config['simulation']['num_organisms']
        assert 'git_hash' in logger.metadata
        assert 'config_hash' in logger.metadata
    
    def test_logger_creates_output_dir(self):
        """Test that logger creates output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = get_default_config()
            output_path = Path(tmpdir) / "logs"
            logger = SimulationLogger(config, run_id=1, output_dir=str(output_path))
            
            assert output_path.exists()
            assert output_path.is_dir()


class TestLoggingTimesteps:
    """Test timestep logging functionality."""
    
    def test_log_single_timestep(self):
        """Test logging a single timestep."""
        config = get_default_config()
        config['simulation']['num_organisms'] = 2
        logger = SimulationLogger(config, run_id=1)
        
        state = {
            'organism_states': [
                {'position': (10, 10), 'energy': 100, 'action': 'MOVE_NORTH', 'alive': True},
                {'position': (5, 5), 'energy': 90, 'action': 'EAT', 'alive': True}
            ],
            'environment_state': {
                'food_positions': [(3, 3), (15, 15)]
            }
        }
        
        logger.log_timestep(0, state)
        
        assert len(logger.organism_data[0]['positions']) == 1
        assert logger.organism_data[0]['positions'][0] == (10, 10)
        assert logger.organism_data[0]['energies'][0] == 100
        assert logger.organism_data[1]['actions'][0] == 'EAT'
    
    def test_log_multiple_timesteps(self):
        """Test logging multiple timesteps."""
        config = get_default_config()
        config['simulation']['num_organisms'] = 1
        logger = SimulationLogger(config, run_id=1)
        
        for t in range(5):
            state = {
                'organism_states': [
                    {'position': (t, t), 'energy': 100 - t, 'action': 'MOVE_NORTH', 'alive': True}
                ],
                'environment_state': {'food_positions': [(t, t + 1)]}
            }
            logger.log_timestep(t, state)
        
        assert len(logger.organism_data[0]['positions']) == 5
        assert logger.organism_data[0]['energies'][0] == 100
        assert logger.organism_data[0]['energies'][4] == 96


class TestEventLogging:
    """Test event logging functionality."""
    
    def test_log_food_consumed_event(self):
        """Test logging food consumption events."""
        config = get_default_config()
        logger = SimulationLogger(config, run_id=1)
        
        logger.log_event('food_consumed', (10, 0, (5, 5)))  # timestep, org_id, position
        logger.log_event('food_consumed', (15, 1, (8, 8)))
        
        assert len(logger.events['food_consumed']) == 2
        assert logger.events['food_consumed'][0] == (10, 0, (5, 5))
    
    def test_log_attack_event(self):
        """Test logging attack events."""
        config = get_default_config()
        logger = SimulationLogger(config, run_id=1)
        
        logger.log_event('attacks', (20, 0, 1))  # timestep, attacker, target
        
        assert len(logger.events['attacks']) == 1
        assert logger.events['attacks'][0] == (20, 0, 1)
    
    def test_log_death_event(self):
        """Test logging death events."""
        config = get_default_config()
        logger = SimulationLogger(config, run_id=1)
        
        logger.log_event('deaths', (50, 0))  # timestep, org_id
        
        assert len(logger.events['deaths']) == 1
        assert logger.events['deaths'][0] == (50, 0)


class TestHDF5Saving:
    """Test HDF5 file saving and structure."""
    
    def test_save_to_hdf5_basic(self):
        """Test basic HDF5 saving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = get_default_config()
            config['simulation']['num_organisms'] = 2
            logger = SimulationLogger(config, run_id=1, output_dir=tmpdir)
            
            # Log some data
            for t in range(3):
                state = {
                    'organism_states': [
                        {'position': (t, t), 'energy': 100, 'action': 'MOVE_NORTH', 'alive': True},
                        {'position': (t + 1, t + 1), 'energy': 95, 'action': 'EAT', 'alive': True}
                    ],
                    'environment_state': {'food_positions': [(5, 5)]}
                }
                logger.log_timestep(t, state)
            
            # Save to HDF5
            filepath = logger.save_to_hdf5()
            
            assert filepath.exists()
            assert filepath.suffix == '.hdf5'
    
    def test_hdf5_metadata_attributes(self):
        """Test that metadata is saved as HDF5 attributes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = get_default_config()
            logger = SimulationLogger(config, run_id=42, output_dir=tmpdir)
            
            # Log minimal data
            state = {
                'organism_states': [{'position': (0, 0), 'energy': 100, 'action': 'NONE', 'alive': True}] * 2,
                'environment_state': {'food_positions': []}
            }
            logger.log_timestep(0, state)
            
            filepath = logger.save_to_hdf5()
            
            # Read and verify metadata
            with h5py.File(filepath, 'r') as f:
                assert f.attrs['run_id'] == 42
                assert 'git_hash' in f.attrs
                assert 'config_hash' in f.attrs
                assert 'timestamp_start' in f.attrs
                assert 'config' in f.attrs
    
    def test_hdf5_organism_data_structure(self):
        """Test HDF5 organism data structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = get_default_config()
            config['simulation']['num_organisms'] = 2
            logger = SimulationLogger(config, run_id=1, output_dir=tmpdir)
            
            # Log data
            for t in range(5):
                state = {
                    'organism_states': [
                        {'position': (t, t), 'energy': 100 - t, 'action': 'MOVE_NORTH', 'alive': True},
                        {'position': (t + 1, t + 1), 'energy': 90 - t, 'action': 'EAT', 'alive': True}
                    ],
                    'environment_state': {'food_positions': []}
                }
                logger.log_timestep(t, state)
            
            filepath = logger.save_to_hdf5()
            
            # Verify structure
            with h5py.File(filepath, 'r') as f:
                assert 'organisms' in f
                assert 'organism_0' in f['organisms']
                assert 'organism_1' in f['organisms']
                
                org0 = f['organisms']['organism_0']
                assert 'positions' in org0
                assert 'energies' in org0
                assert 'actions' in org0
                assert 'alive' in org0
                
                # Check dimensions
                assert org0['positions'].shape == (5, 2)
                assert org0['energies'].shape == (5,)
                assert org0['actions'].shape == (5,)
                assert org0['alive'].shape == (5,)
                
                # Check values
                assert org0['positions'][0, 0] == 0
                assert org0['positions'][0, 1] == 0
                assert org0['energies'][0] == 100
                assert org0['energies'][4] == 96
    
    def test_hdf5_environment_data(self):
        """Test HDF5 environment data structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = get_default_config()
            config['simulation']['num_organisms'] = 1
            logger = SimulationLogger(config, run_id=1, output_dir=tmpdir)
            
            # Log with varying food positions
            for t in range(3):
                state = {
                    'organism_states': [{'position': (0, 0), 'energy': 100, 'action': 'NONE', 'alive': True}],
                    'environment_state': {
                        'food_positions': [(t, t), (t + 1, t + 1)]
                    }
                }
                logger.log_timestep(t, state)
            
            filepath = logger.save_to_hdf5()
            
            with h5py.File(filepath, 'r') as f:
                assert 'environment' in f
                assert 'food_positions' in f['environment']
                
                food_data = f['environment']['food_positions'][:]
                assert food_data.shape[0] == 3  # 3 timesteps
                assert food_data.shape[2] == 2  # 2D positions
    
    def test_hdf5_events_data(self):
        """Test HDF5 events data structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = get_default_config()
            logger = SimulationLogger(config, run_id=1, output_dir=tmpdir)
            
            # Log minimal timestep data
            state = {
                'organism_states': [{'position': (0, 0), 'energy': 100, 'action': 'NONE', 'alive': True}] * 2,
                'environment_state': {'food_positions': []}
            }
            logger.log_timestep(0, state)
            
            # Log events
            logger.log_event('food_consumed', (10, 0, (5, 5)))
            logger.log_event('attacks', (20, 0, 1))
            logger.log_event('deaths', (30, 0))
            
            filepath = logger.save_to_hdf5()
            
            with h5py.File(filepath, 'r') as f:
                assert 'events' in f
                assert 'food_consumed' in f['events']
                assert 'attacks' in f['events']
                assert 'deaths' in f['events']
                
                assert f['events']['food_consumed'].shape[0] == 1
                assert f['events']['attacks'].shape[0] == 1
                assert f['events']['deaths'].shape[0] == 1


class TestConfigYAMLSaving:
    """Test YAML configuration saving."""
    
    def test_save_config_yaml(self):
        """Test saving config to YAML file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = get_default_config()
            logger = SimulationLogger(config, run_id=1, output_dir=tmpdir)
            
            yaml_path = logger.save_config_yaml()
            
            assert yaml_path.exists()
            assert yaml_path.suffix == '.yaml'


class TestSummaryStatistics:
    """Test summary statistics generation."""
    
    def test_get_summary(self):
        """Test summary statistics generation."""
        config = get_default_config()
        config['simulation']['num_organisms'] = 2
        logger = SimulationLogger(config, run_id=1)
        
        # Log some data
        for t in range(10):
            state = {
                'organism_states': [
                    {'position': (t, t), 'energy': 100 - t, 'action': 'MOVE_NORTH', 'alive': True},
                    {'position': (t + 1, t + 1), 'energy': 95 - t, 'action': 'EAT', 'alive': t < 5}
                ],
                'environment_state': {'food_positions': []}
            }
            logger.log_timestep(t, state)
        
        # Log events
        logger.log_event('food_consumed', (5, 0, (5, 5)))
        logger.log_event('attacks', (7, 0, 1))
        logger.log_event('deaths', (9, 1))
        
        summary = logger.get_summary()
        
        assert summary['run_id'] == 1
        assert summary['total_timesteps'] == 10  # Logged timesteps 0-9
        assert summary['num_organisms'] == 2
        assert 'organism_0' in summary
        assert 'organism_1' in summary
        assert summary['organism_0']['final_energy'] == 91  # 100 - 9 = 91
        assert summary['events']['food_consumed'] == 1
        assert summary['events']['attacks'] == 1
        assert summary['events']['deaths'] == 1


class TestContextManager:
    """Test context manager functionality."""
    
    def test_context_manager(self):
        """Test using logger as context manager."""
        config = get_default_config()
        
        with SimulationLogger(config, run_id=1) as logger:
            assert logger.run_id == 1
            assert logger.metadata['timestamp_end'] is None
        
        # After exit, should have end timestamp
        assert logger.metadata['timestamp_end'] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
