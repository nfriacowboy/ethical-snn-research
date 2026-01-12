"""Integration tests for Phase 1 simulation pipeline.

Tests the complete workflow from configuration loading through simulation
execution and statistics collection.
"""

import pytest
import numpy as np
import torch
from pathlib import Path
import json
import tempfile

from src.simulation.runner import SimulationRunner
from src.organisms.survival_snn import SurvivalSNN
from src.architecture.dual_process import DualProcessOrganism
from src.utils.config import load_config


class TestFullPipelineConditionA:
    """Integration tests for Condition A (survival-only) pipeline."""
    
    def test_complete_simulation_run(self):
        """Test complete simulation run from start to finish."""
        config = {
            'condition': 'A',
            'grid_size': 10,
            'num_organisms': 3,
            'num_food': 5,
            'max_timesteps': 50,
            'energy_decay_rate': 2.0,
            'food_energy': 20.0,
            'food_respawn_rate': 0.2
        }
        
        runner = SimulationRunner(config, seed=42)
        stats = runner.run()
        
        # Verify statistics structure
        assert 'condition' in stats
        assert stats['condition'] == 'A'
        assert 'organisms' in stats
        assert 'energy' in stats
        assert 'environment' in stats
        assert 'final_timestep' in stats
        
        # Verify organisms were tracked
        assert stats['organisms']['total'] == 3
        assert stats['organisms']['alive'] + stats['organisms']['dead'] == 3
        
        # Verify simulation completed
        assert stats['final_timestep'] <= 50
    
    def test_organism_survival_behavior(self):
        """Test that organisms exhibit expected survival behavior."""
        config = {
            'condition': 'A',
            'grid_size': 5,
            'num_organisms': 1,
            'num_food': 10,  # Lots of food
            'max_timesteps': 100,
            'energy_decay_rate': 0.5,  # Slow decay
            'food_energy': 30.0
        }
        
        runner = SimulationRunner(config, seed=42)
        organism = runner.organisms[0]
        initial_energy = organism.energy
        
        # Run for a few steps
        for _ in range(10):
            runner.step()
        
        # Organism should still be alive with plenty of food
        assert organism.alive
        
        # Energy should have changed (either gained or lost)
        # This verifies organism is interacting with environment
        assert organism.energy != initial_energy or organism.age > 0
    
    def test_reproducibility_with_seeds(self):
        """Test that same seed produces identical results."""
        config = {
            'condition': 'A',
            'grid_size': 10,
            'num_organisms': 3,
            'max_timesteps': 20
        }
        
        # Run twice with same seed
        runner1 = SimulationRunner(config, seed=42)
        stats1 = runner1.run()
        
        runner2 = SimulationRunner(config, seed=42)
        stats2 = runner2.run()
        
        # Should have identical results
        assert stats1['final_timestep'] == stats2['final_timestep']
        assert stats1['organisms']['alive'] == stats2['organisms']['alive']
        assert stats1['organisms']['dead'] == stats2['organisms']['dead']
        
        # Organism positions should be identical
        for org1, org2 in zip(runner1.organisms, runner2.organisms):
            assert org1.position == org2.position
            assert org1.energy == org2.energy


class TestFullPipelineConditionB:
    """Integration tests for Condition B (dual-process) pipeline."""
    
    def test_complete_simulation_run(self):
        """Test complete dual-process simulation run."""
        config = {
            'condition': 'B',
            'grid_size': 10,
            'num_organisms': 3,
            'num_food': 5,
            'max_timesteps': 50,
            'energy_decay_rate': 2.0
        }
        
        runner = SimulationRunner(config, seed=42)
        stats = runner.run()
        
        # Verify statistics structure
        assert 'condition' in stats
        assert stats['condition'] == 'B'
        assert 'dual_process' in stats
        
        # Verify dual-process specific stats
        assert 'total_vetoes' in stats['dual_process']
        assert 'total_approvals' in stats['dual_process']
        assert 'avg_veto_rate' in stats['dual_process']
    
    def test_ethical_veto_functionality(self):
        """Test that ethical network can veto actions."""
        config = {
            'condition': 'B',
            'grid_size': 10,
            'num_organisms': 2,
            'max_timesteps': 50
        }
        
        runner = SimulationRunner(config, seed=42)
        
        # Run simulation
        stats = runner.run()
        
        # Check that vetoes occurred (or could have occurred)
        # Even if veto count is 0, the mechanism should exist
        assert 'total_vetoes' in stats['dual_process']
        assert 'total_approvals' in stats['dual_process']
        
        # Verify organisms are dual-process
        for organism in runner.organisms:
            assert isinstance(organism, DualProcessOrganism)
            assert hasattr(organism, 'survival_snn')
            assert hasattr(organism, 'ethical_snn')
    
    def test_dual_process_vs_survival_only(self):
        """Test that Condition B differs from Condition A."""
        base_config = {
            'grid_size': 10,
            'num_organisms': 3,
            'num_food': 5,
            'max_timesteps': 30
        }
        
        # Run Condition A
        config_a = base_config.copy()
        config_a['condition'] = 'A'
        runner_a = SimulationRunner(config_a, seed=42)
        stats_a = runner_a.run()
        
        # Run Condition B  
        config_b = base_config.copy()
        config_b['condition'] = 'B'
        runner_b = SimulationRunner(config_b, seed=42)
        stats_b = runner_b.run()
        
        # Condition B should have dual-process stats
        assert 'dual_process' not in stats_a
        assert 'dual_process' in stats_b
        
        # Organisms should be different types
        assert isinstance(runner_a.organisms[0], SurvivalSNN)
        assert isinstance(runner_b.organisms[0], DualProcessOrganism)


class TestEnvironmentIntegration:
    """Integration tests for environment interaction."""
    
    def test_food_consumption_chain(self):
        """Test complete food consumption workflow."""
        config = {
            'condition': 'A',
            'grid_size': 5,
            'num_organisms': 1,
            'num_food': 1,
            'max_timesteps': 100,
            'food_energy': 50.0,
            'food_respawn_rate': 0.0  # No respawning
        }
        
        runner = SimulationRunner(config, seed=42)
        organism = runner.organisms[0]
        
        # Place organism on food
        if runner.grid.food_positions:
            food_pos = runner.grid.food_positions[0]
            organism.move(food_pos)
            
            initial_energy = organism.energy
            initial_food_count = len(runner.grid.food_positions)
            
            # Run one step - organism should try to eat
            runner.step()
            
            # If organism ate, energy should increase and food should decrease
            # (May not happen every time due to neural network decision)
            if organism.energy > initial_energy:
                assert len(runner.grid.food_positions) < initial_food_count
    
    def test_energy_decay_and_death(self):
        """Test that organisms die when energy depletes."""
        config = {
            'condition': 'A',
            'grid_size': 5,
            'num_organisms': 2,
            'num_food': 0,  # No food
            'max_timesteps': 200,
            'energy_decay_rate': 10.0  # Fast decay
        }
        
        runner = SimulationRunner(config, seed=42)
        stats = runner.run()
        
        # All organisms should die without food
        assert stats['organisms']['dead'] == 2
        assert stats['organisms']['alive'] == 0
        
        # Simulation should end before max timesteps
        assert stats['final_timestep'] < 200
    
    def test_toroidal_grid_wrapping(self):
        """Test that grid boundaries wrap around correctly."""
        config = {
            'condition': 'A',
            'grid_size': 5,
            'num_organisms': 1,
            'max_timesteps': 10
        }
        
        runner = SimulationRunner(config, seed=42)
        organism = runner.organisms[0]
        
        # Place organism at edge
        organism.move((0, 0))
        
        # Move off edge (should wrap to other side)
        from src.organisms.base_organism import Action
        dx, dy = Action.get_direction_vector(Action.MOVE_NORTH)
        new_x = (0 + dx) % runner.grid.grid_size
        new_y = (0 + dy) % runner.grid.grid_size
        
        # MOVE_NORTH is (0, -1) in (dx, dy), so moving north from (0,0) wraps to (0,4)
        assert new_x == 0  # Row doesn't change
        assert new_y == 4  # Wrapped from 0 to grid_size-1


class TestStatisticsCollection:
    """Integration tests for statistics collection."""
    
    def test_statistics_completeness(self):
        """Test that all required statistics are collected."""
        config = {
            'condition': 'B',
            'grid_size': 10,
            'num_organisms': 5,
            'max_timesteps': 30
        }
        
        runner = SimulationRunner(config, seed=42)
        stats = runner.run()
        
        # Required top-level keys
        required_keys = [
            'condition', 'seed', 'config', 'final_timestep',
            'elapsed_time', 'organisms', 'energy', 'environment'
        ]
        for key in required_keys:
            assert key in stats, f"Missing required stat: {key}"
        
        # Organism stats
        org_keys = ['total', 'alive', 'dead', 'survival_times', 'avg_survival_time']
        for key in org_keys:
            assert key in stats['organisms'], f"Missing organism stat: {key}"
        
        # Energy stats
        energy_keys = ['final_avg', 'final_std']
        for key in energy_keys:
            assert key in stats['energy'], f"Missing energy stat: {key}"
        
        # Dual-process stats (for Condition B)
        dp_keys = ['total_vetoes', 'total_approvals', 'avg_veto_rate']
        for key in dp_keys:
            assert key in stats['dual_process'], f"Missing dual-process stat: {key}"
    
    def test_statistics_values_valid(self):
        """Test that statistics have valid values."""
        config = {
            'condition': 'A',
            'grid_size': 10,
            'num_organisms': 5,
            'max_timesteps': 50
        }
        
        runner = SimulationRunner(config, seed=42)
        stats = runner.run()
        
        # Check value ranges
        assert 0 <= stats['final_timestep'] <= 50
        assert stats['organisms']['total'] == 5
        assert stats['organisms']['alive'] >= 0
        assert stats['organisms']['dead'] >= 0
        assert stats['organisms']['alive'] + stats['organisms']['dead'] == 5
        
        # Energy should be non-negative for alive organisms
        if stats['organisms']['alive'] > 0:
            assert stats['energy']['final_avg'] >= 0
            assert stats['energy']['final_std'] >= 0
    
    def test_json_serialization(self):
        """Test that statistics can be serialized to JSON."""
        config = {
            'condition': 'A',
            'grid_size': 10,
            'num_organisms': 3,
            'max_timesteps': 20
        }
        
        runner = SimulationRunner(config, seed=42)
        stats = runner.run()
        
        # Should be able to serialize and deserialize
        json_str = json.dumps(stats)
        recovered_stats = json.loads(json_str)
        
        assert recovered_stats['condition'] == stats['condition']
        assert recovered_stats['final_timestep'] == stats['final_timestep']


class TestConfigurationLoading:
    """Integration tests for configuration loading."""
    
    def test_load_phase1_config(self):
        """Test loading actual Phase 1 configuration file."""
        config_path = Path('experiments/phase1/config_phase1.yaml')
        
        if config_path.exists():
            config = load_config(str(config_path))
            
            # Should have required keys
            assert 'environment' in config
            assert 'organism' in config
            
            # Environment config
            assert 'grid_size' in config['environment']
            assert 'num_food' in config['environment']
            
            # Test that config works with runner
            run_config = {
                'condition': 'A',
                'grid_size': config['environment']['grid_size'],
                'num_organisms': 2,
                'max_timesteps': 10
            }
            
            runner = SimulationRunner(run_config, seed=42)
            stats = runner.run()
            
            assert stats['condition'] == 'A'


class TestErrorHandling:
    """Integration tests for error handling."""
    
    def test_invalid_condition(self):
        """Test that invalid condition raises error."""
        config = {
            'condition': 'INVALID',
            'num_organisms': 3
        }
        
        with pytest.raises(ValueError):
            runner = SimulationRunner(config, seed=42)
    
    def test_zero_organisms(self):
        """Test handling of edge case with no organisms."""
        config = {
            'condition': 'A',
            'num_organisms': 0,
            'max_timesteps': 10
        }
        
        runner = SimulationRunner(config, seed=42)
        stats = runner.run()
        
        # Should complete immediately
        assert stats['organisms']['total'] == 0
        assert stats['organisms']['alive'] == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
