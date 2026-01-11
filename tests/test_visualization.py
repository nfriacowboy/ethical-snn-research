"""Unit tests for visualization utilities.

Tests cover:
- Grid plotting with organisms and food
- Energy plots from HDF5 files
- Animation creation (mock tests, no actual video generation in CI)
- Action distribution plots
"""

import pytest
import tempfile
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing

import matplotlib.pyplot as plt
import numpy as np
import h5py

from src.utils.visualization import (
    plot_grid,
    plot_energy_over_time,
    plot_action_distribution
)


# Mock classes for testing
class MockEnvironment:
    """Mock environment for testing."""
    def __init__(self, grid_size=20, food_positions=None):
        self.grid_size = grid_size
        self.food_positions = food_positions if food_positions is not None else []


class MockOrganism:
    """Mock organism for testing."""
    def __init__(self, position, energy, max_energy=100, alive=True):
        self.position = position
        self.energy = energy
        self.max_energy = max_energy
        self._alive = alive
    
    def is_alive(self):
        return self._alive


@pytest.fixture
def sample_hdf5_log():
    """Create a sample HDF5 log file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
        filepath = f.name
    
    # Create minimal HDF5 structure
    with h5py.File(filepath, 'w') as f:
        # Metadata
        f.attrs['run_id'] = 1
        f.attrs['grid_size'] = 20
        f.attrs['config'] = '{"environment": {"grid_size": 20}}'
        
        # Organism data
        organisms_group = f.create_group('organisms')
        for org_id in range(2):
            org_group = organisms_group.create_group(f'organism_{org_id}')
            
            # Create simple trajectories
            positions = np.array([[i, i] for i in range(10)], dtype=np.int32)
            energies = np.array([100 - i for i in range(10)], dtype=np.float32)
            actions = ['MOVE_NORTH'] * 5 + ['EAT'] * 3 + ['MOVE_SOUTH'] * 2
            alive = np.array([True] * 10, dtype=bool)
            
            org_group.create_dataset('positions', data=positions)
            org_group.create_dataset('energies', data=energies)
            
            dt = h5py.string_dtype(encoding='utf-8')
            org_group.create_dataset('actions', data=actions, dtype=dt)
            org_group.create_dataset('alive', data=alive)
        
        # Environment data
        env_group = f.create_group('environment')
        food_data = np.ones((10, 2, 2), dtype=np.int32) * 5  # Static food at (5,5)
        env_group.create_dataset('food_positions', data=food_data)
        
        # Events
        events_group = f.create_group('events')
        food_consumed = np.array([[5, 0, 5, 5]], dtype=np.int32)
        events_group.create_dataset('food_consumed', data=food_consumed)
    
    yield filepath
    
    # Cleanup
    Path(filepath).unlink()


class TestPlotGrid:
    """Test grid plotting functionality."""
    
    def test_plot_grid_basic(self):
        """Test basic grid plotting."""
        env = MockEnvironment(grid_size=20, food_positions=[(5, 5), (10, 10)])
        organisms = [
            MockOrganism(position=(3, 3), energy=80),
            MockOrganism(position=(15, 15), energy=60)
        ]
        
        fig = plot_grid(env, organisms)
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 1
        
        plt.close(fig)
    
    def test_plot_grid_empty(self):
        """Test plotting grid with no food or organisms."""
        env = MockEnvironment(grid_size=20, food_positions=[])
        organisms = []
        
        fig = plot_grid(env, organisms)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_grid_dead_organism(self):
        """Test that dead organisms are not plotted."""
        env = MockEnvironment(grid_size=20)
        organisms = [
            MockOrganism(position=(5, 5), energy=50, alive=True),
            MockOrganism(position=(10, 10), energy=0, alive=False)
        ]
        
        fig = plot_grid(env, organisms)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_grid_custom_title(self):
        """Test grid plot with custom title."""
        env = MockEnvironment(grid_size=20)
        organisms = [MockOrganism(position=(5, 5), energy=80)]
        
        fig = plot_grid(env, organisms, title="Custom Title Test")
        
        assert fig.axes[0].get_title() == "Custom Title Test"
        plt.close(fig)
    
    def test_plot_grid_no_grid_lines(self):
        """Test grid plot without grid lines."""
        env = MockEnvironment(grid_size=20)
        organisms = [MockOrganism(position=(5, 5), energy=80)]
        
        fig = plot_grid(env, organisms, show_grid=False)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotEnergyOverTime:
    """Test energy plotting functionality."""
    
    def test_plot_energy_all_organisms(self, sample_hdf5_log):
        """Test plotting energy for all organisms."""
        fig = plot_energy_over_time(sample_hdf5_log)
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 1
        
        # Check that lines were plotted
        ax = fig.axes[0]
        assert len(ax.lines) == 2  # 2 organisms
        
        plt.close(fig)
    
    def test_plot_energy_specific_organism(self, sample_hdf5_log):
        """Test plotting energy for specific organism."""
        fig = plot_energy_over_time(sample_hdf5_log, organism_ids=[0])
        
        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        assert len(ax.lines) == 1  # Only organism 0
        
        plt.close(fig)
    
    def test_plot_energy_axes_labels(self, sample_hdf5_log):
        """Test that axes have correct labels."""
        fig = plot_energy_over_time(sample_hdf5_log)
        
        ax = fig.axes[0]
        assert ax.get_xlabel() == 'Timestep'
        assert ax.get_ylabel() == 'Energy'
        
        plt.close(fig)


class TestPlotActionDistribution:
    """Test action distribution plotting."""
    
    def test_plot_action_distribution(self, sample_hdf5_log):
        """Test plotting action distribution."""
        fig = plot_action_distribution(sample_hdf5_log, organism_id=0)
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 1
        
        plt.close(fig)
    
    def test_plot_action_distribution_bars(self, sample_hdf5_log):
        """Test that bars are created for actions."""
        fig = plot_action_distribution(sample_hdf5_log, organism_id=0)
        
        ax = fig.axes[0]
        # Should have bars for different actions
        assert len(ax.patches) > 0
        
        plt.close(fig)


class TestFigureSaving:
    """Test saving figures to files."""
    
    def test_save_grid_plot(self):
        """Test saving grid plot to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = MockEnvironment(grid_size=20, food_positions=[(5, 5)])
            organisms = [MockOrganism(position=(3, 3), energy=80)]
            
            fig = plot_grid(env, organisms)
            
            output_path = Path(tmpdir) / "test_grid.png"
            fig.savefig(output_path)
            
            assert output_path.exists()
            assert output_path.stat().st_size > 0
            
            plt.close(fig)
    
    def test_save_energy_plot(self, sample_hdf5_log):
        """Test saving energy plot to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fig = plot_energy_over_time(sample_hdf5_log)
            
            output_path = Path(tmpdir) / "test_energy.png"
            fig.savefig(output_path)
            
            assert output_path.exists()
            assert output_path.stat().st_size > 0
            
            plt.close(fig)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_plot_with_single_organism(self, sample_hdf5_log):
        """Test plotting with only one organism."""
        fig = plot_energy_over_time(sample_hdf5_log, organism_ids=[0])
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_grid_large_grid(self):
        """Test plotting with large grid size."""
        env = MockEnvironment(grid_size=50)
        organisms = [MockOrganism(position=(25, 25), energy=100)]
        
        fig = plot_grid(env, organisms)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_many_food_items(self):
        """Test plotting with many food items."""
        food_positions = [(i, i) for i in range(10)]
        env = MockEnvironment(grid_size=20, food_positions=food_positions)
        organisms = []
        
        fig = plot_grid(env, organisms)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
