"""Main simulation runner."""

from typing import Dict, Any, Optional, List
import time
from ..environment.grid_world import GridWorld
from ..environment.collision_handler import CollisionHandler
from ..architecture.single_process import SingleProcessArchitecture
from ..architecture.dual_process import DualProcessArchitecture
from .logger import SimulationLogger
from .checkpointer import Checkpointer


class SimulationRunner:
    """Main simulation engine.
    
    Coordinates environment, organisms, and logging for complete simulations.
    """
    
    def __init__(self, config: Dict[str, Any], run_id: int = 0):
        """Initialize simulation runner.
        
        Args:
            config: Configuration dictionary
            run_id: Unique run identifier
        """
        self.config = config
        self.run_id = run_id
        
        # Setup random seed for reproducibility
        import random
        import numpy as np
        import torch
        
        seed = config.get('seed', run_id)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Initialize components
        self.environment = GridWorld(
            size=tuple(config.get('grid_size', [50, 50])),
            food_spawn_rate=config.get('food_spawn_rate', 0.02),
            energy_decay=config.get('energy_decay', 1.0)
        )
        
        self.collision_handler = CollisionHandler(
            collision_penalty=config.get('collision_penalty', 10.0)
        )
        
        # Initialize architecture (single or dual process)
        architecture_type = config.get('architecture', 'single')
        num_organisms = config.get('num_organisms', 10)
        
        if architecture_type == 'single':
            self.architecture = SingleProcessArchitecture(
                num_organisms=num_organisms,
                grid_size=tuple(config.get('grid_size', [50, 50]))
            )
        elif architecture_type == 'dual':
            self.architecture = DualProcessArchitecture(
                num_organisms=num_organisms,
                grid_size=tuple(config.get('grid_size', [50, 50])),
                ethical_weight=config.get('ethical_weight', 0.5)
            )
        else:
            raise ValueError(f"Unknown architecture type: {architecture_type}")
        
        # Logging and checkpointing
        self.logger = SimulationLogger(run_id=run_id)
        self.checkpointer = Checkpointer(run_id=run_id)
        
        # Simulation state
        self.timestep = 0
        self.max_timesteps = config.get('max_timesteps', 1000)
        self.checkpoint_frequency = config.get('checkpoint_frequency', 100)
        
        self.running = False
        self.start_time = None
    
    def run(self) -> Dict[str, Any]:
        """Run complete simulation.
        
        Returns:
            Final simulation statistics
        """
        self.running = True
        self.start_time = time.time()
        
        print(f"Starting simulation run {self.run_id}")
        print(f"Architecture: {self.config.get('architecture', 'single')}")
        print(f"Organisms: {self.config.get('num_organisms', 10)}")
        print(f"Max timesteps: {self.max_timesteps}")
        
        for t in range(self.max_timesteps):
            self.timestep = t
            
            # Step simulation
            self.step()
            
            # Checkpoint
            if t % self.checkpoint_frequency == 0:
                self.checkpoint()
                self._print_progress()
            
            # Check termination
            if self.architecture.get_alive_count() == 0:
                print(f"All organisms dead at timestep {t}")
                break
        
        self.running = False
        elapsed_time = time.time() - self.start_time
        
        print(f"Simulation completed in {elapsed_time:.2f} seconds")
        
        return self.get_final_statistics()
    
    def step(self):
        """Execute one simulation timestep."""
        # Get organism positions
        organism_states = [org.get_state() for org in self.architecture.organisms]
        organism_positions = [state['position'] for state in organism_states if state['alive']]
        
        # Update environment
        env_state = self.environment.step(organism_positions)
        
        # Build state for organisms
        full_env_state = {
            'food_positions': self.environment.food_positions,
            'grid': self.environment.grid,
            'timestep': self.timestep
        }
        
        # Add nearest food for each organism
        for org in self.architecture.organisms:
            if org.alive:
                nearest = self.environment.find_nearest_food(org.position)
                full_env_state['nearest_food'] = nearest
                break
        
        # Organisms take actions
        new_states = self.architecture.step(full_env_state)
        
        # Handle collisions
        collisions = self.collision_handler.detect_collisions(new_states)
        
        # Apply collision penalties
        for collision in collisions:
            for org_id in collision['organism_ids']:
                org = self.architecture.organisms[org_id]
                org.update(-collision['penalty'])
        
        # Update organism energy (decay and food consumption)
        for org in self.architecture.organisms:
            if org.alive:
                # Energy decay
                org.update(-self.environment.energy_decay)
                
                # Food consumption
                energy_gained = self.environment.consume_food(org.position)
                if energy_gained > 0:
                    org.update(energy_gained)
        
        # Log timestep
        self.logger.log_timestep(
            timestep=self.timestep,
            organism_states=new_states,
            environment_state=env_state,
            collisions=collisions
        )
    
    def checkpoint(self):
        """Save simulation checkpoint."""
        checkpoint_data = {
            'timestep': self.timestep,
            'environment_state': self.environment.get_state(),
            'organism_states': [org.get_state() for org in self.architecture.organisms],
            'architecture_stats': self.architecture.get_statistics()
        }
        
        self.checkpointer.save(checkpoint_data, self.timestep)
    
    def _print_progress(self):
        """Print progress update."""
        stats = self.architecture.get_statistics()
        print(f"Timestep {self.timestep}/{self.max_timesteps} - "
              f"Alive: {stats['alive_count']}/{stats['total_organisms']} - "
              f"Avg Energy: {stats['avg_energy']:.2f}")
    
    def get_final_statistics(self) -> Dict[str, Any]:
        """Get final simulation statistics.
        
        Returns:
            Dictionary with comprehensive statistics
        """
        return {
            'run_id': self.run_id,
            'config': self.config,
            'final_timestep': self.timestep,
            'elapsed_time': time.time() - self.start_time if self.start_time else 0,
            'architecture_stats': self.architecture.get_statistics(),
            'environment_stats': {
                'total_food_spawned': self.environment.total_food_spawned,
                'total_food_consumed': self.environment.total_food_consumed
            },
            'collision_stats': self.collision_handler.get_statistics()
        }
