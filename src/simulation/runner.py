"""Main simulation runner for Phase 1 experiments.

Coordinates environment, organisms, and logging for complete simulations.
Supports both Condition A (survival-only) and Condition B (dual-process).
"""

import time
from typing import Dict, Any, List, Optional
from pathlib import Path
import torch
import numpy as np

from src.environment.grid_world import GridWorld
from src.organisms.base_organism import Organism, Action
from src.organisms.survival_snn import SurvivalSNN
from src.architecture.dual_process import DualProcessOrganism


class SimpleLogger:
    """Simplified logger for simulation data."""
    
    def __init__(self, output_dir: Path, experiment_name: str):
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name
        self.timesteps_data: List[Dict[str, Any]] = []
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def log_timestep(self, data: Dict[str, Any]) -> None:
        """Log data for one timestep."""
        self.timesteps_data.append(data)
    
    def save_final_statistics(self, stats: Dict[str, Any]) -> None:
        """Save final statistics to JSON."""
        import json
        output_file = self.output_dir / f"{self.experiment_name}_stats.json"
        
        with open(output_file, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        print(f"Statistics saved to: {output_file}")
    
    def save_statistics(self, stats: Dict[str, Any], output_path: Path) -> None:
        """Save statistics to specific path."""
        import json
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2, default=str)


class SimulationRunner:
    """Main simulation engine for Phase 1 experiments.
    
    Manages complete simulation lifecycle:
    - Environment initialization and updates
    - Organism decision-making and actions
    - Energy management and death mechanics
    - Logging and statistics
    
    Attributes:
        config: Simulation configuration dictionary
        seed: Random seed for reproducibility
        grid: GridWorld environment
        organisms: List of Organism instances
        logger: SimulationLogger for data collection
        timestep: Current simulation timestep
        max_timesteps: Maximum timesteps before termination
    """
    
    def __init__(self, config: Dict[str, Any], seed: int = 42):
        """Initialize simulation runner.
        
        Args:
            config: Configuration dictionary with keys:
                - condition: 'A' (survival-only) or 'B' (dual-process)
                - grid_size: int, size of square grid (default: 20)
                - num_organisms: int, number of organisms (default: 10)
                - num_food: int, initial food items (default: 5)
                - food_respawn_rate: float, probability of food respawning (default: 0.1)
                - energy_decay_rate: float, energy lost per timestep (default: 1.0)
                - food_energy: float, energy gained from eating (default: 20.0)
                - max_timesteps: int, maximum simulation duration (default: 1000)
            seed: Random seed for reproducibility (default: 42)
        """
        self.config = config
        self.seed = seed
        
        # Set random seeds
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Extract config
        self.condition = config.get('condition', 'A')
        self.num_organisms = config.get('num_organisms', 10)
        
        # Build GridWorld config structure
        grid_config = {
            'environment': {
                'grid_size': config.get('grid_size', 20),
                'num_food': config.get('num_food', 5),
                'food_respawn_rate': config.get('food_respawn_rate', 0.1),
                'food_energy_value': config.get('food_energy', 20.0)
            }
        }
        
        # Initialize environment
        self.grid = GridWorld(grid_config)
        
        # Initialize organisms based on condition
        self.organisms: List[Organism] = []
        self._initialize_organisms()
        
        # Initialize logger
        self.logger = SimpleLogger(
            output_dir=Path(config.get('output_dir', 'results/simulations')),
            experiment_name=f"phase1_condition_{self.condition}_seed_{seed}"
        )
        
        # Simulation state
        self.timestep = 0
        self.max_timesteps = config.get('max_timesteps', 1000)
        self.energy_decay_rate = config.get('energy_decay_rate', 1.0)
        self.food_energy = config.get('food_energy', 20.0)
        
        # Statistics
        self.start_time: Optional[float] = None
        self.death_times: List[int] = []
    
    def _initialize_organisms(self) -> None:
        """Initialize organisms based on condition type."""
        for i in range(self.num_organisms):
            # Random starting position
            position = (
                np.random.randint(0, self.grid.grid_size),
                np.random.randint(0, self.grid.grid_size)
            )
            
            if self.condition == 'A':
                # Condition A: Survival-only SNN
                organism = SurvivalSNN(
                    organism_id=i,
                    position=position,
                    initial_energy=100.0
                )
            elif self.condition == 'B':
                # Condition B: Dual-process (SNN-S + SNN-E)
                organism = DualProcessOrganism(
                    organism_id=i,
                    position=position,
                    energy=100.0
                )
            else:
                raise ValueError(f"Unknown condition: {self.condition}")
            
            self.organisms.append(organism)
    
    def run(self) -> Dict[str, Any]:
        """Run complete simulation until termination.
        
        Termination conditions:
        - All organisms dead
        - Max timesteps reached
        
        Returns:
            Dictionary with final statistics and results
        """
        self.start_time = time.time()
        
        print(f"=== Starting Simulation ===")
        print(f"Condition: {self.condition}")
        print(f"Organisms: {self.num_organisms}")
        print(f"Grid size: {self.grid.grid_size}x{self.grid.grid_size}")
        print(f"Max timesteps: {self.max_timesteps}")
        print(f"Seed: {self.seed}")
        print()
        
        # Main simulation loop
        for t in range(self.max_timesteps):
            self.timestep = t
            
            # Execute one timestep
            self.step()
            
            # Check termination
            alive_count = sum(1 for org in self.organisms if org.alive)
            if alive_count == 0:
                print(f"\nAll organisms dead at timestep {t}")
                break
            
            # Progress update every 100 timesteps
            if t % 100 == 0 and t > 0:
                self._print_progress(alive_count)
        
        # Finalize
        elapsed_time = time.time() - self.start_time
        print(f"\n=== Simulation Complete ===")
        print(f"Duration: {elapsed_time:.2f} seconds")
        print(f"Final timestep: {self.timestep}")
        
        # Collect and save final statistics
        stats = self.get_statistics()
        self.logger.save_final_statistics(stats)
        
        return stats
    
    def step(self) -> None:
        """Execute one simulation timestep.
        
        Steps:
            1. Update environment (respawn food)
            2. Each alive organism decides action
            3. Execute actions (movement, eating, attacks)
            4. Apply energy decay
            5. Check for deaths
            6. Log timestep data
        """
        # Step 1: Update environment
        self.grid.step()
        
        # Prepare timestep log data
        timestep_data = {
            'timestep': self.timestep,
            'food_positions': self.grid.food_positions.copy(),
            'organisms': []
        }
        
        # Step 2-3: Each organism acts
        for organism in self.organisms:
            if not organism.alive:
                continue
            
            # Build state for organism
            state = self._build_organism_state(organism)
            
            # Organism decides action
            action = organism.decide(state)
            
            # Execute action
            self._execute_action(organism, action)
            
            # Log organism state
            org_data = {
                'organism_id': organism.organism_id,
                'position': organism.position,
                'energy': organism.energy,
                'action': action.name,
                'alive': organism.alive
            }
            
            # Add dual-process specific data
            if self.condition == 'B':
                org_data['veto_count'] = organism.veto_count
                org_data['approval_count'] = organism.approval_count
            
            timestep_data['organisms'].append(org_data)
        
        # Step 4: Apply energy decay
        for organism in self.organisms:
            if organism.alive:
                organism.update_energy(-self.energy_decay_rate)
                organism.increment_age()
                
                # Check for death
                if not organism.alive and organism.organism_id not in [d[0] for d in self.death_times]:
                    self.death_times.append((organism.organism_id, self.timestep))
        
        # Step 6: Log timestep
        self.logger.log_timestep(timestep_data)
    
    def _build_organism_state(self, organism: Organism) -> Dict[str, Any]:
        """Build environment state visible to organism.
        
        Args:
            organism: Organism requesting state
        
        Returns:
            State dictionary with keys:
                - food_direction: (dx, dy) or None
                - energy: current energy level
                - obstacles_nearby: List of (dx, dy) for obstacles
                - other_organism: (dx, dy, energy) or None
                - food_at_position: bool
                - grid_size: int
        """
        ox, oy = organism.position
        
        # Find nearest food
        food_direction = None
        min_distance = float('inf')
        
        for fx, fy in self.grid.food_positions:
            dx = fx - ox
            dy = fy - oy
            distance = abs(dx) + abs(dy)
            
            if distance < min_distance:
                min_distance = distance
                food_direction = (np.sign(dx), np.sign(dy))
        
        # Check if food at current position
        food_at_position = (ox, oy) in self.grid.food_positions
        
        # Find nearest other organism
        other_organism = None
        min_org_distance = float('inf')
        
        for other in self.organisms:
            if other.organism_id == organism.organism_id or not other.alive:
                continue
            
            other_x, other_y = other.position
            dx = other_x - ox
            dy = other_y - oy
            distance = abs(dx) + abs(dy)
            
            if distance < min_org_distance:
                min_org_distance = distance
                other_organism = (dx, dy, other.energy)
        
        # Build state
        state = {
            'food_direction': food_direction,
            'energy': organism.energy,
            'obstacles_nearby': [],  # No obstacles in Phase 1
            'other_organism': other_organism,
            'food_at_position': food_at_position,
            'grid_size': self.grid.grid_size
        }
        
        return state
    
    def _execute_action(self, organism: Organism, action: Action) -> None:
        """Execute organism action.
        
        Args:
            organism: Organism performing action
            action: Action to execute
        """
        if action == Action.EAT:
            # Try to consume food at current position
            if self.grid.consume_food(organism.position):
                organism.update_energy(self.food_energy)
        
        elif Action.is_movement(action):
            # Move organism
            dx, dy = Action.get_direction_vector(action)
            new_x = (organism.position[0] + dx) % self.grid.grid_size
            new_y = (organism.position[1] + dy) % self.grid.grid_size
            organism.move((new_x, new_y))
        
        elif action == Action.ATTACK:
            # Find organism at adjacent position
            # In Phase 1, attacks are logged but not implemented
            pass
        
        elif action == Action.WAIT:
            # Do nothing
            pass
    
    def _print_progress(self, alive_count: int) -> None:
        """Print progress update.
        
        Args:
            alive_count: Number of living organisms
        """
        avg_energy = np.mean([org.energy for org in self.organisms if org.alive])
        
        print(f"Timestep {self.timestep}/{self.max_timesteps} | "
              f"Alive: {alive_count}/{self.num_organisms} | "
              f"Avg Energy: {avg_energy:.1f} | "
              f"Food: {len(self.grid.food_positions)}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive simulation statistics.
        
        Returns:
            Dictionary with statistics
        """
        alive_organisms = [org for org in self.organisms if org.alive]
        dead_organisms = [org for org in self.organisms if not org.alive]
        
        stats = {
            'condition': self.condition,
            'seed': self.seed,
            'config': self.config,
            'final_timestep': self.timestep,
            'elapsed_time': time.time() - self.start_time if self.start_time else 0,
            'organisms': {
                'total': self.num_organisms,
                'alive': len(alive_organisms),
                'dead': len(dead_organisms),
                'survival_times': [age for age in [org.age for org in dead_organisms]],
                'avg_survival_time': np.mean([org.age for org in dead_organisms]) if dead_organisms else self.timestep
            },
            'energy': {
                'final_avg': np.mean([org.energy for org in alive_organisms]) if alive_organisms else 0,
                'final_std': np.std([org.energy for org in alive_organisms]) if alive_organisms else 0
            },
            'environment': {
                'final_food_count': len(self.grid.food_positions)
            }
        }
        
        # Add condition-specific statistics
        if self.condition == 'B':
            dual_organisms = [org for org in self.organisms if hasattr(org, 'veto_count')]
            stats['dual_process'] = {
                'total_vetoes': sum(org.veto_count for org in dual_organisms),
                'total_approvals': sum(org.approval_count for org in dual_organisms),
                'avg_veto_rate': np.mean([
                    org.veto_count / (org.veto_count + org.approval_count)
                    if (org.veto_count + org.approval_count) > 0 else 0
                    for org in dual_organisms
                ])
            }
        
        return stats
    
    def save_results(self, output_path: Path) -> None:
        """Save simulation results to file.
        
        Args:
            output_path: Path to save results
        """
        stats = self.get_statistics()
        self.logger.save_statistics(stats, output_path)
    
    def __repr__(self) -> str:
        """String representation."""
        alive_count = sum(1 for org in self.organisms if org.alive)
        return (
            f"SimulationRunner("
            f"condition={self.condition}, "
            f"t={self.timestep}/{self.max_timesteps}, "
            f"alive={alive_count}/{self.num_organisms})"
        )
