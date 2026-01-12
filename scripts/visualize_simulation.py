#!/usr/bin/env python3
"""
Animated Visualization Script - Ethical SNN Research
=====================================================

Creates an animated visualization of a simulation run, showing organism
movement, food consumption, and energy levels over time.

Usage:
    uv run python scripts/visualize_simulation.py
    uv run python scripts/visualize_simulation.py --condition B --seed 100
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle
from typing import Dict, List, Tuple, Any

from src.simulation.runner import SimulationRunner
from src.utils.config import load_config


class SimulationAnimator:
    """Creates animated visualizations of simulations."""
    
    def __init__(self, grid_size: int):
        """Initialize the animator.
        
        Args:
            grid_size: Size of the simulation grid
        """
        self.grid_size = grid_size
        self.history: List[Dict[str, Any]] = []
        
    def record_frame(
        self,
        organisms: List[Any],
        food_positions: List[Tuple[int, int]],
        timestep: int
    ) -> None:
        """Record a single frame of the simulation.
        
        Args:
            organisms: List of organism objects
            food_positions: List of food positions
            timestep: Current timestep
        """
        frame = {
            'timestep': timestep,
            'organisms': [],
            'food_positions': list(food_positions)
        }
        
        for org in organisms:
            frame['organisms'].append({
                'position': org.position,
                'energy': org.energy,
                'is_alive': org.is_alive(),
                'id': org.id
            })
        
        self.history.append(frame)
    
    def create_animation(
        self,
        output_path: str = 'simulation_animation.mp4',
        fps: int = 10,
        interval: int = 1
    ) -> None:
        """Create animation from recorded history.
        
        Args:
            output_path: Path to save animation
            fps: Frames per second
            interval: Show every Nth frame (1 = all frames)
        """
        if not self.history:
            print("No frames recorded!")
            return
        
        # Filter frames
        frames = self.history[::interval]
        
        # Setup figure
        fig, (ax_main, ax_energy) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Simulation Visualization', fontsize=14, fontweight='bold')
        
        # Setup main grid axis
        ax_main.set_xlim(-0.5, self.grid_size - 0.5)
        ax_main.set_ylim(-0.5, self.grid_size - 0.5)
        ax_main.set_aspect('equal')
        ax_main.grid(True, alpha=0.3)
        ax_main.set_xlabel('X Position')
        ax_main.set_ylabel('Y Position')
        ax_main.invert_yaxis()  # Match array indexing
        
        # Setup energy axis
        ax_energy.set_xlim(0, len(frames))
        ax_energy.set_ylim(0, 100)
        ax_energy.set_xlabel('Timestep')
        ax_energy.set_ylabel('Energy Level')
        ax_energy.grid(True, alpha=0.3)
        ax_energy.set_title('Organism Energy Over Time')
        
        # Initialize plot elements
        food_scatter = ax_main.scatter([], [], c='green', marker='s', s=100, label='Food')
        organism_scatter = ax_main.scatter([], [], c=[], cmap='plasma', s=150, 
                                          edgecolors='black', linewidths=2, label='Organisms')
        timestamp_text = ax_main.text(0.02, 0.98, '', transform=ax_main.transAxes,
                                      verticalalignment='top', fontsize=10,
                                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Energy lines (one per organism)
        num_organisms = len(frames[0]['organisms'])
        energy_lines = []
        colors = plt.cm.tab10(np.linspace(0, 1, num_organisms))
        
        for i in range(num_organisms):
            line, = ax_energy.plot([], [], '-o', color=colors[i], label=f'Organism {i}', 
                                  markersize=4, alpha=0.7)
            energy_lines.append(line)
        
        ax_main.legend(loc='upper right')
        ax_energy.legend(loc='upper right', fontsize=8)
        
        def init():
            """Initialize animation."""
            food_scatter.set_offsets(np.empty((0, 2)))
            organism_scatter.set_offsets(np.empty((0, 2)))
            timestamp_text.set_text('')
            for line in energy_lines:
                line.set_data([], [])
            return [food_scatter, organism_scatter, timestamp_text] + energy_lines
        
        def animate(frame_idx):
            """Update animation for frame."""
            frame = frames[frame_idx]
            
            # Update food positions
            if frame['food_positions']:
                food_array = np.array(frame['food_positions'])
                # Swap x,y for display (row,col) -> (x,y)
                food_scatter.set_offsets(np.c_[food_array[:, 1], food_array[:, 0]])
            else:
                food_scatter.set_offsets(np.empty((0, 2)))
            
            # Update organisms
            organism_positions = []
            organism_energies = []
            
            for org in frame['organisms']:
                if org['is_alive']:
                    # Swap x,y for display
                    organism_positions.append([org['position'][1], org['position'][0]])
                    organism_energies.append(org['energy'])
            
            if organism_positions:
                organism_scatter.set_offsets(np.array(organism_positions))
                organism_scatter.set_array(np.array(organism_energies))
            else:
                organism_scatter.set_offsets(np.empty((0, 2)))
            
            # Update timestamp
            alive_count = sum(1 for org in frame['organisms'] if org['is_alive'])
            timestamp_text.set_text(
                f"Timestep: {frame['timestep']}\n"
                f"Alive: {alive_count}/{len(frame['organisms'])}"
            )
            
            # Update energy lines
            for org_id, line in enumerate(energy_lines):
                timesteps = []
                energies = []
                
                for f in frames[:frame_idx + 1]:
                    if org_id < len(f['organisms']):
                        org = f['organisms'][org_id]
                        if org['is_alive']:
                            timesteps.append(f['timestep'])
                            energies.append(org['energy'])
                
                if timesteps:
                    line.set_data(timesteps, energies)
                else:
                    line.set_data([], [])
            
            return [food_scatter, organism_scatter, timestamp_text] + energy_lines
        
        # Create animation
        print(f"Creating animation with {len(frames)} frames...")
        anim = animation.FuncAnimation(
            fig, animate, init_func=init,
            frames=len(frames), interval=1000//fps, blit=True
        )
        
        # Save animation
        print(f"Saving to {output_path}...")
        try:
            anim.save(output_path, writer='ffmpeg', fps=fps, dpi=100)
            print(f"✅ Animation saved to: {output_path}")
        except Exception as e:
            print(f"⚠️  Could not save video (ffmpeg not available?): {e}")
            print("Saving as GIF instead...")
            gif_path = output_path.replace('.mp4', '.gif')
            anim.save(gif_path, writer='pillow', fps=fps)
            print(f"✅ Animation saved as GIF: {gif_path}")
        
        plt.close(fig)


def run_with_recording(config: Dict[str, Any], seed: int = 42) -> Tuple[Dict, SimulationAnimator]:
    """Run simulation with frame recording.
    
    Args:
        config: Simulation configuration
        seed: Random seed
        
    Returns:
        Tuple of (statistics, animator)
    """
    from src.architecture.single_process import SingleProcessOrganism
    from src.architecture.dual_process import DualProcessOrganism
    from src.environment.grid_world import GridWorld
    import torch
    
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create environment
    env = GridWorld(
        grid_size=config['grid_size'],
        num_food=config['num_food'],
        food_respawn_rate=config['food_respawn_rate'],
        food_energy_value=config['food_energy'],
        seed=seed
    )
    
    # Create organisms
    organisms = []
    organism_class = (DualProcessOrganism if config['condition'] == 'B' 
                     else SingleProcessOrganism)
    
    for i in range(config['num_organisms']):
        position = (
            np.random.randint(0, config['grid_size']),
            np.random.randint(0, config['grid_size'])
        )
        
        org = organism_class(
            organism_id=i,
            initial_energy=100.0,
            max_energy=100.0,
            position=position
        )
        env.add_organism(org, position)
        organisms.append(org)
    
    # Create animator
    animator = SimulationAnimator(config['grid_size'])
    
    # Run simulation with recording
    print(f"\n=== Recording Simulation ===")
    print(f"Condition: {config['condition']}")
    print(f"Organisms: {config['num_organisms']}")
    print(f"Grid size: {config['grid_size']}x{config['grid_size']}")
    print(f"Seed: {seed}\n")
    
    for t in range(config['max_timesteps']):
        # Record frame every 5 timesteps to reduce size
        if t % 5 == 0:
            animator.record_frame(organisms, env.food_manager.food_positions, t)
        
        # Environment step
        env.step()
        
        # Organism actions
        for org in organisms:
            if org.is_alive():
                # Get state
                local_state = env.get_local_state(org.position, org.energy, org.max_energy)
                
                # Decide action
                action = org.decide(local_state)
                
                # Execute action
                if action < 4:  # Movement
                    moves = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # UP, DOWN, LEFT, RIGHT
                    dy, dx = moves[action]
                    new_pos = ((org.position[0] + dy) % config['grid_size'],
                              (org.position[1] + dx) % config['grid_size'])
                    env.move_organism(org.id, new_pos)
                elif action == 4:  # EAT
                    if org.position in env.food_manager.food_positions:
                        org.consume_food(config['food_energy'])
                        env.food_manager.remove_food(org.position)
                
                # Energy decay
                org.energy -= config['energy_decay_rate']
                if org.energy <= 0:
                    org.die(t)
                    env.remove_organism(org.id)
        
        # Check termination
        if all(not org.is_alive() for org in organisms):
            print(f"All organisms dead at timestep {t}")
            break
        
        # Progress update
        if (t + 1) % 100 == 0:
            alive = sum(1 for org in organisms if org.is_alive())
            print(f"Timestep {t+1}/{config['max_timesteps']} | Alive: {alive}/{len(organisms)}")
    
    # Record final frame
    animator.record_frame(organisms, env.food_manager.food_positions, t)
    
    # Compute statistics
    stats = {
        'condition': config['condition'],
        'seed': seed,
        'final_timestep': t,
        'organisms': {
            'total': len(organisms),
            'alive': sum(1 for org in organisms if org.is_alive()),
            'avg_survival_time': np.mean([org.death_time if org.death_time else t 
                                         for org in organisms])
        }
    }
    
    return stats, animator


def main():
    """Run the visualization."""
    parser = argparse.ArgumentParser(description='Visualize simulation runs')
    parser.add_argument('--condition', type=str, default='A', choices=['A', 'B'],
                       help='Simulation condition (A or B)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path (default: auto-generated)')
    parser.add_argument('--fps', type=int, default=10,
                       help='Frames per second (default: 10)')
    
    args = parser.parse_args()
    
    # Load config
    config_path = Path(__file__).parent.parent / 'experiments' / 'phase1' / 'config_phase1.yaml'
    base_config = load_config(str(config_path))
    
    # Create simulation config
    config = {
        'condition': args.condition,
        'grid_size': base_config['environment']['grid_size'],
        'num_organisms': base_config['simulation']['num_organisms'],
        'num_food': base_config['environment']['num_food'],
        'max_timesteps': 500,  # Shorter for visualization
        'energy_decay_rate': base_config['organism']['energy_decay_rate'],
        'food_energy': base_config['environment']['food_energy_value'],
        'food_respawn_rate': base_config['environment']['food_respawn_rate']
    }
    
    # Run with recording
    stats, animator = run_with_recording(config, seed=args.seed)
    
    print(f"\n=== Simulation Complete ===")
    print(f"Final timestep: {stats['final_timestep']}")
    print(f"Average survival: {stats['organisms']['avg_survival_time']:.1f} timesteps")
    
    # Create animation
    if args.output is None:
        output_dir = Path(__file__).parent.parent / 'results' / 'visualizations'
        output_dir.mkdir(parents=True, exist_ok=True)
        args.output = str(output_dir / f"simulation_condition_{args.condition}_seed_{args.seed}.mp4")
    
    animator.create_animation(args.output, fps=args.fps, interval=1)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nVisualization interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
