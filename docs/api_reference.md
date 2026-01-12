# API Reference - Ethical SNN Research

## Table of Contents

1. [Core Components](#core-components)
2. [Neural Networks](#neural-networks)
3. [Environment](#environment)
4. [Simulation](#simulation)
5. [Analysis](#analysis)
6. [Utilities](#utilities)

## Core Components

### SimulationRunner

Main class for running simulations.

```python
from src.simulation.runner import SimulationRunner

runner = SimulationRunner(config: Dict[str, Any], seed: int = 42)
```

#### Parameters

- **config** (dict): Configuration dictionary containing:
  - `condition` (str): 'A' (survival-only) or 'B' (dual-process)
  - `grid_size` (int): Size of square grid (default: 20)
  - `num_organisms` (int): Number of organisms (default: 10)
  - `num_food` (int): Number of food items (default: 10)
  - `max_timesteps` (int): Maximum simulation duration (default: 1000)
  - `energy_decay_rate` (float): Energy lost per timestep (default: 1.0)
  - `food_energy` (float): Energy gained from eating (default: 20.0)
  - `food_respawn_rate` (float): Food respawn probability (default: 0.1)

- **seed** (int): Random seed for reproducibility

#### Methods

##### run()

Execute the simulation until termination.

```python
stats = runner.run() -> Dict[str, Any]
```

**Returns**: Statistics dictionary containing:
- `condition`: 'A' or 'B'
- `seed`: Random seed used
- `final_timestep`: When simulation ended
- `elapsed_time`: Runtime in seconds
- `organisms`: Organism statistics
- `energy`: Energy statistics
- `environment`: Environment statistics
- `dual_process`: Veto statistics (Condition B only)

#### Example

```python
config = {
    'condition': 'B',
    'grid_size': 20,
    'num_organisms': 10,
    'num_food': 10,
    'max_timesteps': 1000,
    'energy_decay_rate': 1.0,
    'food_energy': 20.0,
    'food_respawn_rate': 0.1
}

runner = SimulationRunner(config, seed=42)
stats = runner.run()

print(f"Avg survival time: {stats['organisms']['avg_survival_time']:.1f} timesteps")
```

## Neural Networks

### SurvivalSNN

Spiking neural network for survival behaviors.

```python
from src.organisms.survival_snn import SurvivalSNN

network = SurvivalSNN(
    input_size: int,
    hidden_size: int = 30,
    output_size: int = 5,
    beta: float = 0.9
)
```

#### Parameters

- **input_size** (int): Size of input layer (typically 8: 4 vision + 4 proprioception)
- **hidden_size** (int): Size of hidden layer (default: 30)
- **output_size** (int): Number of actions (default: 5: MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, EAT)
- **beta** (float): LIF neuron decay parameter (default: 0.9)

#### Methods

##### forward(x)

Process input through network.

```python
output_spikes, _ = network.forward(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple]
```

**Parameters**:
- `x`: Input tensor of shape (batch_size, input_size)

**Returns**:
- `output_spikes`: Spike output tensor of shape (batch_size, output_size)
- `state`: Hidden state tuple (not typically used externally)

#### Example

```python
import torch

# Create network
network = SurvivalSNN(input_size=8, hidden_size=30)

# Create input: [food_up, food_down, food_left, food_right, 
#                pos_x, pos_y, energy, bias]
input_data = torch.tensor([
    [1.0, 0.0, 0.0, 0.5,  # Food vision
     0.5, 0.5, 0.8, 1.0]  # Proprioception
])

# Forward pass
output_spikes, _ = network.forward(input_data)

# Select action (highest spike count)
action = torch.argmax(output_spikes, dim=1).item()
actions = ['MOVE_UP', 'MOVE_DOWN', 'MOVE_LEFT', 'MOVE_RIGHT', 'EAT']
print(f"Action: {actions[action]}")
```

### EthicalSNN

Spiking neural network for ethical evaluation.

```python
from src.organisms.ethical_snn import EthicalSNN

network = EthicalSNN(
    input_size: int,
    hidden_size: int = 20,
    output_size: int = 2,
    beta: float = 0.9
)
```

#### Parameters

- **input_size** (int): Size of input layer (typically 10: state + action)
- **hidden_size** (int): Size of hidden layer (default: 20)
- **output_size** (int): Number of outputs (default: 2: APPROVE, VETO)
- **beta** (float): LIF neuron decay parameter (default: 0.9)

#### Methods

##### forward(x)

Evaluate action ethically.

```python
output_spikes, _ = network.forward(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple]
```

**Returns**:
- `output_spikes[0]`: Approval spike
- `output_spikes[1]`: Veto spike

#### Example

```python
import torch

# Create network
network = EthicalSNN(input_size=10, hidden_size=20)

# Create input: [state (8) + action_one_hot (5)]
# But only 10 total features (simplified)
input_data = torch.rand(1, 10)

# Evaluate action
output_spikes, _ = network.forward(input_data)

# Decision
decision = 'APPROVE' if output_spikes[0, 0] > output_spikes[0, 1] else 'VETO'
print(f"Ethical decision: {decision}")
```

## Environment

### GridWorld

2D toroidal grid environment.

```python
from src.environment.grid_world import GridWorld

env = GridWorld(
    grid_size: int = 20,
    num_food: int = 10,
    food_respawn_rate: float = 0.1,
    food_energy_value: float = 20.0,
    seed: int = 42
)
```

#### Methods

##### add_organism(organism, position)

Add organism to environment.

```python
env.add_organism(
    organism: BaseOrganism,
    position: Tuple[int, int]
) -> None
```

##### remove_organism(organism_id)

Remove organism from environment.

```python
env.remove_organism(organism_id: int) -> None
```

##### step()

Update environment (respawn food, update time).

```python
env.step() -> None
```

##### is_valid_position(position)

Check if position is within grid.

```python
is_valid = env.is_valid_position(position: Tuple[int, int]) -> bool
```

##### get_local_state(position)

Get sensory information at position.

```python
state = env.get_local_state(
    position: Tuple[int, int],
    energy: float,
    max_energy: float
) -> Dict[str, Any]
```

**Returns**: Dictionary with:
- `vision`: Food locations in 4 directions
- `proprioception`: Self state (position, energy)

#### Example

```python
from src.environment.grid_world import GridWorld

# Create environment
env = GridWorld(grid_size=20, num_food=10, seed=42)

# Check position
assert env.is_valid_position((5, 10))

# Get sensory state
state = env.get_local_state(
    position=(10, 10),
    energy=50.0,
    max_energy=100.0
)

print(f"Food visible: {state['vision']}")
print(f"Current energy: {state['proprioception']['energy']}")
```

## Simulation

### Batch Runner

```python
from experiments.phase1.batch_runner import run_condition

results = run_condition(
    condition: str,
    config_path: str,
    num_runs: int,
    start_seed: int = 42
) -> List[Dict[str, Any]]
```

#### Parameters

- **condition** (str): 'A' or 'B'
- **config_path** (str): Path to YAML config
- **num_runs** (int): Number of simulations
- **start_seed** (int): Starting random seed

**Returns**: List of statistics dictionaries

#### Example

```python
from experiments.phase1.batch_runner import run_condition, aggregate_statistics

# Run 10 simulations
results = run_condition(
    condition='A',
    config_path='experiments/phase1/config_phase1.yaml',
    num_runs=10,
    start_seed=42
)

# Aggregate statistics
aggregated = aggregate_statistics(results)

print(f"Mean survival time: {aggregated['avg_survival_time']['mean']:.1f}")
print(f"Std: {aggregated['avg_survival_time']['std']:.1f}")
```

## Analysis

### Statistical Tests

```python
from analysis.phase1.statistical_tests import mann_whitney_test, cohens_d

# Compare two groups
result = mann_whitney_test(
    group_a: np.ndarray,
    group_b: np.ndarray,
    metric_name: str,
    alpha: float = 0.05
) -> Dict[str, Any]
```

**Returns**: Dictionary with:
- `u_statistic`: Mann-Whitney U statistic
- `p_value`: Two-tailed p-value
- `significant`: Whether p < alpha
- `metric_name`: Name of metric tested

#### Example

```python
import numpy as np
from analysis.phase1.statistical_tests import mann_whitney_test, cohens_d

# Survival times from two conditions
condition_a = np.array([120, 95, 110, 88, 105])
condition_b = np.array([130, 105, 115, 98, 125])

# Statistical test
result = mann_whitney_test(condition_a, condition_b, 'survival_time')
print(f"P-value: {result['p_value']:.4f}")
print(f"Significant: {result['significant']}")

# Effect size
effect = cohens_d(condition_a, condition_b)
print(f"Cohen's d: {effect:.4f}")
```

### Visualization

```python
from analysis.phase1.visualization import plot_survival_comparison

plot_survival_comparison(
    data_a: List[float],
    data_b: List[float],
    metric: str,
    output_path: str
) -> None
```

#### Example

```python
from analysis.phase1.visualization import plot_survival_comparison

survival_a = [120, 95, 110, 88, 105]
survival_b = [130, 105, 115, 98, 125]

plot_survival_comparison(
    data_a=survival_a,
    data_b=survival_b,
    metric='Survival Time',
    output_path='results/comparison.png'
)
```

## Utilities

### Configuration Loading

```python
from src.utils.config import load_config

config = load_config(config_path: str) -> Dict[str, Any]
```

#### Example

```python
from src.utils.config import load_config

config = load_config('experiments/phase1/config_phase1.yaml')
print(f"Grid size: {config['environment']['grid_size']}")
```

### Metrics Computation

```python
from src.utils.metrics import compute_survival_metrics

metrics = compute_survival_metrics(
    organisms: List[BaseOrganism]
) -> Dict[str, Any]
```

**Returns**: Dictionary with:
- `total`: Total organism count
- `alive`: Currently alive count
- `dead`: Dead count
- `survival_times`: List of lifespans
- `avg_survival_time`: Mean survival time

#### Example

```python
from src.utils.metrics import compute_survival_metrics

# After simulation
metrics = compute_survival_metrics(organisms)

print(f"Average survival: {metrics['avg_survival_time']:.1f} timesteps")
print(f"Survivors: {metrics['alive']}/{metrics['total']}")
```

## Type Definitions

### Common Types

```python
from typing import Tuple, Dict, List, Any

# Position on grid
Position = Tuple[int, int]

# Action indices
MOVE_UP = 0
MOVE_DOWN = 1
MOVE_LEFT = 2
MOVE_RIGHT = 3
EAT = 4

# Configuration dictionary
Config = Dict[str, Any]

# Statistics dictionary
Stats = Dict[str, Any]
```

## Error Handling

### Common Exceptions

```python
# Invalid configuration
if config['condition'] not in ['A', 'B']:
    raise ValueError(f"Invalid condition: {config['condition']}")

# Invalid position
if not env.is_valid_position(position):
    raise ValueError(f"Position {position} out of bounds")

# Energy constraints
if energy < 0:
    raise ValueError(f"Energy cannot be negative: {energy}")
```

## Best Practices

### Reproducibility

Always set random seeds:
```python
import torch
import numpy as np
import random

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

### Device Management

Handle GPU/CPU gracefully:
```python
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
network = network.to(device)
input_tensor = input_tensor.to(device)
```

### Configuration Validation

Validate before running:
```python
def validate_config(config: Dict) -> None:
    assert config['grid_size'] > 0, "Grid size must be positive"
    assert config['num_organisms'] > 0, "Need at least one organism"
    assert 0 <= config['food_respawn_rate'] <= 1, "Invalid respawn rate"
    assert config['condition'] in ['A', 'B'], "Invalid condition"
```

## Further Reading

- [User Guide](user_guide.md) - Installation and usage
- [Preregistration](preregistration_phase1.md) - Study protocol
- [Methodology](methodology_phase1.md) - Scientific methods
