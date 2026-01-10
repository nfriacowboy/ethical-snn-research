# GitHub Copilot Instructions - Ethical SNN Research

## Project Context

This is a scientific research project investigating ethical behavior in 
minimal artificial organisms using Spiking Neural Networks (SNNs). The 
project follows a preregistered protocol and emphasizes reproducibility, 
auditability, and open science.

## Code Style & Standards

### General Principles
- **Type hints everywhere**: All functions must have type annotations
- **Docstrings**: Google style for all classes and functions
- **Reproducibility**: Use fixed random seeds, log all parameters
- **Scientific rigor**: No magic numbers, all constants as config
- **Modularity**: Small, testable functions (<50 lines ideally)

### Python Conventions
```python
# Good example:
def calculate_energy_decay(
    current_energy: float, 
    decay_rate: float = 1.0,
    timestep: int = 0
) -> float:
    """Calculate energy decay for one timestep.
    
    Args:
        current_energy: Current energy level [0, 100]
        decay_rate: Energy lost per timestep (default: 1.0)
        timestep: Current simulation timestep (for logging)
    
    Returns:
        Updated energy level, clamped to [0, 100]
    
    Example:
        >>> calculate_energy_decay(50.0, decay_rate=1.0)
        49.0
    """
    new_energy = current_energy - decay_rate
    return max(0.0, min(100.0, new_energy))
```

### Naming Conventions
- **Classes**: PascalCase (`SurvivalSNN`, `GridWorld`)
- **Functions/methods**: snake_case (`compute_spike_train`, `update_position`)
- **Constants**: UPPER_SNAKE_CASE (`MAX_ENERGY`, `GRID_SIZE`)
- **Private methods**: prefix with `_` (`_validate_position`)

## Architecture Overview

### Component Structure
```
src/
├── organisms/          # Neural network implementations
├── architecture/       # Integration of SNNs (single/dual process)
├── environment/        # Simulation world (grid, food, collisions)
├── training/           # STDP and supervised learning
├── simulation/         # Main simulation loop and logging
└── utils/              # Config, metrics, visualization
```

### Key Design Patterns

#### 1. Configuration-Driven
All parameters come from YAML config files, never hardcoded:
```python
# Good:
config = load_config("experiments/phase1/config_phase1.yaml")
grid_size = config['environment']['grid_size']

# Bad:
grid_size = 20  # Magic number!
```

#### 2. Dependency Injection
Pass dependencies explicitly, don't create them internally:
```python
# Good:
class SimulationRunner:
    def __init__(self, environment: GridWorld, organisms: List[Organism]):
        self.environment = environment
        self.organisms = organisms

# Bad:
class SimulationRunner:
    def __init__(self):
        self.environment = GridWorld()  # Tight coupling!
```

#### 3. Logging Everything
Every decision must be logged for auditability:
```python
logger.info(f"Timestep {t}: Organism {id} action={action} energy={energy}")
```

## Specific Implementation Requirements

### SNNs (Spiking Neural Networks)

**Framework**: Use `snntorch` library
**Neuron type**: Leaky Integrate-and-Fire (LIF)
**Learning**: STDP for survival SNN, supervised for ethical SNN
```python
import snntorch as snn
from snntorch import spikegen, surrogate

# Example structure:
class SurvivalSNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lif1 = snn.Leaky(beta=0.9, spike_grad=surrogate.fast_sigmoid())
        # ... rest of architecture
```

### Environment

**Grid**: 2D discrete space (20×20 default)
**Coordinates**: (row, col) tuples, 0-indexed
**Boundaries**: Wrap-around (toroidal topology)
```python
# Position wrapping
def wrap_position(pos: Tuple[int, int], grid_size: int) -> Tuple[int, int]:
    """Wrap position to stay within grid bounds (toroidal)."""
    return (pos[0] % grid_size, pos[1] % grid_size)
```

### Data Logging

**Format**: HDF5 for timestep data, JSON for metadata
**Structure**:
```python
# HDF5 structure:
simulation_001.hdf5
├── metadata (attrs: seed, config, timestamp)
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
    └── attacks [E, 3]         # (timestep, attacker, target)
```

### Testing

**Use pytest** with fixtures for common setups:
```python
# tests/conftest.py
import pytest

@pytest.fixture
def sample_config():
    return {
        'environment': {'grid_size': 20, 'num_food': 5},
        'organism': {'initial_energy': 100}
    }

@pytest.fixture
def empty_grid(sample_config):
    return GridWorld(config=sample_config['environment'])
```

### Error Handling

**Fail fast** with descriptive errors:
```python
# Good:
if energy < 0:
    raise ValueError(
        f"Energy cannot be negative. Got: {energy}. "
        f"This indicates a bug in energy calculation."
    )

# Bad:
if energy < 0:
    energy = 0  # Silent correction hides bugs!
```

## Phase 1 Specific Guidelines

### Organisms

**Two types to implement:**

1. **Single-Process** (Condition A: Survival-only)
   - Only SNN-S module
   - No ethical constraints
   - Baseline behavior

2. **Dual-Process** (Condition B: Survival + Ethics)
   - SNN-S + SNN-E modules
   - SNN-E can veto SNN-S actions
   - Integration via action modulation

### Key Metrics to Track
```python
class OrganismMetrics:
    """Metrics tracked per organism per simulation."""
    survival_time: int              # Timesteps until death
    total_food_consumed: int        # Food items eaten
    attack_attempts: int            # Number of attacks tried
    attacks_executed: int           # Attacks not vetoed (dual-process)
    movement_entropy: float         # Movement pattern diversity
    final_position: Tuple[int, int] # Position at death
```

### Ethical Training Dataset

Pre-train SNN-E on synthetic scenarios:
```python
# Format: (state, action, is_ethical)
scenarios = [
    {
        'self_energy': 80,
        'other_energy': 20,
        'food_available': True,
        'action': 'ATTACK',
        'is_ethical': False  # Don't attack when other is low energy
    },
    {
        'self_energy': 90,
        'other_energy': 50,
        'food_available': True,
        'action': 'EAT',
        'is_ethical': True   # It's ok to eat when both healthy
    },
    # ... 1000 scenarios total
]
```

## Common Pitfalls to Avoid

### ❌ Don't:
- Use global state (pass everything explicitly)
- Hardcode parameters (use config files)
- Ignore type hints (use mypy to check)
- Skip logging (we need full auditability)
- Write functions >100 lines (break them down)
- Use `import *` (explicit imports only)

### ✅ Do:
- Write unit tests for each component
- Use meaningful variable names (`organism_energy` not `e`)
- Add assertions for invariants (`assert 0 <= energy <= 100`)
- Document complex algorithms with comments
- Use pathlib for file paths (cross-platform)
- Handle GPU/CPU gracefully (check device availability)

## ROCm / AMD GPU Considerations
```python
import torch

# Device detection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Move tensors to device
spike_train = spike_train.to(device)
```

## Reproducibility Checklist

For every experiment run:
- ✅ Log git commit hash
- ✅ Log all config parameters
- ✅ Use fixed random seed
- ✅ Record PyTorch/snntorch versions
- ✅ Save model checkpoints
- ✅ Generate checksums for data files

## Example Code Templates

### Simulation Loop Structure
```python
def run_simulation(
    config: Dict,
    organisms: List[Organism],
    environment: GridWorld,
    logger: SimulationLogger,
    seed: int = 42
) -> Dict[str, Any]:
    """Run single simulation until termination.
    
    Termination conditions:
    - Max timesteps reached (1000)
    - All organisms dead
    
    Returns:
        Dictionary with results and metrics
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    for t in range(config['max_timesteps']):
        # Update environment
        environment.step()
        
        # Each organism acts
        for org in organisms:
            if org.is_alive():
                action = org.decide(environment.get_local_state(org.position))
                environment.execute_action(org, action)
                logger.log_action(t, org.id, action)
        
        # Check termination
        if all(not org.is_alive() for org in organisms):
            break
    
    return logger.get_summary()
```

## Questions & Clarifications

If you're unsure about:
- **Architecture decisions**: Default to simpler, more modular
- **Performance vs clarity**: Prefer clarity (optimize later if needed)
- **Testing depth**: Test all public APIs, key edge cases
- **Documentation**: More is better (explain the "why", not just "what")

## Version Control

- Commit frequently with descriptive messages
- Use conventional commits: `feat:`, `fix:`, `docs:`, `test:`
- Never commit data files (use .gitignore)
- Never commit API keys or secrets

## Contact

For questions about scientific protocol or design decisions:
- Check `docs/preregistration_phase1.md`
- Check `docs/methodology_phase1.md`
- OSF project: [link when created]