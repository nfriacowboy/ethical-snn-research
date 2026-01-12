# User Guide - Ethical SNN Research

## Table of Contents

1. [Installation](#installation)
2. [Running Simulations](#running-simulations)
3. [Analyzing Results](#analyzing-results)
4. [Configuration](#configuration)
5. [Understanding the Output](#understanding-the-output)
6. [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites

- Python 3.10+
- AMD ROCm (optional, for GPU acceleration)
- 8GB+ RAM recommended

### Setup with uv (Recommended)

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone repository
git clone https://github.com/nfriacowboy/ethical-snn-research
cd ethical-snn-research

# Install dependencies
uv sync

# Verify installation
uv run python -c "import torch; print(f'PyTorch: {torch.__version__}')"
uv run python -c "import snntorch; print(f'snnTorch: {snntorch.__version__}')"
```

### Verify GPU Support

```bash
uv run python scripts/verify_gpu.py
```

Expected output:
```
✓ PyTorch installed: 2.9.1
✓ GPU available: True
✓ Device: cuda
✓ GPU Name: AMD Radeon...
```

## Running Simulations

### Single Simulation

#### Condition A: Survival-Only

```bash
uv run python experiments/phase1/run_survival_only.py
```

This runs a single simulation with organisms that only have the survival neural network (SNN-S).

#### Condition B: Dual-Process (Survival + Ethics)

```bash
uv run python experiments/phase1/run_dual_process.py
```

This runs a simulation with organisms that have both survival (SNN-S) and ethical (SNN-E) networks.

### Batch Experiments

For statistically meaningful results, run multiple simulations:

```bash
# Test run (2 simulations per condition)
uv run python experiments/phase1/batch_runner.py --num_runs 2 --condition both

# Full experiment (100 simulations per condition)
uv run python experiments/phase1/batch_runner.py --num_runs 100 --condition both

# Run only one condition
uv run python experiments/phase1/batch_runner.py --num_runs 50 --condition A
```

#### Batch Runner Options

- `--num_runs`: Number of runs per condition (default: 100)
- `--condition`: Which condition to run - 'A', 'B', or 'both' (default: both)
- `--start_seed`: Starting random seed (default: 42)
- `--config`: Path to config file (default: experiments/phase1/config_phase1.yaml)
- `--output_dir`: Where to save results (default: results/batch_experiments)

Example:
```bash
uv run python experiments/phase1/batch_runner.py \
    --num_runs 50 \
    --condition both \
    --start_seed 42 \
    --output_dir results/my_experiment
```

### Expected Runtime

- **Single simulation**: ~1 second (10 organisms, 1000 timesteps)
- **100 runs per condition**: ~3-5 minutes
- Runtime scales linearly with number of organisms and timesteps

## Analyzing Results

### Automatic Analysis

After running batch experiments, analyze the results:

```bash
uv run python analysis/phase1/analyze_results.py \
    --input results/batch_experiments \
    --output results/analysis_phase1
```

This generates:
- Statistical tests (Mann-Whitney U, Cohen's d)
- Comparison plots (survival time, alive count, energy)
- Analysis report (JSON format)

### Output Files

Analysis creates:
```
results/analysis_phase1/
├── analysis_report.json              # Statistical test results
├── survival_time_comparison.png      # Survival time plots
├── alive_count_comparison.png        # Final alive count plots
├── energy_comparison.png             # Final energy plots
├── all_metrics_overview.png          # Combined metrics
└── veto_rate_distribution.png        # Ethical veto rates (Condition B)
```

### Custom Analysis

For custom analysis, use the Python API:

```python
from analysis.phase1.analyze_results import load_batch_results, extract_metrics_dataframe
from analysis.phase1.statistical_tests import mann_whitney_test, cohens_d

# Load results
data = load_batch_results('results/batch_experiments')

# Extract metrics
df_a = extract_metrics_dataframe(data['condition_a'], 'A')
df_b = extract_metrics_dataframe(data['condition_b'], 'B')

# Perform tests
result = mann_whitney_test(
    df_a['avg_survival_time'].values,
    df_b['avg_survival_time'].values,
    metric_name='survival_time'
)

print(f"P-value: {result['p_value']:.4f}")
print(f"Cohen's d: {cohens_d(df_a['avg_survival_time'], df_b['avg_survival_time']):.4f}")
```

## Configuration

### Main Configuration File

`experiments/phase1/config_phase1.yaml`:

```yaml
environment:
  grid_size: 20              # Size of square grid
  num_food: 10               # Number of food items
  food_respawn_rate: 0.1     # Probability of food respawning per timestep
  food_energy_value: 20.0    # Energy gained from eating

organism:
  initial_energy: 100.0      # Starting energy level
  max_energy: 100.0          # Maximum energy capacity
  energy_decay_rate: 1.0     # Energy lost per timestep

simulation:
  max_timesteps: 1000        # Maximum simulation duration
  num_organisms: 10          # Number of organisms per simulation

neural_network:
  survival_hidden_size: 30   # Hidden layer size for SNN-S
  ethical_hidden_size: 20    # Hidden layer size for SNN-E
  lif_beta: 0.9              # LIF neuron decay rate
```

### Programmatic Configuration

```python
from src.simulation.runner import SimulationRunner

config = {
    'condition': 'B',           # 'A' or 'B'
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

print(f"Final timestep: {stats['final_timestep']}")
print(f"Organisms alive: {stats['organisms']['alive']}")
```

## Understanding the Output

### Simulation Statistics

Each simulation produces a statistics dictionary:

```python
{
    'condition': 'B',                # 'A' or 'B'
    'seed': 42,                      # Random seed used
    'final_timestep': 150,           # When simulation ended
    'elapsed_time': 0.82,            # Runtime in seconds
    
    'organisms': {
        'total': 10,                 # Initial organism count
        'alive': 0,                  # Final alive count
        'dead': 10,                  # Final dead count
        'survival_times': [120, 95, ...],  # Lifespan of each organism
        'avg_survival_time': 108.5   # Average survival time
    },
    
    'energy': {
        'final_avg': 0.0,            # Average energy of alive organisms
        'final_std': 0.0             # Standard deviation
    },
    
    'environment': {
        'final_food_count': 8        # Food items remaining
    },
    
    'dual_process': {                # Only for Condition B
        'total_vetoes': 15,          # Actions vetoed by ethical network
        'total_approvals': 135,      # Actions approved
        'avg_veto_rate': 0.1         # Proportion of vetoes (15/150)
    }
}
```

### Interpreting Results

#### Survival Time
- **Higher is better** - indicates longer survival
- Expected: 50-150 timesteps with default settings
- Affected by: energy decay rate, food availability, organism behavior

#### Veto Rate (Condition B only)
- **Proportion of actions vetoed** by ethical network
- Range: 0.0 to 1.0
- Expected: 0.05-0.20 (5-20% of actions vetoed)
- Higher rate indicates more ethical constraint

#### Final Alive Count
- **Number of organisms surviving** to max_timesteps
- Expected: 0-5 with default settings (most organisms die)
- Higher count indicates better survival strategies

## Troubleshooting

### Common Issues

#### 1. Import Errors

```
ImportError: cannot import name 'SimulationRunner'
```

**Solution**: Make sure you're in the repository root and using `uv run`:
```bash
cd ethical-snn-research
uv run python experiments/phase1/batch_runner.py
```

#### 2. GPU Not Detected

```
GPU available: False
```

**Solution**: 
- Verify ROCm installation: `rocm-smi`
- Check PyTorch ROCm support: `uv run python -c "import torch; print(torch.version.hip)"`
- CPU-only mode works fine but is slower

#### 3. Memory Issues

```
RuntimeError: CUDA out of memory
```

**Solution**:
- Reduce `num_organisms` in config
- Reduce `grid_size`
- Run fewer simulations in parallel

#### 4. Slow Performance

**Optimization tips**:
- Use GPU if available
- Reduce `max_timesteps` for testing
- Run batch experiments with `--num_runs` appropriate for your hardware

### Getting Help

1. Check existing [GitHub Issues](https://github.com/nfriacowboy/ethical-snn-research/issues)
2. Run tests to verify installation: `uv run pytest`
3. Enable verbose logging: Add `--verbose` flag to commands

## Next Steps

- **Run batch experiments**: Generate data for analysis
- **Analyze results**: Compare Condition A vs B
- **Explore notebooks**: Interactive visualization in `notebooks/`
- **Modify configuration**: Experiment with different parameters
- **Read preregistration**: See `docs/preregistration_phase1.md` for study protocol
