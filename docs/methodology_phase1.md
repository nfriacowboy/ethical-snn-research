# Phase 1 Methodology

## Technical Implementation

### Spiking Neural Networks

#### SNN-S (Survival Network)
- **Architecture**: 3-layer recurrent SNN
- **Input layer**: 128 neurons (sensory encoding)
- **Hidden layer**: 256 neurons (LIF model)
- **Output layer**: 8 neurons (action selection)
- **Learning**: STDP (unsupervised)
- **Parameters**:
  - Membrane time constant: τ_m = 20ms
  - Refractory period: 5ms
  - Threshold: -50mV
  - Reset potential: -65mV

#### SNN-E (Ethical Network)
- **Architecture**: 3-layer feedforward SNN
- **Input layer**: 64 neurons (ethical context encoding)
- **Hidden layer**: 128 neurons (LIF model)
- **Output layer**: 3 neurons (ethical valence: positive, neutral, negative)
- **Learning**: Supervised pre-training on synthetic ethical dataset
- **Integration**: Output modulates SNN-S decision weights

### Environment Specification
```python
GridWorld(
    size=(50, 50),
    num_organisms=10,
    food_spawn_rate=0.02,
    energy_decay=1.0,
    collision_penalty=10.0
)
```

### Training Protocol

#### Phase 1: Pre-training
1. Generate synthetic ethical dataset (1000 scenarios)
2. Train SNN-E using supervised learning
3. Validate on held-out test set (20%)

#### Phase 2: Simulation
1. Initialize organisms with trained/untrained networks
2. Run 100 independent simulations per condition
3. Log all states, actions, and network activity
4. Save checkpoints every 100 timesteps

### Computational Requirements
- GPU: AMD Radeon (ROCm support)
- RAM: 16GB minimum
- Storage: ~50GB for full dataset
- Estimated runtime: 2-3 days for 200 simulations

## Reproducibility
- Fixed random seeds for each run (seed = run_id)
- Version control for all code
- Docker container for environment consistency
- Detailed logging of all hyperparameters
