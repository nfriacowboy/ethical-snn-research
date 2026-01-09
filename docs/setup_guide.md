# Setup Guide

## Prerequisites
- Python 3.9+
- AMD GPU with ROCm support (optional, but recommended)
- Git

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/nfriacowboy/ethical-snn-research.git
cd ethical-snn-research
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Setup ROCm (for AMD GPU support)
```bash
# Run the ROCm setup script
./scripts/setup_rocm.sh

# Verify GPU availability
python scripts/verify_gpu.py
```

## Quick Start

### Run a single simulation
```bash
python experiments/phase1/run_survival_only.py --config experiments/phase1/config_phase1.yaml
```

### Run full Phase 1 experiment
```bash
python experiments/phase1/batch_runner.py --num_runs 100
```

### Explore results interactively
```bash
jupyter notebook notebooks/03_results_exploration.ipynb
```

## Configuration

Edit [config_phase1.yaml](../experiments/phase1/config_phase1.yaml) to adjust:
- Grid world size
- Number of organisms
- Network hyperparameters
- Logging frequency

## Troubleshooting

### GPU not detected
- Verify ROCm installation: `rocm-smi`
- Check PyTorch ROCm support: `python -c "import torch; print(torch.cuda.is_available())"`

### Out of memory
- Reduce `num_organisms` in config
- Decrease `hidden_size` in network config
- Enable gradient checkpointing

## Testing
```bash
pytest tests/
```

## Documentation
For detailed methodology, see [methodology_phase1.md](methodology_phase1.md)
