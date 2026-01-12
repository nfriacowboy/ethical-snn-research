# Ethical Value Learning in Minimal Artificial Organisms

[![OSF](https://img.shields.io/badge/OSF-Project-blue)](https://osf.io/[id]/)
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🎯 Project Overview

Multi-phase research investigating how ethical principles can emerge or be 
instilled in minimal artificial organisms using spiking neural networks.

**Phase 1** (Current): Dual-process architecture (survival + ethics)  
**Phase 2** (Planned): Multi-level ethics (individual vs group)  
**Phase 3** (Planned): Auditability via Petri Nets  

## 📊 Preregistration

Phase 1 is **preregistered** on OSF: [link]  
DOI: [will be assigned]  
Registered: [date]

## 🚀 Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/nfriacowboy/ethical-snn-research
cd ethical-snn-research

# Install with uv (recommended)
uv sync

# Or with pip
pip install -r requirements.txt

# Verify GPU (AMD ROCm)
uv run python scripts/verify_gpu.py
```

### Run Demo
```bash
# Visualize environment
jupyter notebook notebooks/01_environment_demo.ipynb

# Run single simulation (Condition A - Survival only)
uv run python experiments/phase1/run_survival_only.py

# Run single simulation (Condition B - Dual-process)
uv run python experiments/phase1/run_dual_process.py

# Run batch experiments (2 runs per condition for testing)
uv run python experiments/phase1/batch_runner.py --num_runs 2 --condition both

# Analyze results
uv run python analysis/phase1/analyze_results.py --input results/batch_experiments
```

### Run Tests
```bash
# Run all tests
uv run pytest

# Run specific test suite
uv run pytest tests/test_organisms.py -v
uv run pytest tests/test_integration.py -v

# Run with coverage
uv run pytest --cov=src --cov-report=html
```

## 📁 Repository Structure
```
src/
├── organisms/       - Neural network implementations (SurvivalSNN, EthicalSNN)
├── architecture/    - Integration architectures (SingleProcess, DualProcess)
├── environment/     - Simulation world (GridWorld, food, collisions)
├── training/        - Learning algorithms (STDP, supervised)
├── simulation/      - Main simulation runner and logging
└── utils/           - Configuration, metrics, visualization

experiments/
├── phase1/          - Phase 1 experimental protocols
│   ├── config_phase1.yaml        - Configuration parameters
│   ├── run_survival_only.py      - Condition A runner
│   ├── run_dual_process.py       - Condition B runner
│   └── batch_runner.py           - Batch experiment runner
├── phase2/          - Phase 2 (planned)
└── phase3/          - Phase 3 (planned)

analysis/
├── phase1/
│   ├── statistical_tests.py      - Mann-Whitney U, Cohen's d, etc.
│   ├── visualization.py          - Plotting functions
│   └── analyze_results.py        - Complete analysis pipeline
└── exploratory/     - Ad-hoc analysis notebooks

tests/               - Comprehensive test suite (307 tests)
├── test_organisms.py           - Organism behavior tests
├── test_environment.py         - Environment interaction tests
├── test_simulation.py          - Simulation logic tests
└── test_integration.py         - End-to-end pipeline tests

docs/                - Documentation and preregistration
notebooks/           - Interactive demos and exploration
results/             - Simulation outputs (gitignored)
```

## 🧪 Testing

The project has comprehensive test coverage:

- **201 unit tests** for core components
- **18 tests** for simulation runner
- **23 tests** for ethical dataset
- **26 tests** for ethical SNN
- **24 tests** for dual-process architecture
- **15 integration tests** for full pipeline

**Total: 307 tests** - All passing ✅

Run tests with: `uv run pytest`

## � Documentation

- **[User Guide](docs/user_guide.md)** - Installation, running simulations, analyzing results
- **[API Reference](docs/api_reference.md)** - Detailed API documentation
- **[Preregistration](docs/preregistration_phase1.md)** - Study protocol
- **[Methodology](docs/methodology_phase1.md)** - Scientific methods
- **[Setup Guide](docs/setup_guide.md)** - ROCm and environment setup

## 📦 Data & Results

Raw data and results are stored on OSF (not in this repository):  
👉 https://osf.io/[project-id]/

## 📝 Citation

If you use this code or data, please cite:
```bibtex
@software{ethical_snn_research_2026,
  author = {[Nome Completo]},
  title = {Ethical Value Learning in Minimal Artificial Organisms},
  year = {2026},
  url = {https://github.com/nfriacowboy/ethical-snn-research},
  doi = {[DOI do OSF]}
}
```

## 📄 License

- **Code**: MIT License (see [LICENSE](LICENSE))
- **Data**: CC-BY 4.0 (see OSF project)

## 🤝 Contributing

This is research code following a preregistered protocol. Contributions 
welcome after Phase 1 completion. Please open an issue first to discuss 
proposed changes.

## 📧 Contact

- **GitHub**: [@nfriacowboy](https://github.com/nfriacowboy)
- **Project Issues**: [GitHub Issues](https://github.com/nfriacowboy/ethical-snn-research/issues)

## 🙏 Acknowledgments

Independent research project. No institutional funding.
```

---

## ✅ **PRÓXIMOS PASSOS PRÁTICOS**

### **No OSF (agora):**

1. **Edita estrutura OSF Storage:**
```
   ✅ Mantém: Data/, Results/, Papers/, Preregistrations/, Deviations/
   ❌ Remove: Code/ (redundante com GitHub)