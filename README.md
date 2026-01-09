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
git clone https://github.com/[username]/ethical-snn-research
cd ethical-snn-research

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Verify GPU (AMD ROCm)
python scripts/verify_gpu.py
```

### Run Demo
```bash
# Visualize environment
jupyter notebook notebooks/01_environment_demo.ipynb

# Test single organism
python src/simulation/runner.py --demo

# Run Phase 1 experiments (post-preregistration only!)
python experiments/phase1/batch_runner.py --config phase1/config_phase1.yaml
```

## 📁 Repository Structure
```
src/          - Core implementation (SNNs, environment, simulation)
experiments/  - Experimental protocols (phase-specific)
analysis/     - Statistical analysis and visualization
notebooks/    - Interactive demos and exploration
tests/        - Unit tests
docs/         - Documentation and preregistration copies
```

## 📦 Data & Results

Raw data and results are stored on OSF (not in this repository):  
👉 https://osf.io/[project-id]/

## 📝 Citation

If you use this code or data, please cite:
```bibtex
@software{[sobrenome]2026ethical,
  author = {[Nome Completo]},
  title = {Ethical Value Learning in Minimal Artificial Organisms},
  year = {2026},
  url = {https://github.com/[username]/ethical-snn-research},
  doi = {[DOI do OSF]}
}
```

## 📄 License

Code: MIT License (see [LICENSE](LICENSE))  
Data: CC-BY 4.0 (see OSF project)

## 🤝 Contributing

This is research code following a preregistered protocol. Contributions 
welcome after Phase 1 completion. See [CONTRIBUTING.md](CONTRIBUTING.md).

## 📧 Contact

- ORCID: [teu-orcid]
- Email: [teu-email]
- OSF Profile: [link]

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