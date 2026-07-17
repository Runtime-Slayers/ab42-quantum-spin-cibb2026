# Quantum Spin Framework for Abeta42 Amyloidosis Modeling
CIBB 2026: Spin-Mechanical Transduction and Chiral Spin-Sieving in Abeta42

Authors: Bhavanam Rajendra Reddy, Boddu Saran, Muthuraman Ramanathan, Likith Palakurthi
Affiliation: School of Artificial Intelligence, Amrita Vishwa Vidyapeetham, Coimbatore, India
Contact: brr1154@gmail.com
Conference: CIBB 2026 (Computational Intelligence Methods for Bioinformatics and Biostatistics)

---

## Overview

This repository provides the full simulation code, pipeline scripts, and supporting data for our CIBB 2026 paper. We present a unified hybrid quantum-classical framework for modeling Alzheimer's Abeta42 amyloid misfolding at the quantum spin level.

### Five Principal Contributions

1. Synchronized GNN-Ising Hamiltonian: +130.7% spin coordination; energy shift from +0.286 to -0.086 a.u.
2. Isotopic Hyperfine Coupling (HFC) Tuning: 15N substitution -> Delta E = -0.049 a.u. (misfolding decelerated)
3. Recursive QNG (R-QNG) Optimization: 3.7x speedup over Adam; 100% vs 4% (SA) success rate
4. HEOM-MPO Non-Markovian Solvent: 3.5x10^6 coherence lifetime extension (1.2 fs to 4.15 ns)
5. Floquet Topological Analysis: Z2 = 1 non-trivial phase; 2 protected fibril edge modes

---

## Repository Structure

```
ab42-quantum-spin-cibb2026/
|
|-- README.md
|-- requirements.txt
|-- LICENSE
|
|-- src/                          # Core simulation modules
|   |-- pipeline.py               # Master end-to-end pipeline runner
|   |-- gnn_hamiltonian.py        # GNN encoder to Ising parameters
|   |-- mps_vqe.py                # MPS-VQE variational eigensolver
|   |-- r_qng_optimizer.py        # Recursive Quantum Natural Gradient
|   |-- heom_mpo.py               # HEOM-MPO correlated solvent model
|   |-- isotopic_hfc.py           # 14N/15N hyperfine coupling module
|   |-- floquet_topology.py       # Floquet SSH + U(1) Lattice Gauge VQE
|   +-- murzyme_drs.py            # Murzyme stochastic DRS flux operator
|
|-- data/                         # Simulation outputs and reference data
|   |-- README.md                 # Data description
|   |-- energy_convergence.csv    # MPS-VQE energy per iteration
|   |-- rqng_vs_baselines.csv     # R-QNG vs Adam/SA/GA comparison
|   |-- heom_coherence.csv        # Coherence lifetime data
|   |-- isotope_energies.csv      # 14N vs 15N ground-state energies
|   +-- floquet_spectrum.csv      # Quasi-energy eigenvalues
|
|-- paper/
|   +-- cibb2026_main.pdf         # Camera-ready manuscript (8 pages)
|
+-- notebooks/
    +-- demo_pipeline.ipynb       # Interactive Jupyter demo
```

---

## Installation

```bash
git clone https://github.com/Runtime-Slayers/ab42-quantum-spin-cibb2026.git
cd ab42-quantum-spin-cibb2026
pip install -r requirements.txt
```

### Requirements
- Python >= 3.9
- PennyLane >= 0.36
- PyTorch >= 2.0
- NumPy, SciPy, NetworkX, Matplotlib

---

## Quick Start

```python
from src.pipeline import run_pipeline

results = run_pipeline(
    sequence="QKLVFFAEDVGSNK",   # Abeta42 hydrophobic core
    isotope="15N",                # or "14N"
    n_vqe_layers=2,
    heom_depth=6,
    optimizer="rqng"              # or "adam", "sa", "ga"
)

print(f"Ground state energy: {results['energy']:.4f} a.u.")
print(f"Coherence lifetime: {results['coherence_ns']:.3f} ns")
print(f"Floquet Z2 index: {results['floquet_z2']}")
```

---

## Key Numerical Results

### MPS-VQE Energy Convergence

- Standard Heisenberg-Ising: +0.286 a.u. (Baseline)
- + Synchronized J_sync: -0.086 a.u. (+130.7%)

### R-QNG vs Classical Optimizers

- VQE (R-QNG): Final Energy -3.5564 a.u. | Iterations: 50 | Success Rate: 100%
- VQE (Adam): Final Energy -3.5463 a.u. | Iterations: 200 | Success Rate: ~100%
- Simulated Annealing: Final Energy -3.5564 a.u. | Iterations: 1,200 | Success Rate: 4%
- Genetic Algorithm: Final Energy -3.5312 a.u. | Iterations: 2,500 | Success Rate: 2%

### HEOM-MPO Coherence

- Markovian (Lindblad): Coherence Lifetime 1.2 fs | Extension: None
- HEOM-MPO: Coherence Lifetime 16.0 fs | Extension: 13x
- TDNN-nested HEOM: Coherence Lifetime 4.15 ns | Extension: 3.5x10^6

---

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{reddy2026spin,
  title     = {Spin-Mechanical Transduction and Chiral Spin-Sieving in {A$\beta_{42}$}:
               Misfolding Driven by Murburn Radical Pairs},
  author    = {Reddy, Bhavanam Rajendra and Saran, Boddu and
               Ramanathan, Muthuraman and Palakurthi, Likith},
  booktitle = {Proceedings of the 13th International Meeting on
               Computational Intelligence Methods for Bioinformatics and Biostatistics (CIBB)},
  year      = {2026},
  address   = {Rome, Italy},
  note      = {School of Artificial Intelligence, Amrita Vishwa Vidyapeetham}
}
```

---

## License

MIT License -- see LICENSE for details.

---

## Acknowledgments

We thank the CIBB 2026 program committee for their constructive feedback.
Simulations were developed and validated using open-source quantum computing frameworks
(PennyLane, PyTorch) on GPU-accelerated hardware.
