# ðŸ§¬ Quantum Spin Framework for AÎ²â‚„â‚‚ Amyloidosis Modeling
### CIBB 2026 â€” *Spin-Mechanical Transduction and Chiral Spin-Sieving in AÎ²â‚„â‚‚*

**Authors:** Bhavanam Rajendra Reddy\*, Boddu Saran, Muthuraman Ramanathan, Likith Palakurthi  
**Affiliation:** School of Artificial Intelligence, Amrita Vishwa Vidyapeetham, Coimbatore, India  
**Contact:** brr1154@gmail.com  
**Conference:** CIBB 2026 (Computational Intelligence Methods for Bioinformatics and Biostatistics)

---

## ðŸ“– Overview

This repository provides the full simulation code, pipeline scripts, and supporting data for our CIBB 2026 paper. We present a **unified hybrid quantum-classical framework** for modeling Alzheimer's AÎ²â‚„â‚‚ amyloid misfolding at the quantum spin level.

### Five Principal Contributions

| # | Contribution | Key Result |
|---|---|---|
| 1 | **Synchronized GNN-Ising Hamiltonian** | +130.7% spin coordination; energy shift from +0.286 â†’ âˆ’0.086 a.u. |
| 2 | **Isotopic Hyperfine Coupling (HFC) Tuning** | Â¹âµN substitution â†’ Î”E = âˆ’0.049 a.u. (misfolding decelerated) |
| 3 | **Recursive QNG (R-QNG) Optimization** | 3.7Ã— speedup over Adam; 100% vs 4% (SA) success rate |
| 4 | **HEOM-MPO Non-Markovian Solvent** | 3.5Ã—10â¶ coherence lifetime extension (1.2 fs â†’ 4.15 ns) |
| 5 | **Floquet Topological Analysis** | Zâ‚‚ = 1 non-trivial phase; 2 protected fibril edge modes |

---

## ðŸ—‚ï¸ Repository Structure

```
ab42-quantum-spin-cibb2026/
â”‚
â”œâ”€â”€ README.md
â”œâ”€â”€ requirements.txt
â”œâ”€â”€ LICENSE
â”‚
â”œâ”€â”€ src/                          # Core simulation modules
â”‚   â”œâ”€â”€ pipeline.py               # Master end-to-end pipeline runner
â”‚   â”œâ”€â”€ gnn_hamiltonian.py        # GNN encoder â†’ Ising parameters
â”‚   â”œâ”€â”€ mps_vqe.py                # MPS-VQE variational eigensolver
â”‚   â”œâ”€â”€ r_qng_optimizer.py        # Recursive Quantum Natural Gradient
â”‚   â”œâ”€â”€ heom_mpo.py               # HEOM-MPO correlated solvent model
â”‚   â”œâ”€â”€ isotopic_hfc.py           # Â¹â´N/Â¹âµN hyperfine coupling module
â”‚   â”œâ”€â”€ floquet_topology.py       # Floquet SSH + U(1) Lattice Gauge VQE
â”‚   â””â”€â”€ murzyme_drs.py            # Murzyme stochastic DRS flux operator
â”‚
â”œâ”€â”€ data/                         # Simulation outputs and reference data
â”‚   â”œâ”€â”€ README.md                 # Data description
â”‚   â”œâ”€â”€ energy_convergence.csv    # MPS-VQE energy per iteration
â”‚   â”œâ”€â”€ rqng_vs_baselines.csv     # R-QNG vs Adam/SA/GA comparison
â”‚   â”œâ”€â”€ heom_coherence.csv        # Coherence lifetime data
â”‚   â”œâ”€â”€ isotope_energies.csv      # Â¹â´N vs Â¹âµN ground-state energies
â”‚   â””â”€â”€ floquet_spectrum.csv      # Quasi-energy eigenvalues
â”‚
â”œâ”€â”€ paper/
â”‚   â””â”€â”€ cibb2026_main.pdf         # Camera-ready manuscript (8 pages)
â”‚
â””â”€â”€ notebooks/
    â””â”€â”€ demo_pipeline.ipynb       # Interactive Jupyter demo
```

---

## âš™ï¸ Installation

```bash
git clone https://github.com/BRR1154/ab42-quantum-spin-cibb2026.git
cd ab42-quantum-spin-cibb2026
pip install -r requirements.txt
```

### Requirements
- Python â‰¥ 3.9
- PennyLane â‰¥ 0.36
- PyTorch â‰¥ 2.0
- NumPy, SciPy, NetworkX, Matplotlib

---

## ðŸš€ Quick Start

```python
from src.pipeline import run_pipeline

results = run_pipeline(
    sequence="QKLVFFAEDVGSNK",   # AÎ²â‚„â‚‚ hydrophobic core
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

## ðŸ“Š Key Numerical Results

### MPS-VQE Energy Convergence

| Hamiltonian | Ground-State Energy | Order Parameter |
|---|---|---|
| Standard Heisenberg-Ising | +0.286 a.u. | Baseline |
| + Synchronized J_sync | âˆ’0.086 a.u. | +130.7% |

### R-QNG vs Classical Optimizers

| Method | Final Energy | Iterations | Success Rate |
|---|---|---|---|
| VQE (R-QNG) | **âˆ’3.5564 a.u.** | **50** | **100%** |
| VQE (Adam) | âˆ’3.5463 a.u. | 200 | ~100% |
| Simulated Annealing | âˆ’3.5564 a.u. | 1,200 | 4% |
| Genetic Algorithm | âˆ’3.5312 a.u. | 2,500 | 2% |

### HEOM-MPO Coherence

| Model | Coherence Lifetime | Extension |
|---|---|---|
| Markovian (Lindblad) | 1.2 fs | â€” |
| HEOM-MPO | 16.0 fs | 13Ã— |
| TDNN-nested HEOM | **4.15 ns** | **3.5Ã—10â¶** |

---

## ðŸ“œ Citation

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

## ðŸ“„ License

MIT License â€” see [LICENSE](LICENSE) for details.

---

## ðŸ™ Acknowledgments

We thank the CIBB 2026 program committee for their constructive feedback.
Simulations were developed and validated using open-source quantum computing frameworks
(PennyLane, PyTorch) on GPU-accelerated hardware.
