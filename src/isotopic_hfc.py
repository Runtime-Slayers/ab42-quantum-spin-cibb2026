"""
isotopic_hfc.py — Isotopic Hyperfine Coupling (HFC) module.
CIBB 2026: Spin-Mechanical Transduction and Chiral Spin-Sieving in Aβ₄₂

Computes the effect of ¹⁴N → ¹⁵N isotopic substitution on the
Aβ₄₂ LVFFA core spin Hamiltonian and ground-state energy.

Authors: Bhavanam Rajendra Reddy, Boddu Saran, Muthuraman Ramanathan, Likith Palakurthi
Affiliation: School of Artificial Intelligence, Amrita Vishwa Vidyapeetham, Coimbatore, India
"""

from __future__ import annotations
import numpy as np
from typing import Literal

# ¹⁵N/¹⁴N hyperfine coupling ratio (nuclear magnetic moments)
XI_HF_15N = 1.4026   # γ(¹⁵N) / γ(¹⁴N) effective ratio for HFC

# LVFFA core residue indices within the 14-residue QKLVFFAEDVGSNK sequence
LVFFA_IDX = [2, 3, 4, 5, 6]   # L, V, F, F, A → indices 2-6

# Amino acid baseline hyperfine fields h_i [a.u.] from GNN predictions
BASELINE_FIELDS = {
    "Q": 0.50, "K": 0.45, "L": 0.72, "V": 0.68,
    "F": 0.89, "A": 0.58, "E": 0.61, "D": 0.64,
    "G": 0.42, "S": 0.55, "N": 0.60, "I": 0.70,
}
SEQUENCE_14 = "QKLVFFAEDVGSNK"


def apply_isotopic_hfc(
    h: np.ndarray,
    sequence: str = SEQUENCE_14,
    isotope: Literal["14N", "15N"] = "15N",
    g_perp: float = 0.15,
) -> np.ndarray:
    """
    Apply isotopic HFC amplification to LVFFA core residues.

    For ¹⁵N substitution (Eq. 3 of paper):
        h_i → ξ_hf · h_i + g_⊥   for i ∈ LVFFA

    Parameters
    ----------
    h        : (N,) baseline local field array [a.u.]
    sequence : Amino acid sequence
    isotope  : '14N' (no change) or '15N' (amplified)
    g_perp   : Transverse field correction [a.u.]

    Returns
    -------
    h_iso : modified local field array
    """
    h_iso = h.copy()
    if isotope == "15N":
        for idx in LVFFA_IDX:
            if idx < len(h_iso):
                h_iso[idx] = XI_HF_15N * h_iso[idx] + g_perp
    return h_iso


def ising_ground_state_energy(
    h: np.ndarray,
    J: np.ndarray,
    n_samples: int = 80,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Estimate ground-state energy and standard deviation via random spin sampling
    + local greedy minimization (fast approximate for benchmarking).

    Returns (mean_energy, std_energy).
    """
    rng = np.random.default_rng(seed)
    N = len(h)
    energies = []

    for _ in range(n_samples):
        spins = rng.choice([-1, 1], size=N).astype(float)
        # Local greedy flips
        for _ in range(300):
            i = rng.integers(N)
            delta_E = 2 * spins[i] * (h[i] + J[i] @ spins)
            if delta_E < 0:
                spins[i] *= -1
        E = -float(h @ spins) - 0.5 * float(spins @ J @ spins)
        energies.append(E)

    return float(np.mean(energies)), float(np.std(energies))


def isotope_comparison(
    sequence: str = SEQUENCE_14,
    n_samples: int = 80,
    lambda_sync: float = 0.18,
) -> dict:
    """
    Compare ¹⁴N vs ¹⁵N ground-state energies.

    Returns
    -------
    dict with 14N_energy, 15N_energy, delta_E, hfc_fields_14N, hfc_fields_15N
    """
    # Build baseline fields and couplings
    h_14 = np.array([BASELINE_FIELDS.get(aa, 0.55) for aa in sequence])
    N = len(sequence)
    J = np.zeros((N, N))
    for i in range(N - 1):
        J[i, i + 1] = 0.35 * np.exp(-abs(h_14[i] - h_14[i + 1]))
        J[i + 1, i] = J[i, i + 1]
    for i in range(N - 2):
        J_s = lambda_sync * np.exp(-abs(h_14[i] - h_14[i + 2]))
        J[i, i + 2] += J_s
        J[i + 2, i] += J_s

    h_15 = apply_isotopic_hfc(h_14, sequence, isotope="15N")

    E_14, sd_14 = ising_ground_state_energy(h_14, J, n_samples)
    E_15, sd_15 = ising_ground_state_energy(h_15, J, n_samples)

    delta_E = E_15 - E_14

    return {
        "14N_mean_energy_au":   E_14,
        "14N_std_au":           sd_14,
        "15N_mean_energy_au":   E_15,
        "15N_std_au":           sd_15,
        "delta_E_au":           delta_E,
        "hfc_amplification":    XI_HF_15N,
        "hfc_fields_14N":       h_14.tolist(),
        "hfc_fields_15N":       h_15.tolist(),
        "kinetic_consequence":  "misfolding decelerated" if delta_E < 0 else "misfolding accelerated",
    }


if __name__ == "__main__":
    print("Isotopic HFC Analysis: ¹⁴N vs ¹⁵N substitution at LVFFA core\n")
    result = isotope_comparison()
    print(f"  ¹⁴N mean ground-state energy : {result['14N_mean_energy_au']:+.4f} ± {result['14N_std_au']:.4f} a.u.")
    print(f"  ¹⁵N mean ground-state energy : {result['15N_mean_energy_au']:+.4f} ± {result['15N_std_au']:.4f} a.u.")
    print(f"  Isotopic ΔE (¹⁵N − ¹⁴N)     : {result['delta_E_au']:+.4f} a.u.")
    print(f"  HFC amplification factor     : {result['hfc_amplification']}×")
    print(f"  Kinetic consequence          : {result['kinetic_consequence']}")
    print(f"\n  LVFFA fields (¹⁴N): {[f'{v:.3f}' for v in result['hfc_fields_14N'][2:7]]}")
    print(f"  LVFFA fields (¹⁵N): {[f'{v:.3f}' for v in result['hfc_fields_15N'][2:7]]}")
