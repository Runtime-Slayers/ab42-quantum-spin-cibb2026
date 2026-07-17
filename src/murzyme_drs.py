"""
murzyme_drs.py — Murzyme stochastic DRS flux operator.
CIBB 2026: Spin-Mechanical Transduction and Chiral Spin-Sieving in Aβ₄₂

Implements the Murzyme DRS Flux VQE, computing the effect of Diffusible
Reactive Species (superoxide, hydroxyl) on the Aβ₄₂ spin Hamiltonian.

Authors: Bhavanam Rajendra Reddy, Boddu Saran, Muthuraman Ramanathan, Likith Palakurthi
Affiliation: School of Artificial Intelligence, Amrita Vishwa Vidyapeetham, Coimbatore, India
"""

from __future__ import annotations
import numpy as np

# DRS flux intensity: concentration of radical species in typical murburn context
SUPEROXIDE_FLUX  = 0.042   # Φ(O₂•⁻) [a.u.]
HYDROXYL_FLUX    = 0.031   # Φ(OH•)   [a.u.]
MFC_THRESHOLD    = 0.009   # Murburn Flux Coefficient threshold for misfolding

SEQUENCE_14 = "QKLVFFAEDVGSNK"

# Per-residue DRS susceptibility (high for aromatic F, charged K/E/D)
DRS_SUSCEPTIBILITY: dict[str, float] = {
    "Q": 0.42, "K": 0.55, "L": 0.30, "V": 0.28,
    "F": 0.75, "A": 0.20, "E": 0.60, "D": 0.58,
    "G": 0.18, "S": 0.35, "N": 0.40, "I": 0.32,
}


def murburn_flux_coefficient(
    phi_superoxide: float = SUPEROXIDE_FLUX,
    phi_hydroxyl: float = HYDROXYL_FLUX,
    alpha: float = 0.6,
    beta: float = 0.4,
) -> float:
    """
    Murburn Flux Coefficient (MFC) combining both radical species.
        MFC = α·Φ(O₂•⁻) + β·Φ(OH•)
    """
    return alpha * phi_superoxide + beta * phi_hydroxyl


def build_drs_operator(
    sequence: str = SEQUENCE_14,
    mfc: float | None = None,
    seed: int = 42,
) -> np.ndarray:
    """
    Build the stochastic Murzyme DRS operator Ω^{DRS} ∈ ℝ^{N×N}.

    Ω^{DRS}_{ij} = MFC · χ_i · χ_j · ξ_{ij}

    where χ_i = DRS susceptibility of residue i,
    ξ_{ij} ~ N(0,1) stochastic radical encounter matrix.

    Parameters
    ----------
    sequence : Amino acid sequence
    mfc      : Murburn Flux Coefficient (computed if None)
    """
    if mfc is None:
        mfc = murburn_flux_coefficient()

    N = len(sequence)
    rng = np.random.default_rng(seed)
    chi = np.array([DRS_SUSCEPTIBILITY.get(aa, 0.35) for aa in sequence])

    # Stochastic radical encounter matrix
    xi = rng.standard_normal((N, N))
    xi = 0.5 * (xi + xi.T)   # symmetrise

    omega = mfc * np.outer(chi, chi) * xi
    return omega


def drs_energy_shift(
    H_sync: np.ndarray,
    sequence: str = SEQUENCE_14,
    mfc: float | None = None,
    n_samples: int = 100,
    seed: int = 42,
) -> dict:
    """
    Compute mean DRS-induced ground-state energy shift via Monte Carlo sampling
    of the stochastic Murzyme operator.

    Returns
    -------
    dict with mean_shift_au, drs_suppression_pct, dominant_hotspot,
         mfc, transitions_per_residue
    """
    if mfc is None:
        mfc = murburn_flux_coefficient()

    N = len(sequence)
    rng = np.random.default_rng(seed)
    E_0 = float(np.linalg.eigvalsh(H_sync)[0])

    shifts = []
    transitions = np.zeros(N)

    for trial in range(n_samples):
        omega = build_drs_operator(sequence, mfc, seed=seed + trial)
        H_drs = H_sync + omega
        E_drs = float(np.linalg.eigvalsh(H_drs)[0])
        shift = E_drs - E_0
        shifts.append(shift)

        # Track qubit transitions: |δE_i / E_0|
        for i in range(N):
            transitions[i] += abs(omega[i, i]) / (abs(E_0) + 1e-10)

    transitions /= n_samples
    mean_shift = float(np.mean(shifts))

    # DRS stability suppression: fraction of ground-state stability removed
    suppression = abs(mean_shift) / (abs(E_0) + 1e-10) * 100.0

    # Dominant hotspot = residue with highest transition probability
    hotspot_idx = int(np.argmax(transitions))

    return {
        "mean_drs_shift_au":       mean_shift,
        "std_drs_shift_au":        float(np.std(shifts)),
        "ground_state_energy_au":  E_0,
        "drs_suppression_pct":     suppression,
        "murburn_flux_coefficient": mfc,
        "misfolding_predicted":    mfc > MFC_THRESHOLD,
        "dominant_hotspot_residue": sequence[hotspot_idx],
        "dominant_hotspot_idx":    hotspot_idx,
        "transitions_per_residue": transitions.tolist(),
    }


def zero_field_singlet_yield(
    sequence: str = SEQUENCE_14,
    B_local: float = 50e-6,    # local field [Tesla equivalent in a.u.]
    t_max: float = 200.0,
    n_steps: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Compute time-averaged singlet yield Φ_S in zero-field ST basis
    for each residue pair, at physiological B ≈ 50 μT.

    Returns
    -------
    dict with mean_singlet_yield, per_residue_phi_s, radical_hotspots
    """
    rng = np.random.default_rng(seed)
    N = len(sequence)
    t = np.linspace(0, t_max, n_steps)
    dt = t[1] - t[0]

    chi = np.array([DRS_SUSCEPTIBILITY.get(aa, 0.35) for aa in sequence])

    phi_s_per_residue = []
    for i in range(N):
        a_i = chi[i] * 0.5   # HFC constant proxy [a.u.]
        J_ex = 0.01           # exchange coupling [a.u.]

        # Two-spin Hamiltonian in zero-field basis
        # |S⟩ ↔ |T₀⟩ oscillations driven by HFC
        omega_osc = np.sqrt(a_i ** 2 + J_ex ** 2 + B_local ** 2)
        phi_s_t = 0.25 * (1 + np.cos(omega_osc * t))

        phi_s_per_residue.append(float(np.mean(phi_s_t)))

    mean_phi_s = float(np.mean(phi_s_per_residue))
    hotspots = [sequence[i] for i, p in enumerate(phi_s_per_residue) if p > 0.35]

    return {
        "mean_singlet_yield":      mean_phi_s,
        "per_residue_phi_s":       phi_s_per_residue,
        "radical_hotspots":        hotspots,
        "zeeman_basis_prediction": 0.50,
        "suppression_pct":         (0.50 - mean_phi_s) / 0.50 * 100.0,
    }


if __name__ == "__main__":
    from pipeline import build_hamiltonian

    h, J = build_hamiltonian(SEQUENCE_14, isotope="14N")

    # Build synchronized Hamiltonian matrix
    N = len(SEQUENCE_14)
    H_sync = np.diag(h) + J

    mfc = murburn_flux_coefficient()
    print(f"Murburn Flux Coefficient (MFC)  : {mfc:.4f}")
    print(f"Misfolding threshold            : {MFC_THRESHOLD}")
    print(f"Misfolding predicted            : {mfc > MFC_THRESHOLD}")

    print("\n─── DRS Energy Shift Analysis ───")
    res = drs_energy_shift(H_sync, SEQUENCE_14, mfc)
    print(f"  Mean DRS energy shift   : {res['mean_drs_shift_au']:+.4f} a.u.")
    print(f"  DRS suppression         : {res['drs_suppression_pct']:.1f}%")
    print(f"  Dominant hotspot        : {res['dominant_hotspot_residue']} "
          f"(idx {res['dominant_hotspot_idx']})")

    print("\n─── Zero-Field Singlet Yield ───")
    zy = zero_field_singlet_yield(SEQUENCE_14)
    print(f"  Mean Φ_S (B=50μT)      : {zy['mean_singlet_yield']:.3f} ± SE")
    print(f"  Zeeman basis Φ_S       : {zy['zeeman_basis_prediction']:.2f}")
    print(f"  Suppression            : {zy['suppression_pct']:.1f}%")
    print(f"  Radical hotspots       : {zy['radical_hotspots']}")
