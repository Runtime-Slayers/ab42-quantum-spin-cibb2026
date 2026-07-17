"""
floquet_topology.py — Floquet SSH chain + U(1) Lattice Gauge VQE.
CIBB 2026: Spin-Mechanical Transduction and Chiral Spin-Sieving in Aβ₄₂

Computes the Z2 topological index and protected edge modes of the Aβ₄₂
fibril modeled as a periodically driven SSH tight-binding chain.

Authors: Bhavanam Rajendra Reddy, Boddu Saran, Muthuraman Ramanathan, Likith Palakurthi
Affiliation: School of Artificial Intelligence, Amrita Vishwa Vidyapeetham, Coimbatore, India
"""

from __future__ import annotations
import numpy as np
from scipy.linalg import expm


# ─────────────────────────────────────────────────────────────────────────────
def build_ssh_hamiltonian(
    N: int = 14,
    J: float = 0.35,
    delta: float = 0.12,
) -> np.ndarray:
    """
    Build the Su-Schrieffer-Heeger (SSH) tight-binding Hamiltonian.
    Alternating hopping: J(1-δ) weak, J(1+δ) strong.

    Parameters
    ----------
    N     : number of sites
    J     : mean hopping amplitude [a.u.]
    delta : dimerization parameter (>0 → topological phase)
    """
    H = np.zeros((N, N))
    for i in range(N - 1):
        t = J * (1 + delta) if i % 2 == 0 else J * (1 - delta)
        H[i, i + 1] = -t
        H[i + 1, i] = -t
    return H


# ─────────────────────────────────────────────────────────────────────────────
def build_floquet_operator(
    H_ssh: np.ndarray,
    V_drive: float = 0.1,
    Omega: float = 2.0 * np.pi * 0.8,
    N_steps: int = 60,
) -> np.ndarray:
    """
    Compute the Floquet time-evolution operator over one period T = 2π/Ω
    via second-order Trotter decomposition (60 slices).

    U_F = ∏_{k=1}^{N_steps} exp(-i H(t_k) Δt)

    Parameters
    ----------
    H_ssh   : SSH Hamiltonian matrix
    V_drive : Drive amplitude [a.u.]
    Omega   : Drive frequency [rad/a.u.]
    N_steps : Trotter slices
    """
    T_period = 2.0 * np.pi / Omega
    dt = T_period / N_steps
    N = H_ssh.shape[0]
    # Drive: uniform on-site modulation (diagonal)
    V = V_drive * np.eye(N)

    U_F = np.eye(N, dtype=complex)
    for k in range(N_steps):
        t_k = (k + 0.5) * dt
        H_t = H_ssh + V * np.cos(Omega * t_k)
        U_F = expm(-1j * H_t * dt) @ U_F

    return U_F


# ─────────────────────────────────────────────────────────────────────────────
def quasi_energy_spectrum(U_F: np.ndarray, Omega: float = 2.0 * np.pi * 0.8) -> np.ndarray:
    """
    Extract the quasi-energy spectrum: ε_n = arg(λ_n(U_F)) / T
    """
    T_period = 2.0 * np.pi / Omega
    eigvals = np.linalg.eigvals(U_F)
    quasi_energies = np.angle(eigvals) / T_period
    return np.sort(quasi_energies.real)


# ─────────────────────────────────────────────────────────────────────────────
def compute_z2_index(quasi_energies: np.ndarray, tol: float = 0.05) -> int:
    """
    Compute Z2 topological index from quasi-energy spectrum.
    Presence of zero-energy edge modes (|ε| < tol) → Z2 = 1 (topological).

    Returns 1 (topological) or 0 (trivial).
    """
    zero_modes = np.sum(np.abs(quasi_energies) < tol)
    return 1 if zero_modes >= 2 else 0


# ─────────────────────────────────────────────────────────────────────────────
def bulk_gap(quasi_energies: np.ndarray, tol: float = 0.05) -> float:
    """
    Compute the quasi-energy bulk gap, excluding zero-energy edge modes.
    """
    bulk_modes = quasi_energies[np.abs(quasi_energies) >= tol]
    if len(bulk_modes) == 0:
        return 0.0
    return float(np.min(np.abs(bulk_modes)))


# ─────────────────────────────────────────────────────────────────────────────
def lattice_gauge_vqe(
    H_floquet: np.ndarray,
    gnn_flux_vectors: np.ndarray,
    gamma_chiral: float = 0.085,
    n_iter: int = 250,
    lr: float = 0.02,
) -> tuple[float, float, list[float]]:
    """
    U(1) Lattice Gauge VQE with GNN-derived Wilson line phases.

    State ansatz (Eq. of paper):
        |ψ_LG⟩ = N Σ_i cos(θ_i) exp(i Σ_{k<i} φ_{k,k+1}) |i⟩

    where φ_{ij} = γ_chiral (F_i · F_j) · u_{ij}  (Wilson line)

    Parameters
    ----------
    H_floquet        : Static Floquet Hamiltonian (H_SSH at t=0)
    gnn_flux_vectors : (N, 3) array of GNN-predicted DRS flux vectors
    gamma_chiral     : Helical chirality scaling factor [a.u.]

    Returns
    -------
    E_LG      : LG-VQE ground energy [a.u.]
    E_exact   : Brute-force exact ground energy [a.u.]
    history   : Energy per iteration
    """
    N = H_floquet.shape[0]

    # Compute Wilson line phases from GNN flux vectors
    phi = np.zeros(N)
    for i in range(N - 1):
        u_ij = np.array([1.0, 0.0, 0.0])  # simplified displacement
        phi[i + 1] = gamma_chiral * np.dot(gnn_flux_vectors[i], gnn_flux_vectors[i + 1])

    # Brute-force exact ground state
    E_exact = float(np.linalg.eigvalsh(H_floquet)[0])

    # Variational optimization of θ_i
    theta = np.random.uniform(0, 2 * np.pi, N)
    history = []

    for it in range(n_iter):
        # Construct gauge-dressed state
        phase_accum = np.cumsum(phi)
        psi = np.cos(theta) * np.exp(1j * phase_accum)
        norm = np.linalg.norm(psi)
        if norm < 1e-10:
            psi = np.ones(N, dtype=complex) / np.sqrt(N)
        else:
            psi /= norm

        # Energy expectation
        E = float(np.real(psi.conj() @ H_floquet @ psi))
        history.append(E)

        # Gradient via finite differences
        grad = np.zeros(N)
        for j in range(N):
            theta_p = theta.copy(); theta_p[j] += 1e-4
            psi_p = np.cos(theta_p) * np.exp(1j * phase_accum)
            psi_p /= (np.linalg.norm(psi_p) + 1e-10)
            E_p = float(np.real(psi_p.conj() @ H_floquet @ psi_p))
            grad[j] = (E_p - E) / 1e-4

        theta -= lr * grad

    E_LG = history[-1]
    return E_LG, E_exact, history


# ─────────────────────────────────────────────────────────────────────────────
def run_floquet_analysis(
    N: int = 14,
    J: float = 0.35,
    delta: float = 0.12,
    V_drive: float = 0.1,
    Omega: float = 2.0 * np.pi * 0.8,
) -> dict:
    """
    Full Floquet topological analysis pipeline.

    Returns
    -------
    dict with quasi_energies, z2_index, bulk_gap, edge_modes,
         lg_vqe_energy, exact_energy, relative_error, convergence_history
    """
    H_ssh = build_ssh_hamiltonian(N, J, delta)
    U_F   = build_floquet_operator(H_ssh, V_drive, Omega)
    qe    = quasi_energy_spectrum(U_F, Omega)
    z2    = compute_z2_index(qe)
    gap   = bulk_gap(qe)
    n_edge_modes = int(np.sum(np.abs(qe) < 0.05))

    # Dummy GNN flux vectors (uniform for demo; replace with actual GNN output)
    flux = np.random.randn(N, 3) * 0.1

    E_LG, E_exact, hist = lattice_gauge_vqe(H_ssh, flux)
    rel_err = abs(E_LG - E_exact) / (abs(E_exact) + 1e-10) * 100

    return {
        "quasi_energies":       qe,
        "z2_index":             z2,
        "bulk_gap_rad_per_T":   gap,
        "n_edge_modes":         n_edge_modes,
        "lg_vqe_energy_au":     E_LG,
        "exact_energy_au":      E_exact,
        "relative_error_pct":   rel_err,
        "convergence_history":  hist,
    }


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Running Floquet Topological Analysis on Aβ₄₂ SSH chain (N=14)...")
    res = run_floquet_analysis()
    print(f"\n  Z2 topological index    : {res['z2_index']} "
          f"({'Non-trivial' if res['z2_index'] else 'Trivial'})")
    print(f"  Topological edge modes  : {res['n_edge_modes']}")
    print(f"  Bulk quasi-energy gap   : {res['bulk_gap_rad_per_T']:.4f} rad/T")
    print(f"  LG-VQE energy           : {res['lg_vqe_energy_au']:+.4f} a.u.")
    print(f"  Exact ground energy     : {res['exact_energy_au']:+.4f} a.u.")
    print(f"  Relative error          : {res['relative_error_pct']:.2f}%")
