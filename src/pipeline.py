"""
pipeline.py
===========
Master pipeline for Aβ₄₂ amyloid misfolding simulation.

Integrates:
  - GNN Hamiltonian encoder
  - MPS-VQE variational eigensolver
  - R-QNG optimizer
  - HEOM-MPO non-Markovian solvent
  - Isotopic HFC tuning (14N / 15N)
  - Floquet SSH + U(1) Lattice Gauge VQE
  - Murzyme DRS stochastic operator

Paper reference:
  "Quantum-Biological Modeling of Aβ₄₂ Amyloid Misfolding"
  Sequence: QKLVFFAEDVGSNK (14 residues)
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

import numpy as np

# ── Internal modules ───────────────────────────────────────────────────────
from gnn_hamiltonian import GNNHamiltonian, build_residue_graph
from mps_vqe import MPSVQE
from r_qng_optimizer import RQNG
from heom_mpo import HEOMMPO
from isotopic_hfc import IsotopicHFC
from floquet_topology import FloquetSSH
from murzyme_drs import MurzymeDS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paper constants ────────────────────────────────────────────────────────
DEFAULT_SEQUENCE: str = "QKLVFFAEDVGSNK"   # 14-residue Aβ₄₂ core
DEFAULT_CHI: int = 8                         # MPS bond dimension
DEFAULT_N_PARAMS: int = 56                   # variational parameters
DEFAULT_CNOT_LAYERS: int = 2
TARGET_ENERGY: float = -3.5564              # a.u., paper result
TARGET_COHERENCE_NS: float = 4.15          # ns, TDNN-HEOM prediction


@dataclass
class PipelineResult:
    """Container for all pipeline outputs."""
    energy: float = 0.0
    coherence_ns: float = 0.0
    floquet_z2: int = 0
    sync_order_param: float = 0.0
    drs_suppression: float = 0.0
    delta_e_isotope: float = 0.0
    n_vqe_iterations: int = 0
    elapsed_s: float = 0.0
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "energy": self.energy,
            "coherence_ns": self.coherence_ns,
            "floquet_z2": self.floquet_z2,
            "sync_order_param": self.sync_order_param,
            "drs_suppression": self.drs_suppression,
            "delta_e_isotope": self.delta_e_isotope,
            "n_vqe_iterations": self.n_vqe_iterations,
            "elapsed_s": self.elapsed_s,
            **self.extras,
        }


def run_pipeline(
    sequence: str = DEFAULT_SEQUENCE,
    isotope: str = "14N",
    n_vqe_layers: int = DEFAULT_CNOT_LAYERS,
    heom_depth: int = 6,
    optimizer: str = "rqng",
    chi: int = DEFAULT_CHI,
    lambda_sync: float = 0.18,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run the full quantum-biology pipeline for Aβ₄₂ misfolding.

    Parameters
    ----------
    sequence : str
        Amino-acid single-letter sequence (default: QKLVFFAEDVGSNK).
    isotope : str
        Nitrogen isotope label: '14N' or '15N'.
    n_vqe_layers : int
        Number of CNOT entangling layers in the MPS-VQE ansatz.
    heom_depth : int
        Hierarchy depth N for HEOM (default 6).
    optimizer : str
        Optimiser choice: 'rqng', 'adam', 'sa', or 'ga'.
    chi : int
        MPS bond dimension.
    lambda_sync : float
        Synchronisation coupling λ (default 0.18 a.u.).
    verbose : bool
        Print progress to stdout.

    Returns
    -------
    dict
        Keys: energy, coherence_ns, floquet_z2, sync_order_param,
              drs_suppression, delta_e_isotope, n_vqe_iterations, elapsed_s.
    """
    if isotope not in ("14N", "15N"):
        raise ValueError(f"isotope must be '14N' or '15N', got {isotope!r}")

    t0 = time.perf_counter()
    result = PipelineResult()
    n = len(sequence)

    # ── Step 1: GNN → Ising Hamiltonian parameters ─────────────────────────
    log.info("[1/7] GNN encoder → h_i, J_ij …")
    gnn = GNNHamiltonian(n_residues=n, hidden_dim=64, n_gcn_layers=2)
    graph_data = build_residue_graph(sequence)
    h_i, J_ij = gnn.encode(graph_data)          # (n,), (n,n) numpy arrays
    result.extras["h_i"] = h_i.tolist()
    result.extras["J_ij_norm"] = float(np.linalg.norm(J_ij))

    # ── Step 2: Isotopic HFC tuning ────────────────────────────────────────
    log.info("[2/7] Isotopic HFC tuning (isotope=%s) …", isotope)
    hfc = IsotopicHFC(sequence=sequence, isotope=isotope)
    h_i_hfc, J_ij_hfc, delta_e = hfc.apply(h_i, J_ij)
    result.delta_e_isotope = float(delta_e)
    result.extras["xi_hf"] = hfc.xi_hf

    # ── Step 3: MPS-VQE ground state ───────────────────────────────────────
    log.info("[3/7] MPS-VQE (chi=%d, layers=%d, opt=%s) …", chi, n_vqe_layers, optimizer)
    vqe = MPSVQE(
        n_qubits=n,
        chi=chi,
        n_layers=n_vqe_layers,
        h_i=h_i_hfc,
        J_ij=J_ij_hfc,
    )

    if optimizer == "rqng":
        opt = RQNG(vqe, mu=1e-3, max_iter=50)
    else:
        opt = RQNG(vqe, mu=1e-3, max_iter=200, backend=optimizer)

    energy, iters, mps_tensors = opt.optimize()
    result.energy = float(energy)
    result.n_vqe_iterations = iters
    result.extras["mps_bond_dim"] = chi

    # ── Step 4: HEOM-MPO coherence lifetime ────────────────────────────────
    log.info("[4/7] HEOM-MPO (depth=%d) …", heom_depth)
    heom = HEOMMPO(
        n_system=n,
        heom_depth=heom_depth,
        K=4,
        chi_mpo=12,
        lambda_dl=0.035,
        gamma_dl=0.005,
    )
    coherence_ns = heom.run(h_i_hfc, J_ij_hfc)
    result.coherence_ns = float(coherence_ns)
    result.extras["heom_memory_kernel_p"] = heom.p_learned

    # ── Step 5: Floquet SSH + LG-VQE ──────────────────────────────────────
    log.info("[5/7] Floquet SSH topology …")
    fssh = FloquetSSH(
        n_sites=n,
        delta=0.12,
        J=0.35,
        omega=2 * np.pi * 0.8,
        n_trotter=60,
        J_ij=J_ij_hfc,
    )
    z2_index, quasi_energies = fssh.compute_z2_index()
    result.floquet_z2 = int(z2_index)
    result.extras["quasi_energies"] = quasi_energies[:4].tolist()

    # ── Step 6: Synchronisation order parameter ────────────────────────────
    log.info("[6/7] Synchronisation order parameter (λ=%.3f) …", lambda_sync)
    J_sync = lambda_sync * np.abs(J_ij_hfc)
    r_sync = float(np.mean(np.abs(np.exp(1j * np.angle(J_sync + 1e-12)))))
    result.sync_order_param = r_sync
    result.extras["lambda_sync"] = lambda_sync

    # ── Step 7: Murzyme DRS ────────────────────────────────────────────────
    log.info("[7/7] Murzyme DRS stochastic operator …")
    murzyme = MurzymeDS(sequence=sequence, J_ij=J_ij_hfc, h_i=h_i_hfc)
    drs_suppression, mfc = murzyme.compute()
    result.drs_suppression = float(drs_suppression)
    result.extras["mfc"] = float(mfc)

    result.elapsed_s = time.perf_counter() - t0
    if verbose:
        _print_summary(result, sequence, isotope)

    return result.to_dict()


def _print_summary(result: PipelineResult, sequence: str, isotope: str) -> None:
    """Pretty-print pipeline summary."""
    sep = "═" * 60
    print(f"\n{sep}")
    print("  QUANTUM-BIOLOGY PIPELINE — Aβ₄₂ AMYLOID MISFOLDING")
    print(sep)
    print(f"  Sequence  : {sequence}")
    print(f"  Isotope   : {isotope}")
    print(f"  VQE energy: {result.energy:+.4f} a.u.  (target {TARGET_ENERGY})")
    print(f"  Coherence : {result.coherence_ns:.3f} ns  (target {TARGET_COHERENCE_NS} ns)")
    print(f"  Floquet Z2: {result.floquet_z2}")
    print(f"  Sync ψ    : {result.sync_order_param:.4f}")
    print(f"  DRS suppr.: {result.drs_suppression:.4f}")
    print(f"  ΔE isotope: {result.delta_e_isotope:+.6f} a.u.")
    print(f"  Iterations: {result.n_vqe_iterations}")
    print(f"  Wall time : {result.elapsed_s:.2f} s")
    print(sep + "\n")


# ── CLI entry-point ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Aβ₄₂ quantum-biology pipeline"
    )
    parser.add_argument("--sequence", default=DEFAULT_SEQUENCE)
    parser.add_argument("--isotope", default="14N", choices=["14N", "15N"])
    parser.add_argument("--n_vqe_layers", type=int, default=2)
    parser.add_argument("--heom_depth", type=int, default=6)
    parser.add_argument("--optimizer", default="rqng",
                        choices=["rqng", "adam", "sa", "ga"])
    parser.add_argument("--chi", type=int, default=8)
    args = parser.parse_args()

    out = run_pipeline(
        sequence=args.sequence,
        isotope=args.isotope,
        n_vqe_layers=args.n_vqe_layers,
        heom_depth=args.heom_depth,
        optimizer=args.optimizer,
        chi=args.chi,
    )
    print(json.dumps(out, indent=2))
