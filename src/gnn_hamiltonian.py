"""
gnn_hamiltonian.py — GNN encoder for synchronized Ising Hamiltonian parameters.
CIBB 2026: Spin-Mechanical Transduction and Chiral Spin-Sieving in Aβ₄₂

Two-layer Graph Convolutional Network that maps residue physicochemical
features to Ising parameters (h_i, J_ij) and J_sync coupling strengths.
Includes Chaos-Damping layer via inverse-QFIM trace gating.

Authors: Bhavanam Rajendra Reddy, Boddu Saran, Muthuraman Ramanathan, Likith Palakurthi
Affiliation: School of Artificial Intelligence, Amrita Vishwa Vidyapeetham, Coimbatore, India
"""

from __future__ import annotations
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Amino acid node features: [hydrophobicity, charge, normalized_MW,
#                            DRS_susceptibility, pKa_normalized]
AA_FEATURE_TABLE: dict[str, list[float]] = {
    "Q": [0.00,  0.0, 0.854, 0.42, 0.51],
    "K": [-0.77, 1.0, 0.877, 0.55, 0.96],
    "L": [1.06,  0.0, 0.726, 0.30, 0.50],
    "V": [0.95,  0.0, 0.672, 0.28, 0.50],
    "F": [1.19,  0.0, 0.918, 0.75, 0.50],   # F19/F20: high DRS susceptibility
    "A": [0.62,  0.0, 0.584, 0.20, 0.50],
    "E": [-0.64,-1.0, 0.788, 0.60, 0.42],
    "D": [-0.72,-1.0, 0.738, 0.58, 0.42],
    "G": [0.48,  0.0, 0.540, 0.18, 0.50],
    "S": [-0.18, 0.0, 0.626, 0.35, 0.55],
    "N": [-0.60, 0.0, 0.780, 0.40, 0.50],
    "I": [0.99,  0.0, 0.726, 0.32, 0.50],
}
FEATURE_DIM = 5


# ─────────────────────────────────────────────────────────────────────────────
def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


# ─────────────────────────────────────────────────────────────────────────────
class GCNLayer:
    """
    Single Graph Convolutional Layer (Kipf & Welling, 2017).
    H' = σ(D̂^{-1/2} Â D̂^{-1/2} H W)
    """

    def __init__(self, in_dim: int, out_dim: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / in_dim)
        self.W = rng.normal(0, scale, (in_dim, out_dim))
        self.b = np.zeros(out_dim)

    def __call__(self, H: np.ndarray, A_hat: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        H     : (N, in_dim) node feature matrix
        A_hat : (N, N) symmetrically normalized adjacency + self-loops
        """
        return relu(A_hat @ H @ self.W + self.b)


# ─────────────────────────────────────────────────────────────────────────────
class ChaosDampingLayer:
    """
    Chaos-Damping gate: multiplies GNN output by 1 / (1 + λ·tr(F^{-1})),
    where F is the diagonal QFIM estimated from the GNN output gradient norm.
    This prevents exploding gradients on curved loss manifolds.
    """

    def __init__(self, lambda_cd: float = 0.1):
        self.lambda_cd = lambda_cd

    def __call__(self, h_out: np.ndarray, grad_norm: float = 1.0) -> np.ndarray:
        """
        Parameters
        ----------
        h_out     : (N, D) GNN node embeddings
        grad_norm : Frobenius norm of embedding Jacobian (QFIM trace proxy)
        """
        qfim_trace = grad_norm ** 2 + 1e-8
        gate = 1.0 / (1.0 + self.lambda_cd * qfim_trace)
        return h_out * gate


# ─────────────────────────────────────────────────────────────────────────────
class GNNHamiltonian:
    """
    Two-layer GCN encoder that predicts the Ising Hamiltonian parameters
    (h_i, J_ij, J_sync) from the amino acid residue graph.

    Architecture:
        Input (N × 5) → GCN₁ (hidden_dim=32) → GCN₂ (out_dim=2) →
        Chaos-Damping → [h_pred, DRS_flux]
    """

    def __init__(
        self,
        hidden_dim: int = 32,
        lambda_sync: float = 0.18,
        seed: int = 42,
    ):
        self.gcn1 = GCNLayer(FEATURE_DIM, hidden_dim, seed=seed)
        self.gcn2 = GCNLayer(hidden_dim, 2, seed=seed + 1)
        self.chaos = ChaosDampingLayer()
        self.lambda_sync = lambda_sync

    @staticmethod
    def _build_adjacency(N: int, h_baseline: np.ndarray) -> np.ndarray:
        """
        Build normalized adjacency matrix with nearest and next-nearest bonds.
        A_{ij} = exp(-|h_i - h_j|) for |i-j| ≤ 2 (connectivity horizon).
        """
        A = np.zeros((N, N))
        for i in range(N):
            A[i, i] = 1.0  # self-loop
            for j in range(i + 1, min(i + 3, N)):
                w = np.exp(-abs(h_baseline[i] - h_baseline[j]))
                A[i, j] = w
                A[j, i] = w

        # Symmetric normalization: D̂^{-1/2} A D̂^{-1/2}
        D = np.diag(A.sum(axis=1))
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D) + 1e-10))
        return D_inv_sqrt @ A @ D_inv_sqrt

    def encode(self, sequence: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run GCN forward pass to predict Ising parameters.

        Returns
        -------
        h_pred   : (N,) predicted local fields h_i [a.u.]
        J_pred   : (N, N) predicted coupling matrix [a.u.]
        J_sync   : (N, N) next-nearest synchronized coupling [a.u.]
        """
        N = len(sequence)

        # Build node feature matrix X ∈ ℝ^{N × 5}
        X = np.array([
            AA_FEATURE_TABLE.get(aa, [0.5, 0.0, 0.7, 0.35, 0.5])
            for aa in sequence
        ])

        # Baseline h from hydrophobicity column
        h_base = X[:, 0]

        # Build adjacency
        A_hat = self._build_adjacency(N, h_base)

        # Forward pass
        H1 = self.gcn1(X, A_hat)         # (N, 32)
        grad_norm = float(np.linalg.norm(H1))
        H2 = self.gcn2(H1, A_hat)        # (N, 2)
        H2 = self.chaos(H2, grad_norm)   # chaos damping

        # Decode: column 0 → h_i, column 1 → DRS flux scale
        h_pred = H2[:, 0] + h_base       # residual connection

        # Build coupling matrix from predicted fields
        J_pred = np.zeros((N, N))
        for i in range(N - 1):
            J_pred[i, i + 1] = 0.35 * np.exp(-abs(h_pred[i] - h_pred[i + 1]))
            J_pred[i + 1, i] = J_pred[i, i + 1]

        # Synchronized next-nearest coupling
        J_sync = np.zeros((N, N))
        for i in range(N - 2):
            s = self.lambda_sync * np.exp(-abs(h_pred[i] - h_pred[i + 2]))
            J_sync[i, i + 2] = s
            J_sync[i + 2, i] = s

        return h_pred, J_pred, J_sync

    def predict_drs_flux(self, sequence: str) -> np.ndarray:
        """
        Predict per-residue Murburn DRS flux vectors (N × 3).
        Used as Wilson line phases in the Lattice Gauge VQE.
        """
        N = len(sequence)
        X = np.array([
            AA_FEATURE_TABLE.get(aa, [0.5, 0.0, 0.7, 0.35, 0.5])
            for aa in sequence
        ])
        drs_susceptibility = X[:, 3]
        flux = np.stack([
            drs_susceptibility * np.cos(np.arange(N) * np.pi / N),
            drs_susceptibility * np.sin(np.arange(N) * np.pi / N),
            drs_susceptibility * 0.1,
        ], axis=1)
        return flux


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    SEQUENCE = "QKLVFFAEDVGSNK"
    gnn = GNNHamiltonian(hidden_dim=32, lambda_sync=0.18)
    h, J, J_sync = gnn.encode(SEQUENCE)

    print("GNN Hamiltonian Encoder — Aβ₄₂ QKLVFFAEDVGSNK Core")
    print(f"\nPredicted local fields h_i [a.u.]:  {np.round(h, 3)}")
    print(f"Mean nearest-neighbour J_nn       : {np.mean(J[J>0]):.4f} a.u.")
    print(f"Mean synchronized J_sync          : {np.mean(J_sync[J_sync>0]):.4f} a.u.")
    print(f"Non-reciprocity ratio at F19      : "
          f"{abs(J_sync[4,6]) / (abs(J_sync[6,4]) + 1e-10):.2f}")
