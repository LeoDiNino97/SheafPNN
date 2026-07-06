"""
Synthetic data generation for Flat Bundle Neural Network experiments.

Ground-truth model (Section 3 of the notes):
  - V nodes in R^2, k-NN RBF graph: w_{ij} = exp(-‖p_i-p_j‖²/σ²)
  - Per-node SO(n) rotation frames O₀_v (Haar-uniform)
  - Connection Laplacian  L₀ₙ = O₀ᵀ (L₀ ⊗ Iₙ) O₀  ∈ R^{Vn×Vn}
  - Signals: x_i ~ N(0, L₀ₙ†)
  - Labels from eq (13) in the notes (polynomial flat-bundle filter + mean readout):
      y_i = (1/Vn) 1ᵀ [Σ_j h₀_j L₀ₙʲ] x_i + noise
"""

import numpy as np
import torch
from itertools import combinations


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def sample_so_matrix(n: int, rng: np.random.Generator) -> torch.Tensor:
    """Haar-uniform element of SO(n) via QR."""
    Z = rng.standard_normal((n, n)).astype(np.float32)
    Q, R = np.linalg.qr(Z)
    Q = Q * np.sign(np.diag(R))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return torch.from_numpy(Q)


def complete_graph_edge_index(V: int) -> torch.Tensor:
    """(2, V(V-1)/2) edge_index for the complete graph on V nodes (i > j)."""
    rows, cols = [], []
    for i in range(V):
        for j in range(i):
            rows.append(i); cols.append(j)
    return torch.tensor([rows, cols], dtype=torch.long)


# ---------------------------------------------------------------------------
# RBF connection-graph generator
# ---------------------------------------------------------------------------

def generate_rbf_connection_graph(
    V: int,
    n: int,
    k_nn: int,
    rng: np.random.Generator,
    sigma_scale: float = 0.5,
) -> dict:
    """Random connected RBF graph with SO(n) local frames.

    Nodes are placed uniformly in [0,1]², edges connect each node to its
    k_nn nearest neighbours (union of directed → undirected).  Edge weights
    follow an RBF kernel.  Per-node SO(n) frames are Haar-uniform.

    Args:
        V: number of nodes
        n: stalk dimension
        k_nn: neighbours per node
        rng: reproducible numpy Generator
        sigma_scale: σ = sigma_scale * median_nn_distance
    Returns:
        dict with keys: pos, edge_index, w0, L0, O0_blocks, O0_blk, L0n
    """
    from ..operators.laplacian import build_laplacian, build_blockdiag

    pos = rng.uniform(0, 1, size=(V, 2)).astype(np.float32)
    diff = pos[:, None, :] - pos[None, :, :]          # (V,V,2)
    dist2 = (diff ** 2).sum(-1)                        # (V,V)

    # k-NN graph (directed → undirected)
    k = min(k_nn, V - 1)
    knn_idx = np.argsort(dist2, axis=1)[:, 1:k+1]     # (V, k)
    edge_set = set()
    for i in range(V):
        for j in knn_idx[i]:
            a, b = (int(i), int(j)) if i > j else (int(j), int(i))
            edge_set.add((a, b))
    edges = sorted(edge_set)
    E = len(edges)

    ei = torch.tensor([a for a, b in edges], dtype=torch.long)
    ej = torch.tensor([b for a, b in edges], dtype=torch.long)
    edge_index = torch.stack([ei, ej], dim=0)   # (2, E)

    # RBF edge weights: σ = sigma_scale * median of k-NN distances
    nn_dists = np.sqrt(dist2[np.arange(V)[:, None], knn_idx])  # (V, k)
    sigma = sigma_scale * float(np.median(nn_dists)) + 1e-8
    w0_np = np.array([
        np.exp(-dist2[a, b] / (2 * sigma**2)) for a, b in edges
    ], dtype=np.float32)
    w0 = torch.from_numpy(w0_np)

    # Scalar Laplacian
    L0 = build_laplacian(w0, edge_index, V)

    # Per-node SO(n) frames
    O0_blocks = torch.stack([sample_so_matrix(n, rng) for _ in range(V)])  # (V,n,n)
    O0_blk = build_blockdiag(O0_blocks)                                     # (Vn,Vn)

    # Connection Laplacian: L₀ₙ = O₀ᵀ (L₀ ⊗ Iₙ) O₀
    L0n = O0_blk.t() @ torch.kron(L0, torch.eye(n)) @ O0_blk

    return dict(
        pos=torch.from_numpy(pos),
        edge_index=edge_index,
        w0=w0,
        L0=L0,
        O0_blocks=O0_blocks,
        O0_blk=O0_blk,
        L0n=L0n,
    )


# ---------------------------------------------------------------------------
# Signal sampling  x_i ~ N(0, L₀ₙ†)
# ---------------------------------------------------------------------------

def sample_connection_signals(
    L0n: torch.Tensor,
    T: int,
    rng: np.random.Generator,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Sample T signals from the improper Gaussian N(0, L₀ₙ†).

    Returns X: (Vn, T).
    """
    eigvals, eigvecs = torch.linalg.eigh(L0n)
    inv_sqrt = torch.where(eigvals > eps, eigvals.pow(-0.5), torch.zeros_like(eigvals))
    Z = torch.from_numpy(rng.standard_normal((L0n.shape[0], T)).astype(np.float32))
    return eigvecs @ (inv_sqrt.unsqueeze(1) * (eigvecs.t() @ Z))


# ---------------------------------------------------------------------------
# Label generation  (eq. 33 in the notes: stationary spectral filter)
# ---------------------------------------------------------------------------

def generate_labels_from_spectral_filter(
    X: torch.Tensor,
    L0: torch.Tensor,
    O0_blk: torch.Tensor,
    V: int,
    n: int,
    seed: int = 0,
    SNR_dB: float = 10.0,
    k1: float = 1.0,
) -> tuple:
    """Generate regression labels using the stationary spectral filter eq. (33).

    f(x) = U F(Λ) Uᵀ x,   where   F(Λ) = k1 − k2(Λ−λ1)²(Λ−λ2)²

    expanded to the full flat-bundle space via:
      f(x) = O₀ᵀ ( [U F(Λ) Uᵀ] ⊗ Iₙ ) O₀ x

    λ1 = 3rd non-zero eigenvalue of the scalar Laplacian L0
    λ2 = 12th non-zero eigenvalue of L0
    k2 chosen so that F(λ) ≥ 0 for all λ in the spectrum (with a safety margin).
    y_i = (1/Vn) 1ᵀ f(x_i) + noise,  SNR = SNR_dB dB.

    Args:
        X:    (Vn, T) signal matrix
        L0:   (V, V) scalar combinatorial Laplacian
        O0_blk: (Vn, Vn) block-diagonal SO(n) matrix
        V, n: graph dimensions
        seed, SNR_dB, k1: generation controls
    Returns:
        y:    (T,) regression labels
        meta: dict with lambda1, lambda2, k1, k2, F_Lambda
    """
    # Eigendecompose the scalar Laplacian (ascending order)
    eigvals, U = torch.linalg.eigh(L0)       # (V,), (V,V)

    # Locate non-zero eigenvalues
    nonzero_idx = (eigvals > 1e-6).nonzero(as_tuple=False).squeeze(-1)
    if len(nonzero_idx) < 12:
        raise ValueError(
            f"Need ≥12 non-zero eigenvalues of L0 but found {len(nonzero_idx)}. "
            "Use a denser graph (increase k_nn)."
        )
    lambda1 = eigvals[nonzero_idx[2]]    # 3rd non-zero
    lambda2 = eigvals[nonzero_idx[11]]   # 12th non-zero

    # k2: ensure F(λ) = k1 − k2(λ−λ1)²(λ−λ2)² ≥ 0 for all λ in spectrum
    poly_vals = (eigvals - lambda1) ** 2 * (eigvals - lambda2) ** 2
    max_poly = poly_vals.max().item()
    k2 = float(k1 / (2.0 * max_poly)) if max_poly > 0 else 0.0

    F_Lambda = k1 - k2 * (eigvals - lambda1) ** 2 * (eigvals - lambda2) ** 2  # (V,)

    # Scalar spectral filter matrix H = U diag(F(Λ)) Uᵀ  (V×V)
    H_scalar = U @ torch.diag(F_Lambda) @ U.t()

    # Extend to full flat-bundle space: Hₙ = H_scalar ⊗ Iₙ  (Vn×Vn)
    In = torch.eye(n, dtype=L0.dtype)
    H_n = torch.kron(H_scalar, In)

    # Apply flat-bundle filter: f(x) = O₀ᵀ Hₙ O₀ x
    Vn, T = X.shape
    f_x = O0_blk.t() @ H_n @ O0_blk @ X   # (Vn, T)

    # Graph-level readout: mean over all Vn node-stalk entries
    y_clean = f_x.mean(0)                   # (T,)

    sig_power = y_clean.var().item()
    if sig_power < 1e-12:
        sig_power = 1.0
    noise_std = float(np.sqrt(sig_power / (10 ** (SNR_dB / 10.0))))
    rng = np.random.default_rng(seed + 77777)
    noise = torch.from_numpy(rng.standard_normal(T).astype(np.float32)) * noise_std

    meta = dict(lambda1=lambda1.item(), lambda2=lambda2.item(),
                k1=k1, k2=k2, F_Lambda=F_Lambda)
    return y_clean + noise, meta


# ---------------------------------------------------------------------------
# Full task generator
# ---------------------------------------------------------------------------

def generate_sheaf_task(
    V: int,
    n: int,
    k_nn: int,
    T_train: int,
    T_test: int,
    seed: int,
    SNR_dB: float = 10.0,
) -> dict:
    """Generate a flat-bundle regression task.

    Returns a dict with:
      X_train, X_test : (Vn, T_*, 1)
      y_train, y_test : (T_*,)
      graph           : output of generate_rbf_connection_graph
      complete_ei     : (2, V(V-1)/2) complete graph edge_index
      h0              : (K_label+1,) true filter taps
    """
    rng = np.random.default_rng(seed)
    graph = generate_rbf_connection_graph(V, n, k_nn, rng)
    Vn = V * n
    T_total = T_train + T_test

    X_all = sample_connection_signals(graph["L0n"], T_total, rng)  # (Vn, T_total)
    y_all, label_meta = generate_labels_from_spectral_filter(
        X_all, graph["L0"], graph["O0_blk"], V, n, seed=seed, SNR_dB=SNR_dB
    )

    perm = rng.permutation(T_total)
    tr_idx = torch.from_numpy(perm[:T_train].copy())
    te_idx = torch.from_numpy(perm[T_train:].copy())

    return dict(
        X_train=X_all[:, tr_idx].unsqueeze(-1),
        y_train=y_all[tr_idx],
        X_test=X_all[:, te_idx].unsqueeze(-1),
        y_test=y_all[te_idx],
        graph=graph,
        complete_ei=complete_graph_edge_index(V),
        label_meta=label_meta,
    )


# ---------------------------------------------------------------------------
# Baseline Laplacian estimators (used by Kron-Fixed)
# ---------------------------------------------------------------------------

def project_to_scalar(X: torch.Tensor, V: int, n: int) -> torch.Tensor:
    """Average stalk dims: (Vn, T, F) → (V, T, F)."""
    Vn, T, F = X.shape
    return X.reshape(V, n, T, F).mean(1)


def estimate_laplacian_sample(X: torch.Tensor, V: int, n: int) -> torch.Tensor:
    """V×V Laplacian from the pseudo-inverse of the sample covariance on projected signal."""
    Xs = project_to_scalar(X, V, n)
    V_, T, F = Xs.shape
    Xf = Xs.permute(1, 2, 0).reshape(T * F, V_)
    C = Xf.t() @ Xf / (T * F)
    ev, evec = torch.linalg.eigh(C)
    inv_lam = torch.where(ev > 1e-6, 1.0 / ev, torch.zeros_like(ev))
    Theta = evec @ torch.diag(inv_lam) @ evec.t()
    return _precision_to_laplacian(Theta)


def estimate_laplacian_gl(
    X: torch.Tensor,
    V: int,
    n: int,
    lam: float = 0.1,
    eta: float = 0.01,
    max_iter: int = 500,
    eps: float = 1e-4,
) -> torch.Tensor:
    """V×V Laplacian via graphical lasso on the averaged (scalar) signal."""
    Xs = project_to_scalar(X, V, n)
    V_, T, F = Xs.shape
    Xf = Xs.permute(1, 2, 0).reshape(T * F, V_)
    C = Xf.t() @ Xf / (T * F)

    Theta = torch.eye(V, dtype=C.dtype)
    mask_off = 1.0 - torch.eye(V, dtype=C.dtype)
    for _ in range(max_iter):
        Theta_old = Theta.clone()
        grad = -torch.linalg.inv(Theta + eps * torch.eye(V)) + C
        Theta = Theta - eta * grad
        Theta_thresh = torch.sign(Theta) * torch.clamp(Theta.abs() - eta * lam, min=0.0)
        Theta = Theta * (1 - mask_off) + Theta_thresh * mask_off
        ev, evec = torch.linalg.eigh(Theta)
        Theta = evec @ torch.diag(torch.clamp(ev, min=1e-6)) @ evec.t()
        if (Theta - Theta_old).norm() / (Theta_old.norm() + 1e-12) < 1e-5:
            break
    return _precision_to_laplacian(Theta)


def _precision_to_laplacian(Theta: torch.Tensor) -> torch.Tensor:
    """Convert symmetric PSD matrix to valid graph Laplacian."""
    S = 0.5 * (Theta + Theta.t())
    off = S.abs() * (1.0 - torch.eye(S.shape[0], dtype=S.dtype))
    return torch.diag(off.sum(1)) - off
