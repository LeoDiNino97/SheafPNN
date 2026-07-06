"""
Algorithmic update steps for the Flat Bundle Neural Network.

The joint optimization problem (eq. 14–17 in the notes) is:

  min_{h,w,O,z,P,Q,B}
    α F(h,w,O)
    + (1-α)[Tr(C P^T Ln(z) P) - logdet(L(z)+εI) + λ ψ(z)]
    + γ1/2 ‖w - z‖²
    + γ2/2 ‖O - P‖²_F
    + γ3/2 ‖P - Q + B‖²_F
  s.t.  w, z ≥ 0
        Q = blkdiag({Q_v}_{v∈V}),  Q_v ∈ SO(n)

The alternating steps are:
  1. Neural update  (h, w, O) — gradient descent via autograd
  2. Update z       — projected proximal gradient (MM linearisation)
  3. Update P       — closed-form linear solve (spectral implementation)
  4. Update Q       — Kabsch retraction to SO(n) per block
  5. Update B       — ADMM dual accumulation

References: notes Section 3.1, eqs. (20)–(32).
"""

import torch
from ..operators.laplacian import (
    build_laplacian,
    build_connection_laplacian,
    adjoint_L,
    adjoint_Ln,
)


# ---------------------------------------------------------------------------
# Covariance
# ---------------------------------------------------------------------------

def compute_sample_covariance(X: torch.Tensor) -> torch.Tensor:
    """Compute empirical covariance C = (1/T) X_flat X_flat^T ∈ R^{Vn×Vn}.

    Args:
        X: (Vn, T, F_in).  If F_in > 1 each (sample, feature) pair is treated
           as an independent observation.
    Returns:
        C: (Vn, Vn) symmetric PSD matrix
    """
    Vn, T, F_in = X.shape
    X_flat = X.permute(1, 2, 0).reshape(T * F_in, Vn)   # (T*F_in, Vn)
    C = X_flat.t() @ X_flat / (T * F_in)                 # (Vn, Vn)
    return C


# ---------------------------------------------------------------------------
# GL loss (for monitoring)
# ---------------------------------------------------------------------------

def compute_gl_loss(
    C_sample: torch.Tensor,
    P: torch.Tensor,
    z: torch.Tensor,
    epsilon: float,
    alpha: float,
    lam: float,
    edge_index: torch.Tensor,
    V: int,
    n: int,
) -> torch.Tensor:
    """Evaluate the GL term G(h,L) (eq. 16 in notes).

    G = Tr(C P^T Ln(z) P) - logdet(L(z)+εI) + λ ‖z‖_1
    """
    Ln_z = build_connection_laplacian(z, edge_index, V, n)
    L_z = build_laplacian(z, edge_index, V)
    PCP_T = P @ C_sample @ P.t()
    tr_term = torch.trace(PCP_T @ Ln_z)
    L_reg = L_z + epsilon * torch.eye(V, device=z.device, dtype=z.dtype)
    logdet_term = torch.logdet(L_reg)
    l1_term = z.sum()
    return tr_term - logdet_term + lam * l1_term


# ---------------------------------------------------------------------------
# Update z  (proximal gradient, eq. 24)
# ---------------------------------------------------------------------------

def update_z(
    z: torch.Tensor,
    w: torch.Tensor,
    P: torch.Tensor,
    C_sample: torch.Tensor,
    alpha: float,
    gamma1: float,
    lam: float,
    epsilon: float,
    eta_z: float,
    edge_index: torch.Tensor,
    V: int,
    n: int,
    penalty: str = "l1",
) -> torch.Tensor:
    """Proximal gradient step on z (MM linearisation of GL objective).

    Full gradient of the z-subproblem:
      (1-α)[Ln^*(P C P^T) - L^*((L(z)+εI)^{-1}) + λ ∇ψ(z)] + γ1(z - w)

    Then project onto R^+.

    Args:
        z: current edge-weight estimate (|E|,)
        w: neural edge weights (|E|,)
        P: current block-diagonal proxy (Vn, Vn)
        C_sample: (Vn, Vn) empirical covariance
        alpha: task/model trade-off in [0,1]
        gamma1: proximity penalty between w and z
        lam: sparsity penalty strength
        epsilon: diagonal loading for logdet stability
        eta_z: step size
        edge_index: (2, |E|) with i > j
        V: number of nodes
        n: stalk dimension
        penalty: 'l1' | 'log' sparsity inducing penalty
    Returns:
        z_new: (|E|,) updated and projected weights
    """
    z = z.detach()

    # --- gradient of Tr(C P^T Ln(z) P) w.r.t. z ---
    PCP_T = (P.detach() @ C_sample @ P.detach().t()).detach()
    grad_tr = adjoint_Ln(PCP_T, edge_index, V, n)

    # --- gradient of -logdet(L(z)+εI) w.r.t. z ---
    L_z = build_laplacian(z, edge_index, V)
    L_reg = L_z + epsilon * torch.eye(V, device=z.device, dtype=z.dtype)
    L_inv = torch.linalg.inv(L_reg)
    grad_logdet = -adjoint_L(L_inv, edge_index, V)

    # --- gradient of sparsity penalty ψ(z) ---
    if penalty == "l1":
        grad_psi = torch.ones_like(z)
    elif penalty == "log":
        grad_psi = 1.0 / (z + 1e-6)
    else:
        raise ValueError(f"Unknown penalty: {penalty}")

    # --- coupling term ---
    grad_coupling = gamma1 * (z - w.detach())

    full_grad = (1 - alpha) * (grad_tr + grad_logdet + lam * grad_psi) + grad_coupling
    z_new = torch.clamp(z - eta_z * full_grad, min=0.0)
    return z_new.detach()


# ---------------------------------------------------------------------------
# Update P  (linear solve, eq. 26, spectral implementation)
# ---------------------------------------------------------------------------

def update_P(
    O: torch.Tensor,
    Q: torch.Tensor,
    B: torch.Tensor,
    C_sample: torch.Tensor,
    z: torch.Tensor,
    alpha: float,
    gamma2: float,
    gamma3: float,
    edge_index: torch.Tensor,
    V: int,
    n: int,
    eps_eig: float = 1e-8,
) -> torch.Tensor:
    """Closed-form update for P via spectral linear solve.

    Solves the Sylvester-like equation (from KKT of eq. 25):
      2(1-α) Ln(z) P C + (γ2+γ3) P = γ2 O + γ3(Q - B)

    Using eigendecompositions of L(z) (V×V) and C_sample (Vn×Vn) to avoid
    forming the full (Vn)²×(Vn)² Kronecker system.

    Since Ln(z) = L(z) ⊗ I_n, eigenvalues of Ln are those of L repeated n
    times, and the eigenvectors are Ũ_L ⊗ I_n.

    Args:
        O: (Vn, Vn) unconstrained block-diagonal (current neural O)
        Q: (Vn, Vn) block-diagonal SO(n) (current ADMM Q)
        B: (Vn, Vn) ADMM dual variable
        C_sample: (Vn, Vn) empirical covariance
        z: (|E|,) current GL edge weights
        alpha, gamma2, gamma3: hyperparameters
        edge_index: (2, |E|)
        V: number of nodes
        n: stalk dimension
        eps_eig: regularisation added to eigenvalue denominators
    Returns:
        P: (Vn, Vn) updated matrix
    """
    Vn = V * n
    device = z.device
    dtype = z.dtype

    RHS = gamma2 * O.detach() + gamma3 * (Q.detach() - B.detach())

    # Eigendecompose L(z) (V×V)
    L_z = build_laplacian(z.detach(), edge_index, V)
    eig_L, U_L = torch.linalg.eigh(L_z)          # (V,), (V,V)

    # Eigendecompose C_sample (Vn×Vn)
    eig_C, U_C = torch.linalg.eigh(C_sample)      # (Vn,), (Vn,Vn)

    # Eigenvalues of Ln(z) = eig_L[i] repeated n times → (Vn,)
    lam_L = eig_L.repeat_interleave(n)            # (Vn,)

    # Transform RHS: P̃ = (U_L^T ⊗ I_n) RHS U_C
    # (U_L^T ⊗ I_n) @ M  with M (Vn, Vn):
    #   reshape M → (V, n, Vn), apply U_L^T along first dim, reshape back
    RHS_r = RHS.reshape(V, n, Vn)
    RHS_tilde = torch.einsum("ij,jkl->ikl", U_L.t(), RHS_r).reshape(Vn, Vn) @ U_C

    # Element-wise denominator: 2(1-α) λ_L_i λ_C_j + (γ2+γ3)
    denom = (
        2.0 * (1.0 - alpha) * lam_L.unsqueeze(1) * eig_C.unsqueeze(0)
        + (gamma2 + gamma3)
        + eps_eig
    )  # (Vn, Vn)

    P_tilde = RHS_tilde / denom   # (Vn, Vn)

    # Back-transform: P = (U_L ⊗ I_n) P̃ U_C^T
    P_r = (P_tilde @ U_C.t()).reshape(V, n, Vn)
    P = torch.einsum("ij,jkl->ikl", U_L, P_r).reshape(Vn, Vn)
    return P.detach()


# ---------------------------------------------------------------------------
# SO(n) retraction  (Kabsch algorithm)
# ---------------------------------------------------------------------------

def so_retraction(A: torch.Tensor) -> torch.Tensor:
    """Project n×n matrix A onto SO(n).

    Uses the SVD-based Kabsch algorithm:
      A = U Σ V^T  →  Q = U diag(1,...,1,det(UV^T)) V^T

    This ensures det(Q) = +1.
    """
    U, _, Vt = torch.linalg.svd(A)
    d = torch.det(U @ Vt)
    correction = torch.ones(A.shape[0], device=A.device, dtype=A.dtype)
    correction[-1] = d
    return U @ torch.diag(correction) @ Vt


# ---------------------------------------------------------------------------
# Update Q  (Kabsch retraction per block, eq. 30–31)
# ---------------------------------------------------------------------------

def update_Q(P: torch.Tensor, B: torch.Tensor, V: int, n: int) -> torch.Tensor:
    """Retract P+B to the block-diagonal SO(n) manifold.

    For each node v, applies Kabsch to the v-th diagonal n×n block of P+B.

    Args:
        P: (Vn, Vn) current P
        B: (Vn, Vn) ADMM dual variable
        V: number of nodes
        n: stalk dimension
    Returns:
        Q: (Vn, Vn) block-diagonal SO(n) matrix
    """
    PB = (P + B).detach()
    Q = torch.zeros_like(PB)
    for v in range(V):
        sl = slice(v * n, (v + 1) * n)
        Q[sl, sl] = so_retraction(PB[sl, sl])
    return Q


# ---------------------------------------------------------------------------
# Update B  (ADMM dual accumulation, eq. 32)
# ---------------------------------------------------------------------------

def update_B(B: torch.Tensor, P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """ADMM dual variable update: B ← B + (P - Q)."""
    return (B + P - Q).detach()


# ---------------------------------------------------------------------------
# Update T  (Kabsch retraction for the O splitting variable)
# ---------------------------------------------------------------------------

def update_T(O: torch.Tensor, K: torch.Tensor, V: int, n: int) -> torch.Tensor:
    """Retract (O + K) to block-diagonal SO(n) — analogous to update_Q for (P + B).

    For each node v, apply so_retraction to the v-th diagonal n×n block of O + K.

    Args:
        O: (Vn, Vn) current neural O matrix (block-diagonal, unconstrained)
        K: (Vn, Vn) dual variable for the O splitting
        V: number of nodes
        n: stalk dimension
    Returns:
        T: (Vn, Vn) block-diagonal SO(n) matrix
    """
    OK = (O + K).detach()
    T = torch.zeros_like(OK)
    for v in range(V):
        sl = slice(v * n, (v + 1) * n)
        T[sl, sl] = so_retraction(OK[sl, sl])
    return T


# ---------------------------------------------------------------------------
# Update K  (ADMM dual for O splitting, analogous to Update B for P)
# ---------------------------------------------------------------------------

def update_K(K: torch.Tensor, O: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """ADMM dual variable update: K ← K + (O − T)."""
    return (K + O - T).detach()


# ---------------------------------------------------------------------------
# Standalone graph pretraining (used by DISJ variants in the experiment)
# ---------------------------------------------------------------------------

def pretrain_graph(
    C_sample: torch.Tensor,
    edge_index: torch.Tensor,
    V: int,
    n: int,
    *,
    learn_O: bool = False,
    alpha: float = 0.0,
    gamma1: float = 1.0,
    gamma2: float = 1.0,
    gamma3: float = 1.0,
    gamma4: float = 1.0,
    lam: float = 0.05,
    epsilon: float = 1e-3,
    eta_z: float = 3e-4,
    eta_O: float = 0.01,
    penalty: str = "l1",
    nEpochs: int = 30,
    it_z: int = 8,
    device: str = "cpu",
) -> tuple:
    """Learn graph structure (z→w, O) without neural training (DISJ preprocessing).

    Runs the alternating graph-learning steps (z, P, Q, B, T, K) with α=0
    (no downstream task loss).  The neural filter taps h are never updated.

    Args:
        C_sample: empirical covariance — (V×V) for GL/Kron after projection,
                  or (Vn×Vn) for Kron/FB on full signal
        edge_index: (2, |E|) edge connectivity for the learned graph
        V: number of nodes
        n: stalk dimension (1 for GL, 2 for Kron/FB)
        learn_O: if True update O (FB mode); otherwise O stays at I (GL/Kron)
        alpha: task weight (0 = pure graph learning; kept as param for generality)
        gamma1..gamma4, lam, epsilon, eta_z, eta_O: algorithm hyperparameters
        nEpochs, it_z: iteration counts
        device: torch device
    Returns:
        w: (|E|,) learned edge weights
        O_blocks: (V, n, n) learned per-node frames
    """
    from ..operators.laplacian import build_blockdiag

    E = edge_index.shape[1]
    Vn = V * n

    edge_index = edge_index.to(device)
    C_sample = C_sample.to(device)

    w = torch.ones(E, device=device) * 0.1
    O_blocks = torch.eye(n, device=device).unsqueeze(0).expand(V, -1, -1).clone()
    z = torch.ones(E, device=device) * 0.1

    P = torch.eye(Vn, device=device)
    Q = torch.eye(Vn, device=device)
    B = torch.zeros(Vn, Vn, device=device)
    T = torch.eye(Vn, device=device)
    K = torch.zeros(Vn, Vn, device=device)

    for _ in range(nEpochs):
        O_blk = build_blockdiag(O_blocks)

        # z update (graph-side proximal gradient with P encoding current structure)
        for _ in range(it_z):
            z = update_z(
                z, w, P, C_sample,
                alpha=alpha, gamma1=gamma1, lam=lam,
                epsilon=epsilon, eta_z=eta_z,
                edge_index=edge_index, V=V, n=n, penalty=penalty,
            )

        # Keep w close to z (proximity step, no task loss)
        w = torch.clamp(w - eta_z * gamma1 * (w - z), min=0.0)

        if n > 1:
            # P, Q, B updates (needed for Kron and FB)
            P = update_P(
                O_blk.detach(), Q, B, C_sample, z,
                alpha=alpha, gamma2=gamma2, gamma3=gamma3,
                edge_index=edge_index, V=V, n=n,
            )
            Q = update_Q(P, B, V=V, n=n)
            B = update_B(B, P, Q)

        if learn_O and n > 1:
            # O gradient step: proximity to P and T (no task loss term)
            grad_O = (
                gamma2 * (O_blk - P.detach())
                + gamma4 * (O_blk - T.detach() + K.detach())
            )
            O_blk_new = (O_blk - eta_O * grad_O).detach()
            for v in range(V):
                sl = slice(v * n, (v + 1) * n)
                O_blocks[v] = O_blk_new[sl, sl]

            O_blk = build_blockdiag(O_blocks)
            T = update_T(O_blk, K, V, n)
            K = update_K(K, O_blk, T)

    return w.detach(), O_blocks.detach()
