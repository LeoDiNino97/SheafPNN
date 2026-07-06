"""
Laplacian operators for flat-bundle / connection-graph learning.

Notation follows the notes:
  V    — number of graph nodes
  n    — stalk dimension (per-node orthogonal group dimension)
  |E|  — number of (undirected) edges; edge k = (i_k, j_k) with i_k > j_k
  w    — R^{|E|}_+ edge-weight vector
  L(w)   — V×V combinatorial Laplacian  (eq. 19 in notes)
  Ln(w)  — Vn×Vn block Laplacian = L(w) ⊗ I_n  (eq. 18 in notes)

All functions support autograd through w.
"""

import torch


# ---------------------------------------------------------------------------
# Laplacian builders
# ---------------------------------------------------------------------------

def build_laplacian(w: torch.Tensor, edge_index: torch.Tensor, V: int) -> torch.Tensor:
    """Build V×V combinatorial Laplacian from edge weights.

    Args:
        w: (|E|,) non-negative edge weights
        edge_index: (2, |E|) with edge_index[0,k] > edge_index[1,k]
        V: number of nodes
    Returns:
        L: (V, V) symmetric Laplacian
    """
    i_idx = edge_index[0]
    j_idx = edge_index[1]

    # All non-zero (row, col) positions and their signed contributions
    # Diagonal: +w at (i,i) and +w at (j,j)
    # Off-diagonal: -w at (i,j) and -w at (j,i)
    all_rows = torch.cat([i_idx, j_idx, i_idx, j_idx])
    all_cols = torch.cat([i_idx, j_idx, j_idx, i_idx])
    all_vals = torch.cat([w, w, -w, -w])

    idx_flat = all_rows * V + all_cols
    L_flat = torch.zeros(V * V, device=w.device, dtype=w.dtype)
    L_flat = L_flat.scatter_add(0, idx_flat, all_vals)
    return L_flat.reshape(V, V)


def build_connection_laplacian(
    w: torch.Tensor, edge_index: torch.Tensor, V: int, n: int
) -> torch.Tensor:
    """Build Vn×Vn block connection Laplacian Ln(w) = L(w) ⊗ I_n.

    For a consistent (flat) connection graph, Ln(w) = L(w) ⊗ I_n,
    meaning off-diagonal n×n blocks equal -w_k I_n and diagonal blocks
    equal (sum of incident weights) I_n.

    Args:
        w: (|E|,) non-negative edge weights
        edge_index: (2, |E|) with edge_index[0,k] > edge_index[1,k]
        V: number of nodes
        n: stalk dimension
    Returns:
        Ln: (Vn, Vn) connection Laplacian
    """
    L = build_laplacian(w, edge_index, V)
    In = torch.eye(n, device=w.device, dtype=w.dtype)
    return torch.kron(L, In)


# ---------------------------------------------------------------------------
# Adjoint operators  L^* and Ln^*
# ---------------------------------------------------------------------------

def adjoint_L(M: torch.Tensor, edge_index: torch.Tensor, V: int) -> torch.Tensor:
    """Adjoint of the scalar Laplacian map  L^*(M) ∈ R^{|E|}.

    [L^*(M)]_k = M_{i_k i_k} + M_{j_k j_k} - M_{i_k j_k} - M_{j_k i_k}

    Args:
        M: (V, V) matrix
        edge_index: (2, |E|) with edge_index[0] = i_idx, edge_index[1] = j_idx
        V: number of nodes (unused but kept for API symmetry)
    Returns:
        grad_w: (|E|,)
    """
    i_idx = edge_index[0]
    j_idx = edge_index[1]
    return M[i_idx, i_idx] + M[j_idx, j_idx] - M[i_idx, j_idx] - M[j_idx, i_idx]


def adjoint_Ln(M: torch.Tensor, edge_index: torch.Tensor, V: int, n: int) -> torch.Tensor:
    """Adjoint of the block Laplacian map  Ln^*(M) ∈ R^{|E|}.

    [Ln^*(M)]_k = Tr(M_{i_k,i_k}) + Tr(M_{j_k,j_k}) - Tr(M_{i_k,j_k}) - Tr(M_{j_k,i_k})
    where M_{ab} denotes the (a,b)-th n×n block of M.

    Since Ln(w) = L(w) ⊗ I_n, the adjoint reduces to:
      [Ln^*(M)]_k = Tr(M[i*n:(i+1)*n, i*n:(i+1)*n])  +  ...

    Args:
        M: (Vn, Vn) matrix (block-structured)
        edge_index: (2, |E|)
        V: number of nodes
        n: stalk dimension
    Returns:
        grad_w: (|E|,)
    """
    i_idx = edge_index[0]
    j_idx = edge_index[1]

    # Reshape M to (V, n, V, n) for block indexing
    M_blk = M.reshape(V, n, V, n)

    # Gather all four block-pairs at once.
    # M_blk[i_idx, :, i_idx, :] → (|E|, n, n) blocks (i,i)
    # Using advanced indexing: dims 0 and 2 are indexed by i_idx/j_idx;
    # dims 1 and 3 are full slices → result shape (|E|, n, n).
    M_ii = M_blk[i_idx, :, i_idx, :]   # (|E|, n, n)
    M_jj = M_blk[j_idx, :, j_idx, :]   # (|E|, n, n)
    M_ij = M_blk[i_idx, :, j_idx, :]   # (|E|, n, n)
    M_ji = M_blk[j_idx, :, i_idx, :]   # (|E|, n, n)

    tr = lambda B: torch.diagonal(B, dim1=1, dim2=2).sum(-1)   # (|E|,)
    return tr(M_ii) + tr(M_jj) - tr(M_ij) - tr(M_ji)


# ---------------------------------------------------------------------------
# Block-diagonal builder
# ---------------------------------------------------------------------------

def build_blockdiag(blocks: torch.Tensor) -> torch.Tensor:
    """Build Vn×Vn block-diagonal matrix from V per-node n×n matrices.

    Args:
        blocks: (V, n, n) tensor of per-node matrices
    Returns:
        O: (Vn, Vn) block-diagonal matrix (differentiable w.r.t. blocks)
    """
    V = blocks.shape[0]
    return torch.block_diag(*[blocks[v] for v in range(V)])
