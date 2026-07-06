"""operators.py

Linear operators mapping edge-weight vectors to (combinatorial / connection)
graph Laplacians and back, plus their adjoints.

Extracted from "Structured Learning of Consistent Connection Laplacians with
Spectral Constraints", Di Nino L., D'Acunto G., et al., 2025.
"""

import numpy as np

__all__ = [
    "L", "L_adjoint", "L_inv",
    "LKron", "LKron_adjoint", "L_spy",
]

#############################################################################
####################  COMBINATORIAL LAPLACIAN OPERATORS  ##################
#############################################################################


def L(w: np.ndarray, V: int) -> np.ndarray:
    """Map a vector of non-negative edge weights to a combinatorial graph Laplacian.

    Parameters
    ----------
    w : np.ndarray
        Edge weights vector of dimension V(V-1)/2.
    V : int
        Number of nodes in the graph.

    Returns
    -------
    np.ndarray
        V x V combinatorial graph Laplacian.
    """
    assert len(w) == (V * (V - 1)) // 2, "Invalid vector size for given dimension of the Laplacian"

    L_mat = np.zeros((V, V))
    upper_indices = np.triu_indices(V, k=1)
    L_mat[upper_indices] = -w
    L_mat += L_mat.T
    np.fill_diagonal(L_mat, -L_mat.sum(axis=1))

    return L_mat


def L_adjoint(M: np.ndarray) -> np.ndarray:
    """Adjoint of the linear operator L(w).

    Parameters
    ----------
    M : np.ndarray
        V x V matrix for which the adjoint must be computed.

    Returns
    -------
    np.ndarray
        Edge weights vector.
    """
    N = M.shape[1]
    j_idx, l_idx = np.triu_indices(N, k=1)
    return M[j_idx, j_idx] + M[l_idx, l_idx] - M[l_idx, j_idx] - M[j_idx, l_idx]


def L_inv(M: np.ndarray) -> np.ndarray:
    """Inverse of L(w): map a Laplacian-like matrix back to a vector of edge weights.

    Parameters
    ----------
    M : np.ndarray
        Input Laplacian-like matrix of dimension V x V.

    Returns
    -------
    np.ndarray
        Edge weights vector of dimension V(V-1)/2.
    """
    N = M.shape[1]
    i_idx, j_idx = np.triu_indices(N, k=1)
    return np.maximum(0, -M[i_idx, j_idx])


#############################################################################
######################  CONNECTION LAPLACIAN OPERATORS  ###################
#############################################################################


def LKron(w: np.ndarray, V: int, d: int) -> np.ndarray:
    """Map a vector of non-negative edge weights to a d-Kronecker combinatorial Laplacian.

    Parameters
    ----------
    w : np.ndarray
        Edge weights vector of dimension V(V-1)/2.
    V : int
        Number of nodes in the graph.
    d : int
        Dimension of the stalks over the nodes.

    Returns
    -------
    np.ndarray
        dV x dV d-Kronecker combinatorial graph Laplacian.
    """
    return np.kron(L(w, V), np.eye(d))


def LKron_adjoint(M: np.ndarray, d: int) -> np.ndarray:
    """Adjoint of the linear operator mapping edge weights to a d-Kronecker Laplacian.

    Parameters
    ----------
    M : np.ndarray
        dV x dV matrix for which the adjoint must be computed.
    d : int
        Dimension of the stalks over the nodes.

    Returns
    -------
    np.ndarray
        Edge weights vector of dimension V(V-1)/2.
    """
    N = M.shape[1] // d
    j_idx, l_idx = np.triu_indices(N, k=1)

    block_diag_traces = np.diag(M).reshape(N, d).sum(axis=1)

    r = np.arange(d)
    j_off = j_idx[:, None] * d + r[None, :]
    l_off = l_idx[:, None] * d + r[None, :]
    trace_jl = M[j_off, l_off].sum(axis=1)
    trace_lj = M[l_off, j_off].sum(axis=1)

    return block_diag_traces[j_idx] + block_diag_traces[l_idx] - trace_jl - trace_lj


def L_spy(L_mat: np.ndarray, d: int) -> np.ndarray:
    """Return the sparsity pattern of a Laplacian-like matrix.

    Parameters
    ----------
    L_mat : np.ndarray
        Laplacian-like matrix.
    d : int
        Stalk dimension.

    Returns
    -------
    np.ndarray
        Binary vector flagging non-zero edges.
    """
    N = L_mat.shape[1] // d
    i_idx, j_idx = np.triu_indices(N, k=1)
    r = np.arange(d)

    i_rows = (i_idx[:, None] * d + r[None, :])[:, :, None]
    j_cols = (j_idx[:, None] * d + r[None, :])[:, None, :]
    blocks = L_mat[i_rows, j_cols]

    return (~np.all(np.isclose(blocks, 0, atol=1e-8), axis=(1, 2))).astype(int)
