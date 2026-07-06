"""updates.py

Per-block update (proximal / Riemannian / ADMM / isotonic) steps for the SCGL
alternating optimization scheme.

Extracted from "Structured Learning of Consistent Connection Laplacians with
Spectral Constraints", Di Nino L., D'Acunto G., et al., 2025.
"""

import numpy as np
import autograd.numpy as anp
import pymanopt
from pymanopt import Problem
from pymanopt.manifolds import Stiefel, Product
from pymanopt.optimizers import ConjugateGradient, SteepestDescent, TrustRegions
from scipy.linalg import block_diag

from .operators import L, LKron, LKron_adjoint

__all__ = [
    "Update_Z", "Update_w", "Update_O_RG", "Update_O_SOC",
    "Update_U", "Update_Lambda",
]


def Update_Z(
    w: np.ndarray,
    O: np.ndarray,
    gamma: float,
    X: np.ndarray,
    V: int,
    d: int,
) -> np.ndarray:
    """IIR-filter update for the denoised signals Z.

    Parameters
    ----------
    w : np.ndarray
        Edge weights of the graph.
    O : np.ndarray
        Local node frames.
    gamma : float
        Regularization parameter.
    X : np.ndarray
        Signals to be filtered, shape (Vd, num_samples).
    V : int
        Number of nodes.
    d : int
        Stalk dimension.

    Returns
    -------
    np.ndarray
        Filtered signals.
    """
    LL_hat = O.T @ LKron(w=w, V=V, d=d) @ O
    return np.linalg.solve(gamma * np.eye(V * d) + LL_hat, gamma * X)


def Update_w(
    w: np.ndarray,
    U: np.ndarray,
    S: np.ndarray,
    O: np.ndarray,
    lambda_: np.ndarray,
    alpha: float,
    beta: float,
    gamma: float,
    V: int,
    d: int,
    its: int = 1,
    eps: float = 1e-8,
    proximal_mode: str = "Proximal-L1",
    exact_linesearch: bool = False,
) -> np.ndarray:
    """Proximal / projected-gradient MM step in the edge weights w.

    Parameters
    ----------
    w : np.ndarray
        Current iterate value for w.
    U : np.ndarray
        Current iterate value for U (eigenvectors of the connection Laplacian).
    S : np.ndarray
        Empirical covariance matrix of the observed 0-cochains.
    O : np.ndarray
        Current iterate value for O (block-diagonal local node frames).
    lambda_ : np.ndarray
        Current iterate value for the Laplacian spectrum.
    alpha : float
        Sparsity penalization strength.
    beta : float
        Consistency / spectral regularization strength.
    gamma : float
        Reconstruction-error weight.
    V : int
        Number of nodes in the graph.
    d : int
        Stalk dimension.
    its : int
        Number of descent steps to perform on w.
    proximal_mode : str
        One of 'Proximal-L1', 'Proximal-LOG', 'ReweightedL1'.
    exact_linesearch : bool
        Whether to use an exact (Armijo-free) linesearch or the reciprocal of
        the operator norm as the step size.

    Returns
    -------
    np.ndarray
        Refined estimate of w.
    """
    assert proximal_mode in ["Proximal-L1", "Proximal-LOG", "ReweightedL1"], \
        "Please choose a valid proximal modality"

    S_hat = U @ (np.kron(np.diag(lambda_), np.eye(d))) @ U.T - (1 / beta) * O @ S @ O.T

    def PositiveProxL1(x, th):
        return np.maximum(0, x - th)

    for _ in range(its):
        grad = LKron_adjoint(LKron(w, V, d) - S_hat, d)

        if exact_linesearch:
            c = LKron_adjoint(S_hat, d)
            L_kron_adg_grad = LKron_adjoint(LKron(grad, V, d), d)
            mu = np.dot(grad, L_kron_adg_grad) / (np.dot(w, L_kron_adg_grad) - np.dot(c, grad))
        else:
            mu = 2 * V * d

        if proximal_mode == "Proximal-L1":
            w_hat = PositiveProxL1(w - (1 / mu) * grad, (alpha / beta))

        elif proximal_mode == "Proximal-LOG":
            grad = grad + (alpha / beta) * 1 / (w + eps)
            w_hat = np.maximum(0, w - (1 / mu) * grad)

        elif proximal_mode == "ReweightedL1":
            w_hat = PositiveProxL1(w - (1 / mu) * grad, (alpha / beta) * 1 / (w + 1e-10))

        w = w_hat

    return w


def _build_O_problem(
    A: np.ndarray,
    C: np.ndarray,
    V: int,
    d: int,
    bases: dict,
    sign: int = 1,
) -> tuple:
    """Build a pymanopt Problem for sign * trace(O A O^T C) over block-diagonal SO(d).

    Shared by Update_O_RG (sign=+1, minimize smoothness) and the QP
    initialization (sign=-1, maximize alignment with S_pinv).

    Parameters
    ----------
    A : np.ndarray
        Symmetric matrix (covariance S or pseudo-inverse S_pinv).
    C : np.ndarray
        Symmetric Kronecker Laplacian LKron(w, V, d).
    V : int
        Number of nodes.
    d : int
        Stalk dimension.
    bases : dict
        Fixed bases for a subset of nodes {v: O_v}; None means all free.
    sign : int
        +1 to minimize trace, -1 to maximize (i.e. minimize -trace).

    Returns
    -------
    tuple
        (map_idx, free_vs, problem), or (None, None, None) if all bases are fixed.
    """
    bases_resolved = bases if bases is not None else {}
    free_vs = [v for v in range(V) if v not in bases_resolved]

    if not free_vs:
        return None, None, None

    manifold = Product([Stiefel(d, d, retraction="polar") for _ in free_vs])
    map_idx = {v: i for i, v in enumerate(free_vs)}

    def _assemble(*O_blocks):
        O_mat = anp.zeros((d * V, d * V), dtype=anp.float64)
        for v in range(V):
            if v in bases_resolved:
                O_mat[v * d:(v + 1) * d, v * d:(v + 1) * d] = anp.array(bases_resolved[v])
            else:
                block = O_blocks[map_idx[v]]
                val = block if isinstance(block, anp.ndarray) else block._value
                O_mat[v * d:(v + 1) * d, v * d:(v + 1) * d] = anp.array(val)
        return O_mat

    @pymanopt.function.autograd(manifold)
    def cost(*O_blocks):
        O_mat = _assemble(*O_blocks)
        return sign * anp.trace(O_mat @ A @ O_mat.T @ C)

    @pymanopt.function.autograd(manifold)
    def euclidean_gradient(*O_blocks):
        O_mat = _assemble(*O_blocks)
        gradients = []
        for v in free_vs:
            g = np.zeros((d, d), dtype=np.float64)
            g += (sign * 2 *
                  C[v * d:(v + 1) * d, v * d:(v + 1) * d] @
                  O_mat[v * d:(v + 1) * d, v * d:(v + 1) * d] @
                  A[v * d:(v + 1) * d, v * d:(v + 1) * d])
            for m in range(V):
                if m != v:
                    g += sign * (
                        C[v * d:(v + 1) * d, m * d:(m + 1) * d] @
                        O_mat[m * d:(m + 1) * d, m * d:(m + 1) * d] @
                        A[m * d:(m + 1) * d, v * d:(v + 1) * d]
                        +
                        A[m * d:(m + 1) * d, v * d:(v + 1) * d] @
                        O_mat[m * d:(m + 1) * d, m * d:(m + 1) * d].T @
                        C[v * d:(v + 1) * d, m * d:(m + 1) * d]
                    )
            gradients.append(g)
        return gradients

    problem = Problem(manifold, cost=cost, euclidean_gradient=euclidean_gradient)
    return map_idx, free_vs, problem


def Update_O_RG(
    O: np.ndarray,
    S: np.ndarray,
    w: np.ndarray,
    V: int,
    d: int,
    O_init: bool = True,
    max_its: int = 10,
    bases: dict = None,
    solver: str = "RCG",
) -> np.ndarray:
    """Riemannian-gradient step in the block-diagonal node frames O (via pymanopt).

    Parameters
    ----------
    O : np.ndarray
        Current iterate value for O.
    S : np.ndarray
        Empirical covariance matrix of the observed 0-cochains.
    w : np.ndarray
        Current iterate value for w.
    V : int
        Number of nodes in the graph.
    d : int
        Stalk dimension.
    O_init : bool
        Whether to warm-start the Riemannian solver from O.
    max_its : int
        Maximum number of iterations of the Riemannian subroutine.
    bases : dict
        Optional prior knowledge fixing a subset of node frames.
    solver : str
        One of 'RCG', 'RSD', 'TR'.

    Returns
    -------
    np.ndarray
        Refined estimate of O.
    """
    assert solver in ["RCG", "RSD", "TR"], "Invalid identifier for the Riemannian solver"

    C = LKron(w, V, d)
    map_idx, free_vs, problem = _build_O_problem(S, C, V, d, bases, sign=1)

    if problem is None:
        return O

    if solver == "RCG":
        solver_obj = ConjugateGradient(verbosity=0, max_iterations=max_its)
    elif solver == "RSD":
        solver_obj = SteepestDescent(verbosity=0, max_iterations=max_its)
    else:
        solver_obj = TrustRegions(verbosity=0, max_iterations=max_its)

    if O_init:
        init_point = [O[v * d:(v + 1) * d, v * d:(v + 1) * d] for v in free_vs]
        result = solver_obj.run(problem, initial_point=init_point).point
    else:
        result = solver_obj.run(problem).point

    bases_resolved = bases if bases is not None else {}
    O_full = anp.zeros((d * V, d * V))
    for v in range(V):
        if v in bases_resolved:
            O_full[v * d:(v + 1) * d, v * d:(v + 1) * d] = bases_resolved[v]
        else:
            O_full[v * d:(v + 1) * d, v * d:(v + 1) * d] = result[map_idx[v]]

    return O_full


def Update_O_SOC(
    O: np.ndarray,
    Z: np.ndarray,
    w: np.ndarray,
    V: int,
    d: int,
    rho: float,
    MAX_ITER: int = 1000,
    abs_tol: float = 1e-4,
    rel_tol: float = 1e-4,
) -> np.ndarray:
    """Splitting-Orthogonality-Constraint (ADMM) step in the node frames O.

    Parameters
    ----------
    O : np.ndarray
        Local node frames.
    Z : np.ndarray
        Current signal estimate, shape (Vd, num_samples).
    w : np.ndarray
        Current estimate of edge weights.
    V : int
        Number of nodes.
    d : int
        Stalk dimension.
    rho : float
        Convexification parameter.
    MAX_ITER : int
        Maximum number of ADMM iterations.
    abs_tol : float
        Absolute tolerance for convergence.
    rel_tol : float
        Relative tolerance for convergence.

    Returns
    -------
    np.ndarray
        Refined estimate of O, guaranteed to lie in block-diagonal SO(d).
    """

    def block_retraction(M: np.ndarray, V: int, d: int) -> np.ndarray:
        """Block-wise polar-factor retraction onto block-diagonal SO(d)."""
        P = M.reshape(V, d, V, d)
        P = P[np.arange(V), :, np.arange(V), :]
        U, _, Vt = np.linalg.svd(P)

        P_temp = U @ Vt
        dets = np.linalg.det(P_temp)
        mask = dets < 0
        if np.any(mask):
            U[mask, :, -1] *= -1
            P_temp[mask] = U[mask] @ Vt[mask]
        return block_diag(*list(P_temp))

    B = np.zeros_like(O)
    P = np.copy(O)
    n = V * d

    LambdaZ, UZ = np.linalg.eigh(Z @ Z.T / Z.shape[1])
    LambdaL, UL = np.linalg.eigh(LKron(w, V, d))

    LambdaZL = LambdaL[:, None] * LambdaZ[None, :] + rho

    for _ in range(MAX_ITER):
        P_old = np.copy(P)

        O = UL @ ((UL.T @ (P - B) @ UZ) * rho / LambdaZL) @ UZ.T
        P = block_retraction(O + B, V, d)

        B += (O - P)

        r_norm = np.linalg.norm(O - P, "fro")
        s_norm = np.linalg.norm(rho * (P - P_old), "fro")

        eps_pri = np.sqrt(n ** 2) * abs_tol + rel_tol * max(np.linalg.norm(O, "fro"), np.linalg.norm(P, "fro"))
        eps_dual = np.sqrt(n ** 2) * abs_tol + rel_tol * np.linalg.norm(rho * B, "fro")

        if r_norm <= eps_pri and s_norm <= eps_dual:
            break

    # P (not O) is returned: P always lies exactly in block-diagonal SO(d) by
    # construction of block_retraction, while O is the unconstrained ADMM
    # iterate and may have non-zero off-diagonal blocks if the loop exits
    # before full convergence.
    return P


def Update_U(
    w: np.ndarray,
    k: int,
    V: int,
    d: int,
) -> tuple:
    """Eigenvector update for U via the Von Neumann trace inequality solution.

    Parameters
    ----------
    w : np.ndarray
        Current iterate value for w.
    k : int
        Number of connected components in the graph.
    V : int
        Number of nodes in the graph.
    d : int
        Stalk dimension.

    Returns
    -------
    tuple
        (U_hat, eigenvalues) -- eigenvectors restricted to St(dV, d(V-k)) and
        the full eigenvalue array.
    """
    LC_hat = L(w, V)
    eigenvalues, U_hat = np.linalg.eigh(LC_hat)
    U_hat = np.kron(U_hat, np.eye(d))

    U_hat = U_hat[:, k * d:]
    return U_hat, eigenvalues


def Update_Lambda(
    U: np.ndarray,
    w: np.ndarray,
    beta: float,
    c1: float,
    c2: float,
    V: int,
    k: int,
    d: int,
) -> np.ndarray:
    """Isotonic-regression update for the estimated eigenvalues.

    Parameters
    ----------
    U : np.ndarray
        Current iterate value for U.
    w : np.ndarray
        Current iterate value for w.
    beta : float
        Consistency / spectral regularization strength.
    c1 : float
        Assumed lower bound on the eigenvalues.
    c2 : float
        Assumed upper bound on the eigenvalues.
    V : int
        Number of nodes in the graph.
    k : int
        Number of connected components in the graph.
    d : int
        Stalk dimension.

    Returns
    -------
    np.ndarray
        Refined estimate of lambda_.
    """
    M = U.T @ LKron(w, V, d) @ U

    q = V - k
    lambda_hat = np.zeros(q)

    D = np.zeros(q)
    for i in range(q):
        T = np.trace(M[i * d:(i + 1) * d, i * d:(i + 1) * d])
        D[i] = T
        lambda_hat[i] = 1 / (2 * d) * (T + np.sqrt(T ** 2 + (4 * d ** 2) / beta))

    counter = 0

    # Isotonic heuristic (Palomar D., 2019)
    while not (
        np.all(lambda_hat >= c1)
        and np.all(lambda_hat <= c2)
        and np.all(lambda_hat[:-1] <= lambda_hat[1:])
    ):
        if counter < V - k + 1:
            if np.any(lambda_hat < c1):
                r = np.max(np.where(lambda_hat < c1)[0])
                lambda_hat[:r + 1] = c1

            if np.any(lambda_hat > c2):
                s = np.min(np.where(lambda_hat > c2)[0])
                lambda_hat[s:] = c2

            if np.any(lambda_hat[:-1] > lambda_hat[1:]):
                for i in range(q - 1):
                    if lambda_hat[i] > lambda_hat[i + 1]:
                        m = i
                        while m + 1 < q and lambda_hat[m] > lambda_hat[m + 1]:
                            m += 1

                        d_avg = np.mean(D[i:m + 1])
                        new_val = (1 / (2 * d)) * (d_avg + np.sqrt(d_avg ** 2 + (4 * d ** 2) / beta))
                        lambda_hat[i:m + 1] = new_val
                        break

            counter += 1

        else:
            return lambda_hat

    return lambda_hat
