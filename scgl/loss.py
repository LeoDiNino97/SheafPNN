"""loss.py

Objective function value for the SCGL problem, useful for monitoring
convergence.

Extracted from "Structured Learning of Consistent Connection Laplacians with
Spectral Constraints", Di Nino L., D'Acunto G., et al., 2025.
"""

import numpy as np

from .operators import LKron

__all__ = ["loss_"]


def loss_(
    V: int,
    d: int,
    X: np.ndarray,
    Z: np.ndarray,
    U: np.ndarray,
    O: np.ndarray,
    w: np.ndarray,
    S: np.ndarray,
    lambda_: np.ndarray,
    gamma: float,
    beta: float,
    alpha: float,
    noisy: bool,
    proximal_mode: str,
) -> float:
    """Compute the SCGL loss value given the current state of the algorithm.

    Parameters
    ----------
    V : int
        Number of nodes.
    d : int
        Stalk dimension.
    X : np.ndarray
        Observed signals.
    Z : np.ndarray
        Current estimate of the denoised signals.
    U : np.ndarray
        Current estimate of the Laplacian eigenvectors.
    O : np.ndarray
        Current estimate of the node frames.
    w : np.ndarray
        Current estimate of the edge weights.
    S : np.ndarray
        Current estimate of the covariance matrix.
    lambda_ : np.ndarray
        Current estimate of the Laplacian eigenvalues.
    gamma : float
        Reconstruction-error weight.
    beta : float
        Consistency / spectral regularization strength.
    alpha : float
        Sparsity regularization strength.
    noisy : bool
        Whether the noisy reconstruction term is included.
    proximal_mode : str
        Sparsifying penalty mode ('Proximal-L1' or otherwise log-penalized).

    Returns
    -------
    float
        Loss function value.
    """
    reg = alpha * np.linalg.norm(w, ord=1) if proximal_mode == "Proximal-L1" else np.sum(alpha * np.log(w + 1e-8))

    base = (
        -d * np.sum(np.log(lambda_))
        + np.trace(S @ O.T @ LKron(w, V, d) @ O)
        + 0.5 * beta * np.linalg.norm(LKron(w, V, d) - U @ np.kron(np.diag(lambda_), np.eye(d)) @ U.T) ** 2
        + reg
    )

    if noisy:
        base += gamma * np.linalg.norm(X - Z) ** 2 / X.shape[1]

    return base
