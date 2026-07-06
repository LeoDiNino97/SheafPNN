"""detection.py

Model-order / kernel-dimension detection used to estimate the number of
connected components (and hence the noise floor) from an empirical covariance.

Extracted from "Structured Learning of Consistent Connection Laplacians with
Spectral Constraints", Di Nino L., D'Acunto G., et al., 2025.
"""

import numpy as np

__all__ = ["minAICdetector"]


def minAICdetector(
    C: np.ndarray,
    M: int,
    d: int = 1,
    eps: float = 1e-12,
) -> int:
    """Detect the kernel dimension of a covariance matrix by minimizing the AIC.

    Parameters
    ----------
    C : np.ndarray
        Covariance matrix.
    M : int
        Number of observed signals.
    d : int
        Local observation (stalk) dimension.
    eps : float
        Numerical stability corrector on the eigenvalues.

    Returns
    -------
    int
        Estimated kernel dimension.
    """
    eigvals = np.sort(np.linalg.eigvalsh(C))[::-1]
    eigvals = np.maximum(eigvals, eps)
    V = len(eigvals)

    Kmax = V // d

    AICs = np.zeros(Kmax)

    for k in range(Kmax):
        rank_k = k * d

        noise_eigs = eigvals[rank_k:]
        p_k = len(noise_eigs)

        log_geo = np.mean(np.log(noise_eigs))
        log_arith = np.log(np.mean(noise_eigs))
        ratio_log = log_geo - log_arith

        AICs[k] = -2 * M * p_k * ratio_log + 2 * rank_k * (2 * V - rank_k) + 1

    k_AIC = np.argmin(AICs)

    # Emit the kernel dimension
    return Kmax - k_AIC
