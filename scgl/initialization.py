"""initialization.py

Initialization routines for the edge weights w and node frames O, used to
warm-start the SCGL alternating optimization.

Extracted from "Structured Learning of Consistent Connection Laplacians with
Spectral Constraints", Di Nino L., D'Acunto G., et al., 2025.
"""

import numpy as np
from scipy.optimize import minimize
from tqdm.auto import tqdm

from .operators import LKron, LKron_adjoint
from .updates import _build_O_problem
from .detection import minAICdetector
from pymanopt.optimizers import ConjugateGradient
import autograd.numpy as anp

__all__ = ["Initialization"]


def Initialization(
    S: np.ndarray,
    d: int,
    V: int,
    mode: str = "QP",
    noisy: bool = False,
    beta_0: int = None,
    loss_tracker: bool = False,
    bases: dict = None,
    MAX_ITER: int = 1000,
    reltol: float = 1e-4,
    abstol: float = 1e-6,
    seed: int = 42,
):
    """Initialize w and O for the SCGL main loop.

    Supported modes
    ----------------
    - ``'ID'``:     w is all ones, O_v = I_d for all v.
    - ``'RANDOM'``: w is all ones, O_v is a random rotation in O(d) for all v.
    - ``'QP'``:     jointly initializes w and O by solving
                    ``||pinv(S) - O^T LKron(w) O||^2`` under the structural
                    constraints on both w and O.
    - ``'ID-QP'``:  same objective, but O_v = I_d is fixed for all v (only w
                    is optimized).

    Parameters
    ----------
    S : np.ndarray
        Empirical covariance matrix of the observed 0-cochains.
    d : int
        Stalk dimension.
    V : int
        Number of nodes in the graph.
    mode : str
        Initialization modality, one of 'ID', 'ID-QP', 'QP', 'RANDOM'.
    noisy : bool
        Whether to account for observation noise when building pinv(S).
    beta_0 : int
        Assumed number of connected components, used to estimate the noise
        variance when ``noisy=True``. If left as ``None`` in noisy mode it is
        estimated automatically from ``S`` via :func:`minAICdetector`.
    loss_tracker : bool
        If True (QP mode only), also return the per-iteration loss history.
    bases : dict
        Optional prior knowledge on a subset of node frames, {v: O_v}.
    MAX_ITER : int
        Maximum number of iterations for the QP subroutine.
    reltol : float
        Relative tolerance on primal residuals to declare convergence in QP.
    abstol : float
        Absolute tolerance on primal residuals to declare convergence in QP.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple
        (w, O) or (w, O, sigma_2_hat) if ``noisy=True``; additionally the
        loss history if ``loss_tracker=True`` (QP mode only, noise-free case).
    """
    np.random.seed(seed)

    assert mode in ["ID", "ID-QP", "QP", "RANDOM"], "Invalid initialization modality"

    def RandomOn():
        Q, _ = np.linalg.qr(np.random.randn(d, d))
        if np.linalg.det(Q) < 0:
            Q[0, :] *= -1
        return Q

    if not noisy:
        S_pinv = np.linalg.pinv(S)
    else:
        if beta_0 is None:
            beta_0 = minAICdetector(S, V, d)

        Lambda, U = np.linalg.eigh(S)
        sigma_2_hat = np.mean(Lambda[0: d * beta_0])

        Lambda_hat = np.zeros_like(Lambda)
        Lambda_hat[d * beta_0:] = 1 / (Lambda[d * beta_0:] - sigma_2_hat)

        S_pinv = U @ np.diag(Lambda_hat) @ U.T

    def w_Objective(w, O):
        return 0.5 * np.linalg.norm(LKron(w, V, d) - O @ S_pinv @ O.T, "fro") ** 2

    def w_grad(w, O):
        return LKron_adjoint(LKron(w, V, d) - O @ S_pinv @ O.T, d)

    def O_Update(O_init, w, V, d, bases):
        C = LKron(w, V, d)
        map_idx, free_vs, problem = _build_O_problem(S_pinv, C, V, d, bases, sign=-1)

        bases_resolved = bases if bases is not None else {}

        if problem is None:
            O_full = anp.zeros((d * V, d * V))
            for v in range(V):
                O_full[v * d:(v + 1) * d, v * d:(v + 1) * d] = bases_resolved[v]
            return O_full

        solver_obj = ConjugateGradient(verbosity=0)
        result = solver_obj.run(problem, initial_point=O_init).point

        O_full = anp.zeros((d * V, d * V))
        for v in range(V):
            if v in bases_resolved:
                O_full[v * d:(v + 1) * d, v * d:(v + 1) * d] = bases_resolved[v]
            else:
                O_full[v * d:(v + 1) * d, v * d:(v + 1) * d] = result[map_idx[v]]

        return O_full

    def O_spack(O, d, V, bases=bases):
        O_blocks = []
        bases_local = bases if bases is not None else {}
        for v in range(V):
            if v not in bases_local:
                O_blocks.append(O[v * d:(v + 1) * d, v * d:(v + 1) * d])
        return O_blocks

    w = np.ones(int(0.5 * (V - 1) * V))
    O = np.eye(d * V)

    if mode in ("ID", "ID-QP"):
        if bases is not None:
            for v in bases.keys():
                O[v * d:(v + 1) * d, v * d:(v + 1) * d] = bases[v]

        if mode == "ID-QP":
            w = minimize(
                w_Objective, w, args=(O,), jac=w_grad,
                method="L-BFGS-B", bounds=[(0, None)] * len(w),
            ).x

        if noisy:
            return w, O, sigma_2_hat
        return w, O

    if mode == "RANDOM":
        for v in range(V):
            if bases is not None and v in bases.keys():
                O[v * d:(v + 1) * d, v * d:(v + 1) * d] = bases[v]
            else:
                O[v * d:(v + 1) * d, v * d:(v + 1) * d] = RandomOn()

        if noisy:
            return w, O, sigma_2_hat
        return w, O

    # mode == 'QP'
    bounds = [(0, None)] * len(w)
    loss = np.zeros(MAX_ITER)
    iteration = 0

    for iteration in tqdm(range(MAX_ITER), desc="SCGL init (QP)"):
        w_hat = minimize(
            w_Objective, w, args=(O,), jac=w_grad,
            method="L-BFGS-B", bounds=bounds,
        ).x

        O_hat = O_Update(O_spack(O, d, V, bases=bases), w, V, d, bases=bases)

        loss[iteration] = np.linalg.norm(S_pinv - O.T @ LKron(w, V, d) @ O) ** 2

        w_err = np.abs(w - w_hat)
        O_err = np.abs(O - O_hat)

        converged_w = np.all(w_err <= 0.5 * reltol * (w + w_hat)) or np.all(w_err <= abstol)
        converged_O = np.all(O_err <= 0.5 * reltol * (O + O_hat)) or np.all(O_err <= abstol)

        w = w_hat
        O = O_hat

        if converged_w and converged_O:
            break

    if loss_tracker:
        return w, O, loss[:iteration]

    if noisy:
        return w, O, sigma_2_hat
    return w, O
