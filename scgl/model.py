"""model.py

Structured Connection Graph Learning (SCGL): alternating-optimization solver
for connection Laplacians under sparsity + spectral consistency constraints.

Extracted and repackaged from "Structured Learning of Consistent Connection
Laplacians with Spectral Constraints", Di Nino L., D'Acunto G., et al., 2025.
@Author: Leonardo Di Nino
"""

import warnings

import numpy as np
from tqdm.auto import tqdm

from .initialization import Initialization
from .updates import Update_Z, Update_w, Update_O_RG, Update_O_SOC, Update_U, Update_Lambda
from .loss import loss_
from .operators import LKron

__all__ = ["SCGL", "learn_connection_laplacian"]


class SCGL:
    """Structured Connection Graph Learning.

    Learns a sparse connection Laplacian (edge weights ``w`` and block-diagonal
    node frames ``O``) that is jointly consistent with observed signals and
    with a k-component spectral prior.

    Parameters
    ----------
    V : int
        Number of nodes.
    d : int
        Stalk dimension.
    k : int
        Number of connected components (spectral prior).
    alpha : float
        Sparsity regularization strength.
    beta : float
        Consistency / spectral regularization strength (initial value if
        beta scheduling is enabled).
    eps : float
        Numerical stability control.
    initialization_mode : str
        One of 'ID', 'ID-QP', 'QP', 'RANDOM'.
    initialization_seed : int
        Seed for initialization.
    max_init_its : int
        Maximum number of iterations for the QP initialization.
    w_inits : np.ndarray
        Edge weights initialization, if given (skips the initialization routine).
    O_inits : np.ndarray
        Node basis initialization, if given (skips the initialization routine).
    max_w_its : int
        Number of descent steps per outer iteration for w.
    proximal_mode : str
        Sparsity-inducing penalty, one of 'Proximal-L1', 'Proximal-LOG', 'ReweightedL1'.
    exact_linesearch : bool
        Whether to use an exact linesearch in the w update.
    update_frames : bool
        Whether to learn the node bases O, or keep them fixed.
    max_O_its : int
        Number of descent steps per outer iteration for O.
    SOC : bool
        Whether to update O via Splitting-Orthogonality-Constraint ADMM (True)
        or Riemannian gradient descent (False).
    rho : float
        Convexification strength in the SOC routine.
    R_solver : str
        Riemannian solver identifier ('RCG', 'RSD', 'TR'), used when SOC=False.
    bases : dict
        Prior knowledge fixing a subset of node frames, {v: O_v}.
    noisy : bool
        Whether to jointly denoise the observed signals.
    c1, c2 : float
        Lower / upper bound on the Laplacian eigenvalues.
    beta_factor : float
        Beta scheduling multiplier.
    fix_beta : bool
        Disable beta scheduling.
    beta_min, beta_max : float
        Bounds for beta scheduling.
    rel_tol, abs_tol : float
        Relative / absolute tolerance to declare convergence on residuals.
    loss_tol : float
        Tolerance on loss decrease to reset the plateau-patience counter.
    patience : int
        Number of stagnant iterations before declaring convergence on a loss plateau.
    MAX_ITER : int
        Maximum number of outer iterations. Set to 0 to only run the initialization.
    verbose : bool
        Print progress information.
    logger : callable, optional
        Optional callback ``logger(metrics: dict, step: int)`` invoked once
        per outer iteration (e.g. to forward metrics to W&B, MLflow, a
        dataframe, ...). No monitoring backend is bundled with this library.
    show_progress : bool
        Whether to display a tqdm progress bar over outer iterations.
    """

    def __init__(
        self,
        V: int,
        d: int,
        k: int,
        alpha: float,
        beta: float,
        eps: float = 1e-8,
        initialization_mode: str = "QP",
        initialization_seed: int = 42,
        max_init_its: int = 1000,
        w_inits: np.ndarray = None,
        O_inits: np.ndarray = None,
        max_w_its: int = 1,
        proximal_mode: str = "Proximal-L1",
        exact_linesearch: bool = False,
        update_frames: bool = True,
        max_O_its: int = 100,
        SOC: bool = True,
        rho: float = 100,
        R_solver: str = "RCG",
        bases: dict = None,
        noisy: bool = False,
        c1: float = 1e-5,
        c2: float = 1e4,
        beta_factor: float = 5e-2,
        fix_beta: bool = False,
        beta_min: float = 10,
        beta_max: float = 3000,
        rel_tol: float = 1e-8,
        abs_tol: float = 1e-6,
        loss_tol: float = 1e-6,
        patience: int = 1000,
        MAX_ITER: int = 20000,
        verbose: bool = False,
        logger=None,
        show_progress: bool = True,
    ):
        self.V = V
        self.d = d
        self.k = k

        self.alpha = alpha
        self.beta = beta
        self.eps = eps

        self.initialization_mode = initialization_mode
        self.initialization_seed = initialization_seed
        self.max_init_its = max_init_its
        self.w_inits = w_inits
        self.O_inits = O_inits

        self.max_w_its = max_w_its
        self.proximal_mode = proximal_mode
        self.exact_linesearch = exact_linesearch

        self.update_frames = update_frames
        self.max_O_its = max_O_its
        self.SOC = SOC
        self.rho = rho
        self.R_solver = R_solver
        self.bases = bases

        self.noisy = noisy

        self.c1 = c1
        self.c2 = c2

        self.beta_factor = beta_factor
        self.fix_beta = fix_beta
        self.beta_min = beta_min
        self.beta_max = beta_max

        self.rel_tol = rel_tol
        self.abs_tol = abs_tol
        self.loss_tol = loss_tol
        self.patience = patience
        self.MAX_ITER = MAX_ITER

        self.verbose = verbose
        self.logger = logger
        self.show_progress = show_progress

        self.gamma = 1

    def _initialize(self, X: np.ndarray):
        """Initialization routine.

        Parameters
        ----------
        X : np.ndarray
            Observed signals.
        """
        S = X @ X.T / X.shape[1]
        Z = np.copy(X)

        if self.verbose:
            print("Initializing w and O...")

        if self.w_inits is None or self.O_inits is None:
            init_args = Initialization(
                S=S, d=self.d, V=self.V,
                noisy=self.noisy, beta_0=self.k,
                mode=self.initialization_mode,
                MAX_ITER=self.max_init_its,
                bases=self.bases,
                seed=self.initialization_seed,
            )

            w, O = init_args[0], init_args[1]

            if self.noisy:
                sigma_2_hat = init_args[2]
                self.gamma = 1 / (2 * sigma_2_hat)
            else:
                self.gamma = 1
        else:
            w = self.w_inits
            O = self.O_inits
            if self.noisy:
                sigma_2_hat = np.mean(np.linalg.eigvalsh(X @ X.T / X.shape[1])[0: self.d * self.k])
                self.gamma = 1 / (2 * sigma_2_hat)
            else:
                self.gamma = 1

        if self.noisy:
            Z = Update_Z(w, O, self.gamma, X, self.V, self.d)

        if self.verbose:
            print("Initializing U...")

        U, _ = Update_U(w=w, k=self.k, V=self.V, d=self.d)

        if self.verbose:
            print("Initializing lambda...")

        lambda_ = Update_Lambda(
            U=U, w=w, beta=self.beta, c1=self.c1, c2=self.c2,
            V=self.V, k=self.k, d=self.d,
        )

        return w, U, X, Z, O, lambda_

    def _main_loop(
        self,
        w: np.ndarray,
        U: np.ndarray,
        X: np.ndarray,
        Z: np.ndarray,
        O: np.ndarray,
        lambda_: np.ndarray,
    ):
        """Main alternating-optimization loop."""
        loss = np.zeros(self.MAX_ITER)
        S = Z @ Z.T / Z.shape[1]
        plateau_counter = 0

        if self.MAX_ITER == 0:
            return O, w, None, None, None, None

        iterator = tqdm(range(self.MAX_ITER)) if self.show_progress else range(self.MAX_ITER)

        t = 0
        for t in iterator:
            if self.noisy:
                Z_hat = Update_Z(w=w, O=O, gamma=self.gamma, X=X, V=self.V, d=self.d)
            else:
                Z_hat = X

            S = Z_hat @ Z_hat.T / Z_hat.shape[1]

            w_hat = Update_w(
                w=w, U=U, S=S, O=O, lambda_=lambda_,
                alpha=self.alpha, beta=self.beta, gamma=self.gamma,
                V=self.V, d=self.d, its=self.max_w_its,
                proximal_mode=self.proximal_mode,
                exact_linesearch=self.exact_linesearch,
            )

            if self.update_frames:
                if not self.SOC:
                    O_hat = Update_O_RG(
                        O=O, S=S, w=w_hat, V=self.V, d=self.d,
                        O_init=True, max_its=self.max_O_its,
                        bases=self.bases, solver=self.R_solver,
                    )
                else:
                    O_hat = Update_O_SOC(
                        O=O, Z=Z_hat, w=w_hat, V=self.V, d=self.d,
                        rho=self.rho, MAX_ITER=self.max_O_its,
                        abs_tol=self.abs_tol, rel_tol=self.rel_tol,
                    )
            else:
                O_hat = O

            U_hat, eigenvalues_hat = Update_U(w=w_hat, k=self.k, V=self.V, d=self.d)

            # Beta scheduling before lambda update so that lambda_ and the
            # 1/beta term in S_hat always use the same beta value.
            #
            # The spectral gradient LKron_adjoint(sum mu_i u_i u_i^T) is
            # always >= 0 for every edge. Inter-cluster edges have large
            # gradient (Fiedler vectors flip sign) and get zeroed; within-
            # cluster edges have nearly-zero gradient and survive -- but only
            # if their weights are large enough. Increasing beta when
            # n_zeros < k therefore risks zeroing ALL edges when w is small
            # (isolated nodes). The safer schedule relaxes beta when
            # n_zeros < k (let the data term reconnect/structure the graph)
            # and increases it when n_zeros >= k (lock the found k-component
            # structure in).
            if not self.fix_beta:
                n_zero_eigenvalues = np.sum(np.isclose(np.abs(eigenvalues_hat), 0, atol=1e-9))
                if self.k <= n_zero_eigenvalues:
                    if self.verbose:
                        print("Strengthening beta...", self.beta)
                    self.beta *= (1 + self.beta_factor)
                else:
                    if self.verbose:
                        print("Relaxing beta...", self.beta)
                    self.beta /= (1 + self.beta_factor)

                self.beta = min(max(self.beta, self.beta_min), self.beta_max)

            lambda_hat = Update_Lambda(
                U=U_hat, w=w_hat, beta=self.beta, c1=self.c1, c2=self.c2,
                V=self.V, k=self.k, d=self.d,
            )

            # Convergence check (primal residual on w)
            w_err = np.abs(w - w_hat)
            O_err = np.abs(O - O_hat)
            lambda_err = np.abs(lambda_ - lambda_hat)

            converged = (
                np.all(w_err <= 0.5 * self.rel_tol * (w + w_hat))
                or np.all(w_err <= self.abs_tol)
            )

            w = w_hat
            Z = Z_hat

            if self.update_frames:
                O = O_hat

            U = U_hat
            lambda_ = lambda_hat

            if converged:
                if self.verbose:
                    print(f"Convergence reached in {t} iterations on the residuals")
                break

            loss[t] = loss_(
                V=self.V, d=self.d, X=X, Z=Z, U=U, O=O, w=w, S=S,
                lambda_=lambda_, gamma=self.gamma, beta=self.beta,
                alpha=self.alpha, noisy=self.noisy,
                proximal_mode=self.proximal_mode,
            )

            if self.logger is not None:
                self.logger({
                    "loss": loss[t],
                    "iteration": t,
                    "beta": self.beta,
                    "w_update_norm": np.linalg.norm(w_err),
                    "O_update_norm": np.linalg.norm(O_err),
                    "lambda_update_norm": np.linalg.norm(lambda_err),
                }, t)

            if t > 0:
                relative_loss_change = np.abs(loss[t] - loss[t - 1]) / (np.abs(loss[t - 1]) + 1e-8)
                if relative_loss_change < self.loss_tol:
                    plateau_counter += 1
                    if plateau_counter >= self.patience:
                        if self.verbose:
                            print(f"Convergence assumed on the loss plateau at iteration {t}")
                        break
                else:
                    plateau_counter = 0

        return O, w, Z, U, lambda_, loss[:t + 1]

    def fit(self, X: np.ndarray) -> dict:
        """Run initialization followed by the main optimization loop.

        Parameters
        ----------
        X : np.ndarray
            Observed signals, shape (Vd, num_samples).

        Returns
        -------
        dict
            ``{"Initialization": {...}, "SCGL": {...}, "Loss-log": np.ndarray}``
            where ``SCGL`` holds the final ``w``, ``O``, ``Z``, ``U``, ``lambda``.
        """
        w_init, U_init, X_init, Z_init, O_init, lambda_init = self._initialize(X)
        O, w, Z, U, lambda_, loss_log = self._main_loop(
            w=w_init, U=U_init, X=X_init, Z=Z_init, O=O_init, lambda_=lambda_init,
        )

        return {
            "Initialization": {"w": w_init, "O": O_init, "Z": Z_init},
            "SCGL": {"w": w, "O": O, "Z": Z, "U": U, "lambda": lambda_},
            "Loss-log": loss_log,
        }


def learn_connection_laplacian(X: np.ndarray, V: int, d: int, k: int, alpha: float, beta: float, **kwargs) -> dict:
    """Convenience one-call entry point for using SCGL as a single pipeline step.

    Fits :class:`SCGL` on ``X`` and returns the learned connection Laplacian
    matrix directly alongside the raw factors, so this can be dropped into a
    larger pipeline without touching the ``SCGL`` class itself.

    Parameters
    ----------
    X : np.ndarray
        Observed signals, shape (Vd, num_samples).
    V : int
        Number of nodes.
    d : int
        Stalk dimension.
    k : int
        Number of connected components (spectral prior).
    alpha : float
        Sparsity regularization strength.
    beta : float
        Consistency / spectral regularization strength.
    **kwargs
        Any other :class:`SCGL` constructor argument (e.g. ``MAX_ITER``,
        ``noisy``, ``proximal_mode``, ``verbose``, ``show_progress``, ``logger``).

    Returns
    -------
    dict
        ``{"laplacian": dV x dV connection Laplacian,
           "edge_weights": w,
           "node_frames": O,
           "eigenvalues": lambda_,
           "eigenvectors": U,
           "denoised_signals": Z,
           "loss_log": np.ndarray}``
    """
    if "WANDB_monitor" in kwargs:
        warnings.warn(
            "WANDB_monitor is not part of this library's SCGL API; pass a "
            "`logger` callback instead if you want per-iteration metrics.",
            stacklevel=2,
        )
        kwargs.pop("WANDB_monitor")

    model = SCGL(V=V, d=d, k=k, alpha=alpha, beta=beta, **kwargs)
    result = model.fit(X)

    w = result["SCGL"]["w"]
    O = result["SCGL"]["O"]
    laplacian = O @ LKron(w, V, d) @ O.T

    return {
        "laplacian": laplacian,
        "edge_weights": w,
        "node_frames": O,
        "eigenvalues": result["SCGL"]["lambda"],
        "eigenvectors": result["SCGL"]["U"],
        "denoised_signals": result["SCGL"]["Z"],
        "loss_log": result["Loss-log"],
    }
