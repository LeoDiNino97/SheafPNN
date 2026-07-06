"""
Joint training of Flat Bundle Neural Network parameters and connection graph.

Implements the alternating descent procedure from Section 3.1 of the notes:

  Neural update   – gradient steps on (h, w, O) w.r.t. task loss + proximity
  Update z        – proximal gradient for GL edge weights
  Update P        – spectral linear solve
  Update Q        – Kabsch SO(n) retraction per block
  Update B        – ADMM dual accumulation

Data convention (matches src/neural/layers.py):
  X : (Vn, T, F_in)  — connection-graph signal, nodes first, then samples
  y : (T,)  or  (T, F_out)  — targets

edge_index convention: (2, |E|) with edge_index[0,k] > edge_index[1,k] for all k.
"""

import numpy as np
import torch
from torch import nn, optim

from ..neural.model import SPNN
from ..operators.laplacian import build_laplacian, build_blockdiag
from .utils import (
    compute_sample_covariance,
    update_z,
    update_P,
    update_Q,
    update_B,
    update_T,
    update_K,
    compute_gl_loss,
)
from .scgl_init import scgl_warmstart


def _compute_loss(
    Loss,
    prec_type: str,
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    *,
    w: torch.Tensor,
    z: torch.Tensor,
    O_blk: torch.Tensor,
    P: torch.Tensor,
    T_admm: torch.Tensor,
    K_admm: torch.Tensor,
    gamma1: float,
    gamma2: float,
    gamma4: float,
    alpha: float,
) -> torch.Tensor:
    """Task loss plus proximity terms that involve differentiable variables."""
    loss = alpha * Loss(y_pred, y_true)
    if prec_type == "joint":
        loss = loss + (gamma1 / 2.0) * torch.sum((w - z.detach()) ** 2)
        loss = loss + (gamma2 / 2.0) * torch.sum((O_blk - P.detach()) ** 2)
        # γ4/2 ‖O − T + K‖²_F  (new O-splitting term from updated notes)
        loss = loss + (gamma4 / 2.0) * torch.sum((O_blk - (T_admm - K_admm).detach()) ** 2)
    return loss


def train_sheaf_pnn(
    X: torch.Tensor,
    y: torch.Tensor,
    edge_index: torch.Tensor,
    V: int,
    n: int,
    *,
    # Model hyperparameters
    hidden_sizes=None,
    hidden_mlp_sizes=None,
    K: int = 2,
    bias: bool = True,
    dropout: float = 0.0,
    task_level: str = "graph",
    node_readout: str = "mean",
    task: str = "regression",
    use_batch_norms: bool = False,
    # Optimisation hyperparameters
    nEpochs: int = 50,
    it_h: int = 20,
    it_z: int = 10,
    lr: float = 0.01,
    eta_z: float = 0.001,
    # Joint learning hyperparameters
    alpha: float = 0.5,
    gamma1: float = 1.0,
    gamma2: float = 1.0,
    gamma3: float = 1.0,
    lam: float = 0.1,
    epsilon: float = 1e-4,
    penalty: str = "l1",
    prec_type: str = "joint",
    gamma4: float = 1.0,
    # Optional fixed-graph initialization
    # If prec_type='fixed': these are frozen throughout training.
    # If prec_type='joint': used as warm-start values.
    w_init: torch.Tensor = None,
    O_blocks_init: torch.Tensor = None,
    # SCGL-based warm-start: if True and w_init/O_blocks_init are not given,
    # (w, O) are initialized via scgl.initialization.Initialization on the
    # empirical covariance of X, instead of the naive ones/identity default.
    scgl_init: bool = False,
    scgl_init_kwargs: dict = None,
    # Pre-computed V×V Laplacian for baselines that don't use edge-weight
    # parametrisation (e.g. sample-covariance or dense GL estimates).
    # When set, overrides build_laplacian(w, edge_index) for forward passes.
    L_fixed: torch.Tensor = None,
    # If False: O_blocks is frozen at O_blocks_init throughout (Kron-Joint).
    learn_O: bool = True,
    # Misc
    seed: int = 0,
    device: str = "cpu",
    verbose: bool = True,
):
    """Train a Flat Bundle Neural Network with joint connection-graph learning.

    Args:
        X: (Vn, T, F_in) input signals on the connection graph
        y: (T,) or (T, F_out) regression/classification targets
        edge_index: (2, |E|) with i > j
        V: number of graph nodes
        n: stalk dimension
        hidden_sizes: list of hidden channel sizes for graph filters
        hidden_mlp_sizes: list of hidden sizes for the readout MLP
        K: polynomial filter order
        task: 'regression' | 'classification'
        nEpochs: outer alternating-descent iterations
        it_h: gradient steps per neural update
        it_z: proximal gradient steps per z update
        lr: learning rate for neural parameters
        eta_z: step size for z proximal gradient
        alpha: weight of task loss (1-alpha weights GL objective)
        gamma1: ‖w-z‖² penalty
        gamma2: ‖O-P‖²_F penalty
        gamma3: ‖P-Q+B‖²_F ADMM penalty
        lam: sparsity penalty on z
        epsilon: diagonal loading for logdet stability
        penalty: sparsity inducing penalty ('l1' or 'log')
        prec_type: 'joint' (full alternating) | 'fixed' (no graph update)
        scgl_init: if True, and w_init/O_blocks_init are not given, warm-start
            them via SCGL's joint QP initialization (see scgl_init.py) instead
            of the naive ones/identity default
        scgl_init_kwargs: forwarded to scgl_warmstart / scgl.initialization.Initialization
            (e.g. mode='QP'|'ID-QP'|'ID'|'RANDOM', noisy, seed, MAX_ITER)
        seed: random seed
        verbose: print epoch progress
    Returns:
        model: trained SPNN
        w: final edge weights (|E|,)
        O_blocks: final per-node frames (V, n, n)
        z: final GL edge weights (|E|,)
        history: dict with training metrics
    """
    torch.manual_seed(seed)
    if hidden_sizes is None:
        hidden_sizes = [32]
    if hidden_mlp_sizes is None:
        hidden_mlp_sizes = [32]

    Vn, T, F_in = X.shape
    assert Vn == V * n, f"Expected Vn={V*n}, got {Vn}"
    E = edge_index.shape[1]

    X = X.to(device)
    y = y.to(device)
    edge_index = edge_index.to(device)

    # -----------------------------------------------------------------------
    # Determine output size
    # -----------------------------------------------------------------------
    if task == "classification":
        F_out = int(y.max().item()) + 1
        Loss = nn.NLLLoss()
    else:
        F_out = 1
        Loss = nn.MSELoss()
        MAE = nn.L1Loss()

    # -----------------------------------------------------------------------
    # Compute empirical covariance once (used for z and P updates)
    # -----------------------------------------------------------------------
    C_sample = compute_sample_covariance(X).detach()

    # -----------------------------------------------------------------------
    # Initialise variables
    # -----------------------------------------------------------------------
    _scgl_w = _scgl_O = None
    if scgl_init and (w_init is None or O_blocks_init is None):
        _scgl_w, _scgl_O = scgl_warmstart(
            X, edge_index, V, n, **(scgl_init_kwargs or {}),
        )

    _w_default = (
        w_init.to(device).clone() if w_init is not None
        else _scgl_w.to(device) if _scgl_w is not None
        else torch.ones(E, device=device) * 0.1
    )
    _O_default = (
        O_blocks_init.to(device).clone() if O_blocks_init is not None
        else _scgl_O.to(device) if _scgl_O is not None
        else torch.eye(n, device=device).unsqueeze(0).expand(V, -1, -1).clone()
    )

    if prec_type == "fixed":
        # Graph is given; only filter taps h are learnable.
        w = _w_default.detach()
        O_blocks = _O_default.detach()
    else:
        # w is always jointly optimised in joint mode.
        w = nn.Parameter(_w_default)
        # O is learnable only when learn_O=True (False → Kron-Joint style).
        if learn_O:
            O_blocks = nn.Parameter(_O_default)
        else:
            O_blocks = _O_default.detach()

    # z: GL edge weights (proximal gradient, not a Parameter)
    z = _w_default.clone().detach()

    # ADMM variables P, Q, B for the P-splitting (eq. 27–32)
    P = torch.eye(Vn, device=device)
    Q = torch.eye(Vn, device=device)
    B = torch.zeros(Vn, Vn, device=device)
    # ADMM variables T, K for the O-splitting (new in updated notes)
    T_admm = torch.eye(Vn, device=device)
    K_admm = torch.zeros(Vn, Vn, device=device)

    # -----------------------------------------------------------------------
    # Model and optimiser
    # -----------------------------------------------------------------------
    model = SPNN(
        input_size=F_in,
        output_size=F_out,
        hidden_sizes=hidden_sizes,
        hidden_mlp_sizes=hidden_mlp_sizes,
        K=K,
        bias=bias,
        dropout=dropout,
        task_level=task_level,
        node_readout=node_readout,
        device=device,
        use_batch_norms=use_batch_norms,
        task=task,
    ).to(device)

    if prec_type == "fixed":
        neural_params = list(model.parameters())
    elif learn_O:
        neural_params = list(model.parameters()) + [w, O_blocks]
    else:
        neural_params = list(model.parameters()) + [w]
    optimizer = optim.Adam(neural_params, lr=lr, weight_decay=1e-4)

    # -----------------------------------------------------------------------
    # Training history
    # -----------------------------------------------------------------------
    history = {"train_loss": [], "train_mae": [], "gl_loss": []}

    # -----------------------------------------------------------------------
    # Alternating descent loop
    # -----------------------------------------------------------------------
    for epoch in range(nEpochs):

        # ===================================================================
        # Step 1 — Neural update: optimise (h, w, O) via autograd
        # ===================================================================
        model.train()
        epoch_losses = []

        for _ in range(it_h):
            optimizer.zero_grad()

            # Project w to non-negative before each forward pass (joint only)
            if prec_type != "fixed":
                with torch.no_grad():
                    w.clamp_(min=0.0)

            # Build differentiable operators
            if L_fixed is not None:
                L_w = L_fixed.to(device)
            else:
                L_w = build_laplacian(w, edge_index, V)       # (V, V)
            O_blk = build_blockdiag(O_blocks)                 # (Vn, Vn)

            y_hat = model(X, L_w, O_blk)                     # (T, F_out)

            if task == "regression":
                y_hat = y_hat.squeeze(-1)

            loss = _compute_loss(
                Loss, prec_type, y, y_hat,
                w=w, z=z, O_blk=O_blk, P=P,
                T_admm=T_admm, K_admm=K_admm,
                gamma1=gamma1, gamma2=gamma2, gamma4=gamma4, alpha=alpha,
            )
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        # Ensure w stays non-negative after the full set of gradient steps
        if prec_type != "fixed":
            with torch.no_grad():
                w.clamp_(min=0.0)

        # ===================================================================
        # Steps 2–5 — Connection graph updates (only for joint training)
        # ===================================================================
        if prec_type == "joint":

            O_blk_det = build_blockdiag(O_blocks).detach()

            # Step 2: Update z (proximal gradient, eq. 24)
            for _ in range(it_z):
                z = update_z(
                    z, w.detach(), P, C_sample,
                    alpha=alpha, gamma1=gamma1, lam=lam,
                    epsilon=epsilon, eta_z=eta_z,
                    edge_index=edge_index, V=V, n=n, penalty=penalty,
                )

            # Step 3: Update P (spectral linear solve, eq. 26)
            P = update_P(
                O_blk_det, Q, B, C_sample, z,
                alpha=alpha, gamma2=gamma2, gamma3=gamma3,
                edge_index=edge_index, V=V, n=n,
            )

            # Step 4: Update Q (Kabsch retraction, eq. 31)
            Q = update_Q(P, B, V=V, n=n)

            # Step 5: Update B (ADMM dual, eq. 32)
            B = update_B(B, P, Q)

            # Steps 6–7: Update T and K (O-splitting, updated notes)
            O_blk_det_for_T = build_blockdiag(
                O_blocks if not isinstance(O_blocks, nn.Parameter) else O_blocks.detach()
            )
            T_admm = update_T(O_blk_det_for_T, K_admm, V=V, n=n)
            K_admm = update_K(K_admm, O_blk_det_for_T, T_admm)

        # ===================================================================
        # Monitoring
        # ===================================================================
        gl = compute_gl_loss(
            C_sample, P, z, epsilon=epsilon, alpha=alpha, lam=lam,
            edge_index=edge_index, V=V, n=n,
        ).item()

        mean_loss = float(np.mean(epoch_losses))
        history["train_loss"].append(mean_loss)
        history["gl_loss"].append(gl)

        if task == "regression":
            with torch.no_grad():
                if L_fixed is not None:
                    L_w_eval = L_fixed.to(device)
                else:
                    _w_eval = w if prec_type == "fixed" else w.detach()
                    L_w_eval = build_laplacian(_w_eval, edge_index, V)
                O_blk_eval = build_blockdiag(O_blocks if prec_type == "fixed" else O_blocks.detach())
                y_hat_eval = model(X, L_w_eval, O_blk_eval).squeeze(-1)
                mae = MAE(y_hat_eval, y).item()
            history["train_mae"].append(mae)
            if verbose:
                print(
                    f"Epoch {epoch+1:3d}/{nEpochs}  "
                    f"loss={mean_loss:.4f}  mae={mae:.4f}  gl={gl:.4f}"
                )
        else:
            if verbose:
                print(
                    f"Epoch {epoch+1:3d}/{nEpochs}  loss={mean_loss:.4f}  gl={gl:.4f}"
                )

    return model, w.detach(), O_blocks.detach(), z, history


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_sheaf_pnn(
    model: SPNN,
    X: torch.Tensor,
    y: torch.Tensor,
    w: torch.Tensor,
    O_blocks: torch.Tensor,
    edge_index: torch.Tensor,
    V: int,
    n: int,
    task: str = "regression",
    L_fixed: torch.Tensor = None,
) -> dict:
    """Evaluate a trained SPNN on held-out data.

    Args:
        X: (Vn, T, F_in) test signals
        y: (T,) or (T, F_out) test targets
        w: (|E|,) learned edge weights
        O_blocks: (V, n, n) learned per-node frames
        edge_index: (2, |E|)
        V, n: graph structure
        task: 'regression' | 'classification'
    Returns:
        dict with 'loss' and 'mae' (regression) or 'accuracy' (classification)
    """
    model.eval()
    L_w = L_fixed if L_fixed is not None else build_laplacian(w, edge_index, V)
    O_blk = build_blockdiag(O_blocks)
    y_hat = model(X, L_w, O_blk)

    results = {}
    if task == "regression":
        y_hat = y_hat.squeeze(-1)
        results["mae"] = nn.L1Loss()(y_hat, y).item()
        results["mse"] = nn.MSELoss()(y_hat, y).item()
    else:
        loss = nn.NLLLoss()(y_hat, y)
        preds = y_hat.argmax(1)
        results["accuracy"] = (preds == y).float().mean().item()
        results["loss"] = loss.item()
    return results
