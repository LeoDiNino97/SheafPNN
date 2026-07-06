"""
Warm-start (w, O) for the joint flat-bundle training loop via SCGL.

Wraps ``scgl.initialization.Initialization`` — the joint QP warm-start of
edge weights and node frames from "Structured Learning of Consistent
Connection Laplacians with Spectral Constraints" (Di Nino, D'Acunto et al.,
2025) — and adapts its output (complete-graph triu-ordered w, Vn×Vn
block-diagonal O) to the ``edge_index`` / ``(V, n, n)`` conventions used by
``train_sheaf_pnn``.

The ``scgl`` package pulls in ``pymanopt``/``autograd``, which are not part
of this project's default environment, so the import is deferred until
this warm-start is actually requested.
"""

import numpy as np
import torch

from .utils import compute_sample_covariance

__all__ = ["scgl_warmstart"]


def scgl_warmstart(
    X: torch.Tensor,
    edge_index: torch.Tensor,
    V: int,
    n: int,
    mode: str = "QP",
    noisy: bool = False,
    seed: int = 42,
    rescale: bool = True,
    target_scale: float = 0.1,
    **init_kwargs,
) -> tuple:
    """Compute a warm-start (w, O_blocks) via SCGL's joint QP initialization.

    SCGL's 'QP'/'ID-QP' modes fit an *unregularized* least-squares match to
    pinv(S); the resulting w can be arbitrarily large whenever the sample
    covariance is close to rank-deficient (small eigenvalues blow up under
    pinv). That's harmless inside SCGL's own alternating loop, where the
    alpha/beta-regularized main loop pulls it back down, but it is the wrong
    scale for train_sheaf_pnn's step sizes and proximity penalties (gamma1,
    lr, ...), which are calibrated around the previous ones*0.1 default. When
    ``rescale=True`` (default) the returned w is uniformly rescaled so its
    mean over strictly positive entries equals ``target_scale`` — this only
    changes the overall magnitude, not the relative edge structure. O_blocks
    is orthogonal and never rescaled.

    Args:
        X: (Vn, T, F_in) connection-graph signal
        edge_index: (2, |E|) with edge_index[0,k] > edge_index[1,k]
        V: number of nodes
        n: stalk dimension
        mode: SCGL initialization mode, one of 'ID', 'ID-QP', 'QP', 'RANDOM'
        noisy: whether SCGL should account for observation noise
        seed: random seed forwarded to SCGL's Initialization
        rescale: uniformly rescale w to a train_sheaf_pnn-compatible magnitude
        target_scale: target mean value for the positive entries of w when
            ``rescale=True``
        **init_kwargs: forwarded to ``scgl.initialization.Initialization``
            (e.g. ``bases``, ``MAX_ITER``, ``reltol``, ``abstol``)
    Returns:
        w_init: (|E|,) edge weights, restricted/reordered to edge_index
        O_blocks_init: (V, n, n) per-node frames
    """
    try:
        from scgl.initialization import Initialization
    except ImportError as exc:
        raise ImportError(
            "SCGL-based initialization requires the `scgl` package together "
            "with its `pymanopt`/`autograd` dependencies (see "
            "scgl/pyproject.toml). Install them, or run in an environment "
            "where they're available, to use scgl_init=True."
        ) from exc

    device = X.device
    dtype = X.dtype

    C = compute_sample_covariance(X)
    S_np = C.detach().cpu().numpy().astype(np.float64)

    result = Initialization(
        S=S_np, d=n, V=V, mode=mode, noisy=noisy, seed=seed, **init_kwargs,
    )
    w_scgl, O_scgl = result[0], result[1]

    # SCGL assumes the complete graph and orders w via np.triu_indices(V, k=1)
    # (i.e. pairs (i, j) with i < j, lexicographically). Build a lookup so we
    # can restrict/reorder that vector to whatever edge_index is in use.
    i_idx, j_idx = np.triu_indices(V, k=1)
    lut = np.full((V, V), -1, dtype=np.int64)
    lut[i_idx, j_idx] = np.arange(len(i_idx))

    ei_np = edge_index.detach().cpu().numpy()
    rows = np.minimum(ei_np[0], ei_np[1])
    cols = np.maximum(ei_np[0], ei_np[1])
    lut_idx = lut[rows, cols]
    if np.any(lut_idx < 0):
        raise ValueError("edge_index contains self-loops or out-of-range node indices")

    w_init = torch.as_tensor(w_scgl[lut_idx], dtype=dtype, device=device)

    if rescale:
        positive = w_init[w_init > 0]
        if positive.numel() > 0:
            w_init = w_init * (target_scale / positive.mean())

    O_blocks_np = np.stack(
        [O_scgl[v * n:(v + 1) * n, v * n:(v + 1) * n] for v in range(V)]
    )
    O_blocks_init = torch.as_tensor(O_blocks_np, dtype=dtype, device=device)

    return w_init, O_blocks_init
