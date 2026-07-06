# scgl

Structured Connection Graph Learning — learns a sparse **connection
Laplacian** (edge weights + per-node orthogonal frames) that is consistent
with observed graph signals and with a k-component spectral prior.

Repackaged from the reference implementation of *"Structured Learning of
Consistent Connection Laplacians with Spectral Constraints"*, Di Nino L.,
D'Acunto G., et al., 2025, so it can be dropped into another project as a
single pipeline step, with no monitoring / notebook-only dependencies
required.

## What changed vs. the original scripts

- Split into a proper package (`operators`, `updates`, `initialization`,
  `loss`, `model`) instead of two flat files with `import *`.
- **`wandb` removed as a hard dependency.** Pass a `logger` callback to `SCGL`
  (or `learn_connection_laplacian`) if you want per-iteration metrics
  forwarded anywhere (W&B, MLflow, a list, ...):
  ```python
  history = []
  model = SCGL(..., logger=lambda metrics, step: history.append(metrics))
  ```
- Added `learn_connection_laplacian(...)`, a one-call functional entry point
  that returns the learned Laplacian matrix directly — useful when Laplacian
  learning is just one step in a larger pipeline.
- `show_progress` toggles the tqdm bar independently of `verbose`.
- Restored the `minAICdetector` helper (AIC-based kernel-dimension detector),
  so `Initialization(noisy=True, beta_0=None)` again estimates the component
  count automatically from the covariance. You can still pass `beta_0`
  explicitly to override it.

The math and numerics are untouched — every update step is copied verbatim.

## Install

```bash
pip install ./scgl        # from this directory (contains pyproject.toml)
```

Dependencies: `numpy`, `scipy`, `autograd`, `pymanopt`, `tqdm`.

## Usage

### As a single pipeline step

```python
from scgl import learn_connection_laplacian

# X: np.ndarray of shape (V * d, num_samples)
out = learn_connection_laplacian(
    X, V=20, d=3, k=2, alpha=0.1, beta=50,
    MAX_ITER=5000, show_progress=False,
)

L = out["laplacian"]        # (V*d, V*d) connection Laplacian
w = out["edge_weights"]     # sparse edge weights, length V(V-1)/2
O = out["node_frames"]      # block-diagonal node frames
```

### Full control via the solver class

```python
from scgl import SCGL

model = SCGL(V=20, d=3, k=2, alpha=0.1, beta=50, MAX_ITER=5000, show_progress=False)
result = model.fit(X)

result["SCGL"]["w"]           # edge weights
result["SCGL"]["O"]           # node frames
result["Initialization"]      # w/O/Z at initialization, for inspection
result["Loss-log"]            # per-iteration loss trace
```

### Standalone operators

The linear operators mapping edge weights to (connection) Laplacians are also
exposed directly, in case you only need those:

```python
from scgl import LKron, L

Laplacian = LKron(w, V=20, d=3)   # dV x dV connection Laplacian from edge weights
```

## Notes

- `noisy=True` jointly denoises the input signals; it needs `beta_0` (the
  assumed number of connected components). By default this is taken from `k`
  when using the `SCGL` class, or auto-detected via `minAICdetector` when you
  call `Initialization(...)` directly with `beta_0=None`.
- `SOC=True` (default) updates node frames via ADMM (Splitting Orthogonality
  Constraint); `SOC=False` uses Riemannian gradient descent over the Stiefel
  manifold via `pymanopt` (`R_solver` in `{'RCG', 'RSD', 'TR'}`).
