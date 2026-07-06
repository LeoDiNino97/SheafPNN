"""scgl

Structured Connection Graph Learning: learn sparse connection Laplacians
under sparsity and spectral-consistency constraints.

Reference
---------
"Structured Learning of Consistent Connection Laplacians with Spectral
Constraints", Di Nino L., D'Acunto G., et al., 2025.

Quick start
-----------
>>> from scgl import learn_connection_laplacian
>>> out = learn_connection_laplacian(X, V=20, d=3, k=2, alpha=0.1, beta=50, show_progress=False)
>>> out["laplacian"]        # dV x dV connection Laplacian
>>> out["edge_weights"]     # sparse edge weights w

Or use the full solver directly for more control / to inspect intermediate
state (initialization, loss log, etc.):

>>> from scgl import SCGL
>>> model = SCGL(V=20, d=3, k=2, alpha=0.1, beta=50, show_progress=False)
>>> result = model.fit(X)
"""

from .model import SCGL, learn_connection_laplacian
from .operators import L, L_adjoint, L_inv, LKron, LKron_adjoint, L_spy
from .updates import Update_Z, Update_w, Update_O_RG, Update_O_SOC, Update_U, Update_Lambda
from .initialization import Initialization
from .detection import minAICdetector
from .loss import loss_

__version__ = "0.1.0"

__all__ = [
    "SCGL",
    "learn_connection_laplacian",
    "L", "L_adjoint", "L_inv", "LKron", "LKron_adjoint", "L_spy",
    "Update_Z", "Update_w", "Update_O_RG", "Update_O_SOC", "Update_U", "Update_Lambda",
    "Initialization",
    "minAICdetector",
    "loss_",
]
