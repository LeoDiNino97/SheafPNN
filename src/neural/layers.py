import torch
from torch import nn
import math


def filtering(X, H, L, O, bias=None):
    """
    Flat-bundle filtering: U = O^T (sum_j H_j L^j ⊗ I_n) O X

    Filters are polynomials of the combinatorial Laplacian L, expanded
    to the full connection-graph dimension via the Kronecker form (L^j ⊗ I_n)
    and sandwiched between the orthogonal operators O and O^T.

    Args:
        X: (Vn, T, F_in)  signal on connection graph (V nodes, stalk dim n)
        H: (F_out, K, F_in)  polynomial filter taps (K = degree + 1)
        L: (V, V)  combinatorial Laplacian of the underlying graph
        O: (Vn, Vn)  block-diagonal orthogonal matrix (stacked per-node frames)
        bias: (F_out,) optional bias
    """

    F_out, K, F_in = H.shape
    Vn, T, _ = X.shape
    V = L.shape[0]
    n = Vn // V

    # Apply O to input signal: y = O x  (Vn, T, F_in)
    Y = torch.matmul(O, X.reshape(Vn, T * F_in)).reshape(Vn, T, F_in)
    # Reshape to expose node and stalk dimensions separately for Kronecker action
    Y = Y.reshape(V, n, T, F_in)

    # Build Z = [L^0 Y, L^1 Y, ..., L^{K-1} Y] using (L^j ⊗ I_n) implicitly:
    # (L^j ⊗ I_n) acting on (V, n, T, F_in) is just L^j applied along node dim.
    Z = [Y]
    Yk = Y
    for _ in range(1, K):
        Yk = torch.matmul(L, Yk.reshape(V, n * T * F_in)).reshape(V, n, T, F_in)
        Z.append(Yk)

    Z = torch.stack(Z).reshape(K, Vn, T, F_in)  # (K, Vn, T, F_in)

    # Polynomial combination across taps and input features
    U = H.reshape(F_out, K * F_in) @ Z.permute(0, 3, 1, 2).reshape(K * F_in, Vn * T)
    U = U.reshape(F_out, Vn, T).permute(1, 2, 0)  # (Vn, T, F_out)

    # Apply O^T to output
    U = torch.matmul(O.t(), U.reshape(Vn, T * F_out)).reshape(Vn, T, F_out)

    if bias is not None:
        U = U + bias

    return U


class ConvFilter(nn.Module):
    """
    Flat-bundle convolutional filter: polynomial in the combinatorial Laplacian
    expanded via Kronecker form and applied between the O / O^T operators.
    """

    def __init__(self, in_feat, out_feat, K, bias=True):
        """
        Args:
            in_feat: number of input features
            out_feat: number of output features
            K: polynomial degree (filter has K+1 taps)
            bias: whether to add a learnable bias
        """

        super().__init__()
        self.K = K
        self.H = nn.parameter.Parameter(torch.Tensor(out_feat, K + 1, in_feat))

        if bias:
            self.bias = nn.parameter.Parameter(torch.Tensor(out_feat))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.K)
        self.H.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)


    def forward(self, X, L, O):
        """
        Args:
            X: (Vn, T, F_in)
            L: (V, V) combinatorial Laplacian
            O: (Vn, Vn) block-diagonal orthogonal matrix
        """
        return filtering(X, self.H, L, O, bias=self.bias)

