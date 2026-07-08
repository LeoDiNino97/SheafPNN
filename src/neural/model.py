import torch
from torch import nn
# from .layers import ConvFilter
from layers import ConvFilter
import torch.nn.functional as F

class SPNN(nn.Module):
    """
    Class for a convolutional GNN with shift operator in a Sheaf Laplacian

    Arguments
    input_size: node feature input size
    output_size: node feature output size (if task_level==graph, node features are aggregated)
    hidden_sizes: list of hidden sizes
    hidden_mlp_sizes: list of readout MLP hidden sizes. The first value needs to be the same as the last
        entry of hidden_sizes. Can be empty for no readout MLP

    EXAMPLE 1:
    input_size = 8
    hidden_sizes = [16,32]
    hidden_mlp_sizes = [32,64]
    output_size = 2

    This creates 2 convolutional filterbanks of shape 8->16, 16->32,
    followed by a 2 layer readout MLP of shape 32->64, 64->2.

    EXAMPLE 2: 
    input_size = 4
    hidden_sizes = [2]
    hidden_mlp_sizes = []
    output_size = 8

    This creates 2 convolutional filterbanks of shape 4->2, 2->8,
    with no readout MLP.

    EXAMPLE 3: 
    input_size = 16
    hidden_sizes = [4]
    hidden_mlp_sizes = [4]
    output_size = 32

    This creates 1 convolutional filterbank of shape 16->4,
    followed by a 1 layer readout MLP of shape 4->32.
    """

    def __init__(
                    self, input_size, output_size,
                    hidden_sizes, hidden_mlp_sizes,
                    K, bias, dropout,
                    task_level="graph", node_readout="mean",
                    device="cpu", use_batch_norms=False,
                    use_layer_norms=False,
                    task="classification"
                ):
        super().__init__()

        self.dropout = dropout
        self.nonlinearity = nn.LeakyReLU(0.1)
        self.task_level = task_level
        self.node_readout = node_readout
        self.device = device
        self.use_batch_norms = use_batch_norms
        self.use_layer_norms = use_layer_norms 
        self.task = task

        self.logSoftmax = nn.LogSoftmax(dim=1)

        self.layer_norms = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        self.gfl = nn.ModuleList()
        self.gfl.append(ConvFilter(input_size, hidden_sizes[0], K, bias=bias))
        self.batch_norms.append(nn.BatchNorm1d(hidden_sizes[0]))
        self.layer_norms.append(nn.LayerNorm(hidden_sizes[0]))
            
        if len(hidden_sizes) > 1:
            for l in range(1, len(hidden_sizes)):
                self.gfl.append(ConvFilter(hidden_sizes[l-1], hidden_sizes[l], K, bias=bias))
                self.batch_norms.append(nn.BatchNorm1d(hidden_sizes[l]))
                self.layer_norms.append(nn.LayerNorm(hidden_sizes[l]))

        self.mlp = nn.ModuleList()

        for l in range(1, len(hidden_mlp_sizes)):
            self.mlp.append(nn.Linear(hidden_mlp_sizes[l-1], hidden_mlp_sizes[l], bias=bias))
            self.init_weights(self.mlp[-1])

        if len(hidden_mlp_sizes) > 0:
            self.mlp.append(nn.Linear(hidden_mlp_sizes[-1], output_size, bias=bias))
            self.init_weights(self.mlp[-1])
        else: # no MLP, just filters
            self.gfl.append(ConvFilter(hidden_sizes[-1], output_size, K, bias=bias))

        # Check that layers are as expected:
        # print([(gf.H.shape) for gf in self.gfl])
        # print(self.mlp)
        # input()        


    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)



    def forward(self, X, L, O):
        """
        Args:
            X: (Vn, T, F_in)  signal on the connection graph
            L: (V, V)  combinatorial Laplacian of the underlying graph
            O: (Vn, Vn)  block-diagonal orthogonal matrix of per-node frames
        """

        for i, gf in enumerate(self.gfl):
            X = gf(X, L, O)
            if i < len(self.gfl) - 1 or self.mlp: # no non-linearity at last layer
                if self.use_batch_norms:
                    X = self.batch_norms[i](X.permute((1,2,0))).permute((2,0,1))
                if self.use_layer_norms:
                    X = self.layer_norms[i](X)
                X = self.nonlinearity(X)
                X = F.dropout(X, p=self.dropout, training=self.training)

        if self.task_level == "graph":
            # Aggregate over Vn nodes (dim 0); X is (Vn, T, F)
            if self.node_readout == "max":
                X = X.max(0)[0]
            elif self.node_readout == "mean":
                X = X.mean(0)
            elif self.node_readout == "sum":
                X = X.sum(0)
            else:
                raise NotImplementedError("Node readout function not implemented")


        for i, l in enumerate(self.mlp):
            X = l(X)
            if i < len(self.mlp) - 1: # no non-linearity at last layer
                X = self.nonlinearity(X)
                X = F.dropout(X, p=self.dropout, training=self.training)

        if self.task == "classification":
            X = self.logSoftmax(X)
        else:
            X = X.squeeze(-1)

        return X
