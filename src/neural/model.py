import torch
from torch import nn
from layers import *
import torch.nn.functional as F

class SPNN(nn.Module):
    """
    Class for a convolutional GNN with shift operator in a Sheaf Laplacian
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
        
        first_filter_out_size = hidden_sizes[0] if hidden_sizes else output_size
        self.gfl = nn.ModuleList()
        self.gfl.append(ConvFilter(input_size, first_filter_out_size, K, bias=bias))
        self.batch_norms.append(nn.BatchNorm1d(first_filter_out_size).to(self.device))
        self.layer_norms.append(nn.LayerNorm(first_filter_out_size).to(self.device))
            
        if hidden_sizes:
            for l in range(1, len(hidden_sizes)):
                self.gfl.append(ConvFilter(hidden_sizes[l-1], hidden_sizes[l], K, bias=bias))
                
                self.batch_norms.append(nn.BatchNorm1d(hidden_sizes[l]).to(self.device))
                self.layer_norms.append(nn.LayerNorm(hidden_sizes[l]).to(self.device))


        self.mlp = nn.ModuleList()
        if hidden_mlp_sizes: # we have a readout layer
            first_linear_out_size = hidden_sizes[-1] if len(hidden_mlp_sizes) > 1 else output_size
            self.mlp.append(nn.Linear(hidden_mlp_sizes[0], first_linear_out_size))            
            for l in range(1, len(hidden_mlp_sizes)):
                self.mlp.append(nn.Linear(hidden_mlp_sizes[l-1], hidden_mlp_sizes[l]))
                self.init_weights(self.mlp[-1])
            if len(hidden_mlp_sizes) > 1:
                self.mlp.append(nn.Linear(hidden_mlp_sizes[-1], output_size))            

        else: # no readout layer
            self.gfl.append(ConvFilter(hidden_sizes[-1], output_size, K, bias=bias))


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
            # Aggregate node outputs
            if self.node_readout == "max":
                X = X.max(1)[0]
            elif self.node_readout == "mean":
                X = X.mean(1)
            elif self.node_readout == "sum":
                X = X.sum(1)
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
    

