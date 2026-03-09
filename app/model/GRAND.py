import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
import torch_geometric.nn as gnn
import torch_geometric as torchg


class GRANDDiffusionODEFunc(gnn.MessagePassing):
    def __init__(self, vert_dim, model_dim, num_heads):
        super().__init__(flow="target_to_source")

        self.model_dim = model_dim
        self.vert_dim = vert_dim
        self.num_heads = num_heads

        self.k = nn.Linear(vert_dim, num_heads * model_dim)
        self.q = nn.Linear(vert_dim, num_heads * model_dim)

    def update(self, aggr_out):
        return aggr_out

    def message(self, x_j, x_i, index):
        Q_i = self.q(x_i).view(-1, self.num_heads, self.model_dim)
        K_j = self.k(x_j).view(-1, self.num_heads, self.model_dim)

        scores = (Q_i * K_j).sum(dim=-1) / torch.sqrt(self.model_dim)
        scores = scores.mean(dim=1)
        alpha = torchg.utils.softmax(scores, index)
        return (x_j - x_i) * alpha.unsqueeze(1)

    def forward(self, edge_index, x):
        return self.propagate(edge_index, x=x)


class ODEFunc(nn.Module):
    def __init__(self, func, edge_index):
        super().__init__()
        self.func = func
        self.edge_index = edge_index

    def forward(self, t, x):
        return self.func(self.edge_index, x)


class GRANDAutoencoder(nn.Module):
    def __init__(self, edge_index, vert_dim, model_dim, num_heads):
        super().__init__()
        self.encoder = GRANDDiffusionODEFunc(vert_dim, model_dim, num_heads)
        self.decoder = GRANDDiffusionODEFunc(vert_dim, model_dim, num_heads)

        self.encoder_ode = ODEFunc(self.encoder, edge_index)
        self.decoder_ode = ODEFunc(self.decoder, edge_index)

    def forward(self, x, t):

        z = odeint(self.encoder_ode, x, t)[-1]
        x_hat = odeint(self.decoder_ode, z, t)[-1]

        return x_hat, z
