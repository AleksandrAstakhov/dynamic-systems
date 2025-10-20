import torch_geometric as torchg
import torch
import torch.nn as nn
import torch_geometric.nn as gnn

import matplotlib.pyplot as plt

from torchdiffeq import odeint

from tqdm.auto import tqdm

class GRANDBlock(nn.Module):
    def __init__(self, num_heads, num_chanels, featurs_dim, out_dim):
        super().__init__()

        self.num_heads = num_heads
        self.num_chanels = num_chanels
        self.featurs_dim = featurs_dim
        self.out_dim = out_dim

        self.register_buffer("eye", torch.eye(self.num_chanels))

        self.k = nn.Linear(self.featurs_dim, self.out_dim * self.num_heads)
        self.q = nn.Linear(self.featurs_dim, self.out_dim * self.num_heads)
        self.v = nn.Linear(self.featurs_dim, self.out_dim * self.num_heads)

        self.w0 = nn.Linear(self.out_dim * self.num_heads, self.out_dim)

    def forward(
        self,
        x,
    ):  # just x cause we hawe complete graph where edge weight is att score
        nodes = torch.arange(x.shape[0])
        edge_idx = torch.stack(
            [nodes.repeat(len(nodes)), nodes.repeat_interleave(len(nodes))], dim=0
        )

        k = self.k(x).reshape(-1, self.num_heads, self.num_chanels, self.out_dim)
        q = self.q(x).reshape(-1, self.num_heads, self.num_chanels, self.out_dim)
        v = self.v(x).reshape(-1, self.num_heads, self.num_chanels, self.out_dim)

        att_scores = torch.einsum("bhcf,bhdf->bhcd", q, k) / (self.out_dim**0.5)
        vals = torch.einsum(
            "bhcd,bhdf->bhcf",
            torch.softmax(att_scores, dim=-1) - self.eye,
            v,
        )

        vals = vals.permute(0, 2, 1, 3)  # [B, C, H, F]
        vals = vals.reshape(vals.size(0), vals.size(1), self.num_heads * self.out_dim)

        return self.w0(vals)


class ODEFunc(nn.Module):
    def __init__(self, func, **kwargs):
        super().__init__()

        self.func = func
        self.kwargs = kwargs

    def forward(self, t, x):
        return self.func(x)


class GRAND(nn.Module):
    def __init__(self, num_heads, num_chanels, featurs_dim, out_dim):
        super().__init__()

        self.grand_encoder = ODEFunc(
            GRANDBlock(num_heads, num_chanels, featurs_dim, out_dim)
        )
        self.grand_decoder = ODEFunc(
            GRANDBlock(num_heads, num_chanels, featurs_dim, out_dim)
        )
        self.ln = nn.Linear(num_chanels, 3)

    def forward(self, x, t):
        z = odeint(self.grand_encoder, x, t, method="rk4")[-1]
        x_hat = odeint(self.grand_decoder, x, t, method="rk4")[-1]

        return self.ln(z.permute(0, 2, 1)).permute(0, 2, 1), x_hat