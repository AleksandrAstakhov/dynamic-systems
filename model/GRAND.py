import torch
import torch.nn as nn
from torchdiffeq import odeint


class GRANDBlock(nn.Module):
    def __init__(self, num_heads, num_chanels, featurs_dim, out_dim, rho=0.7):
        super().__init__()

        self.num_heads = num_heads
        self.num_chanels = num_chanels
        self.featurs_dim = featurs_dim
        self.out_dim = out_dim
        self.rho = rho

        self.register_buffer("eye", torch.eye(self.num_chanels))

        self.k = nn.Linear(self.featurs_dim, self.out_dim * self.num_heads)
        self.q = nn.Linear(self.featurs_dim, self.out_dim * self.num_heads)
        self.v = nn.Linear(self.featurs_dim, self.out_dim)

    def forward(
        self,
        x,
    ):

        k = self.k(x).reshape(-1, self.num_heads, self.num_chanels, self.out_dim)
        q = self.q(x).reshape(-1, self.num_heads, self.num_chanels, self.out_dim)
        v = self.v(x)

        att_scores = torch.einsum("bhcf,bhdf->bhcd", q, k) / (self.out_dim**0.5)

        att_map = torch.softmax(
            (torch.softmax(att_scores, dim=-1) > self.rho).long() * att_scores, dim=-1
        ).mean(dim=1)

        out = torch.einsum(
            "bcd,bdf->bcf",
            att_map - self.eye,
            v,
        )

        return out


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

    def forward(self, x, t):
        z = odeint(self.grand_encoder, x, t, method="rk4")[-1]

        return z


class GRANDAutoencoder(nn.Module):
    def __init__(self, num_heads, num_chanels, featurs_dim, latent_dim):
        super().__init__()

        self.encoder = ODEFunc(
            GRANDBlock(num_heads, num_chanels, featurs_dim, latent_dim)
        )
        self.decoder = ODEFunc(
            GRANDBlock(num_heads, num_chanels, latent_dim, featurs_dim)
        )

        self.enc_linear = nn.Linear(num_chanels, 3)

    def forward(self, x, t):
        z = odeint(self.encoder, x, t, method="rk4")[-1]
        z_lin = self.enc_linear(z.permute(0, 2, 1)).permute(0, 2, 1)

        x_hat = odeint(self.decoder, z, t, method="rk4")[-1]

        return z_lin, x_hat
