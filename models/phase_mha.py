from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhaseConditionalSpatial(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        num_channels: int,
        d_head: int = 16,
        n_modes: int = 2,
        untied: bool = True,
        routing_hidden: int = 64,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_channels = num_channels
        self.d_head = d_head
        self.n_modes = n_modes
        self.untied = untied

        self.W_q = nn.Parameter(torch.randn(n_modes, latent_dim, d_head) * 0.05)
        if untied:
            self.W_k = nn.Parameter(torch.randn(n_modes, latent_dim, d_head) * 0.05)
        else:
            self.W_k = self.W_q

        self.routing = nn.Sequential(
            nn.Linear(latent_dim, routing_hidden),
            nn.GELU(),
            nn.Linear(routing_hidden, n_modes),
        )

        with torch.no_grad():
            self.routing[-1].weight.zero_()
            self.routing[-1].bias.zero_()

    def phase_weights(self, z: torch.Tensor) -> torch.Tensor:
        z_global = z.mean(dim=1)
        logits = self.routing(z_global)
        return F.softmax(logits, dim=-1)

    def per_mode_attn(self, z: torch.Tensor) -> torch.Tensor:
        Q = torch.einsum("bcl,kld->bkcd", z, self.W_q)
        K = torch.einsum("bcl,kld->bkcd", z, self.W_k)
        return torch.einsum("bkid,bkjd->bkij", Q, K) / math.sqrt(self.d_head)

    def attn(self, z: torch.Tensor) -> torch.Tensor:
        per = self.per_mode_attn(z)
        w = self.phase_weights(z)
        return torch.einsum("bk,bkij->bij", w, per)

    def aux_loss(self, z: torch.Tensor) -> torch.Tensor:
        w = self.phase_weights(z)
        p_bar = w.mean(dim=0)
        H = -(p_bar * (p_bar + 1e-8).log()).sum()
        return math.log(self.n_modes) - H

    def forward(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        A = self.attn(z)
        return torch.einsum("bij,bj->bi", A, h)
