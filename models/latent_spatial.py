from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _spectral_normalize(
    A: torch.Tensor, max_radius: float = 1.0, n_iter: int = 1
) -> torch.Tensor:

    B, C, _ = A.shape
    v = torch.randn(B, C, 1, device=A.device, dtype=A.dtype)
    v = v / (v.norm(dim=1, keepdim=True) + 1e-8)
    for _ in range(n_iter):
        u = A @ v
        u = u / (u.norm(dim=1, keepdim=True) + 1e-8)
        v = A.transpose(-1, -2) @ u
        v = v / (v.norm(dim=1, keepdim=True) + 1e-8)
    sigma = (u.transpose(-1, -2) @ A @ v).squeeze(-1).squeeze(-1)
    scale = (sigma / max_radius).clamp(min=1.0)
    return A / scale.view(B, 1, 1)


class CorrelationLatentSpatial(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        num_channels: int,
        init_corr: torch.Tensor | None = None,
        normalize: bool = True,
    ):
        super().__init__()
        self.C = num_channels
        if init_corr is None:
            init_corr = torch.eye(num_channels)
        else:
            init_corr = init_corr.detach().clone().float()
        if normalize:
            with torch.no_grad():
                eig = torch.linalg.eigvalsh(init_corr)
                rho = eig.abs().max().clamp(min=1e-6)
                init_corr = init_corr / rho
        self.register_buffer("A_fixed", init_corr, persistent=True)
        self.gain = nn.Parameter(torch.tensor(1.0))
        self.W_v = nn.Linear(latent_dim, latent_dim, bias=False)
        with torch.no_grad():
            self.W_v.weight.copy_(
                torch.eye(latent_dim) + torch.randn_like(self.W_v.weight) * 0.01
            )

    def attn(self, z: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        return (self.gain * self.A_fixed).unsqueeze(0).expand(B, -1, -1)

    def propagate(self, z: torch.Tensor) -> torch.Tensor:
        A = self.attn(z)
        V = self.W_v(z)
        return torch.einsum("bij,bjl->bil", A, V)

    def aux_loss(self, z: torch.Tensor) -> torch.Tensor:
        return torch.zeros((), device=z.device)


class _BankedAttention(nn.Module):

    def __init__(
        self,
        latent_dim: int,
        num_channels: int,
        d_head: int = 16,
        n_modes: int = 2,
        routing_hidden: int = 64,
        untied: bool = True,
        spec_norm: bool = True,
        max_radius: float = 1.5,
    ):
        super().__init__()
        self.C = num_channels
        self.L = latent_dim
        self.d_head = d_head
        self.n_modes = n_modes
        self.spec_norm = spec_norm
        self.max_radius = max_radius
        self.W_q = nn.Parameter(torch.randn(n_modes, latent_dim, d_head) * 0.1)
        if untied:
            self.W_k = nn.Parameter(torch.randn(n_modes, latent_dim, d_head) * 0.1)
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
        self.temperature = 1.0

    def phase_weights(self, z: torch.Tensor) -> torch.Tensor:
        z_global = z.mean(dim=1)
        logits = self.routing(z_global)
        if self.training:
           
            return F.gumbel_softmax(logits, tau=self.temperature, hard=True)
        else:
            idx = logits.argmax(dim=-1)
            return F.one_hot(idx, num_classes=self.n_modes).float()

    def attn(self, z: torch.Tensor) -> torch.Tensor:
        Q = torch.einsum("bcl,kld->bkcd", z, self.W_q)
        K = torch.einsum("bcl,kld->bkcd", z, self.W_k)
        per_mode = torch.einsum("bkid,bkjd->bkij", Q, K) / math.sqrt(self.d_head)
        w = self.phase_weights(z)
        A = torch.einsum("bk,bkij->bij", w, per_mode)
        if self.spec_norm:
            A = _spectral_normalize(A, max_radius=self.max_radius, n_iter=1)
        return A

    def aux(self, z: torch.Tensor) -> torch.Tensor:
        w = self.phase_weights(z)
        p_bar = w.mean(dim=0)
        H = -(p_bar * (p_bar + 1e-8).log()).sum()
        return math.log(self.n_modes) - H


class GRANDDiffLatentSpatial(nn.Module):

    def __init__(
        self,
        latent_dim: int,
        num_channels: int,
        d_head: int = 16,
        n_modes: int = 2,
        routing_hidden: int = 64,
        untied: bool = True,
        max_radius: float = 1.5,
        t_max: float = 1.0,
        method: str = "rk4",
        n_steps: int = 4,
    ):
        super().__init__()
        self.attn_mod = _BankedAttention(
            latent_dim,
            num_channels,
            d_head=d_head,
            n_modes=n_modes,
            routing_hidden=routing_hidden,
            untied=untied,
            spec_norm=True,
            max_radius=max_radius,
        )
        self.W_v = nn.Linear(latent_dim, latent_dim, bias=False)
        with torch.no_grad():
            self.W_v.weight.copy_(
                torch.eye(latent_dim) + torch.randn_like(self.W_v.weight) * 0.01
            )

    def attn(self, z: torch.Tensor) -> torch.Tensor:
        return self.attn_mod.attn(z)

    def propagate(self, z: torch.Tensor) -> torch.Tensor:
        A = self.attn(z)
        V = self.W_v(z)
        return torch.einsum("bij,bjl->bil", A, V)

    def aux_loss(self, z: torch.Tensor) -> torch.Tensor:
        return self.attn_mod.aux(z)


class GRANDFullLatentSpatial(GRANDDiffLatentSpatial):

    def __init__(
        self,
        latent_dim: int,
        num_channels: int,
        d_head: int = 16,
        n_modes: int = 2,
        routing_hidden: int = 64,
        untied: bool = True,
        max_radius: float = 1.5,
        mlp_hidden: int | None = None,
        t_max: float = 1.0,
        method: str = "rk4",
        n_steps: int = 4,
    ):
        super().__init__(
            latent_dim,
            num_channels,
            d_head=d_head,
            n_modes=n_modes,
            routing_hidden=routing_hidden,
            untied=untied,
            max_radius=max_radius,
        )
        mlp_hidden = mlp_hidden or (2 * latent_dim)
        self.reaction = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, latent_dim),
        )
        with torch.no_grad():
            self.reaction[-1].weight.zero_()
            self.reaction[-1].bias.zero_()

    def propagate(self, z: torch.Tensor) -> torch.Tensor:
        A = self.attn(z)
        V = self.W_v(z)
        diff = torch.einsum("bij,bjl->bil", A, V)
        return diff + self.reaction(z)
