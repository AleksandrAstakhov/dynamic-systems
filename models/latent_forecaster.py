from __future__ import annotations

import torch
import torch.nn as nn

from .latent_spatial import (
    CorrelationLatentSpatial,
    GRANDDiffLatentSpatial,
    GRANDFullLatentSpatial,
)
from .vae import Block as VAEBlock
from .vae import SequenceVAE


class TemporalPerSensor(nn.Module):
    def __init__(
        self, latent_dim: int, n_blocks: int = 1, n_heads: int = 4, ff_mult: int = 2
    ):
        super().__init__()
        d_model = latent_dim
        while n_heads > 1 and d_model % n_heads != 0:
            n_heads -= 1
        self.blocks = nn.ModuleList(
            [
                VAEBlock(d_model, n_heads=n_heads, ff_mult=ff_mult)
                for _ in range(n_blocks)
            ]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B, T, C, L = z.shape
        h = z.permute(0, 2, 1, 3).reshape(B * C, T, L)
        for blk in self.blocks:
            h = blk(h)
        return h.view(B, C, T, L).permute(0, 2, 1, 3).contiguous()


def _make_spatial(
    kind: str,
    latent_dim: int,
    num_channels: int,
    spatial_kwargs: dict,
    corr_matrix: torch.Tensor | None,
):
    sk = spatial_kwargs or {}
    if kind == "correlation":
        return CorrelationLatentSpatial(
            latent_dim=latent_dim,
            num_channels=num_channels,
            init_corr=corr_matrix,
        )
    if kind == "grand_diff":
        return GRANDDiffLatentSpatial(
            latent_dim=latent_dim,
            num_channels=num_channels,
            d_head=sk.get("d_head", 16),
            n_modes=sk.get("n_modes", 2),
            untied=sk.get("untied", True),
            max_radius=sk.get("max_radius", 1.5),
        )
    if kind == "grand_full":
        return GRANDFullLatentSpatial(
            latent_dim=latent_dim,
            num_channels=num_channels,
            d_head=sk.get("d_head", 16),
            n_modes=sk.get("n_modes", 2),
            untied=sk.get("untied", True),
            max_radius=sk.get("max_radius", 1.5),
            mlp_hidden=sk.get("mlp_hidden", None),
        )
    raise ValueError(f"unknown spatial_kind={kind}")


class LatentForecaster(nn.Module):
    def __init__(
        self,
        m_embed: int,
        latent_dim: int,
        num_channels: int,
        warmup: int,
        horizon: int,
        spatial_kind: str,
        vae_dim: int = 32,
        vae_blocks: int = 2,
        vae_spatial_blocks: int = 1,
        temporal_blocks: int = 1,
        spatial_kwargs: dict | None = None,
        corr_matrix: torch.Tensor | None = None,
        deterministic_encoder: bool = False,
        use_phase_loss: bool = False,
    ):
        super().__init__()
        self.warmup = warmup
        self.horizon = horizon
        self.num_channels = num_channels
        self.latent_dim = latent_dim
        self.spatial_kind = spatial_kind

        self.encoder = SequenceVAE(
            m_embed=m_embed,
            latent_dim=latent_dim,
            warmup=warmup,
            d_model=vae_dim,
            n_blocks=vae_blocks,
            n_spatial_blocks=vae_spatial_blocks,
            deterministic=deterministic_encoder,
        )
        self.temporal = TemporalPerSensor(latent_dim, n_blocks=temporal_blocks)
        self.use_phase_loss = use_phase_loss
        if use_phase_loss:
            self.phase_head = nn.Linear(latent_dim, 1)
        self.spatial = _make_spatial(
            spatial_kind,
            latent_dim,
            num_channels,
            spatial_kwargs or {},
            corr_matrix,
        )
        self.decoder = nn.Linear(latent_dim, 1)

        self.alpha_logit = nn.Parameter(torch.tensor(0.0))

    def step(self, z: torch.Tensor) -> torch.Tensor:
        
        a = torch.sigmoid(self.alpha_logit)
        return z + a * self.spatial.propagate(z)

    def forward(self, x: torch.Tensor, x_raw: torch.Tensor) -> dict:
        vae_out = self.encoder(x)
        z_enc = vae_out["z"]
        z_t = self.temporal(z_enc)

        z_curr = z_t[:, :-1]
        B, Tm1, C, L = z_curr.shape
        z_curr_flat = z_curr.reshape(B * Tm1, C, L)
        z_next_flat = self.step(z_curr_flat)
        in_window_pred = self.decoder(z_next_flat).squeeze(-1).view(B, Tm1, C)
        in_window_target = x_raw[:, self.warmup + 1 :]

        z = z_t[:, -1]
        preds = []
        for _ in range(self.horizon):
            z = self.step(z)
            preds.append(self.decoder(z).squeeze(-1))
        pred = torch.stack(preds, dim=1)

        A_last = self.spatial.attn(z_t[:, -1])
        z_for_A = z_t.reshape(B * z_t.shape[1], C, L)
        A_seq = self.spatial.attn(z_for_A).view(B, z_t.shape[1], C, C)

        phase_logit = None
        if self.use_phase_loss:
            z_global = z_t[:, -1].mean(dim=1)
            phase_logit = self.phase_head(z_global).squeeze(-1)

        return dict(
            pred=pred,
            in_window_pred=in_window_pred,
            in_window_target=in_window_target,
            z=z_t,
            z_last=z_t[:, -1],
            A_seq=A_seq,
            A_last=A_last,
            phase_logit=phase_logit,
            **{k: v for k, v in vae_out.items() if k != "z"},
        )

    def coupling(self, z_last: torch.Tensor) -> torch.Tensor:
        return self.spatial.attn(z_last)
