from __future__ import annotations

import torch
import torch.nn as nn

from .correlation import CorrelationSpatial
from .phase_mha import PhaseConditionalSpatial
from .vae import SequenceVAE


class SpatialForecaster(nn.Module):
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
        spatial_kwargs: dict | None = None,
        corr_matrix: torch.Tensor | None = None,
    ):
        super().__init__()
        self.warmup = warmup
        self.horizon = horizon
        self.num_channels = num_channels
        self.latent_dim = latent_dim
        self.spatial_kind = spatial_kind

        self.vae = SequenceVAE(
            m_embed=m_embed,
            latent_dim=latent_dim,
            warmup=warmup,
            d_model=vae_dim,
            n_blocks=vae_blocks,
        )

        spatial_kwargs = spatial_kwargs or {}
        if spatial_kind == "phase_mha":
            self.spatial = PhaseConditionalSpatial(
                latent_dim=latent_dim,
                num_channels=num_channels,
                **spatial_kwargs,
            )
        elif spatial_kind == "correlation":
            self.spatial = CorrelationSpatial(
                num_channels=num_channels,
                init_corr=corr_matrix,
            )
        else:
            raise ValueError(f"unknown spatial_kind={spatial_kind}")

        self.alpha = nn.Parameter(torch.tensor(0.5))

    def attn_seq(self, z: torch.Tensor) -> torch.Tensor:
        """Apply spatial attention to every latent timestep.
        z: [B, T', C, L] -> A_seq: [B, T', C, C]"""
        B, Tp, C, L = z.shape
        z_flat = z.reshape(B * Tp, C, L)
        A_flat = self.spatial.attn(z_flat)
        return A_flat.view(B, Tp, C, C)

    def step(self, A: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """One coupling step: h <- (1-α) h + α A h."""
        a = torch.sigmoid(self.alpha)
        return (1 - a) * h + a * torch.einsum("...ij,...j->...i", A, h)

    def rollout_with_A(
        self, A_last: torch.Tensor, h_last: torch.Tensor
    ) -> torch.Tensor:
        """Roll `horizon` steps using the *same* A (per-sample, last timestep)."""
        preds = []
        h = h_last
        for _ in range(self.horizon):
            h = self.step(A_last, h)
            preds.append(h)
        return torch.stack(preds, dim=1)

    def forward(self, x: torch.Tensor, x_raw: torch.Tensor) -> dict:
        """
        x     : [B, W, C, m]    Takens windows
        x_raw : [B, W, C]       raw scalar values aligned with the Takens window
        """
        vae_out = self.vae(x)
        z = vae_out["z"]
        B, Tp, C, _ = z.shape

        A_seq = self.attn_seq(z)

        x_aligned = x_raw[:, self.warmup :]
        x_curr = x_aligned[:, :-1]
        x_next = x_aligned[:, 1:]
        A_for_pred = A_seq[:, :-1]
        in_window_pred = self.step(A_for_pred, x_curr)

        A_last = A_seq[:, -1]
        h_last = x_raw[:, -1]
        pred = self.rollout_with_A(A_last, h_last)

        return dict(
            pred=pred,
            in_window_pred=in_window_pred,
            in_window_target=x_next,
            z=z,
            z_last=z[:, -1],
            A_seq=A_seq,
            A_last=A_last,
            **{k: v for k, v in vae_out.items() if k != "z"},
        )

    def coupling(self, z_last: torch.Tensor) -> torch.Tensor:
        return self.spatial.attn(z_last)
