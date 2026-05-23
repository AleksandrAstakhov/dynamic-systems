from __future__ import annotations

import torch
import torch.nn as nn


def empirical_correlation(x: torch.Tensor | "ndarray") -> torch.Tensor:
    """x: [T, C] (numpy or torch) -> Pearson correlation [C, C]."""
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x, dtype=torch.float32)
    x = x - x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True) + 1e-8
    x = x / std
    return (x.T @ x) / x.shape[0]


class CorrelationSpatial(nn.Module):
    """A is a fixed [C, C] matrix derived from the data correlation.

    A learnable scalar `gain` allows the optimizer to scale the matrix
    (so this baseline is not unfairly handicapped by the absolute scale
    of correlation). A learnable bias per sensor is added too.

    Note: the empirical correlation matrix can have spectral radius
    far above 1 (up to C for fully correlated sensors). Applying it
    directly inside the rollout (1-alpha)*I + alpha*gain*A leads to
    immediate numerical blowup at training start. We therefore
    normalize the matrix to spectral radius 1 before storing it;
    the learnable `gain` keeps the same expressive scale as before.
    """

    def __init__(
        self,
        num_channels: int,
        init_corr: torch.Tensor | None = None,
        normalize: bool = True,
    ):
        super().__init__()
        self.num_channels = num_channels
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
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def attn(self, z: torch.Tensor) -> torch.Tensor:

        B = z.shape[0]
        return (self.gain * self.A_fixed).unsqueeze(0).expand(B, -1, -1)

    def aux_loss(self, z: torch.Tensor) -> torch.Tensor:
        return torch.zeros(1, device=z.device).squeeze()

    def forward(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        A = self.gain * self.A_fixed
        return h @ A.T + self.bias
