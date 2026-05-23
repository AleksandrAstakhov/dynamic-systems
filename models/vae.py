from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        assert n_heads * self.d_head == d_model
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o = nn.Linear(d_model, d_model)

    def forward(self, x):
        N, T, D = x.shape
        qkv = self.qkv(x).view(N, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        scores = scores.masked_fill(mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(N, T, D)
        return self.o(out)


class PosEnc(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].size(1)])
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(1)]


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4, ff_mult: int = 2):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Linear(ff_mult * d_model, d_model),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class SequenceVAE(nn.Module):
    """Per-sensor causal Transformer encoder over Takens-window sequence.

    Encoder collapses sensors into batch axis (per-sensor independent),
    so spatial coupling is left entirely to the downstream spatial module.
    """

    def __init__(
        self,
        m_embed: int,
        latent_dim: int,
        warmup: int,
        d_model: int = 32,
        n_heads: int = 4,
        n_blocks: int = 2,
    ):
        super().__init__()
        self.warmup = warmup
        self.m_embed = m_embed
        self.latent_dim = latent_dim

        self.in_proj = nn.Linear(m_embed, d_model)
        self.pos = PosEnc(d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads) for _ in range(n_blocks)])
        self.ln = nn.LayerNorm(d_model)
        self.mu_head = nn.Linear(d_model, latent_dim)
        self.logvar_head = nn.Linear(d_model, latent_dim)
        self.decoder = nn.Linear(latent_dim, m_embed)

    def encode(self, x):
        B, T, C, m = x.shape
        flat = x.permute(0, 2, 1, 3).contiguous().view(B * C, T, m)
        h = self.pos(self.in_proj(flat))
        for blk in self.blocks:
            h = blk(h)
        h = self.ln(h)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)

        def back(t_):
            return t_.view(B, C, T, -1).permute(0, 2, 1, 3).contiguous()

        return back(mu), back(logvar)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)
        return mu

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)

        mu = mu[:, self.warmup :]
        logvar = logvar[:, self.warmup :]
        z = z[:, self.warmup :]
        recon = recon[:, self.warmup :]
        target = x[:, self.warmup :]
        return dict(z=z, mu=mu, logvar=logvar, recon=recon, target=target)

    @staticmethod
    def losses(out, beta: float = 1e-3):
        rec = F.mse_loss(out["recon"], out["target"])
        mu, logvar = out["mu"], out["logvar"]
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return rec, kl
