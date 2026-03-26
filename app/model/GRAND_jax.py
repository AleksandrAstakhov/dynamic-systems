import jax
import jax.numpy as jnp
import flax.linen as nn
import jraph
from diffrax import ODETerm, SaveAt, Tsit5, diffeqsolve
import diffrax
from dataclasses import field
from typing import Type


class GrandDiffuser(nn.Module):
    num_heads: int
    head_dim: int
    latent_dim: int
    out_dim: int

    @nn.compact
    def __call__(self, t, k, q, v):

        k = nn.Dense(self.num_heads * self.head_dim)(k)
        q = nn.Dense(self.num_heads * self.head_dim)(q)
        v = nn.Dense(self.num_heads * self.head_dim)(v)

        def split_heads(t):
            B, C, D = t.shape
            return t.reshape(B, C, self.num_heads, self.head_dim)

        k = split_heads(k)
        q = split_heads(q)
        v = split_heads(v)

        attn = nn.softmax(jnp.einsum("bchd,bqhd->bchq", k, q), axis=-1)
        B, C, H, _ = attn.shape

        out = jnp.einsum(
            "bchq,bqhd->bchd",
            attn - jnp.eye(C).reshape(1, C, 1, C),
            v,
        )
        D = v.shape[-1]

        return nn.Dense(self.out_dim)(out.reshape(B, C, H * D))


class Grand(nn.Module):
    num_heads: int
    head_dim: int
    latent_dim: int
    out_dim: int
    diffuser: Type[nn.Module]
    solver_cls: diffrax.AbstractERK = field(default_factory=Tsit5)

    @nn.compact
    def __call__(self, x, t):

        diffuser = self.diffuser(
            self.num_heads, self.head_dim, self.latent_dim, self.out_dim
        )

        diffusion_term = ODETerm(lambda t, y, args: diffuser(t, y, y, y))

        solver = self.solver_cls
        saveat = SaveAt(ts=t)

        print(x.shape)

        z = diffeqsolve(
            diffusion_term, solver, dt0=0.2, t0=t[0], t1=t[-1], y0=x, saveat=saveat
        ).ys[-1]

        return z
