import jax
import jax.numpy as jnp
import flax.linen as nn

from STFormerBlocks import STFormerBlock, DiffGraphSTFormerBlock
from VAE import VAE
import matplotlib.pyplot as plt
import optax
from typing import Any
from flax.training import train_state, checkpoints
from torch import Tensor
import torch
from jax import grad, value_and_grad
from tqdm import tqdm
import numpy as np
from eegdash.dataset import DS006940
from functools import partial
from flax import nnx
from utils import create_v_model

base_key = jax.random.key(42)


class STFormer(nnx.Module):

    def __init__(
        self,
        in_dim,
        out_dim,
        model_dim,
        num_heads,
        head_dim,
        vae_latent,
        num_layers,
        num_chanels,
        *,
        rngs: nnx.Rngs,
    ):

        self.model_layers = nnx.Sequential(
            *[
                STFormerBlock(
                    in_dim=vae_latent if i == 0 else model_dim,
                    out_dim=model_dim,
                    model_dim=model_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    num_chanels=num_chanels,
                    rngs=rngs,
                )
                for i in range(num_layers)
            ]
        )

        backup = nnx.split_rngs(rngs, splits=num_chanels, only="params")

        self.vae = create_v_model(
            rngs, VAE, model_args={"latent_dim": vae_latent, "input_dim": in_dim}
        )

        self.mu_proj = create_v_model(
            rngs,
            nnx.Linear,
            model_args={"in_features": model_dim, "out_features": vae_latent},
        )

        self.logvar_prog = create_v_model(
            rngs,
            nnx.Linear,
            model_args={"in_features": model_dim, "out_features": vae_latent},
        )

        self.out_proj = create_v_model(
            rngs,
            nnx.Linear,
            model_args={"in_features": vae_latent, "out_features": out_dim},
        )

        nnx.restore_rngs(backup)

        self.rngs = nnx.Rngs(rngs.params())

    def __call__(self, x):

        B, S, C, D = x.shape

        x_ = x.reshape(B * S, C, D)

        recon, mu, logvar = nnx.vmap(lambda vae, x: vae(x), in_axes=(0, 1), out_axes=1)(
            self.vae, x_
        )

        recon = recon.reshape(B, S, C, D)

        # mu_ = mu.reshape(B, S, C, -1)
        # logvar_ = logvar.reshape(B, S, C, -1)

        # z = mu_ + logvar_ * rngs
        z = mu.reshape(B, S, C, -1)

        p = self.model_layers(z).reshape(B * S, C, -1)

        mu_pred = nnx.vmap(lambda proj, h: proj(h), in_axes=(0, 1), out_axes=1)(
            self.mu_proj, p
        )
        logvar_pred = nnx.vmap(lambda proj, h: proj(h), in_axes=(0, 1), out_axes=1)(
            self.logvar_prog, p
        )

        sigma = jax.nn.softplus(logvar_pred) + 1e-4
        eps = self.rngs.normal(shape=mu_pred.shape)

        z_next = mu_pred + sigma * eps

        out = nnx.vmap(lambda proj, h: proj(h), in_axes=(0, 1), out_axes=1)(
            self.out_proj, z_next
        ).reshape(B, S, C, -1)

        return {
            "prediction": out,
            "reconstruction": recon,
            "mu": mu,
            "logvar": logvar,
            "mu_pred": mu_pred,
            "sigma": sigma,
        }


class DiffGraphSTFormer(STFormer):

    def __init__(
        self,
        in_dim,
        out_dim,
        model_dim,
        num_heads,
        head_dim,
        vae_latent,
        num_layers,
        num_chanels,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            model_dim=model_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            vae_latent=vae_latent,
            num_layers=num_layers,
            num_chanels=num_chanels,
            rngs=rngs,
        )

        self.model_layers = nnx.Sequential(
            *[
                DiffGraphSTFormerBlock(
                    in_dim=vae_latent if i == 0 else model_dim,
                    out_dim=model_dim,
                    model_dim=model_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    num_chanels=num_chanels,
                    rngs=rngs,
                )
                for i in range(num_layers)
            ]
        )
