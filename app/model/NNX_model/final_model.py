import jax
import jax.numpy as jnp

from STFormerBlocks import (
    STFormerBlock,
    DiffGraphSTFormerBlock,
    TFormerBlock,
    LightGConvSTFormerBlock,
    LightSTFormerBlock,
    LightTFormerBlock,
    LightDiffGraphSTFormerBlock,
    MLP,
)
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
import diffrax

from terms import DriftDiffusionTerm

base_key = jax.random.key(42)


class STFormer(nnx.Module):

    def __init__(
        self,
        in_dim,
        vae_latent,
        out_dim,
        model_dim,
        num_heads,
        head_dim,
        num_layers,
        num_chanels,
        edge_index,
        decoder,
        encoder,
        block_cls,
        *,
        rngs: nnx.Rngs,
    ):

        self.model_layers = nnx.Sequential(
            *[
                block_cls(
                    in_dim=3 * vae_latent if i == 0 else model_dim,
                    out_dim=model_dim,
                    model_dim=model_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    num_chanels=num_chanels,
                    edge_index=edge_index,
                    rngs=rngs,
                )
                for i in range(num_layers)
            ]
        )

        self.encoder = encoder

        backup = nnx.split_rngs(rngs, splits=num_chanels)

        # self.vae = create_v_model(
        #     rngs, VAE, model_args={"latent_dim": vae_latent, "input_dim": in_dim}
        # )

        self.mu_proj = create_v_model(
            rngs,
            MLP,
            model_args={
                "din": model_dim,
                "dmid": model_dim,
                "dout": vae_latent,
            },
        )

        self.logvar_prog = create_v_model(
            rngs,
            MLP,
            model_args={
                "din": model_dim,
                "dmid": model_dim,
                "dout": vae_latent,
            },
        )

        self.decoder = decoder

        nnx.restore_rngs(backup)

        self.rngs = nnx.Rngs(rngs())

    def __call__(self, x):

        mu, logvar = nnx.vmap(
            lambda model, h: model(h), in_axes=(0, 2), out_axes=(2, 2)
        )(self.encoder, x)

        sigma = jax.nn.softplus(logvar) + 1e-4
        eps = self.rngs.normal(shape=mu.shape)

        z0 = mu + sigma * eps

        x = jnp.concat([z0, mu, logvar], axis=-1)

        B, S, C, D = x.shape

        p = self.model_layers(z0)

        mu_pred = nnx.vmap(lambda proj, h: proj(h), in_axes=(0, 2), out_axes=2)(
            self.mu_proj, p
        )
        logvar_pred = nnx.vmap(lambda proj, h: proj(h), in_axes=(0, 2), out_axes=2)(
            self.logvar_prog, p
        )

        sigma = jax.nn.softplus(logvar_pred) + 1e-4
        eps = self.rngs.normal(shape=mu_pred.shape)

        z_next = mu_pred + sigma * eps

        out = nnx.vmap(lambda model, h: model(h), in_axes=(0, 2), out_axes=2)(
            self.decoder, z_next
        )

        return {
            "prediction": out,
            "mu_pred": mu_pred,
            "sigma": sigma,
        }


def flow_matching_loss(model, x_t, mu_t, logvar_t, mu_target, sigma_target, dt=0.2):

    drift, log_g = model(x_t, mu_t, logvar_t)

    mu_pred = x_t + drift * dt
    sigma_pred = jnp.exp(log_g) * jnp.sqrt(dt)

    var_pred = sigma_pred**2
    var_target = sigma_target**2

    kl = (
        jnp.log(sigma_pred / sigma_target)
        + (var_target + (mu_target - mu_pred) ** 2) / (2.0 * var_pred)
        - 0.5
    )

    return jnp.mean(jnp.sum(kl, axis=-1))
