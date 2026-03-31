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
        out_dim,
        model_dim,
        num_heads,
        head_dim,
        vae_latent,
        num_layers,
        num_chanels,
        edge_index,
        *,
        rngs: nnx.Rngs,
    ):

        self.model_layers = nnx.Sequential(
            *[
                LightSTFormerBlock(
                    in_dim=vae_latent if i == 0 else model_dim,
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

        backup = nnx.split_rngs(rngs, splits=num_chanels)

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

        p = self.model_layers(None, z, None).reshape(B * S, C, -1)

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
        edge_index,
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
                LightDiffGraphSTFormerBlock(
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


class TFormer(nnx.Module):

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
        edge_index,
        *,
        rngs: nnx.Rngs,
    ):

        self.model_layers = nnx.Sequential(
            *[
                LightTFormerBlock(
                    in_dim=vae_latent if i == 0 else model_dim,
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

        backup = nnx.split_rngs(rngs, splits=num_chanels)

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


class GConvSTFormer(nnx.Module):

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
        edge_index,
        *,
        rngs: nnx.Rngs,
    ):

        self.model_layers = nnx.Sequential(
            *[
                LightGConvSTFormerBlock(
                    in_dim=vae_latent if i == 0 else model_dim,
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


class VectorVAE(nnx.Module):

    def __init__(
        self,
        in_dim,
        vae_latent,
        num_chanels,
        *,
        rngs: nnx.Rngs,
    ):

        self.rngs = nnx.Rngs(rngs())

        backup = nnx.split_rngs(rngs, splits=num_chanels)

        self.vae = create_v_model(
            rngs, VAE, model_args={"latent_dim": vae_latent, "input_dim": in_dim}
        )

        nnx.restore_rngs(backup)

    def __call__(self, x: jnp.ndarray):

        B, S, C, D = x.shape

        x_ = x.reshape(B * S, C, D)

        recon, mu, logvar = nnx.vmap(lambda vae, x: vae(x), in_axes=(0, 1), out_axes=1)(
            self.vae, x_
        )

        std = jnp.exp(0.5 * logvar)

        rng = self.rngs()
        eps = jax.random.normal(rng, std.shape)

        z = mu + eps * std

        recon = recon.reshape(B, S, C, D)
        mu = mu.reshape(B, S, C, -1)
        logvar = logvar.reshape(B, S, C, -1)
        z = z.reshape(B, S, C, -1)

        return recon, mu, logvar, z

    @staticmethod
    def loss_fn(x, recon, mu, logvar, beta: float = 1.0):

        recon_loss = jnp.mean((recon - x) ** 2)

        kl = -0.5 * (1 + logvar - mu**2 - jnp.exp(logvar))
        kl = jnp.mean(jnp.sum(kl, axis=-1))

        loss = recon_loss + beta * kl

        return loss


class diff_sigma(nnx.Module):

    def __init__(self, out_dim, vae_latent, model_dim, num_chanels, rngs):

        # backup = nnx.split_rngs(rngs, splits=num_chanels)

        # self.mu_proj = create_v_model(
        #     rngs,
        #     nnx.Linear,
        #     model_args={"in_features": model_dim, "out_features": vae_latent},
        # )

        # self.logvar_prog = create_v_model(
        #     rngs,
        #     nnx.Linear,
        #     model_args={"in_features": model_dim, "out_features": vae_latent},
        # )

        self.mu_proj = nnx.Linear(model_dim, vae_latent, rngs=rngs)
        self.logvar_prog = nnx.Linear(model_dim, vae_latent, rngs=rngs)

        self.model_dim = model_dim

        # nnx.restore_rngs(backup)

    def __call__(self, x):
        B, S, C, D = x.shape

        x = x.reshape(B * S, C, D)

        a, b = x[:, :, : self.model_dim], x[:, :, self.model_dim :]

        # mu_pred = nnx.vmap(lambda proj, h: proj(h), in_axes=(0, 1), out_axes=1)(
        #     self.mu_proj, a
        # )

        # logvar_pred = nnx.vmap(lambda proj, h: proj(h), in_axes=(0, 1), out_axes=1)(
        #     self.logvar_prog, b
        # )

        mu_pred = self.mu_proj(a)
        logvar_pred = self.logvar_prog(b)

        sigma = jnp.exp(0.5 * logvar_pred)
        return mu_pred.reshape(B, S, C, -1), jnp.clip(sigma, 1e-4, 5.0).reshape(
            B, S, C, -1
        )


class SDESTFormer(nnx.Module):

    def __init__(
        self,
        out_dim,
        model_dim,
        num_heads,
        head_dim,
        vae_latent,
        num_layers,
        num_chanels,
        edge_index,
        model_cls=LightSTFormerBlock,
        *,
        rngs: nnx.Rngs,
    ):

        backup = nnx.split_rngs(rngs, splits=num_chanels)

        self.out_proj = create_v_model(
            rngs,
            nnx.Linear,
            model_args={"in_features": vae_latent, "out_features": out_dim},
        )

        nnx.restore_rngs(backup)

        # self.k = diff_sigma(
        #     model_dim=model_dim,
        #     out_dim=out_dim,
        #     num_chanels=num_chanels,
        #     vae_latent=vae_latent,
        #     rngs=rngs,
        # )

        self.rngs = nnx.Rngs(rngs())

        models = [
            model_cls(
                in_dim=vae_latent if i == 0 else model_dim,
                out_dim=model_dim if i != num_layers - 1 else 2 * model_dim,
                model_dim=model_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                num_chanels=num_chanels,
                edge_index=edge_index,
                rngs=rngs,
            )
            for i in range(num_layers)
        ] + [
            diff_sigma(
                model_dim=model_dim,
                out_dim=out_dim,
                num_chanels=num_chanels,
                vae_latent=vae_latent,
                rngs=rngs,
            )
        ]

        self.stmodel = nnx.Sequential(*models)

    def __call__(self, z0, ts=jnp.linspace(0, 0.1, 2), only_last=False):

        def __drift_diffusion(t, y, arg):
            drift, diff = self.stmodel(t, y, arg)
            return drift, diff

        def f(t, y, arg):
            drift, _ = __drift_diffusion(t, y, arg)
            return drift

        def g(t, y, arg):
            _, diff = __drift_diffusion(t, y, arg)
            return diff

        B, S, C, D = z0.shape

        rng = self.rngs()
        bm = diffrax.VirtualBrownianTree(
            t0=ts[0],
            t1=ts[-1],
            tol=1e-3,
            shape=z0.shape,
            key=rng,
        )

        term = diffrax.MultiTerm(
            diffrax.ODETerm(f),
            diffrax.ControlTerm(g, bm),
        )

        solver = diffrax.EulerHeun()

        edge_index = None

        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0=ts[0],
            t1=ts[-1],
            dt0=ts[1] - ts[0],
            y0=z0,
            args=edge_index,
            saveat=diffrax.SaveAt(t1=True),
        )

        z_t = sol.ys[-1].reshape(B * S, C, -1)

        out = nnx.vmap(lambda proj, h: proj(h), in_axes=(0, 1), out_axes=1)(
            self.out_proj, z_t
        ).reshape(B, S, C, -1)

        return out