from flax import nnx
import jax.numpy as jnp
from utils import create_v_model
from STFormerBlocks import Transformer, MLP
from typing import Tuple
import jax


class STFormerBlock(nnx.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        temporal_model_dim: int,
        spatial_model_dim: int,
        temporal_num_heads: int,
        spatial_num_heads: int,
        temporal_head_dim: int,
        spatial_head_dim: int,
        temporal_multichanel: bool = True,
        num_chanels: int = 0,
        spatial_model_cls: nnx.Module | None = None,
        spatial_model_extra_params: dict = dict(),
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):

        self.rngs = nnx.Rngs(rngs())
        self.ln = nnx.LayerNorm(num_features=temporal_model_dim, rngs=self.rngs)

        if temporal_multichanel:
            backup = nnx.split_rngs(self.rngs, splits=num_chanels)

            self.temporal_model = create_v_model(
                self.rngs,
                Transformer,
                model_args={
                    "in_dim": in_dim,
                    "out_dim": temporal_model_dim,
                    "model_dim": temporal_model_dim,
                    "num_heads": temporal_num_heads,
                    "head_dim": temporal_head_dim,
                    "need_pos_enc": True,
                },
            )

            nnx.restore_rngs(backup)

        else:
            self.temporal_model = Transformer(
                model_args={
                    "in_dim": in_dim,
                    "out_dim": temporal_model_dim,
                    "model_dim": temporal_model_dim,
                    "num_heads": temporal_num_heads,
                    "head_dim": temporal_head_dim,
                    "need_pos_enc": True,
                },
                rngs=self.rngs,
            )

        if spatial_model_cls:
            self.spatial_model = spatial_model_cls(
                rngs=self.rngs,
                **{
                    "in_dim": temporal_model_dim,
                    "out_dim": temporal_model_dim,
                    "model_dim": spatial_model_dim,
                    "num_heads": spatial_num_heads,
                    "head_dim": spatial_head_dim,
                    **spatial_model_extra_params,
                },
            )
        else:
            self.spatial_model = None

        if temporal_multichanel:

            backup = nnx.split_rngs(self.rngs, splits=num_chanels)

            self.out_model = create_v_model(
                self.rngs,
                MLP,
                model_args={
                    "din": temporal_model_dim,
                    "dmid": temporal_model_dim,
                    "dout": out_dim,
                },
            )

            nnx.restore_rngs(backup)

        else:

            self.out_model = MLP(
                **{
                    "din": temporal_model_dim,
                    "dmid": temporal_model_dim,
                    "dout": out_dim,
                },
                rngs=self.rngs,
            )
        self.temporal_multichanel = temporal_multichanel

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:

        if self.temporal_multichanel:
            z = nnx.vmap(lambda model, h: model(h), in_axes=(0, 2), out_axes=2)(
                self.temporal_model,
                x,
            )
        else:
            z = self.temporal_model(x)

        B, S, C, _ = z.shape

        if self.spatial_model:
            z = self.spatial_model(z.reshape(B * S, C, -1))

            z = z.reshape(B, S, C, -1)

        z = self.ln(z)

        if self.temporal_multichanel:

            z = nnx.vmap(lambda model, h: model(h), in_axes=(0, 2), out_axes=2)(
                self.out_model,
                z,
            )
        else:
            z = self.out_model(z)

        return z


class STFormer(nnx.Module):

    def __init__(
        self,
        in_dim: int,
        latent_dim: int,
        out_dim: int,
        temporal_model_dim: int,
        spatial_model_dim: int,
        num_blocks: int,
        temporal_num_heads: int,
        spatial_num_heads: int,
        temporal_head_dim: int,
        spatial_head_dim: int,
        temporal_multichanel: bool = True,
        num_chanels: int = 0,
        encoder: nnx.Module | None = None,
        spatial_model_cls: nnx.Module | None = None,
        spatial_model_extra_params: dict = dict(),
        dt: int = 1 / 250,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):

        self.rngs = nnx.Rngs(rngs())
        self.dt = dt

        self.encoder = encoder

        self.stmodel = nnx.Sequential(
            *[
                STFormerBlock(
                    in_dim=in_dim if i == 0 else temporal_model_dim,
                    out_dim=temporal_model_dim,
                    temporal_model_dim=temporal_model_dim,
                    spatial_model_dim=spatial_model_dim,
                    temporal_num_heads=temporal_num_heads,
                    spatial_num_heads=spatial_num_heads,
                    temporal_head_dim=temporal_head_dim,
                    spatial_head_dim=spatial_head_dim,
                    temporal_multichanel=temporal_multichanel,
                    num_chanels=num_chanels,
                    spatial_model_cls=spatial_model_cls,
                    spatial_model_extra_params=spatial_model_extra_params,
                    rngs=rngs,
                )
                for i in range(num_blocks)
            ]
        )

        if temporal_multichanel:

            backup = nnx.split_rngs(self.rngs, splits=num_chanels)

            self.drift_model = create_v_model(
                self.rngs,
                MLP,
                model_args={
                    "din": temporal_model_dim,
                    "dmid": temporal_model_dim,
                    "dout": out_dim,
                },
            )

            nnx.restore_rngs(backup)

        else:
            self.drift_model = MLP(
                **{
                    "din": temporal_model_dim,
                    "dmid": temporal_model_dim,
                    "dout": out_dim,
                },
                rngs=self.rngs,
            )

        self.temporal_multichanel = temporal_multichanel

    def __call__(self, z):

        if self.encoder:

            mu, logvar = nnx.vmap(
                lambda model, h: model(h), in_axes=(0, 2), out_axes=(2, 2)
            )(self.encoder, z)

            z = mu + (jax.nn.softplus(logvar) + 1e-4) * self.rngs.normal(mu.shape)

        z = self.stmodel(z)

        if self.temporal_multichanel:

            dz = nnx.vmap(lambda model, h: model(h), in_axes=(0, 2), out_axes=2)(
                self.drift_model, z
            )

        else:

            dz = self.drift_model(z)

        return dz
