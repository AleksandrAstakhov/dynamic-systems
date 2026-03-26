import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx import bridge


from positional_encoding import TimeSeriesPositionalEncoding
from GRAND_jax import Grand, GrandDiffuser

from attention import MultiHeadAttention
from utils import create_v_model


class MLP(nnx.Module):
    def __init__(self, din: int, dmid: int, dout: int, *, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(din, dmid, rngs=rngs)
        self.linear2 = nnx.Linear(dmid, dout, rngs=rngs)

    def __call__(self, x: jax.Array):
        x = nnx.gelu(self.linear1(x))
        return self.linear2(x)


class Transformer(nnx.Module):

    def __init__(
        self,
        in_dim,
        out_dim,
        model_dim,
        num_heads,
        head_dim,
        *,
        rngs: nnx.Rngs,
        need_pos_enc=True
    ):
        self.need_pos_enc = need_pos_enc

        self.pos_enc = TimeSeriesPositionalEncoding()

        self.mha = MultiHeadAttention(
            in_dim=model_dim,
            out_dim=model_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            rngs=rngs,
        )

        self.init_lin = nnx.Linear(in_dim, model_dim, rngs=rngs)

        self.ln1 = nnx.LayerNorm(model_dim, rngs=rngs)
        self.ln2 = nnx.LayerNorm(model_dim, rngs=rngs)

        self.mlp = MLP(model_dim, model_dim, out_dim, rngs=rngs)
        self.rngs = nnx.Rngs(rngs.params())

    def __call__(self, x):

        x = self.init_lin(x)

        if self.need_pos_enc:
            x = self.pos_enc(x)

        h = self.ln1(x)
        h = self.mha(h, h, h)
        x = x + h

        h = self.ln2(x)
        h = self.mlp(h)
        x = x + h

        return x


class STFormerBlock(nnx.Module):

    def __init__(
        self,
        in_dim,
        out_dim,
        model_dim,
        num_heads,
        head_dim,
        num_chanels,
        *,
        rngs: nnx.Rngs
    ):

        self.spatial_transformer = Transformer(
            in_dim=model_dim,
            out_dim=model_dim,
            num_heads=num_heads,
            model_dim=model_dim,
            head_dim=head_dim,
            need_pos_enc=False,
            rngs=rngs,
        )

        self.temporal_transformer = Transformer(
            in_dim=model_dim,
            out_dim=model_dim,
            num_heads=num_heads,
            model_dim=model_dim,
            head_dim=head_dim,
            need_pos_enc=True,
            rngs=rngs,
        )

        self.ln = nnx.LayerNorm(model_dim, rngs=rngs)

        self.mlp = MLP(model_dim, model_dim, out_dim, rngs=rngs)

        # backup = nnx.split_rngs(rngs, splits=num_chanels, only="params")

        # self.temporal_transformer = create_v_model(
        #     rngs,
        #     Transformer,
        #     model_args={
        #         "in_dim": in_dim,
        #         "out_dim": model_dim,
        #         "num_heads": num_heads,
        #         "model_dim": model_dim,
        #         "num_heads": num_heads,
        #         "head_dim": head_dim,
        #         "need_pos_enc": True,
        #     },
        # )

        # nnx.restore_rngs(backup)

        # self.rngs = nnx.Rngs(rngs.params())

    def __call__(self, x):
        B, S, C, D = x.shape

        x = x.transpose(0, 2, 1, 3).reshape(B * C, S, D)

        # h = nnx.vmap(lambda model, x: model(x), in_axes=(0, 2), out_axes=2)(
        #     self.temporal_transformer, x
        # )

        h = self.temporal_transformer(x)

        # h = nnx.vmap(lambda d: self.spatial_transformer(d), in_axes=1, out_axes=1)(h)

        h = h.reshape(B, C, S, D).transpose(0, 2, 1, 3).reshape(B * S, C, D)

        h = self.spatial_transformer(h)

        res = h
        h = self.ln(h)
        h = self.mlp(h)

        return (h + res).reshape(B, S, C, D)


class DiffGraphSTFormerBlock(nnx.Module):

    def __init__(
        self,
        in_dim,
        out_dim,
        model_dim,
        num_heads,
        head_dim,
        num_chanels,
        *,
        rngs: nnx.Rngs
    ):

        self.spatial_model = Grand(
            in_dim=model_dim,
            model_dim=model_dim,
            out_dim=model_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            num_chanels=num_chanels,
            rngs=rngs,
        )

        self.ln = nnx.LayerNorm(model_dim, rngs=rngs)

        self.mlp = MLP(model_dim, model_dim, out_dim, rngs=rngs)

        self.temporal_transformer = Transformer(
            in_dim=model_dim,
            out_dim=model_dim,
            num_heads=num_heads,
            model_dim=model_dim,
            head_dim=head_dim,
            need_pos_enc=True,
            rngs=rngs,
        )

        # backup = nnx.split_rngs(rngs, splits=num_chanels, only="params")

        # self.temporal_transformer = create_v_model(
        #     rngs,
        #     Transformer,
        #     model_args={
        #         "in_dim": in_dim,
        #         "out_dim": model_dim,
        #         "model_dim": model_dim,
        #         "num_heads": num_heads,
        #         "head_dim": head_dim,
        #         "need_pos_enc": True,
        #     },
        # )

        # nnx.restore_rngs(backup)

        self.rngs = nnx.Rngs(rngs.params())

    def __call__(self, x: jax.Array) -> jax.Array:
        B, S, C, D = x.shape

        # h = nnx.vmap(lambda model, x: model(x), in_axes=(0, 2), out_axes=2)(
        #     self.temporal_transformer, x
        # )

        x = x.transpose(0, 2, 1, 3).reshape(B * C, S, D)

        h = self.temporal_transformer(x).reshape(B, C, S, D).transpose(0, 2, 1, 3)

        t_grid = jnp.linspace(0, 0.5, 4)

        h = self.spatial_model(h, t_grid)

        res = h
        h = self.ln(h)
        out = self.mlp(h)

        return out + res


class TFormerBlock(nnx.Module):

    def __init__(
        self,
        in_dim,
        out_dim,
        model_dim,
        num_heads,
        head_dim,
        num_chanels,
        *,
        rngs: nnx.Rngs
    ):

        self.spatial_transformer = Transformer(
            in_dim=model_dim,
            out_dim=model_dim,
            num_heads=num_heads,
            model_dim=model_dim,
            head_dim=head_dim,
            need_pos_enc=False,
            rngs=rngs,
        )

        self.ln = nnx.LayerNorm(model_dim, rngs=rngs)

        self.mlp = MLP(model_dim, model_dim, out_dim, rngs=rngs)

        backup = nnx.split_rngs(rngs, splits=num_chanels, only="params")

        # self.temporal_transformer = create_v_model(
        #     rngs,
        #     Transformer,
        #     model_args={
        #         "in_dim": in_dim,
        #         "out_dim": model_dim,
        #         "num_heads": num_heads,
        #         "model_dim": model_dim,
        #         "num_heads": num_heads,
        #         "head_dim": head_dim,
        #         "need_pos_enc": True,
        #     },
        # )

        self.temporal_transformer = Transformer(
            in_dim=in_dim,
            out_dim=out_dim,
            num_heads=num_heads,
            model_dim=model_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            need_pos_enc=True,
        )

        nnx.restore_rngs(backup)

        self.rngs = nnx.Rngs(rngs.params())

    def __call__(self, x):
        B, S, C, D = x.shape

        h = x.transpose(0, 2, 1, 3).reshape(B * C, S, D)

        res = h
        h = self.ln(h)
        h = self.mlp(h)

        out = (h + res).reshape(B, C, S, D).transpose(0, 2, 1, 3)

        return out
