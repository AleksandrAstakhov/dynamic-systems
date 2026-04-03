import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx import bridge


from positional_encoding import TimeSeriesPositionalEncoding
from graph_modules import Grand, GrandDiffuser
from jraphx.nn.conv import TransformerConv

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
        self.rngs = nnx.Rngs(rngs())

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

        self.ln = nnx.LayerNorm(model_dim, rngs=rngs)

        self.mlp = MLP(model_dim, model_dim, out_dim, rngs=rngs)

        backup = nnx.split_rngs(rngs, splits=num_chanels)

        self.temporal_transformer = create_v_model(
            rngs,
            Transformer,
            model_args={
                "in_dim": in_dim,
                "out_dim": model_dim,
                "num_heads": num_heads,
                "model_dim": model_dim,
                "head_dim": head_dim,
                "need_pos_enc": True,
            },
        )

        nnx.restore_rngs(backup)

        self.rngs = nnx.Rngs(rngs())

    def __call__(self, x):

        B, S, C, D = x.shape

        h = nnx.vmap(lambda model, x: model(x), in_axes=(0, 2), out_axes=2)(
            self.temporal_transformer, x
        )

        h = h.reshape(B * S, C, -1)

        h = self.spatial_transformer(h).reshape(B, S, C, -1)

        res = h
        h = self.ln(h)
        h = self.mlp(h)

        return h


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

        backup = nnx.split_rngs(rngs, splits=num_chanels)

        self.temporal_transformer = create_v_model(
            rngs,
            Transformer,
            model_args={
                "in_dim": in_dim,
                "out_dim": model_dim,
                "model_dim": model_dim,
                "num_heads": num_heads,
                "head_dim": head_dim,
                "need_pos_enc": True,
            },
        )

        nnx.restore_rngs(backup)

        self.rngs = nnx.Rngs(rngs())

    def __call__(self, x: jax.Array) -> jax.Array:
        B, S, S, D = x.shape

        h = nnx.vmap(lambda model, x: model(x), in_axes=(0, 2), out_axes=2)(
            self.temporal_transformer, x
        )

        t_grid = jnp.linspace(0, 0.5, 4)

        h = self.spatial_model(h, t_grid)

        res = h
        h = self.ln(h)
        out = self.mlp(h)

        return out


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

        backup = nnx.split_rngs(rngs, splits=num_chanels)

        self.temporal_transformer = create_v_model(
            rngs,
            Transformer,
            model_args={
                "in_dim": in_dim,
                "out_dim": model_dim,
                "num_heads": num_heads,
                "model_dim": model_dim,
                "head_dim": head_dim,
                "need_pos_enc": True,
            },
        )

        nnx.restore_rngs(backup)

        self.rngs = nnx.Rngs(rngs())

    def __call__(self, x):

        h = nnx.vmap(lambda model, x: model(x), in_axes=(0, 2), out_axes=2)(
            self.temporal_transformer, x
        )

        res = h
        h = self.ln(h)
        h = self.mlp(h)

        return h + res


class LightSTFormerBlock(nnx.Module):

    def __init__(
        self,
        in_dim,
        out_dim,
        model_dim,
        num_heads,
        head_dim,
        num_chanels,
        edge_index,
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

        self.temporal_transformer = Transformer(
            in_dim=in_dim,
            out_dim=model_dim,
            num_heads=num_heads,
            model_dim=model_dim,
            head_dim=head_dim,
            need_pos_enc=True,
            rngs=rngs,
        )

        self.rngs = nnx.Rngs(rngs())

    def __call__(self, x):
        B, S, C, D = x.shape

        x = x.transpose(0, 2, 1, 3).reshape(B * C, S, -1)

        h = (
            self.temporal_transformer(x)
            .reshape(B, C, S, -1)
            .transpose(0, 2, 1, 3)
            .reshape(B * S, C, -1)
        )

        h = self.spatial_transformer(h).reshape(B, S, C, -1)

        res = h
        h = self.ln(h)
        h = self.mlp(h)

        return h


class LightDiffGraphSTFormerBlock(nnx.Module):

    def __init__(
        self,
        in_dim,
        out_dim,
        model_dim,
        num_heads,
        head_dim,
        num_chanels,
        edge_index,
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
            in_dim=in_dim,
            out_dim=model_dim,
            num_heads=num_heads,
            model_dim=model_dim,
            head_dim=head_dim,
            need_pos_enc=True,
            rngs=rngs,
        )

        self.rngs = nnx.Rngs(rngs())

    def __call__(self, x) -> jax.Array:
        B, S, C, D = x.shape

        x = x.transpose(0, 2, 1, 3).reshape(B * C, S, D)

        h = self.temporal_transformer(x).reshape(B, C, S, -1).transpose(0, 2, 1, 3)

        t_grid = jnp.linspace(0, 0.5, 4)

        h = self.spatial_model(h, t_grid)

        res = h
        h = self.ln(h)
        out = self.mlp(h)

        return out


class LightTFormerBlock(nnx.Module):

    def __init__(
        self,
        in_dim,
        out_dim,
        model_dim,
        num_heads,
        head_dim,
        num_chanels,
        edge_index,
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

        self.temporal_transformer = Transformer(
            in_dim=in_dim,
            out_dim=model_dim,
            num_heads=num_heads,
            model_dim=model_dim,
            head_dim=head_dim,
            need_pos_enc=True,
            rngs=rngs,
        )

        self.rngs = nnx.Rngs(rngs())

    def __call__(self, x):
        B, S, C, D = x.shape

        x = x.transpose(0, 2, 1, 3).reshape(B * C, S, D)

        h = self.temporal_transformer(x).reshape(B, C, S, -1).transpose(0, 2, 1, 3)

        res = h
        h = self.ln(h)
        h = self.mlp(h)

        return h


class LightGConvSTFormerBlock(nnx.Module):

    def __init__(
        self,
        in_dim,
        out_dim,
        model_dim,
        num_heads,
        head_dim,
        edge_index,
        *,
        rngs: nnx.Rngs
    ):

        self.spatial_model = TransformerConv(
            in_features=model_dim,
            out_features=model_dim,
            heads=num_heads,
            rngs=rngs,
        )

        self.ln = nnx.LayerNorm(model_dim, rngs=rngs)

        self.mlp = MLP(model_dim, model_dim, out_dim, rngs=rngs)

        self.temporal_transformer = Transformer(
            in_dim=in_dim,
            out_dim=model_dim,
            num_heads=num_heads,
            model_dim=model_dim,
            head_dim=head_dim,
            need_pos_enc=True,
            rngs=rngs,
        )

        self.rngs = nnx.Rngs(rngs())

        self.edge_index = edge_index

    def __batch_edge_index(edge_index: jnp.ndarray, num_nodes: int, batch_size: int):

        E = edge_index.shape[1]

        offsets = jnp.arange(batch_size) * num_nodes

        offsets = offsets[:, None]

        edge_index_expanded = edge_index[None, :, :]

        edge_index_batched = edge_index_expanded + offsets[:, None, :]

        edge_index_batched = edge_index_batched.transpose(1, 0, 2).reshape(2, -1)

        return edge_index_batched

    def __call__(
        self,
        x: jnp.ndarray,
        edge_index: jnp.ndarray,
    ) -> jax.Array:
        B, S, C, D = x.shape

        x = x.transpose(0, 2, 1, 3).reshape(B * C, S, D)

        h = (
            self.temporal_transformer(x)
            .reshape(B, C, S, -1)
            .transpose(0, 2, 1, 3)
            .reshape(B * S * C, -1)
        )

        edge_index = self.__batch_edge_index(self.edge_index, C, B * S)

        h = self.spatial_model(h, edge_index).reshape(B, S, C, -1)

        res = h
        h = self.ln(h)
        out = self.mlp(h)

        return out + res
