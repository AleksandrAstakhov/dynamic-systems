import jax
import jax.numpy as jnp
import flax.linen as nn
from attention import MultiHeadAttention
from positional_encoding import TimeSeriesPositionalEncoding


class MLP(nn.Module):
    d_model: int
    hidden_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.d_model)(x)
        return x


class Transformer(nn.Module):
    num_heads: int
    head_dim: int
    mlp_dim: int
    need_pos_enc: bool = True

    @nn.compact
    def __call__(self, x):
        model_dim = self.num_heads * self.head_dim

        if self.need_pos_enc:
            x = TimeSeriesPositionalEncoding(model_dim)(x)

        h = nn.LayerNorm()(x)
        h = MultiHeadAttention(self.head_dim, self.num_heads)(h, h, h)

        x = x + h

        h = nn.LayerNorm()(x)
        h = MultiHeadAttention(self.head_dim, self.num_heads)(h, h, h)

        x = x + h

        h = nn.LayerNorm()(x)
        h = MLP(model_dim, self.mlp_dim)(h)

        x = x + h

        return x


class STFormer(nn.Module):
    num_heads: int
    head_dim: int
    mlp_dim: int

    @nn.compact
    def __call__(self, x):
        B, S, C, D = x.shape

        def create_temporal_transformer():
            return Transformer(
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                mlp_dim=self.mlp_dim,
                need_pos_enc=True
            )
        
        temporal_vmap = nn.vmap(
            create_temporal_transformer,
            variable_axes={'params': 0},
            split_rngs={'params': True},
            in_axes=2,
            out_axes=2
        )
        
        h = temporal_vmap(x)

        self.spatial_transformer = Transformer(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            mlp_dim=self.mlp_dim,
            need_pos_enc=False,
        )

        s_list = []
        for i in range(S):
            sp = h[:, i, :, :]
            s_list.append(self.spatial_transformer(sp))
        h = jnp.stack(s_list, axis=1)

        return h
