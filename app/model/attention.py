import jax
import jax.numpy as jnp
import flax.linen as nn


class MultiHeadAttention(nn.Module):
    num_heads: int
    head_dim: int

    @nn.compact
    def __call__(self, k, q, v):

        model_dim = v.shape[-1]

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

        out = jnp.einsum("bchq,bqhd->bchd", attn, v)
        B, C, H, D = out.shape

        return nn.Dense(model_dim)(out.reshape(B, C, H * D))
