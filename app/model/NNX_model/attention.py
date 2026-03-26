import jax
import jax.numpy as jnp
from flax import nnx


class MultiHeadAttention(nnx.Module):

    def __init__(self, in_dim, out_dim, num_heads, head_dim, *, rngs: nnx.Rngs):
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.to_k = nnx.Linear(in_dim, self.num_heads * self.head_dim, rngs=rngs)
        self.to_q = nnx.Linear(in_dim, self.num_heads * self.head_dim, rngs=rngs)
        self.to_v = nnx.Linear(in_dim, self.num_heads * self.head_dim, rngs=rngs)
        self.out_proj = nnx.Linear(self.num_heads * self.head_dim, out_dim, rngs=rngs)

    def __call__(self, k, q, v):
        B, C, D = v.shape

        k = self.to_k(k)
        q = self.to_q(q)
        v = self.to_v(v)

        def split_heads(t):
            B, C, D = t.shape
            return t.reshape(B, C, self.num_heads, self.head_dim)

        k = split_heads(k)
        q = split_heads(q)
        v = split_heads(v)

        scale = jnp.sqrt(self.head_dim).astype(k.dtype)
        attn_logits = jnp.einsum("bchd,bqhd->bchq", k, q) / scale
        attn = nnx.softmax(attn_logits, axis=-1)

        out = jnp.einsum("bchq,bqhd->bchd", attn, v)

        B, C, H, D = out.shape
        out = out.reshape(B, C, H * D)

        return self.out_proj(out)
