import jax
import jax.numpy as jnp
from flax import nnx


class TimeSeriesPositionalEncoding(nnx.Module):
    max_len: int = 10000

    def __call__(self, x):

        B, S, D = x.shape

        if D % 2 != 0:
            raise ValueError(
                f"d_model должен быть четным для sin/cos encoding, got {D}"
            )

        pos = jnp.arange(S)[:, None]
        div_term = jnp.exp(jnp.arange(0, D, 2) * (-jnp.log(10000.0) / D))
        pe = jnp.zeros((S, D), dtype=x.dtype)
        pe = pe.at[:, 0::2].set(jnp.sin(pos * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(pos * div_term))
        pe = pe[None, :, :]

        return x + pe
