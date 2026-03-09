import jax
import jax.numpy as jnp
import flax.linen as nn

class TimeSeriesPositionalEncoding(nn.Module):
    max_len: int = 10000

    @nn.compact
    def __call__(self, x):
        B, S, D = x.shape


        pos = jnp.arange(S)[:, None]

        if D % 2 != 0:
            raise ValueError(
                f"d_model должен быть четным для sin/cos encoding, got {D}"
            )

        div_term = jnp.exp(
            jnp.arange(0, D, 2) * (-jnp.log(10000.0) / D)
        )

        pe = jnp.zeros((S, D))

        pe = pe.at[:, 0::2].set(jnp.sin(pos * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(pos * div_term))

        pe = pe[None, :, :]

        learned = self.param(
            "learned_pos",
            nn.initializers.normal(stddev=0.02),
            (1, self.max_len, D),
        )

        pe = pe + learned[:, :S]

        return (x + pe)