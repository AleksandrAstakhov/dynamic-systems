import jax
import jax.numpy as jnp
import flax.linen as nn

class TimeSeriesPositionalEncoding(nn.Module):
    d_model: int
    max_len: int = 10000

    @nn.compact
    def __call__(self, x):
        B, S, C, D = x.shape

        x_reshaped = x.transpose(0, 2, 1, 3).reshape(B*C, S, D)

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

        return (x_reshaped + pe).reshape(B, C, S, D).transpose(0, 2, 1, 3)