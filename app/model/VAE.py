import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state


# -------------------------
# Encoder
# -------------------------
class Encoder(nn.Module):
    latent_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.latent_dim)(x)
        x = nn.relu(x)

        mu = nn.Dense(self.latent_dim)(x)
        logvar = nn.Dense(self.latent_dim)(x)

        return mu, logvar


# -------------------------
# Decoder
# -------------------------
class Decoder(nn.Module):
    latent_dim: int
    output_dim: int

    @nn.compact
    def __call__(self, z):
        z = nn.Dense(self.latent_dim)(z)
        z = nn.relu(z)

        z = nn.Dense(self.output_dim)(z)

        return z


# -------------------------
# VAE
# -------------------------
class VAE(nn.Module):
    latent_dim: int
    input_dim: int

    def setup(self):
        self.encoder = Encoder(self.latent_dim)
        self.decoder = Decoder(self.input_dim)

    def reparameterize(self, key, mu, logvar):
        std = jnp.exp(0.5 * logvar)
        eps = jax.random.normal(key, std.shape)
        return mu + eps * std

    def __call__(self, x, key):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(key, mu, logvar)
        recon = self.decoder(z)

        return recon, mu, logvar


def vae_loss(recon_x, x, mu, logvar):
    recon_loss = jnp.sum((x - recon_x) ** 2)

    kl = -0.5 * jnp.sum(1 + logvar - mu**2 - jnp.exp(logvar))

    return recon_loss + kl


@jax.jit
def train_step(state, batch, key):

    def loss_fn(params):
        recon, mu, logvar = state.apply_fn({"params": params}, batch, key)
        loss = vae_loss(recon, batch, mu, logvar)
        return loss

    grads = jax.grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)

    return state
