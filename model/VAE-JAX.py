import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
from typing import Sequence


class VAE(nn.Module):
    input_dim: int
    latent_dim: int
    hidden_dim: int = 256

    def setup(self):
        # Encoder
        self.fc1 = nn.Dense(self.hidden_dim)
        self.fc_mu = nn.Dense(self.latent_dim)
        self.fc_logvar = nn.Dense(self.latent_dim)

        # Decoder
        self.fc2 = nn.Dense(self.hidden_dim)
        self.fc_out = nn.Dense(self.input_dim)

    def encode(self, x):
        h = nn.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, key, mu, logvar):
        std = jnp.exp(0.5 * logvar)
        eps = jax.random.normal(key, mu.shape)
        return mu + eps * std

    def decode(self, z):
        h = nn.relu(self.fc2(z))
        return self.fc_out(h)

    def __call__(self, x, key):
        mu, logvar = self.encode(x)
        z = self.reparameterize(key, mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar