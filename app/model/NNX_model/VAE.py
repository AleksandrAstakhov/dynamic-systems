from flax import nnx
import jax
import jax.numpy as jnp


class Encoder(nnx.Module):
    def __init__(self, input_dim, latent_dim, *, rngs: nnx.Rngs):
        self.dense1 = nnx.Linear(input_dim, latent_dim, rngs=rngs)
        self.dense2 = nnx.Linear(latent_dim, latent_dim, rngs=rngs)

        self.mu_proj = nnx.Linear(latent_dim, latent_dim, rngs=rngs)
        self.logvar_proj = nnx.Linear(latent_dim, latent_dim, rngs=rngs)

    def __call__(self, x):
        x = nnx.gelu(self.dense1(x))
        x = nnx.gelu(self.dense2(x))

        mu = self.mu_proj(x)
        logvar = self.logvar_proj(x)

        return mu, logvar


class Decoder(nnx.Module):
    def __init__(self, latent_dim, out_dim, *, rngs: nnx.Rngs):
        self.dense1 = nnx.Linear(latent_dim, latent_dim, rngs=rngs)
        self.out = nnx.Linear(latent_dim, out_dim, rngs=rngs)

    def __call__(self, z):
        z = nnx.relu(self.dense1(z))
        return self.out(z)


class VAE(nnx.Module):
    def __init__(self, latent_dim, input_dim, *, rngs: nnx.Rngs):

        self.encoder = Encoder(input_dim, latent_dim, rngs=rngs)
        self.decoder = Decoder(latent_dim, input_dim, rngs=rngs)
        self.rngs = nnx.Rngs(rngs.params())

    def reparameterize(self, rngs, mu, logvar):
        std = jnp.exp(0.5 * logvar)
        eps = jax.random.normal(rngs(), shape=std.shape)
        return mu + eps * std

    def __call__(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(self.rngs, mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar
