from flax import nnx
import jax
import jax.numpy as jnp
import pickle


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
        self.rngs = nnx.Rngs(rngs())

    def reparameterize(self, rngs, mu, logvar):
        std = jnp.exp(0.5 * logvar)
        eps = jax.random.normal(rngs(), shape=std.shape)
        return mu + eps * std

    def __call__(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(self.rngs, mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar


def loss_fn(model, x):
    recon, mu, logvar = model(x)

    recon_loss = jnp.mean((recon - x) ** 2)
    kl_loss = -0.5 * jnp.mean(1 + logvar - mu**2 - jnp.exp(logvar))

    return recon_loss + kl_loss


@nnx.jit
def train_step(model, optimizer, x):
    loss, grads = nnx.value_and_grad(loss_fn)(model, x)
    optimizer.update(grads)
    return loss


@nnx.jit
def eval_step(model, x):
    return loss_fn(model, x)


def train_model(
    model, optimizer, train_loader, val_loader, epochs=10, save_path="vae.pkl"
):

    best_val_loss = float("inf")

    for epoch in range(epochs):

        # ===== TRAIN =====
        train_loss = 0.0
        train_steps = 0

        for batch in train_loader:
            loss = train_step(model, optimizer, batch)
            train_loss += loss
            train_steps += 1

        train_loss /= train_steps

        # ===== VALIDATION =====
        val_loss = 0.0
        val_steps = 0

        for batch in val_loader:
            loss = eval_step(model, batch)
            val_loss += loss
            val_steps += 1

        val_loss /= val_steps

        print(f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}")

        # ===== SAVE BEST =====
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            state = nnx.state(model)
            with open(save_path, "wb") as f:
                import pickle

                pickle.dump(state, f)

            print(" Saved best model")

    return model


def save_model(model, path):
    state = nnx.state(model)
    with open(path, "wb") as f:
        pickle.dump(state, f)


def load_model(model, path):
    with open(path, "rb") as f:
        state = pickle.load(f)
    nnx.update(model, state)
    return model
