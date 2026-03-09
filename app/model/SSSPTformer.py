import jax
import jax.numpy as jnp
import flax.linen as nn

from STFormer import STFormer
from VAE import VAE, vae_loss
import matplotlib.pyplot as plt
import optax
from typing import Any
from flax.training import train_state, checkpoints
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor
import torch
from jax import grad, value_and_grad
from tqdm import tqdm
import numpy as np
from eegdash.dataset import DS006940

base_key = jax.random.key(42)


class SSSPformer(nn.Module):
    num_heads: int
    head_dim: int
    mlp_dim: int
    latent_dim: int
    num_layers: int

    @nn.compact
    def __call__(self, x, key):

        B, S, C, D = x.shape

        x_ = x.reshape(B * S, C, D)

        recon, mu, logvar = nn.vmap(
            VAE,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=1,
            out_axes=1,
            axis_size=C,
        )(latent_dim=self.latent_dim, input_dim=D)(x_, jax.random.split(key, C))

        recon_reshaped = recon.reshape(B, S, C, -1)

        z = mu.reshape(B, S, C, -1)

        layers = []
        for _ in jnp.arange(self.num_layers):
            layers.append(
                STFormer(
                    num_heads=self.num_heads,
                    head_dim=self.head_dim,
                    mlp_dim=self.mlp_dim,
                )
            )

        z = nn.Sequential(layers)(z)

        final_output = nn.Dense(D)(z)

        return {
            "prediction": final_output,
            "recnstruction": recon_reshaped,
            "mu": mu,
            "logvar": logvar,
        }


def loss_fn(params: dict, model: nn.Module, x: Tensor, y: Tensor):
    base_key, new_key = jax.random.split(base_key)
    out = model.apply(params, x, new_key)

    prediction = out["prediction"]
    recnstruction = out["recnstruction"]
    mu = out["mu"]
    logvar = out["logvar"]

    recon_loss = jnp.mean((x - recnstruction) ** 2)

    kl = -0.5 * jnp.mean(1 + logvar - mu**2 - jnp.exp(logvar))

    prediction_loss = jnp.mean((y[:, :-5, :, :-1] - prediction[:, :-5, :, :-1]) ** 2)

    return recon_loss + kl + prediction_loss


def train_step(params, opt_state, x, y, optimizer):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


def train_model(model: nn.Module, dataloader: DataLoader, dummy_x: jnp.Array, epochs):
    key = jax.random.key(0)
    params = model.init(key, dummy_x)
    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(params)

    for i in tqdm(epochs):
        losses = []
        for x, y in dataloader:
            params, opt_state, loss = train_step(params, opt_state, x, y, optimizer)
            losses.append(loss)
        print(f"Epoch {i + 1}, Loss: ", jnp.array(losses).mean())


def takens_embedding_multichannel(data, embedding_dim=10, delay=5):

    T, C = data.shape

    n_windows = T - (embedding_dim - 1) * delay

    start_indices = jnp.arange(n_windows)[:, None]

    offsets = jnp.arange(embedding_dim)[None, :] * delay

    time_indices = start_indices + offsets

    embedded = data[time_indices]

    return embedded.transpose(0, 2, 1)


def create_dataset(embedded, window=20, horizon=5):
    T = embedded.shape[0]

    n_samples = T - window - horizon + 1

    x = jnp.stack([embedded[i : i + window] for i in range(n_samples)])

    y = jnp.stack(
        [embedded[i + horizon : i + horizon + window] for i in range(n_samples)]
    )

    return x, y


def dataloader(x, y, batch_size, shuffle=True):

    n = x.shape[0]
    idx = np.arange(n)

    if shuffle:
        np.random.shuffle(idx)

    for i in range(0, n, batch_size):
        batch_idx = idx[i : i + batch_size]
        yield x[batch_idx], y[batch_idx]


if __name__ == "__main__":

    try:

        dataset = DS006940(cache_dir="./data")
        raw = dataset.datasets[0].raw
        raw.load_data()
        raw.filter(1, 40)

        data = raw.get_data().T[:2000]

    except ImportError:
        print("DS006940 не найден, используем синтетические данные")

    print("Raw EEG:", data.shape)

    split_norm = int(len(data) * 0.8)

    mean = data[:split_norm].mean(axis=0)
    std = data[:split_norm].std(axis=0) + 1e-8

    data = (data - mean) / std

    embedding_dim = 8
    delay = 3

    X_embedded = takens_embedding_multichannel(
        data, embedding_dim=embedding_dim, delay=delay
    )

    print("Takens shape:", X_embedded.shape)

    window_size = 20
    horizon = 5

    X, y = create_dataset(X_embedded, window=window_size, horizon=horizon)

    split_norm = len(X)

    print("Dataset:", X.shape, y.shape)

    X_train = X[:split_norm]
    y_train = y[:split_norm]

    X_test = X[split_norm:]
    y_test = y[split_norm:]

    print(f"Train: {X_train.shape}, {y_train.shape}")
    print(f"Test: {X_test.shape}, {y_test.shape}")

    model = SSSPformer(4, 8, 64, 64, 3)
    batch_size = 32
    train_model(
        model,
        dataloader(X_train, y_train, 32),
        jnp.ones(X_train[:batch_size].shape),
        15,
    )
