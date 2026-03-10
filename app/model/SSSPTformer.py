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
from functools import partial

base_key = jax.random.key(42)


class SSSPformer(nn.Module):
    num_heads: int
    head_dim: int
    mlp_dim: int
    vae_latent: int
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
            in_axes=(1, 0),
            out_axes=1,
            axis_size=C,
        )(latent_dim=self.vae_latent, input_dim=D)(x_, jax.random.split(key, C))

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


def loss_fn(params: dict, model: nn.Module, x, y, key):
    out = model.apply(params, x, key)

    prediction = out["prediction"]
    recnstruction = out["recnstruction"]
    mu = out["mu"]
    logvar = out["logvar"]

    recon_loss = jnp.mean((x - recnstruction) ** 2)

    kl = -0.5 * jnp.mean(1 + logvar - mu**2 - jnp.exp(logvar))

    prediction_loss = jnp.mean((y[:, :-5, :, :-1] - prediction[:, :-5, :, :-1]) ** 2)

    return recon_loss + kl + prediction_loss


@partial(jax.jit, static_argnums=(0, 6))
def train_step(model, params, key, opt_state, x, y, optimizer):
    loss, grads = jax.value_and_grad(loss_fn)(params, model, x, y, key)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

@partial(jax.jit, static_argnums=(0,))
def eval_step(model, params, key, x, y):
    return loss_fn(params, model, x, y, key)

@partial(jax.jit, static_argnums=(0,))
def predict(model, params, key, x):
    out = model.apply(params, x, key)
    return out["prediction"]


def train_model(model, x_train, y_train, x_test, y_test, batch_size, dummy_x, epochs):

    global base_key
    base_key, new_key, dataloader_key, step_key, eval_key = jax.random.split(base_key, 5)

    params = model.init(base_key, dummy_x, new_key)

    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(params)

    for i in tqdm(range(epochs)):

        losses = []

        dataloader_key, cur_dataloader_key = jax.random.split(dataloader_key)
        train_loader = create_dataloader(cur_dataloader_key, x_train, y_train, batch_size)

        for x, y in zip(train_loader[0], train_loader[1]):

            step_key, cur_step_key = jax.random.split(step_key)

            params, opt_state, loss = train_step(
                model, params, cur_step_key, opt_state, x, y, optimizer
            )

            losses.append(loss)

        train_loss = jnp.array(losses).mean()

        # ---------- validation ----------

        eval_key, cur_eval_key = jax.random.split(eval_key)

        test_loader = create_dataloader(cur_eval_key, x_test, y_test, batch_size, shuffle=False)

        val_losses = []

        for x, y in zip(test_loader[0], test_loader[1]):

            eval_key, cur_eval_key = jax.random.split(eval_key)

            val_loss = eval_step(model, params, cur_eval_key, x, y)

            val_losses.append(val_loss)

        val_loss = jnp.array(val_losses).mean()

        print(f"Epoch {i+1} | train loss {train_loss:.4f} | val loss {val_loss:.4f}")

    return params

def plot_prediction(model, params, x_test, y_test):

    key = jax.random.key(0)

    sample = x_test[0:1]

    pred = predict(model, params, key, sample)

    pred = np.array(pred[0])
    truth = np.array(y_test[0])

    # берём два канала
    channels = [0, 1]

    for ch in channels:

        plt.figure(figsize=(10,4))

        plt.plot(truth[:, ch, 0], label="truth")
        plt.plot(pred[:, ch, 0], label="prediction")

        plt.title(f"Channel {ch}")
        plt.legend()

        plt.show()

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


def create_dataloader(key, x, y, batch_size, shuffle=True):

    n = x.shape[0]

    if shuffle:
        idx = jax.random.permutation(key, n)
        x = x[idx]
        y = y[idx]

    n_batches = n // batch_size

    x = x[: n_batches * batch_size]
    y = y[: n_batches * batch_size]

    x = x.reshape(n_batches, batch_size, *x.shape[1:])
    y = y.reshape(n_batches, batch_size, *y.shape[1:])

    return (x, y)


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

    split_norm = int(len(X) * 0.8)

    print("Dataset:", X.shape, y.shape)

    X_train = X[:split_norm]
    y_train = y[:split_norm]

    X_test = X[split_norm:]
    y_test = y[split_norm:]

    print(f"Train: {X_train.shape}, {y_train.shape}")
    print(f"Test: {X_test.shape}, {y_test.shape}")

    model = SSSPformer(4, 8, 64, 4, 64, 3)
    batch_size = 32
    params = train_model(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        32,
        jnp.ones(X_train[:batch_size].shape),
        40,
    )
    plot_prediction(model, params, X_test, y_test)
