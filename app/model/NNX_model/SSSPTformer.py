import jax
import jax.numpy as jnp
import flax.linen as nn

from STFormer import STFormer, DiffGraphSTFormer, TFormer
from VAE import VAE
import matplotlib.pyplot as plt
import optax
from typing import Any
from flax.training import train_state, checkpoints
from torch import Tensor
import torch
from jax import grad, value_and_grad
from tqdm import tqdm
import numpy as np
from eegdash.dataset import DS006940
from functools import partial
from flax import nnx
from utils import create_v_model


base_key = jax.random.key(42)


# class SSSPformer(nnx.Module):

#     def __init__(
#         self,
#         in_dim,
#         out_dim,
#         model_dim,
#         num_heads,
#         head_dim,
#         vae_latent,
#         num_layers,
#         num_chanels,
#         *,
#         rngs: nnx.Rngs,
#     ):

#         self.model_layers = nnx.Sequential(
#             *[
#                 DiffGraphSTFormer(
#                     in_dim=vae_latent if i == 0 else model_dim,
#                     out_dim=model_dim,
#                     model_dim=model_dim,
#                     num_heads=num_heads,
#                     head_dim=head_dim,
#                     num_chanels=num_chanels,
#                     rngs=rngs,
#                 )
#                 for i in range(num_layers)
#             ]
#         )

#         backup = nnx.split_rngs(rngs, splits=num_chanels, only="params")

#         self.vae = create_v_model(
#             rngs, VAE, model_args={"latent_dim": vae_latent, "input_dim": in_dim}
#         )

#         self.mu_proj = create_v_model(
#             rngs,
#             nnx.Linear,
#             model_args={"in_features": model_dim, "out_features": vae_latent},
#         )

#         self.logvar_prog = create_v_model(
#             rngs,
#             nnx.Linear,
#             model_args={"in_features": model_dim, "out_features": vae_latent},
#         )

#         self.out_proj = create_v_model(
#             rngs,
#             nnx.Linear,
#             model_args={"in_features": vae_latent, "out_features": out_dim},
#         )

#         nnx.restore_rngs(backup)

#         self.rngs = nnx.Rngs(rngs.params())

#     def __call__(self, x):

#         B, S, C, D = x.shape

#         x_ = x.reshape(B * S, C, D)

#         recon, mu, logvar = nnx.vmap(
#             lambda vae, x: vae(x), in_axes=(0, 1), out_axes=1
#         )(self.vae, x_)

#         recon = recon.reshape(B, S, C, D)

#         # mu_ = mu.reshape(B, S, C, -1)
#         # logvar_ = logvar.reshape(B, S, C, -1)

#         # z = mu_ + logvar_ * rngs
#         z = mu.reshape(B, S, C, -1)

#         p = self.model_layers(z).reshape(B * S, C, -1)

#         mu_pred = nnx.vmap(lambda proj, h: proj(h), in_axes=(0, 1), out_axes=1)(
#             self.mu_proj, p
#         )
#         logvar_pred = nnx.vmap(lambda proj, h: proj(h), in_axes=(0, 1), out_axes=1)(
#             self.logvar_prog, p
#         )

#         sigma = jax.nn.softplus(logvar_pred) + 1e-4
#         eps = self.rngs.normal(shape=mu_pred.shape)

#         z_next = mu_pred + sigma * eps

#         out = nnx.vmap(lambda proj, h: proj(h), in_axes=(0, 1), out_axes=1)(
#             self.out_proj, z_next
#         ).reshape(B, S, C, -1)

#         return {
#             "prediction": out,
#             "reconstruction": recon,
#             "mu": mu,
#             "logvar": logvar,
#             "mu_pred": mu_pred,
#             "sigma": sigma,
#         }


def loss_fn(model, x, y):
    out = model(x)

    prediction = out["prediction"]
    reconstruction = out["reconstruction"]

    mu = out["mu"]
    logvar = out["logvar"]

    mu_pred = out["mu_pred"]
    sigma = out["sigma"]

    # -------------------------
    # 1. VAE reconstruction loss
    # -------------------------
    recon_loss = jnp.mean((x - reconstruction) ** 2)

    # -------------------------
    # 2. KL divergence (VAE)
    # -------------------------
    kl = -0.5 * jnp.mean(1 + logvar - mu**2 - jnp.exp(logvar))

    # -------------------------
    # 3. Latent target
    # -------------------------
    # берем target latent (сдвиг по времени)
    z_target = mu[:, 1:]  # teacher forcing

    mu_pred = mu_pred[:, :-1]
    sigma = sigma[:, :-1]

    # -------------------------
    # 4. SDE loss (Gaussian NLL)
    # -------------------------
    sde_loss = jnp.mean(((z_target - mu_pred) ** 2) / (sigma**2) + jnp.log(sigma**2))

    # -------------------------
    # 5. (опционально) prediction loss в x-space
    # -------------------------
    pred_loss = jnp.mean((y[:, -1, :, :] - prediction[:, -1, :, :]) ** 2)

    loss = recon_loss + kl + sde_loss + pred_loss

    return loss


# @partial(jax.jit, static_argnums=(0, 6))
# def train_step(model, params, key, opt_state, x, y, optimizer):
#     loss, grads = jax.value_and_grad(loss_fn)(params, model, x, y, key)
#     updates, opt_state = optimizer.update(grads, opt_state)
#     params = optax.apply_updates(params, updates)
#     return params, opt_state, loss


# @partial(jax.jit, static_argnums=(0,))
# def eval_step(model, params, key, x, y):
#     return loss_fn(params, model, x, y, key)


# @partial(jax.jit, static_argnums=(0,))
# def predict(model, params, key, x):
#     out = model.apply(params, x, key)
#     return out["prediction"]


@nnx.jit
def train_step(model, optimizer, x, y):

    def loss_wrapper(model):
        return loss_fn(model, x, y)

    loss, grads = nnx.value_and_grad(loss_wrapper)(model)

    optimizer.update(model, grads=grads)

    return loss


@nnx.jit
def eval_step(model, x, y):
    return loss_fn(model, x, y)


def train_model(
    model,
    x_train,
    y_train,
    x_test,
    y_test,
    batch_size,
    epochs,
):

    rngs = jax.random.key(0)

    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)

    for epoch in tqdm(range(epochs)):

        # -------------------------
        # TRAIN
        # -------------------------

        train_loader = DataLoader(x_train, y_train, batch_size)

        train_losses = []

        for x, y in train_loader:

            loss = train_step(model, optimizer, x, y)
            train_losses.append(loss)

        train_loss = jnp.mean(jnp.stack(train_losses))

        # -------------------------
        # VALIDATION
        # -------------------------

        test_loader = DataLoader(x_test, y_test, batch_size, shuffle=False)

        val_losses = []

        for x, y in test_loader:

            loss = eval_step(model, x, y)
            val_losses.append(loss)

        val_loss = jnp.mean(jnp.stack(val_losses))

        print(f"Epoch {epoch+1} | train {train_loss:.4f} | val {val_loss:.4f}")

    return model


# def plot_prediction(model, params, x_test, y_test):

#     key = jax.random.key(0)

#     sample = x_test[0:1]

#     pred = predict(model, params, key, sample)

#     pred = np.array(pred[0])
#     truth = np.array(y_test[0])

#     # берём два канала
#     channels = [0, 1]

#     for ch in channels:

#         plt.figure(figsize=(10, 4))

#         plt.plot(truth[:, ch, 0], label="truth")
#         plt.plot(pred[:, ch, 0], label="prediction")

#         plt.title(f"Channel {ch}")
#         plt.legend()

#         plt.show()

import jax
import jax.numpy as jnp
import jax.lax as lax
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Sequence


@jax.jit
def _model_step(model, x_context):
    """Один шаг модели: принимает контекст (B, S, C, D), возвращает (B, 1, C, D)."""
    out = model(x_context)
    next_step = out["prediction"][:, -1:, :, :]
    return next_step


def generate_trajectory(
    model,
    x_init: jnp.ndarray,
    horizon: int,
) -> jnp.ndarray:
    """
    Генерация траектории.

    x_init: (B, S, C, D)
    return: (B, horizon, C, D)
    """

    def step_fn(carry, _):
        current_ctx = carry  # (B, S, C, D)

        next_step = _model_step(model, current_ctx)  # (B, 1, C, D)

        new_ctx = jnp.concatenate(
            [current_ctx[:, 1:, :, :], next_step],
            axis=1,
        )

        return new_ctx, next_step

    _, generated = lax.scan(
        step_fn,
        x_init,
        xs=None,
        length=horizon,
    )

    # generated: (horizon, B, 1, C, D)
    generated = generated.transpose(1, 0, 2, 3, 4)  # (B, horizon, 1, C, D)
    generated = generated.squeeze(axis=2)           # (B, horizon, C, D)

    return generated


def plot_prediction(
    model,
    x_test: jnp.ndarray,
    y_test: jnp.ndarray,
    *,
    sample_idx: int = 0,
    channels: Sequence[int] = (0, 1),
    feature_idx: int = -1,
    horizon: Optional[int] = None,
    figsize: tuple[int, int] = (12, 6),
):
    """
    Визуализация предсказания для одного элемента батча.
    """

    # сохраняем batch dimension (=1)
    x_sample = x_test[sample_idx : sample_idx + 1]
    y_sample = y_test[sample_idx : sample_idx + 1]

    if horizon is None:
        horizon = y_sample.shape[1]

    pred_future = generate_trajectory(model, x_sample, horizon=horizon)

    # → numpy для matplotlib
    x_vals = np.array(x_sample[0, :, channels, feature_idx])
    y_vals = np.array(y_sample[0, :, channels, feature_idx])
    pred_vals = np.array(pred_future[0, :, channels, feature_idx])

    t_context = np.arange(x_vals.shape[0])
    t_future = np.arange(x_vals.shape[0], x_vals.shape[0] + pred_vals.shape[0])

    n_channels = len(channels)
    fig, axes = plt.subplots(n_channels, 1, figsize=figsize, sharex=True)

    if n_channels == 1:
        axes = [axes]

    for i, (ch_idx, ax) in enumerate(zip(channels, axes)):
        ax.plot(
            t_context,
            x_vals[:, i],
            label="Input (context)",
            linewidth=1,
        )

        ax.plot(
            np.arange(x_vals.shape[0], x_vals.shape[0] + y_vals.shape[0]),
            y_vals[:, i],
            label="Ground Truth",
            linewidth=2,
            alpha=0.7,
        )

        ax.plot(
            t_future,
            pred_vals[:, i],
            label="Prediction",
            linestyle="--",
            linewidth=2,
        )

        ax.axvline(
            x=x_vals.shape[0] - 0.5,
            linestyle=":",
            alpha=0.5,
            label="Forecast start",
        )

        ax.set_title(f"Channel {ch_idx} (feature {feature_idx})")
        ax.set_ylabel("Amplitude")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time step")
    plt.tight_layout()
    plt.show()

    # MSE (если есть пересечение)
    if horizon <= y_sample.shape[1]:
        y_truth_overlap = y_sample[0, :horizon, channels, feature_idx]
        mse = np.mean((pred_vals - np.array(y_truth_overlap)) ** 2)
        print(f"MSE на горизонте {horizon}: {mse:.6f}")


def takens_embedding_multichannel(data, embedding_dim=10, delay=5):

    T, C = data.shape

    n_windows = T - (embedding_dim - 1) * delay

    start_indices = jnp.arange(n_windows)[:, None]

    offsets = jnp.arange(embedding_dim)[None, :] * delay

    time_indices = start_indices + offsets

    embedded = data[time_indices]

    return embedded.transpose(0, 2, 1)


def create_dataset(embedded, window=20, horizon=1):
    T = embedded.shape[0]

    n_samples = T - window - horizon + 1

    x = jnp.stack([embedded[i : i + window] for i in range(n_samples)])

    y = jnp.stack(
        [embedded[i + horizon : i + horizon + window] for i in range(n_samples)]
    )

    return x, y


class DataLoader:
    def __init__(self, X, y, batch_size=32, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = X.shape[0]

    def __iter__(self):
        indices = np.arange(self.num_samples)

        if self.shuffle:
            np.random.shuffle(indices)

        for start in range(0, self.num_samples, self.batch_size):
            batch_idx = indices[start : start + self.batch_size]
            yield self.X[batch_idx], self.y[batch_idx]


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

    # -------------------------
    # NORMALIZATION
    # -------------------------
    split_norm = int(len(data) * 0.8)

    mean = data[:split_norm].mean(axis=0)
    std = data[:split_norm].std(axis=0) + 1e-8

    data = (data - mean) / std

    # -------------------------
    # EMBEDDING
    # -------------------------
    embedding_dim = 8
    delay = 3

    X_embedded = takens_embedding_multichannel(
        data, embedding_dim=embedding_dim, delay=delay
    )

    print("Takens shape:", X_embedded.shape)

    # -------------------------
    # DATASET
    # -------------------------
    window_size = 20
    horizon = 5

    X, y = create_dataset(X_embedded, window=window_size, horizon=horizon)

    print("Dataset:", X.shape, y.shape)

    split_norm = int(len(X) * 0.8)

    X_train = X[:split_norm]
    y_train = y[:split_norm]

    X_test = X[split_norm:]
    y_test = y[split_norm:]

    print(f"Train: {X_train.shape}, {y_train.shape}")
    print(f"Test: {X_test.shape}, {y_test.shape}")

    # -------------------------
    # MODEL (NNX)
    # -------------------------

    model = STFormer(
        in_dim=8,
        out_dim=8,
        model_dim=32,
        num_heads=5,
        head_dim=8,
        vae_latent=4,
        num_layers=2,
        num_chanels=64,
        rngs=nnx.Rngs(params=0),
    )

    batch_size = 32

    trained_model = train_model(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        batch_size,
        epochs=40,
    )
    plot_prediction(model, X_test, y_test)
