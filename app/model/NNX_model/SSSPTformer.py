import jax
import jax.numpy as jnp
import flax.linen as nn

from STFormer import (
    STFormer,
    DiffGraphSTFormer,
    TFormer,
    GConvSTFormer,
    SDESTFormer,
    VectorVAE,
)
from STFormerBlocks import (
    DiffGraphSTFormerBlock,
    LightDiffGraphSTFormerBlock,
    LightSTFormerBlock,
    LightTFormerBlock,
)
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


def vae_train_model(
    x_train,
    y_train,
    x_test,
    y_test,
    batch_size,
    in_dim,
    latent_dim,
    num_channels,
    num_epochs=10,
    lr=1e-3,
    seed=0,
):
    train_loader = DataLoader(x_train, y_train, batch_size)
    val_loader = DataLoader(x_test, y_test, batch_size, shuffle=False)

    rngs = nnx.Rngs(seed)

    model = VectorVAE(
        in_dim=in_dim,
        vae_latent=latent_dim,
        num_chanels=num_channels,
        rngs=rngs,
    )

    optimizer = nnx.Optimizer(
        model,
        optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(lr),
        ),
        wrt=nnx.Param,
    )

    def vae_loss_fn(model, x):
        recon, mu, logvar, _ = model(x)

        recon_loss = jnp.mean((x - recon) ** 2)

        kl_loss = -0.5 * jnp.mean(1 + logvar - mu**2 - jnp.exp(logvar))

        loss = recon_loss + kl_loss
        return loss, (recon_loss, kl_loss)

    @nnx.jit
    def train_step(model, optimizer, x):

        def loss_fn(model):
            return vae_loss_fn(model, x)

        (loss, (recon_loss, kl_loss)), grads = nnx.value_and_grad(
            loss_fn, has_aux=True
        )(model)

        optimizer.update(model, grads)

        return loss, recon_loss, kl_loss

    @nnx.jit
    def eval_step(model, x):
        loss, (recon_loss, kl_loss) = vae_loss_fn(model, x)
        return loss, recon_loss, kl_loss

    for epoch in range(num_epochs):

        # ===== TRAIN =====
        train_loss = 0.0
        train_recon = 0.0
        train_kl = 0.0
        n_train = 0

        for x, y in train_loader:

            loss, recon_loss, kl_loss = train_step(model, optimizer, x)

            train_loss += loss
            train_recon += recon_loss
            train_kl += kl_loss
            n_train += 1

        # ===== VALIDATION =====
        val_loss = 0.0
        val_recon = 0.0
        val_kl = 0.0
        n_val = 0

        for x, y in val_loader:

            loss, recon_loss, kl_loss = eval_step(model, x)

            val_loss += loss
            val_recon += recon_loss
            val_kl += kl_loss
            n_val += 1

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss / n_train:.4f}, "
            f"val_loss={val_loss / n_val:.4f} | "
            f"train_recon={train_recon / n_train:.4f}, "
            f"val_recon={val_recon / n_val:.4f} | "
            f"train_kl={train_kl / n_train:.4f}, "
            f"val_kl={val_kl / n_val:.4f}"
        )

    return model


def loss_fn(model, x, y, horizon=1):
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
    # z_target = mu[:, 1:]  # teacher forcing

    # mu_pred = mu_pred[:, :-1]
    # sigma = sigma[:, :-1]

    # -------------------------
    # 4. SDE loss (Gaussian NLL)
    # -------------------------
    # B, S, C, D = y.shape
    # sde_loss = jnp.mean(((y.reshape(B * S, C, -1) - mu_pred) ** 2) / (sigma**2) + jnp.log(sigma**2))

    # -------------------------
    # 5. (опционально) prediction loss в x-space
    # -------------------------
    pred_loss = jnp.mean((y[:, -horizon:, ...] - prediction[:, -horizon:, ...]) ** 2)

    loss = recon_loss + pred_loss + kl

    return loss


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

    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)

    train_history = []
    val_history = []

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

        train_history.append(train_loss)
        val_history.append(val_loss)

        print(f"Epoch {epoch+1} | train {train_loss:.4f} | val {val_loss:.4f}")

    return model, train_history, val_history


@nnx.jit
def eval_step(model, x, y, horizon=20):

    def rollout(x_init):
        x_curr = x_init
        preds = []

        for _ in range(horizon):
            out = model(x_curr)

            pred = out["prediction"][:, -1:]  # (B, 1, C, D)

            preds.append(pred)

            x_curr = jnp.concatenate([x_curr[:, 1:], pred], axis=1)

        return jnp.concatenate(preds, axis=1)  # (B, horizon, C, D)

    preds = rollout(x)

    # сравниваем с реальным будущим
    y_target = y[:, :horizon]

    mse = jnp.mean((preds - y_target) ** 2)

    return mse


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


def plot_losses(train_history, val_history, save_path="losses.png"):

    train_history = np.array(train_history)
    val_history = np.array(val_history)

    plt.figure()
    plt.plot(train_history, label="train")
    plt.plot(val_history, label="val (rollout)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()

    plt.savefig(save_path)
    plt.close()


def mian(mode_cls, batch_size):

    try:
        dataset = DS006940(cache_dir="./data")
        raw = dataset.datasets[0].raw
        raw.load_data()
        raw.filter(1, 40)

        data = raw.get_data().T[:3000]

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
    embedding_dim = 30
    delay = 1

    X_embedded = takens_embedding_multichannel(
        data, embedding_dim=embedding_dim, delay=delay
    )

    print("Takens shape:", X_embedded.shape)

    # -------------------------
    # DATASET
    # -------------------------
    window_size = 50
    horizon = 1

    X_embedded_train = X_embedded[:split_norm]

    X_train, y_train = create_dataset(
        X_embedded_train, window=window_size, horizon=horizon
    )

    X_embedded_test = X_embedded[split_norm:]

    X_test, y_test = create_dataset(
        X_embedded_test, window=window_size, horizon=horizon
    )

    corr = np.corrcoef(data, rowvar=False)

    threshold = 0.7
    adj = np.where(np.abs(corr) >= threshold, 1, 0)

    np.fill_diagonal(adj, 0)

    senders, receivers = np.nonzero(adj)

    edge_index = np.stack([senders, receivers], axis=0)

    print(f"Train: {X_train.shape}, {y_train.shape}")
    print(f"Test: {X_test.shape}, {y_test.shape}")

    # -------------------------
    # MODEL (NNX)
    # -------------------------

    vae = vae_train_model(
        x_train=X_train,
        y_train=y_train,
        x_test=X_test,
        y_test=y_test,
        batch_size=10,
        in_dim=30,
        latent_dim=15,
        num_channels=64,
        num_epochs=200,
    )

    model = mode_cls(
        in_dim=15,
        out_dim=15,
        model_dim=15,
        num_heads=8,
        head_dim=4,
        vae_latent=15,
        num_layers=2,
        num_chanels=64,
        edge_index=edge_index,
        rngs=nnx.Rngs(0),
    )

    trained_model, tl, vl = train_model(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        batch_size,
        epochs=40,
    )
    plot_losses(tl, vl)


if __name__ == "__main__":

    for cls, bs in [
        (TFormer, 32),
        (STFormer, 32),
        (GConvSTFormer, 32),
        (DiffGraphSTFormer, 12),
    ]:

        mian(cls, bs)
