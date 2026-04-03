import jax
import jax.numpy as jnp
import flax.linen as nn

from STFormer_ import VectorVAE
from STFormerBlocks import (
    DiffGraphSTFormerBlock,
    LightDiffGraphSTFormerBlock,
    LightSTFormerBlock,
    LightTFormerBlock,
    Transformer,
)

from STFormer_final import STFormer, STFormerBlock

from VAE import VAE
import matplotlib.pyplot as plt
import optax
from typing import Any
from flax.training import train_state, checkpoints
from jax import grad, value_and_grad
from tqdm import tqdm
import numpy as np
from eegdash.dataset import DS006940
from functools import partial
from flax import nnx
from utils import (
    create_v_model,
    DataLoader,
    FlowDataloader,
    takens_embedding_multichannel,
    create_dataset,
)


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

        recon_loss = jnp.mean((x[:, -1, ...] - recon[:, -1, ...]) ** 2)

        kl_loss = -0.5 * jnp.mean(
            1 + logvar[:, -1, ...] - mu[:, -1, ...] ** 2 - jnp.exp(logvar[:, -1, ...])
        )

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


def train_stmodel(model, X_train, y_train, X_test, y_test, encoder, num_epochs):

    def collect_latents(encoder, dataloader):
        mu_list = []
        logvar_list = []

        for x in dataloader:

            mu, logvar = nnx.vmap(
                lambda model, h: model(h), in_axes=(0, 2), out_axes=(2, 2)
            )(encoder, x)
            mu_list.append(mu[:, -1, ...])
            logvar_list.append(logvar[:, -1, ...])

        mu = jnp.concatenate(mu_list, axis=0)
        logvar = jnp.concatenate(logvar_list, axis=0)

        return mu, logvar

    mu_x_train, logvar_x_train = collect_latents(
        encoder=encoder, dataloader=DataLoader(X_train, shuffle=False)
    )
    mu_y_train, logvar_y_train = collect_latents(
        encoder=encoder, dataloader=DataLoader(y_train, shuffle=False)
    )
    mu_x_test, logvar_x_test = collect_latents(
        encoder=encoder, dataloader=DataLoader(X_test, shuffle=False)
    )
    mu_y_test, logvar_y_test = collect_latents(
        encoder=encoder, dataloader=DataLoader(y_test, shuffle=False)
    )

    train_loader = FlowDataloader(
        mu_x_train,
        logvar_x_train,
        mu_y_train,
        logvar_y_train,
        batch_size=32,
        dt=1 / 250,
        rngs=nnx.Rngs(21231),
    )

    val_loader = FlowDataloader(
        mu_x_test,
        logvar_x_test,
        mu_y_test,
        logvar_y_test,
        batch_size=32,
        dt=1 / 250,
        rngs=nnx.Rngs(21231),
        shuffle=False,
    )

    def flow_loss_fn(model, z, dz):

        dz_cuppa = model(z)

        return jnp.mean((dz - dz_cuppa[:, -1, ...]) ** 2)

    @nnx.jit
    def train_step(
        model: nnx.Module, optimizer: nnx.Optimizer, z: jnp.ndarray, dz: jnp.ndarray
    ) -> jnp.ndarray:

        loss, grads = nnx.value_and_grad(flow_loss_fn)(model, z, dz)
        optimizer.update(model, grads)
        return loss

    @nnx.jit
    def eval_step(model: nnx.Module, z: jnp.ndarray, dz: jnp.ndarray) -> jnp.ndarray:
        dz_cuppa = model(z)
        loss = jnp.mean((dz[:, -1, ...] - dz_cuppa[:, -1, ...]) ** 2)
        return loss

    optimizer = nnx.Optimizer(
        model,
        optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(1e-3),
        ),
        wrt=nnx.Param,
    )

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'='*50}")

        # ========== ОБУЧЕНИЕ ==========
        train_loss = 0.0
        num_train_batches = 0

        for batch in tqdm(train_loader, desc=f"Train Epoch {epoch}"):
            z, dz = batch  # распаковка батча
            loss = train_step(model, optimizer, z, dz)
            train_loss += loss
            num_train_batches += 1

        avg_train_loss = train_loss / num_train_batches
        history["train_loss"].append(float(avg_train_loss))

        # ========== ВАЛИДАЦИЯ ==========
        val_loss = 0.0
        num_val_batches = 0

        for batch in tqdm(val_loader, desc=f"Val Epoch {epoch}"):
            z, dz = batch
            loss = eval_step(model, z, dz)
            val_loss += loss
            num_val_batches += 1

        avg_val_loss = val_loss / num_val_batches
        history["val_loss"].append(float(avg_val_loss))

        # ========== ЛОГИРОВАНИЕ ==========
        print(f"\n📊 Results for epoch {epoch}:")
        print(f"   Train Loss: {avg_train_loss:.6f}")
        print(f"   Val Loss:   {avg_val_loss:.6f}")

        # ========== СОХРАНЕНИЕ ЛУЧШЕЙ МОДЕЛИ ==========
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f" Best model saved! (val_loss: {best_val_loss:.6f})")

    print(f"\n{'='*50}")
    print(f"Training completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"{'='*50}")

    return history


def main():

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

    in_dim = embedding_dim

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

    in_dim = embedding_dim
    latent_dim = embedding_dim // 2 + 1
    num_chanels = 64

    vae = vae_train_model(
        x_train=X_train,
        y_train=y_train,
        x_test=X_test,
        y_test=y_test,
        batch_size=32,
        in_dim=in_dim,
        latent_dim=latent_dim,
        num_channels=num_chanels,
        num_epochs=100,
    )

    st_model = STFormer(
        in_dim=latent_dim,
        latent_dim=latent_dim,
        out_dim=latent_dim,
        temporal_model_dim=latent_dim,
        spatial_model_dim=latent_dim,
        num_blocks=2,
        temporal_num_heads=4,
        temporal_head_dim=4,
        spatial_head_dim=4,
        spatial_num_heads=4,
        num_chanels=64,
        spatial_model_cls=Transformer,
    )

    t_model = STFormer(
        in_dim=latent_dim,
        latent_dim=latent_dim,
        out_dim=latent_dim,
        temporal_model_dim=latent_dim,
        spatial_model_dim=latent_dim,
        num_blocks=2,
        temporal_num_heads=4,
        temporal_head_dim=4,
        spatial_head_dim=4,
        spatial_num_heads=4,
        num_chanels=64,
        spatial_model_cls=None,
    )

    train_stmodel(
        t_model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        encoder=vae.vae.encoder,
        num_epochs=40,
    )

    train_stmodel(
        st_model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        encoder=vae.vae.encoder,
        num_epochs=40,
    )


if __name__ == "__main__":
    main()
