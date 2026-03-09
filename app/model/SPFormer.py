import jax
import jax.numpy as jnp
from flax import linen as nn
import numpy as np
from flax.training import train_state
import optax
from typing import Tuple, List
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


class SelfAttention(nn.Module):
    d_model: int
    num_heads: int

    @nn.compact
    def __call__(self, x):
        head_dim = self.d_model // self.num_heads

        qkv = nn.Dense(self.d_model * 3)(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        def split_heads(t):
            B, T, C = t.shape
            return t.reshape(B, T, self.num_heads, head_dim).transpose(0, 2, 1, 3)

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        attn = jnp.einsum("bhqd,bhkd->bhqk", q, k)
        attn = attn / jnp.sqrt(head_dim)

        attn = nn.softmax(attn, axis=-1)

        out = jnp.einsum("bhqk,bhkd->bhqd", attn, v)

        B, H, T, D = out.shape
        out = out.transpose(0, 2, 1, 3).reshape(B, T, H * D)

        return nn.Dense(self.d_model)(out)


class MultiHeadAttention(nn.Module):
    d_model: int
    num_heads: int

    @nn.compact
    def __call__(self, q, k, v):

        head_dim = self.d_model // self.num_heads

        q = nn.Dense(self.d_model)(q)
        k = nn.Dense(self.d_model)(k)
        v = nn.Dense(self.d_model)(v)

        def split_heads(x):
            B, T, C = x.shape
            x = x.reshape(B, T, self.num_heads, head_dim)
            return x.transpose(0, 2, 1, 3)

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        attn = jnp.einsum("bhqd,bhkd->bhqk", q, k)
        attn = attn / jnp.sqrt(head_dim)

        attn = nn.softmax(attn, axis=-1)

        out = jnp.einsum("bhqk,bhkd->bhqd", attn, v)

        B, H, T, D = out.shape
        out = out.transpose(0, 2, 1, 3).reshape(B, T, H * D)

        return nn.Dense(self.d_model)(out)


class JointAttention(nn.Module):
    d_model: int
    num_heads: int

    @nn.compact
    def __call__(self, x, context):

        head_dim = self.d_model // self.num_heads

        q = nn.Dense(self.d_model)(x)

        k_self = nn.Dense(self.d_model)(x)
        v_self = nn.Dense(self.d_model)(x)

        k_cross = nn.Dense(self.d_model)(context)
        v_cross = nn.Dense(self.d_model)(context)

        k = jnp.concatenate([k_self, k_cross], axis=1)
        v = jnp.concatenate([v_self, v_cross], axis=1)

        def split_heads(t):
            B, T, C = t.shape
            t = t.reshape(B, T, self.num_heads, head_dim)
            return t.transpose(0, 2, 1, 3)

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        attn = jnp.einsum("bhqd,bhkd->bhqk", q, k)
        attn = attn / jnp.sqrt(head_dim)

        attn = nn.softmax(attn, axis=-1)

        out = jnp.einsum("bhqk,bhkd->bhqd", attn, v)

        B, H, T, D = out.shape
        out = out.transpose(0, 2, 1, 3).reshape(B, T, H * D)

        return nn.Dense(self.d_model)(out)


class MLP(nn.Module):
    d_model: int
    hidden_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.d_model)(x)
        return x


# class TransformerBlock(nn.Module):
#     d_model: int
#     num_heads: int
#     mlp_dim: int

#     @nn.compact
#     def __call__(self, x):

#         h = nn.LayerNorm()(x)
#         h = SelfAttention(self.d_model, self.num_heads)(h)
#         x = x + h

#         h = nn.LayerNorm()(x)
#         h = MLP(self.d_model, self.mlp_dim)(h)
#         x = x + h

#         return x


class TransformerBlock(nn.Module):
    d_model: int
    num_heads: int
    mlp_dim: int

    @nn.compact
    def __call__(self, x, context):

        # Self Attention
        h = nn.LayerNorm()(x)
        h = MultiHeadAttention(self.d_model, self.num_heads)(h, h, h)

        x = x + h

        # Cross Attention
        h = nn.LayerNorm()(x)

        h = MultiHeadAttention(self.d_model, self.num_heads)(
            h, context, context  # queries  # keys  # values
        )

        x = x + h

        # Feed Forward
        h = nn.LayerNorm()(x)
        h = MLP(self.d_model, self.mlp_dim)(h)

        x = x + h

        return x
    
class MultiSequenceTransformerBlock(nn.Module):
    d_model: int
    num_heads: int
    mlp_dim: int

    @nn.compact
    def __call__(self, x):

        # x = (B, S, T, D)

        B, S, T, D = x.shape

        # ------------------------
        # Project to model dim
        # ------------------------

        x = nn.Dense(self.d_model)(x)

        # ------------------------
        # Self attention внутри каждой последовательности
        # ------------------------

        # Merge batch and sequence dims
        x_reshaped = x.reshape(B*S, T, self.d_model)

        h = nn.LayerNorm()(x_reshaped)

        h = MultiHeadAttention(
            self.d_model,
            self.num_heads
        )(h, h, h)

        x_reshaped = x_reshaped + h

        x = x_reshaped.reshape(B, S, T, self.d_model)

        # ------------------------
        # Cross-sequence attention
        # ------------------------

        # Mix sequences via time concatenation
        x_perm = jnp.transpose(x, (0, 2, 1, 3))  
        # (B, T, S, D)

        x_perm = x_perm.reshape(B*T, S, self.d_model)

        h = nn.LayerNorm()(x_perm)

        h = MultiHeadAttention(
            self.d_model,
            self.num_heads
        )(h, h, h)

        x_perm = x_perm + h

        x_perm = x_perm.reshape(B, T, S, self.d_model)

        x = jnp.transpose(x_perm, (0, 2, 1, 3))

        # ------------------------
        # MLP
        # ------------------------

        x = x + MLP(
            self.d_model,
            self.mlp_dim
        )(nn.LayerNorm()(x))

        return x
    
class TimeSeriesPositionalEncoding(nn.Module):
    d_model: int
    max_len: int = 10000

    @nn.compact
    def __call__(self, x):
        B, S, T, D = x.shape

        x_reshaped = x.reshape(B * S, T, D)

        pos = jnp.arange(T)[:, None]

        # гарантируем четную размерность
        if D % 2 != 0:
            raise ValueError(
                f"d_model должен быть четным для sin/cos encoding, got {D}"
            )

        div_term = jnp.exp(
            jnp.arange(0, D, 2) * (-jnp.log(10000.0) / D)
        )

        pe = jnp.zeros((T, D))

        pe = pe.at[:, 0::2].set(jnp.sin(pos * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(pos * div_term))

        pe = pe[None, :, :]

        learned = self.param(
            "learned_pos",
            nn.initializers.normal(stddev=0.02),
            (1, self.max_len, D),
        )

        pe = pe + learned[:, :T]

        return (x_reshaped + pe).reshape(B, S, T, D)

# ==================== 1. Генерация данных аттрактора Лоренца ====================

def generate_lorenz_data(
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0/3.0,
    dt: float = 0.01,
    t_span: Tuple[float, float] = (0, 100),
    x0: List[float] = [1.0, 1.0, 1.0]
) -> np.ndarray:
    """
    Генерирует временной ряд аттрактора Лоренца
    """
    def lorenz(t, xyz):
        x, y, z = xyz
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        return [dx, dy, dz]
    
    t_eval = np.arange(t_span[0], t_span[1], dt)
    sol = solve_ivp(lorenz, t_span, x0, t_eval=t_eval, method='RK45')
    
    return sol.y.T  # shape: (n_timesteps, 3)

# ==================== 2. Вложение Такенса ====================

def takens_embedding(
    data: np.ndarray,
    embedding_dim: int = 10,
    delay: int = 5
) -> np.ndarray:
    """
    Создает вложение Такенса для временного ряда
    
    Args:
        data: временной ряд (n_timesteps, n_features)
        embedding_dim: размерность вложения
        delay: задержка между отсчетами
    
    Returns:
        embedded_data: (n_windows, embedding_dim * n_features)
    """
    n_timesteps, n_features = data.shape
    n_windows = n_timesteps - (embedding_dim - 1) * delay
    
    if n_windows <= 0:
        raise ValueError("Слишком большая размерность вложения для данного ряда")
    
    embedded = []
    for i in range(n_windows):
        window = []
        for d in range(embedding_dim):
            idx = i + d * delay
            window.append(data[idx])
        embedded.append(np.concatenate(window))
    
    return np.array(embedded)

def create_windows(
    data: np.ndarray,
    window_size: int = 50,
    pred_len: int = 10,
    stride: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Создает окна для обучения (вход -> предсказание)
    
    Args:
        data: данные (n_samples, features)
        window_size: длина входного окна
        pred_len: длина предсказания
        stride: шаг скольжения
    
    Returns:
        X: входные окна (n_windows, window_size, features)
        y: целевые окна (n_windows, pred_len, features)
    """
    n_samples, n_features = data.shape
    n_windows = (n_samples - window_size - pred_len + 1) // stride
    
    X, y = [], []
    for i in range(0, n_windows * stride, stride):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size:i+window_size+pred_len])
    
    return np.array(X), np.array(y)

# ==================== Dataset ====================

import jax.random as jr


class TimeSeriesDataset:
    def __init__(self, X, y, batch_size=32, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        n = len(self.X)
        idx = np.arange(n)

        if self.shuffle:
            np.random.shuffle(idx)

        for i in range(0, n, self.batch_size):
            batch_idx = idx[i:i+self.batch_size]
            yield self.X[batch_idx], self.y[batch_idx]


# ==================== Model ====================

class TimeSeriesTransformer(nn.Module):
    d_model: int
    num_heads: int
    mlp_dim: int
    num_layers: int
    pred_len: int
    out_dim: int

    @nn.compact
    def __call__(self, x):

        # Input projection (CRITICAL)
        x = nn.Dense(self.d_model)(x)

        # Positional encoding
        x = TimeSeriesPositionalEncoding(self.d_model)(x)

        # Transformer stack
        for _ in range(self.num_layers):

            # Self attention block
            h = nn.LayerNorm()(x)
            h = MultiSequenceTransformerBlock(
                self.d_model,
                self.num_heads,
                mlp_dim=self.mlp_dim
            )(h)

            x = x + h

            # MLP block
            h = nn.LayerNorm()(x)
            h = MLP(self.d_model, self.mlp_dim)(h)

            x = x + h

        x = nn.LayerNorm()(x)

        # Output projection
        x = nn.Dense(self.out_dim)(x)

        return x[..., -self.pred_len:, :]


# ==================== Training ====================

@jax.jit
def train_step(state, batch):

    X, y = batch

    def loss_fn(params):
        preds = state.apply_fn(
            {"params": params},
            X
        )
        loss = jnp.mean((preds - y) ** 2)
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)

    return state, loss


@jax.jit
def eval_step(state, batch):
    X, y = batch

    preds = state.apply_fn(
        {"params": state.params},
        X
    )

    loss = jnp.mean((preds - y) ** 2)
    return loss


# ==================== Training Loop ====================

def train_model(
    X,
    y,
    d_model=64,
    num_heads=4,
    mlp_dim=128,
    num_layers=3,
    lr=1e-3,
    epochs=5,
    batch_size=32,
    pred_len=10
):

    model = TimeSeriesTransformer(
        d_model=d_model,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        num_layers=num_layers,
        pred_len=pred_len,
        out_dim=X.shape[-1]
    )
    rng = jr.PRNGKey(0)

    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=model.init(rng, X[:1])["params"],
        tx=optax.adam(lr)
    )

    dataset = TimeSeriesDataset(X, y, batch_size)

    train_losses = []

    for epoch in range(epochs):

        epoch_loss = 0
        steps = 0

        for batch in dataset:
            state, loss = train_step(state, batch)
            epoch_loss += loss
            steps += 1

        epoch_loss /= max(steps, 1)
        train_losses.append(epoch_loss)

        print(f"Epoch {epoch+1}/{epochs} | Loss = {epoch_loss:.6f}")

    return state, train_losses


# ==================== Visualization ====================

def visualize_prediction(state, X, y, idx=0):

    sample_x = X[idx:idx+1]
    true_y = y[idx]

    pred_y = state.apply_fn(
        {"params": state.params},
        sample_x
    )[0]

    # Plot each feature separately
    n_features = true_y.shape[-1]

    for f in range(3):

        plt.figure(figsize=(10,4))

        # context window + prediction
        context = sample_x[0, ..., f]
        true_future = true_y[0, ..., f]
        pred_future = pred_y[0, ..., f]

        t1 = np.arange(len(context))
        t2 = np.arange(len(context),
                       len(context)+len(true_future))

        plt.plot(t1, context, label="Context")
        plt.plot(t2, true_future, label="True future")
        plt.plot(t2, pred_future, label="Predicted future")

        plt.legend()
        plt.title(f"Feature {f}")
        plt.xlabel("Time")
        plt.ylabel("Value")

        plt.show()


# ==================== Example Usage ====================

def takens_embedding_multichannel(
    data,
    embedding_dim=10,
    delay=5
):
    """
    data : (time, channels)
    returns : (windows, channels, embedding_dim)
    """

    T, C = data.shape

    n_windows = T - (embedding_dim - 1) * delay

    if n_windows <= 0:
        raise ValueError("Embedding too large for signal length")

    embedded = []

    for i in range(n_windows):

        window_channels = []

        for c in range(C):

            channel_emb = []

            for d in range(embedding_dim):

                idx = i + d * delay
                channel_emb.append(data[idx, c])

            window_channels.append(channel_emb)

        embedded.append(window_channels)

    return np.array(embedded)

from eegdash.dataset import DS006940

def create_forecast_dataset(data, window=40, horizon=5, stride=1):

    X, y = [], []

    for i in range(0, len(data) - window - horizon, stride):

        X.append(data[i:i+window])
        y.append(data[i+window:i+window+horizon])

    return np.array(X), np.array(y)

if __name__ == "__main__":

    # ----------------------------
    # Load EEG
    # ----------------------------

    dataset = DS006940(cache_dir="./data")

    raw = dataset.datasets[0].raw

    # VERY IMPORTANT
    raw.load_data()

    # Filter
    raw.filter(1, 40)

    data = raw.get_data().T[:2000]  # (time, channels)

    print("Raw EEG:", data.shape)

    # ----------------------------
    # Normalize (train stats only)
    # ----------------------------

    split_norm = int(len(data) * 0.8)

    mean = data[:split_norm].mean(axis=0)
    std = data[:split_norm].std(axis=0) + 1e-8

    data = (data - mean) / std

    # ----------------------------
    # Takens embedding per channel
    # ----------------------------

    embedding_dim = 8
    delay = 3

    X_embedded = takens_embedding_multichannel(
        data,
        embedding_dim=embedding_dim,
        delay=delay
    )

    print("Takens shape:", X_embedded.shape)

    # ----------------------------
    # Forecast dataset
    # ----------------------------

    window_size = 20
    horizon = 5

    X, y = create_forecast_dataset(
        X_embedded,
        window=window_size,
        horizon=horizon
    )

    # Transformer expects (samples, time, features)

    X = X.transpose(0, 2, 1, 3)
    y = y.transpose(0, 2, 1, 3)

    print("Dataset:", X.shape, y.shape)

    # ----------------------------
    # Train / Test split
    # ----------------------------

    split_idx = int(len(X) * 0.8)

    X_train = X[:split_idx]
    y_train = y[:split_idx]

    X_test = X[split_idx:]
    y_test = y[split_idx:]

    # ----------------------------
    # Train
    # ----------------------------

    state, losses = train_model(
        X_train,
        y_train,
        pred_len=horizon,
        epochs=15
    )

    # ----------------------------
    # Evaluate
    # ----------------------------

    test_loss = eval_step(state, (X_test, y_test))
    print("Test MSE:", float(test_loss))

    # Plot loss
    plt.plot(losses)
    plt.title("Training Loss")
    plt.show()

    if len(X_test) > 0:
        visualize_prediction(state, X_test, y_test, idx=0)