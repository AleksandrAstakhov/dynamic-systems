import jax
import jax.numpy as jnp
import flax.linen as nn

from STFormer import STFormer
from VAE import VAE, vae_loss
import matplotlib.pyplot as plt
import optax
from typing import Any
from flax.training import train_state, checkpoints



class SSSPformer(nn.Module):
    num_heads: int
    head_dim: int
    mlp_dim: int
    latent_dim: int
    num_layers: int

    @nn.compact
    def __call__(self, x):

        B, S, C, D = x.shape

        x_ = x.rashape(B * S, C, D)

        def create_vae():
            return VAE(latent_dim=self.latent_dim, input_dim=D)

        vae_vmap = nn.vmap(
            create_vae,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=1,
            out_axes=1,
        )

        recon, mu, logvar = vae_vmap(x_)

        recon_reshaped = recon.reshape(B, S, C, -1)

        z = mu.reshape(B, S, C, -1)

        for _ in range(self.num_layers):
            z = STFormer(
                num_heads=self.num_heads, head_dim=self.head_dim, mlp_dim=self.mlp_dim
            )(z)

        final_output = nn.Dense(D)(z)

        return {
            "final_output": final_output,
            "vae_recon": recon_reshaped,
            "mu": mu,
            "logvar": logvar,
            "x_orig": x,
        }


def compute_full_loss(outputs, y, horizon, beta=1.0, alpha=2.0):
    """Вычисляет полную функцию потерь с VAE компонентой"""

    x_orig = outputs["x_orig"]
    final_output = outputs["final_output"]
    recon = outputs["vae_recon"]
    mu = outputs["mu"]
    logvar = outputs["logvar"]

    kl_loss = -0.5 * jnp.sum(1 + logvar - mu**2 - jnp.exp(logvar))
    kl_loss = kl_loss / mu.shape[0]

    recon_loss = jnp.mean((recon - x_orig) ** 2)

    predict_loss = jnp.mean(
        (y[:, -horizon:, :, -1] - final_output[:, -horizon:, :, -1]) ** 2
    )

    total_loss = recon_loss + beta * kl_loss + alpha * predict_loss

    return total_loss, {
        "total_loss": total_loss,
        "recon_loss": recon_loss,
        "kl_loss": kl_loss,
        "predict_loss" : predict_loss
    }

class TrainState(optax.train_state.TrainState):
    """Состояние обучения"""
    batch_stats: Any = None


def create_train_state(rng, model, learning_rate, input_shape):
    """Создает состояние обучения"""
    
    variables = model.init(rng, jnp.ones(input_shape))
    params = variables['params']
    
    # Оптимизатор с cosine decay
    scheduler = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=100,
        decay_steps=1000,
        end_value=1e-6
    )
    
    tx = optax.adamw(learning_rate=scheduler, weight_decay=0.01)
    
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )
    
    return state


# ==================== Шаги обучения ====================

@jax.jit
def train_step(state, model, batch, horizon, beta=1.0, alpha=2.0):
    """Один шаг обучения"""
    
    x = batch['x']
    y = batch['y']
    
    def loss_fn(params):
        outputs = model.apply({'params': params}, x)
        loss, metrics = compute_full_loss(outputs, y, horizon, beta, alpha)
        return loss, metrics
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, metrics), grads = grad_fn(state.params)
    
    new_state = state.apply_gradients(grads=grads)
    
    return new_state, metrics


@jax.jit
def eval_step(state, model, batch, horizon, beta=1.0, alpha=2.0):
    """Шаг валидации"""
    
    x = batch['x']
    y = batch['y']
    
    outputs = model.apply({'params': state.params}, x)
    loss, metrics = compute_full_loss(outputs, y, horizon, beta, alpha)
    
    return metrics

def takens_embedding_multichannel(data, embedding_dim=8, delay=3):
    """Takens embedding для многоканальных данных
    
    Args:
        data: [time, channels]
        embedding_dim: размерность вложения
        delay: задержка
    
    Returns:
        [time - (embedding_dim-1)*delay, channels, embedding_dim]
    """
    n_samples, n_channels = data.shape
    n_vectors = n_samples - (embedding_dim - 1) * delay
    
    embedded = jnp.zeros((n_vectors, n_channels, embedding_dim))
    
    for i in range(n_vectors):
        for j in range(embedding_dim):
            idx = i + j * delay
            embedded[i, :, j] = data[idx, :]
    
    return embedded


def create_forecast_dataset(data, window=20, horizon=5):
    """Создает датасет для прогнозирования
    
    Args:
        data: [samples, channels, features] после Takens
        window: размер окна
        horizon: горизонт прогноза
    
    Returns:
        X: [n_samples - window - horizon, window, channels, features]
        y: [n_samples - window - horizon, horizon, channels, features]
    """
    n_samples = len(data)
    X, y = [], []
    
    for i in range(n_samples - window - horizon):
        X.append(data[i:i+window])
        y.append(data[i+window:i+window+horizon])
    
    return jnp.array(X), jnp.array(y)


def normalize_data(X_train, X_test=None):
    """Нормализация данных"""
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    
    X_train_norm = (X_train - mean) / std
    
    if X_test is not None:
        X_test_norm = (X_test - mean) / std
        return X_train_norm, X_test_norm, mean, std
    
    return X_train_norm, mean, std


def prepare_batch(batch_x, batch_y):
    """Подготавливает батч для модели"""
    return {
        'x': jnp.array(batch_x, dtype=jnp.float32),
        'y': jnp.array(batch_y, dtype=jnp.float32)
    }


# ==================== Генератор батчей ====================

def batch_generator(X, y, batch_size, shuffle=True):
    """Генератор батчей"""
    n_samples = len(X)
    indices = jnp.arange(n_samples)
    
    if shuffle:
        jnp.random.shuffle(indices)
    
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_indices = indices[start:end]
        
        batch_x = X[batch_indices]
        batch_y = y[batch_indices]
        
        yield prepare_batch(batch_x, batch_y)


# ==================== Основная функция обучения ====================

def train_model(X_train, y_train, X_val=None, y_val=None, **kwargs):
    """Основная функция обучения
    
    Args:
        X_train: [n_samples, window, channels, features]
        y_train: [n_samples, horizon, channels, features]
        X_val: валидационные данные
        y_val: валидационные цели
        **kwargs: параметры обучения
    """
    
    # Параметры по умолчанию
    config = {
        'num_heads': 4,
        'head_dim': 32,
        'mlp_dim': 128,
        'latent_dim': 16,
        'num_layers': 2,
        'learning_rate': 1e-3,
        'batch_size': 32,
        'epochs': 50,
        'beta': 0.1,  # вес KL
        'alpha': 2.0,  # вес prediction loss
        'patience': 10,
        'seed': 42
    }
    
    # Обновляем параметры
    config.update(kwargs)
    
    # Получаем размерности
    n_samples, window, channels, features = X_train.shape
    horizon = y_train.shape[1]
    
    print(f"Данные для обучения:")
    print(f"  X_train: {X_train.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  window: {window}, horizon: {horizon}")
    print(f"  channels: {channels}, features: {features}")
    
    # Создаем модель
    model = SSSPformer(
        num_heads=config['num_heads'],
        head_dim=config['head_dim'],
        mlp_dim=config['mlp_dim'],
        latent_dim=config['latent_dim'],
        input_dim=features,
        num_layers=config['num_layers']
    )
    
    # Инициализация
    rng = jax.random.PRNGKey(config['seed'])
    input_shape = (config['batch_size'], window, channels, features)
    state = create_train_state(rng, model, config['learning_rate'], input_shape)
    
    # Обучение
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    print("\nНачало обучения...")
    print("=" * 60)
    
    for epoch in range(config['epochs']):
        # Обучение
        epoch_losses = []
        
        for batch in batch_generator(X_train, y_train, config['batch_size']):
            state, metrics = train_step(
                state, model, batch, horizon,
                beta=config['beta'], alpha=config['alpha']
            )
            epoch_losses.append(metrics['total_loss'])
        
        avg_train_loss = jnp.mean(epoch_losses)
        train_losses.append(avg_train_loss)
        
        # Валидация
        if X_val is not None:
            val_metrics = []
            for batch in batch_generator(X_val, y_val, config['batch_size'], shuffle=False):
                metrics = eval_step(
                    state, model, batch, horizon,
                    beta=config['beta'], alpha=config['alpha']
                )
                val_metrics.append(metrics['total_loss'])
            
            avg_val_loss = jnp.mean(val_metrics)
            val_losses.append(avg_val_loss)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Сохраняем лучшую модель
                checkpoints.save_checkpoint(
                    ckpt_dir='./checkpoints',
                    target=state,
                    step=epoch,
                    prefix='best_model_',
                    keep=1
                )
            else:
                patience_counter += 1
                if patience_counter >= config['patience']:
                    print(f"Early stopping на эпохе {epoch+1}")
                    break
            
            print(f"Эпоха {epoch+1:3d}/{config['epochs']} | "
                  f"Train Loss: {avg_train_loss:.6f} | "
                  f"Val Loss: {avg_val_loss:.6f}")
        else:
            print(f"Эпоха {epoch+1:3d}/{config['epochs']} | "
                  f"Train Loss: {avg_train_loss:.6f}")
    
    print("=" * 60)
    print("Обучение завершено!")
    
    return state, train_losses, val_losses if X_val is not None else None


# ==================== Визуализация ====================

def visualize_prediction(state, model, X_test, y_test, idx=0, horizon=None):
    """Визуализация предсказания"""
    
    batch = prepare_batch(X_test[idx:idx+1], y_test[idx:idx+1])
    x = batch['x']
    y_true = batch['y'][0]
    
    # Предсказание
    outputs = model.apply({'params': state.params}, x)
    y_pred = outputs['final_output'][0]
    
    if horizon is not None:
        y_true = y_true[-horizon:]
        y_pred = y_pred[-horizon:]
    
    # Визуализация
    n_channels = min(4, y_true.shape[1])  # покажем до 4 каналов
    
    fig, axes = plt.subplots(n_channels, 1, figsize=(12, 3*n_channels))
    if n_channels == 1:
        axes = [axes]
    
    for ch in range(n_channels):
        ax = axes[ch]
        
        # Истинные значения
        ax.plot(y_true[:, ch, 0], 'b-', label='True', linewidth=2)
        # Предсказанные значения
        ax.plot(y_pred[:, ch, 0], 'r--', label='Predicted', linewidth=2)
        
        ax.set_title(f'Channel {ch+1}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# ==================== Основной блок ====================

if __name__ == "__main__":
    
    # ----------------------------
    # Загрузка ЭЭГ данных
    # ----------------------------
    
    # Предполагаем, что у вас есть класс DS006940
    # Если нет, используйте синтетические данные для теста
    
    try:
        from eegdash.dataset import DS006940
        
        dataset = DS006940(cache_dir="./data")
        raw = dataset.datasets[0].raw
        
        # VERY IMPORTANT
        raw.load_data()
        
        # Фильтрация
        raw.filter(1, 40)
        
        data = raw.get_data().T[:2000]  # (time, channels)
        
    except ImportError:
        print("DS006940 не найден, используем синтетические данные")
        # Синтетические данные
        n_time = 5000
        n_channels = 19
        t = jnp.linspace(0, 10, n_time)
        data = jnp.zeros((n_time, n_channels))
        
        for ch in range(n_channels):
            freq = 5 + ch * 2
            data[:, ch] = jnp.sin(2 * jnp.pi * freq * t) + 0.1 * jnp.random.randn(n_time)
    
    print("Raw EEG:", data.shape)
    
    # ----------------------------
    # Нормализация
    # ----------------------------
    
    split_norm = int(len(data) * 0.8)
    
    mean = data[:split_norm].mean(axis=0)
    std = data[:split_norm].std(axis=0) + 1e-8
    
    data = (data - mean) / std
    
    # ----------------------------
    # Takens embedding
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
    
    # Transformer expects (samples, time, channels, features)
    # X уже имеет форму [samples, window, channels, features]
    
    print("Dataset:", X.shape, y.shape)
    
    # ----------------------------
    # Train / Test split
    # ----------------------------
    
    split_idx = int(len(X) * 0.8)
    
    X_train = X[:split_idx]
    y_train = y[:split_idx]
    
    X_test = X[split_idx:]
    y_test = y[split_idx:]
    
    # Дополнительное разделение на train/val
    val_split = int(len(X_train) * 0.9)
    X_val = X_train[val_split:]
    y_val = y_train[val_split:]
    X_train = X_train[:val_split]
    y_train = y_train[:val_split]
    
    print(f"Train: {X_train.shape}, {y_train.shape}")
    print(f"Val: {X_val.shape}, {y_val.shape}")
    print(f"Test: {X_test.shape}, {y_test.shape}")
    
    # ----------------------------
    # Обучение
    # ----------------------------
    
    state, train_losses, val_losses = train_model(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        epochs=30,
        batch_size=32,
        learning_rate=1e-3,
        num_heads=4,
        head_dim=32,
        mlp_dim=128,
        latent_dim=16,
        num_layers=2,
        beta=0.1,
        alpha=2.0,
        patience=10
    )
    
    # ----------------------------
    # Оценка на тесте
    # ----------------------------
    
    test_metrics = []
    for batch in batch_generator(X_test, y_test, 32, shuffle=False):
        metrics = eval_step(state, state, batch, horizon, beta=0.1, alpha=2.0)
        test_metrics.append(metrics['total_loss'])
    
    test_loss = jnp.mean(test_metrics)
    print(f"\nTest MSE: {test_loss:.6f}")
    
    # ----------------------------
    # Визуализация
    # ----------------------------
    
    # График обучения
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train', linewidth=2)
    if val_losses is not None:
        plt.plot(val_losses, label='Validation', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Предсказания
    plt.subplot(1, 2, 2)
    if len(X_test) > 0:
        # Используем последнюю модель для визуализации
        model_for_viz = SSSPformer(
            num_heads=4, head_dim=32, mlp_dim=128,
            latent_dim=16, input_dim=embedding_dim, num_layers=2
        )
        visualize_prediction(state, model_for_viz, X_test, y_test, idx=0)
    else:
        plt.text(0.5, 0.5, 'Нет тестовых данных для визуализации', 
                 ha='center', va='center')
    
    plt.tight_layout()
    plt.show()
