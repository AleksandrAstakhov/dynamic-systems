from flax import nnx
import numpy as np
import jax.numpy as jnp
import jax

state_axes = nnx.StateAxes({(nnx.Param, "default"): 0, ...: None})


@nnx.vmap(in_axes=(0, None, None), out_axes=0)
def create_v_model(rngs: nnx.Rngs, model_cls, model_args):
    return model_cls(rngs=rngs, **model_args)


class FlowDataloader:
    def __init__(
        self,
        mu_x,
        logvar_x,
        mu_y,
        logvar_y,
        batch_size,
        dt,
        rngs,
        g,
        window_size=50,
        shuffle=True,
    ):
        self.mu_x = mu_x
        self.mu_y = mu_y
        self.rngs = nnx.Rngs(rngs())
        self.batch_size = batch_size
        self.dt = dt
        self.window_size = window_size
        self.shuffle = shuffle
        self.g = g

        self.sigma_x = jax.nn.softplus(logvar_x) + 1e-4
        self.sigma_y = jax.nn.softplus(logvar_y) + 1e-4

        self.valid_indices = np.arange(window_size - 1, mu_x.shape[0])

    def __iter__(self):
        indices = self.valid_indices.copy()
        if self.shuffle:
            np.random.shuffle(indices)

        for start in range(0, len(indices), self.batch_size):
            batch_idx = indices[start : start + self.batch_size]

            x_batch = []
            dz_batch = []

            for idx in batch_idx:
                mu_x_w = self.mu_x[idx - self.window_size + 1 : idx + 1]
                mu_y_w = self.mu_y[idx - self.window_size + 1 : idx + 1]
                sigma_x_w = self.sigma_x[idx - self.window_size + 1 : idx + 1]
                sigma_y_w = self.sigma_y[idx - self.window_size + 1 : idx + 1]

                mu_x_t = mu_x_w[-1]
                mu_y_t = mu_y_w[-1]
                sigma_x_t = sigma_x_w[-1]
                sigma_y_t = sigma_y_w[-1]

                x = mu_x_t + sigma_x_t * self.rngs.normal(mu_x_t.shape)

                dz = (mu_y_t - mu_x_t) / self.dt + (
                    (((sigma_y_t - sigma_x_t) / self.dt) ** 2 - g**2)
                    / (2 * sigma_x_t**2)
                ) * (x - mu_x_t)

                x_batch.append(jnp.concatenate([mu_x_w], axis=0))
                dz_batch.append(dz)

            yield jnp.array(x_batch), jnp.array(dz_batch)


class DataLoader:
    def __init__(
        self, X, y=None, batch_size=32, shuffle=True, encoder=None, window_size=50
    ):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.encoder = encoder
        self.window_size = window_size

        self.valid_indices = np.arange(window_size - 1, X.shape[0])

    def __iter__(self):
        indices = self.valid_indices.copy()

        if self.shuffle:
            np.random.shuffle(indices)

        for start in range(0, len(indices), self.batch_size):
            batch_idx = indices[start : start + self.batch_size]

            X_batch = []
            y_batch = [] if self.y is not None else None

            for idx in batch_idx:
                window = self.X[idx - self.window_size + 1 : idx + 1]
                X_batch.append(window)

                if self.y is not None:
                    y_batch.append(self.y[idx])

            X_batch = jnp.array(X_batch)

            if self.encoder:
                X_batch = self.encoder(X_batch)

            if self.y is not None:
                y_batch = jnp.array(y_batch)
                if self.encoder:
                    y_batch = self.encoder(y_batch)
                yield X_batch, y_batch
            else:
                yield X_batch


def create_dataset(embedded, window=20, horizon=1):
    T = embedded.shape[0]

    n_samples = T - window - horizon + 1

    x = jnp.array(embedded[: len(embedded) - horizon])
    y = jnp.stack(embedded[horizon:])

    return x, y


def takens_embedding_multichannel(data, embedding_dim=10, delay=5):

    T, C = data.shape

    n_windows = T - (embedding_dim - 1) * delay

    start_indices = jnp.arange(n_windows)[:, None]

    offsets = jnp.arange(embedding_dim)[None, :] * delay

    time_indices = start_indices + offsets

    embedded = data[time_indices]

    return embedded.transpose(0, 2, 1)
