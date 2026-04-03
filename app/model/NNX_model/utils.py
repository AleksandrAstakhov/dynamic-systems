from flax import nnx
import numpy as np
import jax.numpy as jnp
import jax

state_axes = nnx.StateAxes({(nnx.Param, "default"): 0, ...: None})


@nnx.vmap(in_axes=(0, None, None), out_axes=0)
def create_v_model(rngs: nnx.Rngs, model_cls, model_args):
    return model_cls(rngs=rngs, **model_args)


class FlowDataloader:

    def __init__(self, mu_x, logvar_x, mu_y, logvar_y, batch_size, dt, rngs, shuffle=True):

        self.mu_x = mu_x
        self.mu_y = mu_y
        self.rngs = nnx.Rngs(rngs())
        self.num_samples = mu_x.shape[0]
        self.batch_size = batch_size
        self.dt = dt
        self.sigma_x = jax.nn.softplus(logvar_x) + 1e-4
        self.sigma_y = jax.nn.softplus(logvar_y) + 1e-4
        self.shuffle = shuffle

    def __iter__(self):
        indices = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(indices)

        for start in range(0, self.num_samples, self.batch_size):
            batch_idx = indices[start : start + self.batch_size]
            mu_x = self.mu_x[batch_idx]
            mu_y = self.mu_y[batch_idx]
            sigma_x = self.sigma_x[batch_idx]
            sigma_y = self.sigma_y[batch_idx]

            x = mu_x + sigma_x * self.rngs.normal(mu_x.shape)

            dz = (mu_y - mu_x) / self.dt + (
                (((sigma_y - sigma_x) / self.dt) ** 2 - jnp.ones(mu_x.shape))
                / (2 * sigma_x**2)
            ) * (x - mu_x)

            yield x, dz


class DataLoader:
    def __init__(self, X, y=None, batch_size=32, shuffle=True, encoder=None):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = X.shape[0]
        self.encoder = encoder

    def __iter__(self):
        indices = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(indices)

        for start in range(0, self.num_samples, self.batch_size):
            batch_idx = indices[start : start + self.batch_size]
            X_batch = self.X[batch_idx]

            if self.encoder:
                X_batch = self.encoder(X_batch)

            if self.y is not None:
                y_batch = self.y[batch_idx]
                if self.encoder:
                    y_batch = self.encoder(y_batch)
                yield jnp.array(X_batch), jnp.array(y_batch)
            else:
                yield jnp.array(X_batch)


def create_dataset(embedded, window=20, horizon=1):
    T = embedded.shape[0]

    n_samples = T - window - horizon + 1

    x = jnp.stack([embedded[i : i + window] for i in range(n_samples)])

    y = jnp.stack(
        [embedded[i + horizon : i + horizon + window] for i in range(n_samples)]
    )

    return x, y


def takens_embedding_multichannel(data, embedding_dim=10, delay=5):

    T, C = data.shape

    n_windows = T - (embedding_dim - 1) * delay

    start_indices = jnp.arange(n_windows)[:, None]

    offsets = jnp.arange(embedding_dim)[None, :] * delay

    time_indices = start_indices + offsets

    embedded = data[time_indices]

    return embedded.transpose(0, 2, 1)
