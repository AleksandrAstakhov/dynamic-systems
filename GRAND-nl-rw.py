import torch_geometric as torchg
import torch
import torch.nn as nn
import torch_geometric.nn as gnn

import matplotlib.pyplot as plt

from torchdiffeq import odeint

from tqdm.auto import tqdm


class GRAND_nl_rw_Block(gnn.MessagePassing):
    def __init__(self, num_heads, num_chanels, featurs_dim, out_dim, rho=0.7):
        super().__init__(flow="target_to_source", aggr="sum")

        self.rho = 0.7
        self.num_heads = num_heads
        self.num_chanels = num_chanels
        self.featurs_dim = featurs_dim
        self.out_dim = out_dim

        self.register_buffer("eye", torch.eye(self.num_chanels))

        self.att = torch.zeros(4, 4)

        self.k = nn.Linear(self.featurs_dim, self.out_dim * self.num_heads)
        self.q = nn.Linear(self.featurs_dim, self.out_dim * self.num_heads)
        self.v = nn.Linear(self.featurs_dim, self.out_dim * self.num_heads)

        self.w0 = nn.Linear(self.out_dim * self.num_heads, self.out_dim)

    def message(self, src, tgt, src_idx, tgt_idx):
        return self.att[src_idx, tgt_idx] * (tgt - src)
    
    def update(self, inputs):
        return inputs

    def forward(
        self,
        x,
    ):  # just x cause we hawe complete graph where edge weight is att score
        nodes = torch.arange(x.shape[0])

        k = self.k(x).reshape(-1, self.num_heads, self.num_chanels, self.out_dim)
        q = self.q(x).reshape(-1, self.num_heads, self.num_chanels, self.out_dim)
        v = self.v(x).reshape(-1, self.num_heads, self.num_chanels, self.out_dim)

        att_scores = torch.einsum("bhcf,bhdf->bhcd", q, k) / (self.out_dim**0.5)
        self.att = torch.softmax(
            (torch.softmax(att_scores, dim=-1) > self.rho).long() * att_scores, dim=-1
        )
        x_v = self.v(x)
        

        vals = torch.einsum(
            "bhcd,bhdf->bhcf",
            att_weights,
            v,
        )
        
        x_v = self.v(x)
        

        vals = vals.permute(0, 2, 1, 3)  # [B, C, H, F]
        vals = vals.reshape(vals.size(0), vals.size(1), self.num_heads * self.out_dim)

        return self.w0(vals)


class ODEFunc(nn.Module):
    def __init__(self, func, **kwargs):
        super().__init__()

        self.func = func
        self.kwargs = kwargs

    def forward(self, t, x):
        return self.func(x)


class GRAND(nn.Module):
    def __init__(self, num_heads, num_chanels, featurs_dim, out_dim):
        super().__init__()

        self.grand_encoder = ODEFunc(
            GRAND_nl_rw_Block(num_heads, num_chanels, featurs_dim, out_dim)
        )
        self.grand_decoder = ODEFunc(
            GRAND_nl_rw_Block(num_heads, num_chanels, featurs_dim, out_dim)
        )

    def forward(self, x, t):
        z = odeint(self.grand_encoder, x, t, method="rk4")[-1]
        x_hat = odeint(self.grand_decoder, z, t, method="rk4")[-1]

        return z, x_hat


def lorenz_attractor(
    y0=(0.1, 0.0, 0.0), t1=50.0, dt=0.01, sigma=10.0, rho=28.0, beta=8 / 3
):
    y0 = torch.Tensor(y0)
    t = torch.arange(0, t1 + dt, dt)
    traj = torch.zeros((len(t), 3))
    traj[0] = y0

    def f(y):
        x, yv, z = y
        return torch.Tensor([sigma * (yv - x), x * (rho - z) - yv, x * yv - beta * z])

    for i in range(len(t) - 1):
        y = traj[i]
        k1 = f(y)
        k2 = f(y + 0.5 * dt * k1)
        k3 = f(y + 0.5 * dt * k2)
        k4 = f(y + dt * k3)
        traj[i + 1] = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return traj


def nonlinear_observation(traj: torch.Tensor) -> torch.Tensor:
    x, y, z = traj[:, 0], traj[:, 1], traj[:, 2]
    x_nl = torch.tanh(x + y**2 - z)
    y_nl = torch.sin(y + z**2 - x)
    z_nl = torch.tanh(z + x**2 - y)
    return torch.stack([x_nl, y_nl, z_nl], axis=1)


def multivariate_delay_embedding(
    traj_nl: torch.Tensor, d: int, tau: int
) -> torch.Tensor:
    T, num_features = traj_nl.shape
    N = T - (d - 1) * tau
    if N <= 0:
        raise ValueError("Too large embedding dimension or lag")
    embeddings = []
    for i in range(num_features):
        signal = traj_nl[:, i]
        emb_i = torch.zeros((N, d))
        for j in range(d):
            emb_i[:, j] = signal[j * tau : j * tau + N]
        embeddings.append(emb_i)
    return torch.concatenate(embeddings, axis=1)


# ------------------------
# Main
# ------------------------
if __name__ == "__main__":
    # ====== Настройка устройства ======
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ====== Генерация данных ======
    Y = lorenz_attractor()
    Y_nl = nonlinear_observation(Y)

    X = multivariate_delay_embedding(Y_nl, d=9, tau=25)
    T, num_chanels = X.shape
    dt = 0.01
    window_sec = 1
    window_size = int(window_sec / dt)
    num_windows = T // window_size
    X_trimmed = X[: num_windows * window_size]

    X_windows = torch.tensor(
        X_trimmed.reshape(num_windows, window_size, num_chanels), dtype=torch.float
    )
    X_windows = X_windows.permute(0, 2, 1).to(
        device
    )  # (num_windows, num_features, window_size)

    num_heads = 4

    # ====== Модель и обучение ======
    model = GRAND(
        num_heads=num_heads,
        num_chanels=num_chanels,
        featurs_dim=window_size,
        out_dim=window_size,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # уменьшил lr
    criterion = nn.MSELoss()

    epochs = 50
    time_to_integrate = torch.linspace(0, 10, 50).to(device)
    losses = []

    for epoch in tqdm(range(epochs)):
        total_loss = 0
        latents_list = []
        x_hat_list = []
        optimizer.zero_grad()

        for window in tqdm(X_windows):
            window = window.to(device)  # обнуляем градиенты

            z, x_hat = model(window.unsqueeze(0), time_to_integrate)
            latents_list.append(z.squeeze(0).T)  # (features, batch/time)
            x_hat_list.append(x_hat)

            total_loss += criterion(x_hat, window)

        # Формируем Z
        Z = torch.cat(latents_list, dim=0)  # (num_windows * features, ...)

        # SVD и уменьшение размерности
        k = 3
        U, S, Vh = torch.linalg.svd(Z, full_matrices=False)
        Z_reduced = U[:, :k] @ torch.diag(S[:k])  # (num_windows*features, k)

        # Приводим Y к нужной форме
        Y_tensor = torch.tensor(
            Y[: Z_reduced.shape[0]].T, dtype=torch.float, device=device
        )

        # Loss на латентные координаты
        total_loss += criterion(Z_reduced.T, Y_tensor)

        # backward + gradient clipping
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss_value = total_loss.item()
        losses.append(total_loss_value)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss_value:.6f}")

        # ====== Визуализация ======
        with torch.no_grad():
            # Оригинальные данные (истинный аттрактор)
            Y_true = (
                torch.tensor(Y[: Z_reduced.shape[0]], dtype=torch.float).cpu().numpy()
            )

            # Восстановленные данные из латентного пространства
            Z_final = Z_reduced.detach().cpu().numpy()

            fig = plt.figure(figsize=(12, 6))

            # 1. До обучения — оригинальный аттрактор
            ax1 = fig.add_subplot(1, 2, 1, projection="3d")
            ax1.plot(Y_true[:, 0], Y_true[:, 1], Y_true[:, 2], color="blue")
            ax1.set_title("Оригинальный аттрактор Лоренца")
            ax1.set_xlabel("X")
            ax1.set_ylabel("Y")
            ax1.set_zlabel("Z")

            # 2. После обучения — латентное представление модели
            ax2 = fig.add_subplot(1, 2, 2, projection="3d")
            ax2.plot(Z_final[:, 0], Z_final[:, 1], Z_final[:, 2], color="red")
            ax2.set_title("Реконструированный аттрактор (GRAND)")
            ax2.set_xlabel("Z₁")
            ax2.set_ylabel("Z₂")
            ax2.set_zlabel("Z₃")

            plt.tight_layout()
            plt.show()

        # ====== График ошибки ======
        plt.figure(figsize=(6, 4))
        plt.plot(losses)
        plt.title("График функции потерь")
        plt.xlabel("Эпоха")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.show()
