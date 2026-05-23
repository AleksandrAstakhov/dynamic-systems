from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from models import LatentForecaster
from torch.utils.data import DataLoader, Dataset, Subset


def empirical_correlation(x):
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x, dtype=torch.float32)
    x = x - x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True) + 1e-8
    x = x / std
    return (x.T @ x) / x.shape[0]


class WindowDataset(Dataset):
    def __init__(
        self, emb: np.ndarray, raw: np.ndarray, window: int, horizon: int, warmup: int
    ):
        T_emb = emb.shape[0]
        offset = raw.shape[0] - T_emb
        assert offset >= 0
        self.emb = emb.astype(np.float32)
        self.raw = raw[offset:].astype(np.float32)
        self.window = window
        self.horizon = horizon
        self.warmup = warmup
        assert window > warmup
        self.N = T_emb - window - horizon + 1

    def __len__(self):
        return self.N

    def __getitem__(self, i):
        x = self.emb[i : i + self.window]
        x_raw = self.raw[i : i + self.window]
        y = self.raw[i + self.window : i + self.window + self.horizon]
        return torch.from_numpy(x), torch.from_numpy(x_raw), torch.from_numpy(y)


def _base(model):
    """Unwrap DataParallel if applied."""
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def run_epoch(
    model, loader, opt, device, lam_rec, lam_kl, lam_aux, lam_in, train: bool
):
    model.train(train)
    ctx = torch.enable_grad() if train else torch.no_grad()
    inner = _base(model)
    tot = dict(L=0.0, fc=0.0, inw=0.0, rec=0.0, kl=0.0, aux=0.0, n=0)
    with ctx:
        for x, x_raw, y in loader:
            x = x.to(device, non_blocking=True)
            x_raw = x_raw.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            out = model(x, x_raw)
            fc = F.mse_loss(out["pred"], y)
            inw = F.mse_loss(out["in_window_pred"], out["in_window_target"])
            rec, kl = inner.encoder.losses(out, beta=1.0)
            z = out["z"]
            B, Tp = z.shape[0], z.shape[1]
            aux = inner.spatial.aux_loss(z.reshape(B * Tp, *z.shape[2:]))
            total = fc + lam_in * inw + lam_rec * rec + lam_kl * kl + lam_aux * aux
            if train:
                opt.zero_grad()
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            bs = x.size(0)
            tot["L"] += float(total) * bs
            tot["fc"] += float(fc) * bs
            tot["inw"] += float(inw) * bs
            tot["rec"] += float(rec) * bs
            tot["kl"] += float(kl) * bs
            tot["aux"] += float(aux) * bs
            tot["n"] += bs
    n = max(tot["n"], 1)
    return {k: v / n for k, v in tot.items() if k != "n"}


def per_horizon_mse(model, loader, device, H):
    mse_h = torch.zeros(H)
    n = 0
    with torch.no_grad():
        for x, x_raw, y in loader:
            x = x.to(device)
            x_raw = x_raw.to(device)
            y = y.to(device)
            p = model(x, x_raw)["pred"].cpu()
            y = y.cpu()
            mse_h += ((p - y) ** 2).mean(dim=(0, 2)) * x.size(0)
            n += x.size(0)
    return (mse_h / n).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--spatial", choices=["correlation", "grand_diff", "grand_full"], required=True
    )
    ap.add_argument("--takens", default="data/takens.npz")
    ap.add_argument("--series", default="data/series.npz")
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--window", type=int, default=24)
    ap.add_argument("--warmup", type=int, default=6)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--vae_dim", type=int, default=32)
    ap.add_argument("--temporal_blocks", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--lam_rec", type=float, default=0.1)
    ap.add_argument("--lam_kl", type=float, default=1e-4)
    ap.add_argument("--lam_aux", type=float, default=0.05)
    ap.add_argument("--lam_in", type=float, default=1.0)
    ap.add_argument("--n_modes", type=int, default=2)
    ap.add_argument("--ode_method", default="rk4")
    ap.add_argument("--ode_steps", type=int, default=4)
    ap.add_argument("--ode_t_max", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.out is None:
        args.out = f"results/{args.spatial}.pt"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    tk = np.load(args.takens)
    sr = np.load(args.series, allow_pickle=True)
    emb = tk["emb"]
    m = int(tk["m"])
    raw = sr["x"]
    T, C = raw.shape
    latent = m

    ds = WindowDataset(emb, raw, args.window, args.horizon, args.warmup)
    cut = int(len(ds) * 0.8)
    pin = args.device == "cuda"
    tr_ld = DataLoader(
        Subset(ds, list(range(cut))),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=2,
        pin_memory=pin,
    )
    va_ld = DataLoader(
        Subset(ds, list(range(cut, len(ds)))),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=pin,
    )

    raw_train = raw[: int(T * 0.8)]
    corr_train = empirical_correlation(raw_train)

    sk = dict(
        t_max=args.ode_t_max,
        method=args.ode_method,
        n_steps=args.ode_steps,
    )
    if args.spatial in ("grand_diff", "grand_full"):
        sk.update(dict(d_head=16, n_modes=args.n_modes, untied=True))

    model = LatentForecaster(
        m_embed=m,
        latent_dim=latent,
        num_channels=C,
        warmup=args.warmup,
        horizon=args.horizon,
        spatial_kind=args.spatial,
        vae_dim=args.vae_dim,
        temporal_blocks=args.temporal_blocks,
        spatial_kwargs=sk,
        corr_matrix=corr_train,
    ).to(args.device)

    n_visible = torch.cuda.device_count() if args.device == "cuda" else 0
    if n_visible > 1:
        print(
            f"[{args.spatial}] DataParallel across {n_visible} GPUs: "
            f"{[torch.cuda.get_device_name(i) for i in range(n_visible)]}",
            flush=True,
        )
        model = torch.nn.DataParallel(model)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=1e-5
    )

    best = float("inf")
    history = []
    for ep in range(args.epochs):
        tr = run_epoch(
            model,
            tr_ld,
            opt,
            args.device,
            args.lam_rec,
            args.lam_kl,
            args.lam_aux,
            args.lam_in,
            True,
        )
        va = run_epoch(
            model,
            va_ld,
            opt,
            args.device,
            args.lam_rec,
            args.lam_kl,
            args.lam_aux,
            args.lam_in,
            False,
        )
        sched.step()
        history.append(
            {
                "epoch": ep,
                "train_fc": tr["fc"],
                "val_fc": va["fc"],
                "train_inw": tr["inw"],
                "val_inw": va["inw"],
            }
        )
        print(
            f"[{args.spatial}] ep {ep:3d}  "
            f"train fc={tr['fc']:.5f} inw={tr['inw']:.5f} rec={tr['rec']:.4f}  |  "
            f"val fc={va['fc']:.5f} inw={va['inw']:.5f}",
            flush=True,
        )
        if va["fc"] < best:
            best = va["fc"]
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            torch.save(
                {
                    "state_dict": _base(model).state_dict(),
                    "args": vars(args),
                    "m": m,
                    "latent": latent,
                    "num_channels": C,
                },
                args.out,
            )

    ck = torch.load(args.out, map_location=args.device, weights_only=False)
    _base(model).load_state_dict(ck["state_dict"])
    mse_h = per_horizon_mse(model, va_ld, args.device, args.horizon)
    metrics = dict(
        spatial=args.spatial,
        best_val_fc=float(best),
        per_horizon_mse=mse_h.tolist(),
        history=history,
    )
    with open(args.out.replace(".pt", "_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(
        f"[{args.spatial}] best val MSE = {best:.5f}; "
        f"per-horizon MSE = {mse_h.round(5).tolist()}"
    )


if __name__ == "__main__":
    main()
