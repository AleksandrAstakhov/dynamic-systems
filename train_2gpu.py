from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import SpatialForecaster
from models.correlation import empirical_correlation
from torch.utils.data import DataLoader, Subset
from train import WindowDataset


def run_epoch(
    model, loader, opt, device, lam_rec, lam_kl, lam_aux, lam_in, train: bool
):
    model.train(train)
    ctx = torch.enable_grad() if train else torch.no_grad()
    inner = model.module if isinstance(model, nn.DataParallel) else model
    tot = dict(L=0.0, fc=0.0, inw=0.0, rec=0.0, kl=0.0, aux=0.0, n=0)
    with ctx:
        for x, x_raw, y in loader:
            x = x.to(device, non_blocking=True)
            x_raw = x_raw.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            out = model(x, x_raw)
            fc = F.mse_loss(out["pred"], y)
            inw = F.mse_loss(out["in_window_pred"], out["in_window_target"])
            rec, kl = inner.vae.losses(out, beta=1.0)
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
    ap.add_argument("--spatial", choices=["phase_mha", "correlation"], required=True)
    ap.add_argument("--takens", default="data/takens.npz")
    ap.add_argument("--series", default="data/series.npz")
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="общий батч; делится поровну между GPU",
    )
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--vae_dim", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--lam_rec", type=float, default=0.1)
    ap.add_argument("--lam_kl", type=float, default=1e-4)
    ap.add_argument("--lam_aux", type=float, default=0.05)
    ap.add_argument("--lam_in", type=float, default=1.0)
    ap.add_argument("--n_modes", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--gpus",
        type=int,
        nargs="+",
        default=[0, 1],
        help="ID видеокарт, например: --gpus 0 1  или  --gpus 0 2",
    )
    args = ap.parse_args()

    if args.out is None:
        args.out = f"results/{args.spatial}_dp.pt"
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA недоступна, запуск 2-GPU невозможен")
    if len(args.gpus) < 2:
        raise ValueError("требуется минимум 2 GPU; передано: %s" % args.gpus)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    main_device = torch.device(f"cuda:{args.gpus[0]}")
    print(
        f"=== Distributed training on GPUs {args.gpus} (main: cuda:{args.gpus[0]}) ==="
    )
    for g in args.gpus:
        name = torch.cuda.get_device_name(g)
        print(f"  cuda:{g} -> {name}")

    tk = np.load(args.takens)
    sr = np.load(args.series, allow_pickle=True)
    emb = tk["emb"]
    m = int(tk["m"])
    raw = sr["x"]
    T, C = raw.shape
    latent = m

    ds = WindowDataset(emb, raw, args.window, args.horizon, args.warmup)
    cut = int(len(ds) * 0.8)
    tr_ds = Subset(ds, np.arange(cut).tolist())
    va_ds = Subset(ds, np.arange(cut, len(ds)).tolist())
    tr_ld = DataLoader(
        tr_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=2,
        pin_memory=True,
    )
    va_ld = DataLoader(
        va_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    raw_train = raw[: int(T * 0.8)]
    corr_train = empirical_correlation(raw_train)

    spatial_kwargs = {}
    if args.spatial == "phase_mha":
        spatial_kwargs = dict(d_head=16, n_modes=args.n_modes, untied=True)

    model = SpatialForecaster(
        m_embed=m,
        latent_dim=latent,
        num_channels=C,
        warmup=args.warmup,
        horizon=args.horizon,
        spatial_kind=args.spatial,
        vae_dim=args.vae_dim,
        spatial_kwargs=spatial_kwargs,
        corr_matrix=corr_train,
    ).to(main_device)

    if len(args.gpus) > 1:
        model = nn.DataParallel(model, device_ids=args.gpus)
        print(f"  DataParallel: batch split across {len(args.gpus)} GPUs")

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
            main_device,
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
            main_device,
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
            f"train: fc={tr['fc']:.5f} inw={tr['inw']:.5f} rec={tr['rec']:.5f}  |  "
            f"val: fc={va['fc']:.5f} inw={va['inw']:.5f}"
        )
        if va["fc"] < best:
            best = va["fc"]
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            inner = model.module if isinstance(model, nn.DataParallel) else model
            torch.save(
                {
                    "state_dict": inner.state_dict(),
                    "args": vars(args),
                    "m": m,
                    "latent": latent,
                    "num_channels": C,
                },
                args.out,
            )

    ck = torch.load(args.out, map_location=main_device, weights_only=False)
    inner = model.module if isinstance(model, nn.DataParallel) else model
    inner.load_state_dict(ck["state_dict"])
    mse_h = per_horizon_mse(model, va_ld, main_device, args.horizon)

    metrics = dict(
        spatial=args.spatial,
        best_val_fc=float(best),
        per_horizon_mse=mse_h.tolist(),
        history=history,
        gpus_used=args.gpus,
    )
    json_path = args.out.replace(".pt", "_metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[{args.spatial}] best val MSE = {best:.5f}")
    print(f"  per-horizon MSE: {mse_h.round(5).tolist()}")
    print(f"  wrote {json_path}")


if __name__ == "__main__":
    main()
