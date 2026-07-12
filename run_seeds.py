#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from models import LatentForecaster
from train import WindowDataset, empirical_correlation

LABELS = {
    "correlation": "корреляция",
    "grand_diff": "GA, асимм.",
    "grand_full": "GA + reaction",
}
COLORS = {
    "correlation": "#1f77b4",
    "grand_diff": "#d62728",
    "grand_full": "#2ca02c",
}

LOSS_KEYS = ["fc", "inw", "rec", "kl", "aux", "L"]
LOSS_LABELS = {
    "fc": "Horizon MSE (fc)",
    "inw": "In-window MSE (inw)",
    "rec": "Reconstruction (rec)",
    "kl": "KL divergence (kl)",
    "aux": "Anti-collapse (aux)",
    "L": "Total loss (L)",
}


def ci95(values: np.ndarray) -> float:

    n = len(values)
    if n < 2:
        return 0.0
    return float(stats.t.ppf(0.975, df=n - 1) * values.std(ddof=1) / np.sqrt(n))


def asym(M: np.ndarray) -> float:
    return float(np.linalg.norm(M - M.T) / (np.linalg.norm(M) + 1e-12))


def load_model_for_eval(ckpt: Path, device: str, kind: str, corr_matrix) -> tuple:
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    a = ck["args"]
    sk = dict(t_max=a["ode_t_max"], method=a["ode_method"], n_steps=a["ode_steps"])
    if kind in ("grand_diff", "grand_full"):
        sk.update(dict(d_head=16, n_modes=a["n_modes"], untied=True))
    model = LatentForecaster(
        m_embed=ck["m"],
        latent_dim=ck["latent"],
        num_channels=ck["num_channels"],
        warmup=a["warmup"],
        horizon=a["horizon"],
        spatial_kind=kind,
        vae_dim=a["vae_dim"],
        temporal_blocks=a["temporal_blocks"],
        spatial_kwargs=sk,
        corr_matrix=corr_matrix,
    ).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, a


def eval_asym(ckpt: Path, device: str, kind: str, emb, raw) -> float:
    raw_train = raw[: int(raw.shape[0] * 0.8)]
    corr_train = empirical_correlation(raw_train)
    ck_meta = torch.load(ckpt, map_location="cpu", weights_only=False)
    a = ck_meta["args"]
    ds = WindowDataset(emb, raw, a["window"], a["horizon"], a["warmup"])
    cut = int(len(ds) * 0.8)
    va_ld = DataLoader(
        Subset(ds, list(range(cut, len(ds)))), batch_size=128, shuffle=False
    )
    model, _ = load_model_for_eval(ckpt, device, kind, corr_train)
    A_list = []
    with torch.no_grad():
        for x, x_raw, _ in va_ld:
            x = x.to(device)
            x_raw = x_raw.to(device)
            out = model(x, x_raw)
            A_list.append(model.coupling(out["z_last"]).cpu().numpy())
    return asym(np.concatenate(A_list, 0).mean(0))


def ckpt_path(out_dir: Path, kind: str, seed: int) -> Path:
    return out_dir / f"{kind}_seed{seed}.pt"


def metrics_path(out_dir: Path, kind: str, seed: int) -> Path:
    return out_dir / f"{kind}_seed{seed}_metrics.json"


def run_one(
    kind: str, seed: int, out_dir: Path, args_extra: list[str], gpus: str | None
) -> Path:
    ckpt = ckpt_path(out_dir, kind, seed)
    mpath = metrics_path(out_dir, kind, seed)
    if ckpt.exists() and mpath.exists():
        print(f"[skip] {kind} seed={seed}: already done")
        return mpath
    env = os.environ.copy()
    if gpus is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpus
    cmd = [
        sys.executable,
        str(ROOT / "train.py"),
        "--spatial",
        kind,
        "--seed",
        str(seed),
        "--out",
        str(ckpt),
        *args_extra,
    ]
    print(f"\n[run] {kind} seed={seed}  ->  {ckpt.name}")
    print("      " + " ".join(cmd))
    subprocess.run(cmd, env=env, check=True)
    return mpath


def collect_metrics(
    out_dir: Path, kinds: list[str], seeds: list[int], device: str, emb, raw
) -> dict:

    data: dict = {}
    for kind in kinds:
        mse_list, asym_list, horizon_list, history_list = [], [], [], []
        for seed in seeds:
            mp = metrics_path(out_dir, kind, seed)
            cp = ckpt_path(out_dir, kind, seed)
            if not mp.exists():
                print(f"[warn] missing {mp}, skipping seed {seed}")
                continue
            with open(mp) as f:
                m = json.load(f)
            mse_list.append(m["best_val_fc"])
            horizon_list.append(m["per_horizon_mse"])
            history_list.append(m.get("history", []))
            asym_list.append(
                eval_asym(cp, device, kind, emb, raw) if cp.exists() else float("nan")
            )
        data[kind] = dict(
            mse=np.array(mse_list),
            asym=np.array(asym_list),
            horizon_mse=np.array(horizon_list) if horizon_list else np.empty((0, 1)),
            history=history_list,
        )
    return data


def _history_matrix(history_list: list, key: str) -> np.ndarray | None:

    rows = []
    for hist in history_list:
        if not hist:
            continue
        col = [h.get(f"val_{key}", float("nan")) for h in hist]
        if not all(np.isnan(col)):
            rows.append(col)
    if not rows:
        return None
    L = max(len(r) for r in rows)
    mat = np.full((len(rows), L), float("nan"))
    for i, r in enumerate(rows):
        mat[i, : len(r)] = r
    return mat


def _band(ax, xs, mat: np.ndarray, color: str, label: str, alpha_fill=0.18):

    n = mat.shape[0]
    mu = np.nanmean(mat, axis=0)
    if n >= 2:
        sd = np.nanstd(mat, axis=0, ddof=1)
        t_val = stats.t.ppf(0.975, df=n - 1)
        half = t_val * sd / np.sqrt(n)
    else:
        half = np.zeros_like(mu)
    ax.plot(xs, mu, color=color, linewidth=1.8, label=label)
    if n >= 2:
        ax.fill_between(xs, mu - half, mu + half, color=color, alpha=alpha_fill)


def plot_mse_bars(data: dict, kinds: list[str], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, kind in enumerate(kinds):
        vals = data[kind]["mse"]
        mu = vals.mean()
        half = ci95(vals)
        ax.bar(
            i,
            mu,
            color=COLORS[kind],
            alpha=0.85,
            yerr=half,
            capsize=6,
            error_kw=dict(linewidth=1.5),
        )
        ax.text(i, mu + half + 0.003, f"{mu:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(np.arange(len(kinds)))
    ax.set_xticklabels([LABELS[k] for k in kinds], fontsize=10)
    ax.set_ylabel("Val MSE (horizon mean)", fontsize=10)
    n = len(data[kinds[0]]["mse"])
    ax.set_title(f"MSE, 95% CI по {n} сидам (t-распределение)", fontsize=11)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.2)
    ax.grid(axis="y", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[plot] {out}")


def plot_asym_bars(data: dict, kinds: list[str], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, kind in enumerate(kinds):
        vals = data[kind]["asym"]
        valid = vals[~np.isnan(vals)]
        if not len(valid):
            continue
        mu = valid.mean()
        half = ci95(valid)
        ax.bar(
            i,
            mu,
            color=COLORS[kind],
            alpha=0.85,
            yerr=half,
            capsize=6,
            error_kw=dict(linewidth=1.5),
        )
        ax.text(i, mu + half + 0.005, f"{mu:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(np.arange(len(kinds)))
    ax.set_xticklabels([LABELS[k] for k in kinds], fontsize=10)
    ax.set_ylabel(r"$\|\hat A - \hat A^\top\|_F\,/\,\|\hat A\|_F$", fontsize=10)
    n = max(len(data[k]["asym"][~np.isnan(data[k]["asym"])]) for k in kinds)
    ax.set_title(f"Асимметрия матрицы связи, 95% CI по {n} сидам", fontsize=11)
    ax.set_ylim(0, max(1.0, ax.get_ylim()[1] * 1.2))
    ax.grid(axis="y", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[plot] {out}")


def plot_horizon_curves(data: dict, kinds: list[str], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for kind in kinds:
        hm = data[kind]["horizon_mse"]
        if hm.ndim < 2 or hm.shape[0] == 0:
            continue
        xs = np.arange(1, hm.shape[1] + 1)
        _band(ax, xs, hm, COLORS[kind], LABELS[kind])
        ax.plot(xs, hm.mean(0), "o", color=COLORS[kind], ms=4)
    ax.set_xlabel("Шаг горизонта", fontsize=10)
    ax.set_ylabel("MSE", fontsize=10)
    ax.set_title("MSE по горизонту, 95% CI", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.35)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[plot] {out}")


def plot_loss_component(key: str, data: dict, kinds: list[str], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    has_data = False
    for kind in kinds:
        mat = _history_matrix(data[kind]["history"], key)
        if mat is None:
            continue
        xs = np.arange(1, mat.shape[1] + 1)
        _band(ax, xs, mat, COLORS[kind], LABELS[kind])
        has_data = True
    if not has_data:
        plt.close(fig)
        return
    ax.set_xlabel("Эпоха", fontsize=10)
    ax.set_ylabel(LOSS_LABELS.get(key, key), fontsize=10)
    ax.set_title(f"Val {LOSS_LABELS.get(key, key)}, 95% CI", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.35)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[plot] {out}")


def plot_training_curve_per_model(
    kind: str, data: dict, seeds: list[int], out: Path
) -> None:
    histories = data[kind]["history"]
    mat = _history_matrix(histories, "fc")
    if mat is None:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    cmap = plt.cm.tab10
    n_kept = 0
    for i, hist in enumerate(histories):
        if not hist:
            continue
        epochs = [h["epoch"] for h in hist]
        vals = [h.get("val_fc", float("nan")) for h in hist]
        ax.plot(
            epochs,
            vals,
            color=cmap(n_kept % 10),
            alpha=0.5,
            linewidth=1.0,
            label=f"seed {seeds[i]}",
        )
        n_kept += 1

    xs = np.arange(mat.shape[1])
    _band(ax, xs, mat, "black", "CI", alpha_fill=0.12)
    ax.set_xlabel("Эпоха", fontsize=10)
    ax.set_ylabel("Val Horizon MSE", fontsize=10)
    ax.set_title(f"Кривая обучения -- {LABELS.get(kind, kind)}", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.35)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[plot] {out}")


def print_table(data: dict, kinds: list[str]) -> None:
    header = (
        f"{'Модель':<18} {'MSE mean':>10} {'CI':>8} " f"{'asym mean':>10} {'CI':>8}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for kind in kinds:
        mse = data[kind]["mse"]
        asym_v = data[kind]["asym"]
        valid_a = asym_v[~np.isnan(asym_v)]
        print(
            f"{LABELS.get(kind, kind):<18} "
            f"{mse.mean():>10.5f} {ci95(mse):>8.5f} "
            f"{valid_a.mean() if len(valid_a) else float('nan'):>10.4f} "
            f"{ci95(valid_a) if len(valid_a) >= 2 else 0.0:>8.4f}"
        )
    print("=" * len(header))
    print("(CI = 95%, Student t, n = number of seeds per cell)")


def save_summary_json(
    data: dict, kinds: list[str], seeds: list[int], out: Path
) -> None:
    summary = {}
    for kind in kinds:
        mse = data[kind]["mse"]
        asym_v = data[kind]["asym"]
        valid_a = asym_v[~np.isnan(asym_v)]
        summary[kind] = {
            "seeds": seeds,
            "n": len(mse),
            "mse_per_seed": mse.tolist(),
            "mse_mean": float(mse.mean()),
            "mse_ci95": float(ci95(mse)),
            "asym_per_seed": [float(a) for a in asym_v],
            "asym_mean": float(valid_a.mean()) if len(valid_a) else None,
            "asym_ci95": float(ci95(valid_a)) if len(valid_a) >= 2 else 0.0,
            "horizon_mse_mean": (
                data[kind]["horizon_mse"].mean(0).tolist()
                if data[kind]["horizon_mse"].ndim == 2
                and data[kind]["horizon_mse"].shape[0] > 0
                else []
            ),
        }
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument(
        "--spatial", nargs="+", default=["correlation", "grand_diff", "grand_full"]
    )
    ap.add_argument("--out_dir", default="results/seeds")
    ap.add_argument("--takens", default="data/takens.npz")
    ap.add_argument("--series", default="data/series.npz")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--gpus",
        default="0",
        help="CUDA_VISIBLE_DEVICES for child processes. "
        "Examples: '0', '0,1', 'none' (don't pin).",
    )
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--n_modes", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--vae_dim", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--lam_kl", type=float, default=1e-4)
    ap.add_argument(
        "--skip_train",
        action="store_true",
        help="Skip training; aggregate existing checkpoints only.",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gpus = None if args.gpus.lower() == "none" else args.gpus

    extra = [
        "--takens",
        args.takens,
        "--series",
        args.series,
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--window",
        str(args.window),
        "--warmup",
        str(args.warmup),
        "--horizon",
        str(args.horizon),
        "--vae_dim",
        str(args.vae_dim),
        "--lr",
        str(args.lr),
        "--lam_kl",
        str(args.lam_kl),
        "--n_modes",
        str(args.n_modes),
        "--device",
        args.device,
    ]

    if not args.skip_train:
        total = len(args.spatial) * len(args.seeds)
        done = 0
        for kind in args.spatial:
            for seed in args.seeds:
                done += 1
                print(f"\n{'-'*60}")
                print(f"[{done}/{total}]  spatial={kind}  seed={seed}")
                print(f"{'-'*60}")
                run_one(kind, seed, out_dir, extra, gpus)
    else:
        print("[skip_train] using existing checkpoints")

    print("\n[aggregate] loading data ...")
    emb = np.load(args.takens)["emb"]
    raw = np.load(args.series, allow_pickle=True)["x"]

    data = collect_metrics(out_dir, args.spatial, args.seeds, args.device, emb, raw)

    print_table(data, args.spatial)
    save_summary_json(data, args.spatial, args.seeds, out_dir / "summary.json")

    plot_mse_bars(data, args.spatial, out_dir / "mse_bars.png")
    plot_asym_bars(data, args.spatial, out_dir / "asym_bars.png")
    plot_horizon_curves(data, args.spatial, out_dir / "horizon_curves.png")

    for key in LOSS_KEYS:
        plot_loss_component(key, data, args.spatial, out_dir / f"loss_{key}.png")

    for kind in args.spatial:
        if data[kind]["history"] and any(data[kind]["history"]):
            plot_training_curve_per_model(
                kind,
                data,
                args.seeds,
                out_dir / f"training_{kind}.png",
            )

    print(f"\n[done] outputs in {out_dir}/")


if __name__ == "__main__":
    main()
