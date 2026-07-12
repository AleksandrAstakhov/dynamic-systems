
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DISPLAY = {
    "correlation": "Corr",
    "grand_diff":  "MoE-GA Diff",
    "grand_full":  "MoE-GA Full",
}
COLORS = {
    "correlation": "#2196F3",
    "grand_diff":  "#FF9800",
    "grand_full":  "#4CAF50",
}
MARKERS = {
    "correlation": "o",
    "grand_diff":  "s",
    "grand_full":  "^",
}


def label(kind: str) -> str:
    return DISPLAY.get(kind, kind)


def color(kind: str) -> str:
    return COLORS.get(kind, "#888888")


def load_summary(results_dir: str) -> dict:
    path = os.path.join(results_dir, "summary.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Не найден {path}. Убедитесь, что эксперимент завершён."
        )
    with open(path) as f:
        return json.load(f)

def _curve(data: dict, key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    mean = np.array(data[key]["mean"])
    std  = np.array(data[key]["std"])
    epochs = np.arange(len(mean))
    return epochs, mean, std


def fill_between(ax, x, mean, std, color, alpha=0.15):
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=alpha)

def plot_val_fc(models: dict, out_path: str):
    fig, ax = plt.subplots(figsize=(8, 5))

    for kind, data in models.items():
        if "history" not in data:
            continue
        hist = data["history"]
        if "val_fc" not in hist:
            continue
        ep, mean, std = _curve(hist, "val_fc")
        c = color(kind)
        ax.plot(ep, mean, color=c, lw=2, label=label(kind))
        fill_between(ax, ep, mean, std, c)

    ax.set_xlabel("Эпоха", fontsize=12)
    ax.set_ylabel("Val forecast MSE", fontsize=12)
    ax.set_title("Ошибка прогноза (val) vs эпоха", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")

LOSS_COMPONENTS = [
    ("train_fc",    "val_fc",    "Forecast MSE"),
    ("train_inw",   "val_inw",   "In-window MSE"),
    ("train_rec",   "val_rec",   "Reconstruction"),
    ("train_phase", "val_phase", "Phase BCE"),
    ("train_aux",   "val_aux",   "Mode entropy"),
    ("train_total", "val_total", "Total loss"),
]


def plot_loss_components(models: dict, out_path: str):
    n_models = len(models)
    n_components = len(LOSS_COMPONENTS)
    fig, axes = plt.subplots(
        n_components, n_models,
        figsize=(5 * n_models, 3 * n_components),
        squeeze=False,
    )

    for col, (kind, data) in enumerate(models.items()):
        hist = data.get("history", {})
        c = color(kind)

        for row, (tr_key, va_key, comp_name) in enumerate(LOSS_COMPONENTS):
            ax = axes[row, col]
            plotted = False

            if tr_key in hist:
                ep, m, s = _curve(hist, tr_key)
                ax.plot(ep, m, color=c, lw=2, linestyle="-", label="train")
                fill_between(ax, ep, m, s, c, alpha=0.12)
                plotted = True
            if va_key in hist:
                ep, m, s = _curve(hist, va_key)
                ax.plot(ep, m, color=c, lw=2, linestyle="--", label="val")
                fill_between(ax, ep, m, s, c, alpha=0.12)
                plotted = True

            if row == 0:
                ax.set_title(label(kind), fontsize=11, fontweight="bold", color=c)
            if col == 0:
                ax.set_ylabel(comp_name, fontsize=9)
            if row == n_components - 1:
                ax.set_xlabel("Эпоха", fontsize=9)
            if plotted:
                ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

    fig.suptitle("Компоненты потери (mean +- std по сидам)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_horizon(models: dict, out_path: str):
    fig, ax = plt.subplots(figsize=(9, 5))

    for kind, data in models.items():
        if "per_horizon_mse_mean" not in data:
            continue
        mean = np.array(data["per_horizon_mse_mean"])
        std  = np.array(data.get("per_horizon_mse_std", np.zeros_like(mean)))
        h = np.arange(1, len(mean) + 1)
        c = color(kind)
        ax.plot(h, mean, color=c, lw=2, marker=MARKERS.get(kind, "o"),
                markersize=4, label=label(kind))
        fill_between(ax, h, mean, std, c)

    ax.axhline(1.0, color="gray", lw=1, linestyle=":", alpha=0.6, label="baseline=1")
    ax.set_xlabel("Горизонт прогноза (шагов)", fontsize=12)
    ax.set_ylabel("MSE", fontsize=12)
    ax.set_title("Ошибка прогноза vs горизонт (mean +- std)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")



def plot_train_val_gap(models: dict, out_path: str):
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 4), squeeze=False)

    for col, (kind, data) in enumerate(models.items()):
        ax = axes[0, col]
        hist = data.get("history", {})
        c = color(kind)

        if "train_fc" in hist and "val_fc" in hist:
            ep_tr, tr_mean, tr_std = _curve(hist, "train_fc")
            ep_va, va_mean, va_std = _curve(hist, "val_fc")

            ax.plot(ep_tr, tr_mean, color=c, lw=2, label="train")
            fill_between(ax, ep_tr, tr_mean, tr_std, c, alpha=0.15)
            ax.plot(ep_va, va_mean, color=c, lw=2, linestyle="--", label="val")
            fill_between(ax, ep_va, va_mean, va_std, c, alpha=0.15)

        ax.set_title(label(kind), fontsize=12, fontweight="bold", color=c)
        ax.set_xlabel("Эпоха", fontsize=10)
        ax.set_ylabel("Forecast MSE", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Train vs Val forecast error (mean +- std по сидам)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_summary_bar(models: dict, out_path: str):
    kinds = list(models.keys())
    means = [models[k]["val_fc_mean"] for k in kinds]
    stds  = [models[k]["val_fc_std"]  for k in kinds]
    colors_list = [color(k) for k in kinds]
    labels_list = [label(k) for k in kinds]

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(kinds))
    bars = ax.bar(x, means, yerr=stds, capsize=6, color=colors_list,
                  alpha=0.85, edgecolor="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_list, fontsize=12)
    ax.set_ylabel("Val forecast MSE", fontsize=12)
    ax.set_title("Лучший val MSE (mean +- std по сидам)", fontsize=13, fontweight="bold")

    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + 0.002,
                f"{mean:.4f}", ha="center", va="bottom", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Построение графиков обучения")
    ap.add_argument("--results_dir", default="results/compressed",
                    help="Директория с summary.json")
    ap.add_argument("--out_dir", default=None,
                    help="Куда сохранять графики (по умолчанию = results_dir)")
    args = ap.parse_args()

    out_dir = args.out_dir or args.results_dir
    os.makedirs(out_dir, exist_ok=True)

    summary = load_summary(args.results_dir)
    models = summary.get("models", {})

    if not models:
        print("summary.json не содержит данных моделей (models: {}).")
        return

    n_seeds = summary.get("n_seeds", "?")
    print(f"\nЗагружено: {len(models)} моделей, сидов={n_seeds}")
    print(f"Сохраняем графики в: {out_dir}/\n")


    print(f"{'Модель':14s} {'val_fc mean':>12s} {'val_fc std':>11s} {'asym mean':>10s}")
    print("-" * 50)
    for kind, data in models.items():
        print(f"{label(kind):14s} "
              f"{data.get('val_fc_mean', float('nan')):12.5f} "
              f"{data.get('val_fc_std',  float('nan')):11.5f} "
              f"{data.get('A_asym_mean', float('nan')):10.3f}")

    plot_val_fc(models,          os.path.join(out_dir, "fig_val_fc.png"))
    plot_loss_components(models, os.path.join(out_dir, "fig_loss_components.png"))
    plot_horizon(models,         os.path.join(out_dir, "fig_horizon.png"))
    plot_train_val_gap(models,   os.path.join(out_dir, "fig_train_val.png"))
    plot_summary_bar(models,     os.path.join(out_dir, "fig_summary_bar.png"))

    print(f"\nВсе графики сохранены в {out_dir}/")


if __name__ == "__main__":
    main()
