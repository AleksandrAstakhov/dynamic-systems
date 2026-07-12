from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import argparse
_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--mat_dir", default="results/compressed_v2/matrices")
_ap.add_argument("--out_dir", default="results/compressed_v2")
_args, _ = _ap.parse_known_args()
MAT_DIR = _args.mat_dir
OUT_DIR = _args.out_dir

def load(name):
    return np.load(os.path.join(MAT_DIR, f"{name}.npy"))

def asym(M):
    return np.linalg.norm(M - M.T) / (np.linalg.norm(M) + 1e-12)

def corr(A, B):
    return float(np.corrcoef(A.flatten(), B.flatten())[0, 1])

def rank(M, tol=1e-6):
    return int(np.linalg.matrix_rank(M, tol=tol))

T_pos = load("T_pos")
T_neg = load("T_neg")
C = T_pos.shape[0]

matrices = {
    "correlation": {
        "A_pos": load("correlation_A_pos"),
        "A_neg": load("correlation_A_neg"),
        "A_mean": load("correlation_A_mean"),
    },
    "grand_diff": {
        "A_pos": load("grand_diff_A_pos"),
        "A_neg": load("grand_diff_A_neg"),
        "A_mean": load("grand_diff_A_mean"),
        "A_modes": load("grand_diff_A_modes"),
        "w": load("grand_diff_mode_weights").mean(0),
    },
    "grand_full": {
        "A_pos": load("grand_full_A_pos"),
        "A_neg": load("grand_full_A_neg"),
        "A_mean": load("grand_full_A_mean"),
        "A_modes": load("grand_full_A_modes"),
        "w": load("grand_full_mode_weights").mean(0),
    },
}

LABELS = {
    "correlation": "Corr (static)",
    "grand_diff":  "MoE-GA Diff",
    "grand_full":  "MoE-GA Full",
}
SPATIALS = list(matrices.keys())

fig1, axes = plt.subplots(2, 4, figsize=(16, 8))

for row, (regime, T_true) in enumerate([("+V", T_pos), ("-V", T_neg)]):
    key = "A_pos" if regime == "+V" else "A_neg"
    vmax_t = np.abs(T_true).max()

    ax = axes[row, 0]
    im = ax.imshow(T_true, cmap="RdBu_r", vmin=-vmax_t, vmax=vmax_t)
    ax.set_title(f"True T ({regime})\nrank={rank(T_true)}  asym={asym(T_true):.3f}", fontsize=9)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

    for col, sp in enumerate(SPATIALS, start=1):
        A = matrices[sp][key]
        vmax_a = np.abs(A).max()
        ax = axes[row, col]
        im = ax.imshow(A, cmap="RdBu_r", vmin=-vmax_a, vmax=vmax_a)
        cr = corr(A, T_true)
        ax.set_title(
            f"{LABELS[sp]} ({regime})\n"
            f"rank={rank(A)}  asym={asym(A):.3f}  corr(T)={cr:+.3f}",
            fontsize=8,
        )
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)

fig1.suptitle(
    f"Learned coupling matrices vs ground truth  (C={C}, latent_dim=2, m=4)",
    fontsize=12, fontweight="bold"
)
fig1.tight_layout()
fig1.savefig(f"{OUT_DIR}/fig1_regimes.png", dpi=150, bbox_inches="tight")
plt.close(fig1)
print(f"Saved {OUT_DIR}/fig1_regimes.png")

grand_models = [sp for sp in SPATIALS if "A_modes" in matrices[sp]]
n_modes = matrices[grand_models[0]]["A_modes"].shape[0]

fig2, axes2 = plt.subplots(len(grand_models), n_modes + 1, figsize=(5 * (n_modes + 1), 5 * len(grand_models)))
if len(grand_models) == 1:
    axes2 = axes2[np.newaxis, :]

for row, sp in enumerate(grand_models):
    A_modes = matrices[sp]["A_modes"]
    w = matrices[sp]["w"]

    A_sum = sum(w[k] * A_modes[k] for k in range(n_modes))
    vmax = np.abs(A_sum).max()
    ax = axes2[row, 0]
    im = ax.imshow(A_sum, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_title(f"{LABELS[sp]}\nweighted sum\nrank={rank(A_sum)}", fontsize=9)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

    for k in range(n_modes):
        A_k = A_modes[k]
        vmax_k = np.abs(A_k).max()
        ax = axes2[row, k + 1]
        im = ax.imshow(A_k, cmap="RdBu_r", vmin=-vmax_k, vmax=vmax_k)
        ax.set_title(
            f"Mode {k+1}  (w={w[k]:.3f})\n"
            f"rank={rank(A_k)}  asym={asym(A_k):.3f}",
            fontsize=9,
        )
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)

fig2.suptitle("Матрицы экспертов MoE-GA (смесь экспертов, усреднение по val)", fontsize=12, fontweight="bold")
fig2.tight_layout()
fig2.savefig(f"{OUT_DIR}/fig2_modes.png", dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"Saved {OUT_DIR}/fig2_modes.png")

fig3, axes3 = plt.subplots(1, 4, figsize=(16, 4))

T_diff = T_pos - T_neg
vmax_t = np.abs(T_diff).max()
ax = axes3[0]
im = ax.imshow(T_diff, cmap="RdBu_r", vmin=-vmax_t, vmax=vmax_t)
ax.set_title(f"True  T_pos - T_neg\nrank={rank(T_diff)}", fontsize=10)
ax.axis("off")
plt.colorbar(im, ax=ax, fraction=0.046)

for col, sp in enumerate(SPATIALS, start=1):
    A_diff = matrices[sp]["A_pos"] - matrices[sp]["A_neg"]
    vmax_a = max(np.abs(A_diff).max(), 1e-9)
    cr = corr(A_diff, T_diff)
    ax = axes3[col]
    im = ax.imshow(A_diff, cmap="RdBu_r", vmin=-vmax_a, vmax=vmax_a)
    ax.set_title(
        f"{LABELS[sp]}\nA_pos - A_neg\ncorr(DT)={cr:+.3f}",
        fontsize=9,
    )
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

fig3.suptitle("Regime sensitivity: A_pos - A_neg  vs  T_pos - T_neg", fontsize=12, fontweight="bold")
fig3.tight_layout()
fig3.savefig(f"{OUT_DIR}/fig3_regime_diff.png", dpi=150, bbox_inches="tight")
plt.close(fig3)
print(f"Saved {OUT_DIR}/fig3_regime_diff.png")

fig4, axes4 = plt.subplots(1, 2, figsize=(14, 5))

for ax, (regime, T_true, key) in zip(axes4, [("+V", T_pos, "A_pos"), ("-V", T_neg, "A_neg")]):
    row0_T = T_true[0]
    ax.plot(row0_T, "k-", lw=2, label=f"True T ({regime}), row 0")
    for sp in SPATIALS:
        A = matrices[sp][key]
        row0_A = A[0] / (np.abs(A[0]).max() + 1e-9) * np.abs(row0_T).max()
        ax.plot(row0_A, "--", label=f"{LABELS[sp]} (normalised)")
    ax.set_xlabel("Sensor index")
    ax.set_ylabel("Coupling weight")
    ax.set_title(f"First row profile, regime {regime}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

fig4.suptitle("Circulant structure: first row of coupling matrices", fontsize=12, fontweight="bold")
fig4.tight_layout()
fig4.savefig(f"{OUT_DIR}/fig4_profiles.png", dpi=150, bbox_inches="tight")
plt.close(fig4)
print(f"Saved {OUT_DIR}/fig4_profiles.png")

print("\n-- Метрики --------------------------------------------------------------")
print(f"{'':18s} {'rank(A+)':>8} {'rank(A-)':>8} {'asym(A+)':>9} {'corr(T+)':>9} {'corr(T-)':>9} {'Dcorr':>8}")
for sp in SPATIALS:
    A_pos = matrices[sp]["A_pos"]
    A_neg = matrices[sp]["A_neg"]
    print(
        f"{LABELS[sp]:18s} "
        f"{rank(A_pos):>8d} {rank(A_neg):>8d} "
        f"{asym(A_pos):>9.3f} "
        f"{corr(A_pos, T_pos):>9.3f} "
        f"{corr(A_neg, T_neg):>9.3f} "
        f"{corr(A_pos, T_pos) - corr(A_neg, T_neg):>8.3f}"
    )
print(f"\n{'True T_pos':18s} rank={rank(T_pos)}  asym={asym(T_pos):.3f}")
print(f"{'True T_neg':18s} rank={rank(T_neg)}  asym={asym(T_neg):.3f}")
