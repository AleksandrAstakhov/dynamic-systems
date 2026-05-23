from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from models import LatentForecaster
from torch.utils.data import DataLoader, Subset
from train import WindowDataset, empirical_correlation

SPATIAL_KINDS = ["correlation", "grand_diff", "grand_full"]

LABELS = {
    "correlation": "корреляция",
    "grand_diff": "GA, асимм.",
    "grand_full": "GA + reaction",
}


def L(kind: str) -> str:
    return LABELS.get(kind, kind)


def load_model(ckpt_path, device, kind, corr_matrix=None):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ck["args"]
    sk = dict(
        t_max=a["ode_t_max"],
        method=a["ode_method"],
        n_steps=a["ode_steps"],
    )
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
    return model, ck["args"]


def corr_mat(a, b):
    return float(np.corrcoef(a.flatten(), b.flatten())[0, 1])


def asym(M):
    return float(np.linalg.norm(M - M.T) / (np.linalg.norm(M) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--takens", default="data/takens.npz")
    ap.add_argument("--series", default="data/series.npz")
    ap.add_argument("--truth", default="data/series_truth.npz")
    ap.add_argument("--ckpt_dir", default="results")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_json", default="results/comparison.json")
    ap.add_argument("--out_png", default="results/comparison.png")
    args = ap.parse_args()

    tk = np.load(args.takens)
    sr = np.load(args.series, allow_pickle=True)
    emb = tk["emb"]
    raw = sr["x"]
    has_v = "v" in sr.files
    v_arr = sr["v"] if has_v else None
    has_truth = os.path.exists(args.truth)
    T_pos = T_neg = T_avg = None
    if has_truth:
        tr = np.load(args.truth)
        T_pos, T_neg = tr["T_pos"], tr["T_neg"]

        if T_pos.shape[0] != raw.shape[1]:
            print(
                f"[warn] stale truth ignored: T has C={T_pos.shape[0]} "
                f"but data has C={raw.shape[1]}"
            )
            has_truth = False
            T_pos = T_neg = T_avg = None
        else:
            T_avg = 0.5 * (T_pos + T_neg)

    any_ckpt = None
    for kind in SPATIAL_KINDS:
        path = f"{args.ckpt_dir}/{kind}.pt"
        if os.path.exists(path):
            any_ckpt = path
            break
    if any_ckpt is None:
        raise FileNotFoundError("no checkpoints found")
    ck = torch.load(any_ckpt, map_location="cpu", weights_only=False)
    a = ck["args"]
    ds = WindowDataset(emb, raw, a["window"], a["horizon"], a["warmup"])
    cut = int(len(ds) * 0.8)
    va_idx = np.arange(cut, len(ds))
    va_ld = DataLoader(Subset(ds, va_idx.tolist()), batch_size=64, shuffle=False)

    raw_train = raw[: int(raw.shape[0] * 0.8)]
    corr_train = empirical_correlation(raw_train)

    if has_v:
        T_emb = emb.shape[0]
        offset = raw.shape[0] - T_emb
        val_current_t = offset + va_idx + a["window"] - 1
        val_v = v_arr[val_current_t]
        mask_pos = val_v > 0
        mask_neg = val_v < 0
        print(
            f"val samples: {len(val_v)}  (+V: {mask_pos.sum()}, -V: {mask_neg.sum()})"
        )
    else:
        mask_pos = mask_neg = None

    results = {}
    A_per_model_pos = {}
    A_per_model_neg = {}
    mse_per_model = {}

    for kind in SPATIAL_KINDS:
        path = f"{args.ckpt_dir}/{kind}.pt"
        if not os.path.exists(path):
            print(f"[skip] {kind}: no checkpoint at {path}")
            continue
        model, _ = load_model(path, args.device, kind, corr_matrix=corr_train)

        per_sample_A = []
        mse_h = []
        with torch.no_grad():
            for x, x_raw, y in va_ld:
                x = x.to(args.device)
                x_raw = x_raw.to(args.device)
                y = y.to(args.device)
                out = model(x, x_raw)
                A_batch = model.coupling(out["z_last"]).cpu().numpy()
                per_sample_A.append(A_batch)
                mse_h.append(
                    ((out["pred"] - y) ** 2).mean(dim=(0, 2)).cpu().numpy() * x.size(0)
                )
        A_all = np.concatenate(per_sample_A, axis=0)
        mse_h = np.stack(mse_h).sum(0) / len(va_idx)
        mse_per_model[kind] = mse_h
        if has_v:
            A_per_model_pos[kind] = A_all[mask_pos].mean(0) if mask_pos.any() else None
            A_per_model_neg[kind] = A_all[mask_neg].mean(0) if mask_neg.any() else None
        else:
            A_per_model_pos[kind] = A_all.mean(0)
            A_per_model_neg[kind] = None

        entry = {
            "val_mse_horizon": mse_h.tolist(),
            "val_mse_mean": float(mse_h.mean()),
            "A_mean_asym": asym(A_all.mean(0)),
        }
        if has_truth and has_v and A_per_model_pos[kind] is not None:
            entry["asym_per_regime"] = {
                "pos": asym(A_per_model_pos[kind]),
                "neg": asym(A_per_model_neg[kind]),
            }
            entry["corr_with_true"] = {
                "A_pos_vs_T_pos": corr_mat(A_per_model_pos[kind], T_pos),
                "A_pos_vs_T_neg": corr_mat(A_per_model_pos[kind], T_neg),
                "A_neg_vs_T_pos": corr_mat(A_per_model_neg[kind], T_pos),
                "A_neg_vs_T_neg": corr_mat(A_per_model_neg[kind], T_neg),
            }
        results[kind] = entry
        print(
            f"[{kind:14s}] mean MSE = {mse_h.mean():.4f}; "
            f"asym_pos={entry.get('asym_per_regime',{}).get('pos','--')}; "
            f"corr(T+,A+)={entry.get('corr_with_true',{}).get('A_pos_vs_T_pos','--')}"
        )

    if has_truth:
        results["ground_truth"] = dict(
            asym_T_pos=asym(T_pos),
            asym_T_neg=asym(T_neg),
            asym_T_avg=asym(T_avg),
        )

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {args.out_json}")

    n_models = len([k for k in SPATIAL_KINDS if k in mse_per_model])
    fig, ax = plt.subplots(
        3 if has_truth else 2,
        n_models + 1,
        figsize=(4 * (n_models + 1), 9 if has_truth else 6),
    )
    if has_truth:
        for r_i, (regime_name, T_true) in enumerate([("+V", T_pos), ("-V", T_neg)]):
            v = np.max(np.abs(T_true)) + 1e-9
            ax[r_i, 0].imshow(T_true, cmap="RdBu_r", vmin=-v, vmax=v)
            ax[r_i, 0].set_title(
                f"True T ({regime_name})\nasym={asym(T_true):.2f}", fontsize=9
            )
            ax[r_i, 0].axis("off")
            for c_i, kind in enumerate(k for k in SPATIAL_KINDS if k in mse_per_model):
                A = (
                    A_per_model_pos[kind]
                    if regime_name == "+V"
                    else A_per_model_neg[kind]
                )
                if A is None:
                    continue
                v_ = np.max(np.abs(A)) + 1e-9
                ax[r_i, c_i + 1].imshow(A, cmap="RdBu_r", vmin=-v_, vmax=v_)
                ax[r_i, c_i + 1].set_title(
                    f"{L(kind)} ({regime_name})\nasym={asym(A):.2f}, "
                    f"corr(T)={corr_mat(A, T_true):+.2f}",
                    fontsize=8,
                )
                ax[r_i, c_i + 1].axis("off")

        row = 2
        for j in range(n_models + 1):
            ax[row, j].axis("off")

        ax_mse = ax[row, 0]
        ax_mse.axis("on")
        H = len(next(iter(mse_per_model.values())))
        xs = np.arange(H)
        width = 0.25
        for i, kind in enumerate(k for k in SPATIAL_KINDS if k in mse_per_model):
            ax_mse.bar(xs + i * width, mse_per_model[kind], width=width, label=L(kind))
        ax_mse.set_xlabel("horizon step")
        ax_mse.set_ylabel("MSE")
        ax_mse.set_title("Forecast MSE per horizon")
        ax_mse.legend(fontsize=8)
        ax_mse.grid(alpha=0.3)
    else:

        for c_i, kind in enumerate(k for k in SPATIAL_KINDS if k in mse_per_model):
            A = A_per_model_pos[kind]
            v_ = np.max(np.abs(A)) + 1e-9
            ax[0, c_i].imshow(A, cmap="RdBu_r", vmin=-v_, vmax=v_)
            ax[0, c_i].set_title(f"{L(kind)} mean A\nasym={asym(A):.2f}", fontsize=9)
            ax[0, c_i].axis("off")
        ax[0, n_models].axis("off")
        ax_mse = ax[1, 0]
        H = len(next(iter(mse_per_model.values())))
        xs = np.arange(H)
        width = 0.25
        for i, kind in enumerate(k for k in SPATIAL_KINDS if k in mse_per_model):
            ax_mse.bar(xs + i * width, mse_per_model[kind], width=width, label=L(kind))
        ax_mse.set_xlabel("horizon step")
        ax_mse.set_ylabel("MSE")
        ax_mse.set_title("Forecast MSE per horizon")
        ax_mse.legend(fontsize=8)
        ax_mse.grid(alpha=0.3)
        for j in range(1, n_models + 1):
            ax[1, j].axis("off")

    bank_matrices = {}
    for kind in SPATIAL_KINDS:
        if kind not in mse_per_model:
            continue
        path = f"{args.ckpt_dir}/{kind}.pt"
        model_, _ = load_model(path, args.device, kind, corr_matrix=corr_train)
        sp = model_.spatial
        if kind == "correlation":
            A_static = (sp.gain * sp.A_fixed).detach().cpu().numpy()
            bank_matrices[kind] = {
                "A_modes": A_static[None],
                "w_mean": np.array([1.0]),
            }
            continue
        if not hasattr(sp, "attn_mod"):
            continue
        attn_mod = sp.attn_mod
        d_head = attn_mod.d_head
        z_list, w_list = [], []
        with torch.no_grad():
            for x, x_raw, y in va_ld:
                x = x.to(args.device)
                x_raw = x_raw.to(args.device)
                out = model_(x, x_raw)
                z = out["z_last"]
                w = attn_mod.phase_weights(z)
                z_list.append(z)
                w_list.append(w)
            z_all = torch.cat(z_list, dim=0)
            w_all = torch.cat(w_list, dim=0)
            Q = torch.einsum("bcl,kld->bkcd", z_all, attn_mod.W_q)
            K_ = torch.einsum("bcl,kld->bkcd", z_all, attn_mod.W_k)
            per_mode = torch.einsum("bkid,bkjd->bkij", Q, K_) / np.sqrt(d_head)
            A_modes_mean = per_mode.mean(dim=0).cpu().numpy()
            w_mean = w_all.mean(dim=0).cpu().numpy()
        bank_matrices[kind] = {"A_modes": A_modes_mean, "w_mean": w_mean}

    if bank_matrices:
        max_K = max(b["A_modes"].shape[0] for b in bank_matrices.values())
        n_rows = len(bank_matrices)
        fig_bank, axb = plt.subplots(
            n_rows,
            max_K,
            figsize=(3.2 * max_K, 3.2 * n_rows),
            squeeze=False,
        )
        for r, (kind, info) in enumerate(bank_matrices.items()):
            A_modes = info["A_modes"]
            w_mean = info["w_mean"]
            K_modes = A_modes.shape[0]
            for c in range(max_K):
                a = axb[r, c]
                if c < K_modes:
                    M = A_modes[c]
                    v_ = float(np.max(np.abs(M))) + 1e-9
                    a.imshow(M, cmap="RdBu_r", vmin=-v_, vmax=v_)
                    if K_modes == 1:
                        title = f"{L(kind)}\nasym={asym(M):.2f}"
                    else:
                        title = (
                            f"{L(kind)} mode {c+1}/{K_modes}\n"
                            f"w_mean={w_mean[c]:.3f}, asym={asym(M):.2f}"
                        )
                    a.set_title(title, fontsize=8)
                a.axis("off")
        fig_bank.suptitle(
            "Bank matrices per model (averaged over val set)", fontsize=10
        )
        fig_bank.tight_layout()
        bank_path = args.out_png.replace(".png", "_bank.png")
        fig_bank.savefig(bank_path, dpi=120)
        print(f"wrote {bank_path}")

    fig.tight_layout()
    fig.savefig(args.out_png, dpi=130)
    print(f"wrote {args.out_png}")


if __name__ == "__main__":
    main()
