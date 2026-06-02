
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

sys.path.insert(0, os.path.dirname(__file__))

from data_gen import make_T, simulate
from models import LatentForecaster
from takens_global import (
    ami_one_sensor,
    embed_field,
    first_local_min,
    fnn_fraction_global,
)

class WindowDataset(Dataset):
    def __init__(self, emb, raw, window, horizon, warmup, v_arr=None):
        T_emb = emb.shape[0]
        offset = raw.shape[0] - T_emb
        self.emb = emb.astype(np.float32)
        self.raw = raw[offset:].astype(np.float32)
        self.window = window
        self.horizon = horizon
        self.warmup = warmup
        self.N = T_emb - window - horizon + 1
        if v_arr is not None:
            v_align = v_arr[offset:]
            self.regime = (v_align > 0).astype(np.float32)
        else:
            self.regime = None

    def __len__(self):
        return self.N

    def __getitem__(self, i):
        x = self.emb[i : i + self.window]
        x_raw = self.raw[i : i + self.window]
        y = self.raw[i + self.window : i + self.window + self.horizon]
        regime = float(self.regime[i + self.window - 1]) if self.regime is not None else 0.0
        return torch.from_numpy(x), torch.from_numpy(x_raw), torch.from_numpy(y), torch.tensor(regime)




def empirical_correlation(x):
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x, dtype=torch.float32)
    x = x - x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True) + 1e-8
    return (x.T @ x) / x.shape[0] / (std.T @ std)


def _base(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model




def run_epoch(model, loader, opt, device, lam_rec, lam_kl, lam_aux, lam_in, lam_phase, train):
    model.train(train)
    ctx = torch.enable_grad() if train else torch.no_grad()
    inner = _base(model)
    tot = dict(L=0.0, fc=0.0, inw=0.0, rec=0.0, kl=0.0, aux=0.0, phase=0.0, n=0)
    with ctx:
        for x, x_raw, y, regime in loader:
            x = x.to(device, non_blocking=True)
            x_raw = x_raw.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            regime = regime.to(device, non_blocking=True)
            out = model(x, x_raw)
            fc = F.mse_loss(out["pred"], y)
            inw = F.mse_loss(out["in_window_pred"], out["in_window_target"])
            rec, kl = inner.encoder.losses(out, beta=1.0)
            z = out["z"]
            B, Tp = z.shape[0], z.shape[1]
            aux = inner.spatial.aux_loss(z.reshape(B * Tp, *z.shape[2:]))
            phase_loss = torch.zeros((), device=device)
            if out["phase_logit"] is not None:
                phase_loss = F.binary_cross_entropy_with_logits(out["phase_logit"], regime)
            total = (fc + lam_in * inw + lam_rec * rec + lam_kl * kl
                     + lam_aux * aux + lam_phase * phase_loss)
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
            tot["phase"] += float(phase_loss) * bs
            tot["n"] += bs
    n = max(tot["n"], 1)
    return {k: v / n for k, v in tot.items() if k != "n"}


def per_horizon_mse(model, loader, device, H):
    mse_h = torch.zeros(H)
    n = 0
    with torch.no_grad():
        for x, x_raw, y, _ in loader:
            x, x_raw, y = x.to(device), x_raw.to(device), y.to(device)
            p = model(x, x_raw)["pred"].cpu()
            mse_h += ((p - y.cpu()) ** 2).mean(dim=(0, 2)) * x.size(0)
            n += x.size(0)
    return (mse_h / n).numpy()


def train_one(
    spatial_kind,
    m_embed,
    latent_dim,
    C,
    tr_ld,
    va_ld,
    corr_train,
    args,
    out_path,
):
    sk = {}
    if spatial_kind in ("grand_diff", "grand_full"):
        sk = dict(d_head=latent_dim, n_modes=args.n_modes, untied=True)

    model = LatentForecaster(
        m_embed=m_embed,
        latent_dim=latent_dim,
        num_channels=C,
        warmup=args.warmup,
        horizon=args.horizon,
        spatial_kind=spatial_kind,
        vae_dim=args.vae_dim,
        vae_blocks=args.vae_blocks,
        vae_spatial_blocks=args.vae_spatial_blocks,
        temporal_blocks=args.temporal_blocks,
        spatial_kwargs=sk,
        corr_matrix=corr_train,
        deterministic_encoder=args.deterministic_encoder,
        use_phase_loss=True,
    ).to(args.device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=1e-5
    )

    best = float("inf")
    history = []
    for ep in range(args.epochs):

        gumbel_tau = max(0.2, 1.0 - (ep / max(args.epochs - 1, 1)) * 0.8)
        for m in model.modules():
            if hasattr(m, "temperature"):
                m.temperature = gumbel_tau

        tr = run_epoch(
            model, tr_ld, opt, args.device,
            args.lam_rec, args.lam_kl, args.lam_aux, args.lam_in, args.lam_phase, True,
        )
        va = run_epoch(
            model, va_ld, opt, args.device,
            args.lam_rec, args.lam_kl, args.lam_aux, args.lam_in, args.lam_phase, False,
        )
        sched.step()
        history.append({
            "epoch": ep,
            "gumbel_tau": gumbel_tau,

            "train_fc": tr["fc"],   "val_fc":  va["fc"],

            "train_inw": tr["inw"], "val_inw": va["inw"],

            "train_rec": tr["rec"], "val_rec": va["rec"],

            "train_kl":  tr["kl"],  "val_kl":  va["kl"],

            "train_aux": tr["aux"], "val_aux": va["aux"],

            "train_phase": tr["phase"], "val_phase": va["phase"],

            "train_total": tr["L"],  "val_total": va["L"],
        })
        print(
            f"  [{spatial_kind}] ep {ep:3d}  tau={gumbel_tau:.2f}  "
            f"train fc={tr['fc']:.5f} phase={tr['phase']:.4f}  |  "
            f"val fc={va['fc']:.5f}",
            flush=True,
        )
        if va["fc"] < best:
            best = va["fc"]
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            torch.save(
                {
                    "state_dict": _base(model).state_dict(),
                    "args": vars(args),
                    "m": m_embed,
                    "latent": latent_dim,
                    "num_channels": C,
                    "spatial_kind": spatial_kind,
                },
                out_path,
            )

    ck = torch.load(out_path, map_location=args.device, weights_only=False)
    _base(model).load_state_dict(ck["state_dict"])
    mse_h = per_horizon_mse(model, va_ld, args.device, args.horizon)

    metrics = dict(
        spatial=spatial_kind,
        best_val_fc=float(best),
        latent_dim=latent_dim,
        m_embed=m_embed,
        compression_ratio=float(latent_dim / m_embed),
        per_horizon_mse=mse_h.tolist(),
        history=history,
    )
    with open(out_path.replace(".pt", "_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(
        f"  [{spatial_kind}] best val fc={best:.5f} | "
        f"latent_dim={latent_dim}/{m_embed} "
        f"(сжатие {latent_dim/m_embed:.1%})"
    )
    return model, best, mse_h, history


def extract_matrices(model, va_ld, device, out_dir, spatial_kind, v_arr, val_idx, warmup, window):
   
    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    A_all, w_all = [], []
    with torch.no_grad():
        for x, x_raw, y, _ in va_ld:
            x, x_raw = x.to(device), x_raw.to(device)
            out = model(x, x_raw)
            A_last = out["A_last"].cpu().numpy()
            A_all.append(A_last)
            if hasattr(_base(model).spatial, "attn_mod"):
                attn_mod = _base(model).spatial.attn_mod
                z = out["z_last"]
                w = attn_mod.phase_weights(z).cpu().numpy()
                w_all.append(w)

    A_all = np.concatenate(A_all, axis=0)
    A_mean = A_all.mean(axis=0)

    np.save(os.path.join(out_dir, f"{spatial_kind}_A_mean.npy"), A_mean)
    np.save(os.path.join(out_dir, f"{spatial_kind}_A_all.npy"), A_all)
    print(f"  [{spatial_kind}] A_mean saved: shape={A_mean.shape}")


    if v_arr is not None:

        mid = window // 2
        v_center = v_arr[val_idx + mid]
        mask_pos = v_center > 0
        mask_neg = v_center < 0
        if mask_pos.sum() > 0:
            A_pos = A_all[mask_pos].mean(axis=0)
            np.save(os.path.join(out_dir, f"{spatial_kind}_A_pos.npy"), A_pos)
        if mask_neg.sum() > 0:
            A_neg = A_all[mask_neg].mean(axis=0)
            np.save(os.path.join(out_dir, f"{spatial_kind}_A_neg.npy"), A_neg)
        print(f"  [{spatial_kind}] +V samples={mask_pos.sum()}, -V samples={mask_neg.sum()}")


    if w_all:
        w_all = np.concatenate(w_all, axis=0)
        np.save(os.path.join(out_dir, f"{spatial_kind}_mode_weights.npy"), w_all)
        attn_mod = _base(model).spatial.attn_mod
        d_head = attn_mod.d_head

        z_list = []
        with torch.no_grad():
            for x, x_raw, _, _ in va_ld:
                x, x_raw = x.to(device), x_raw.to(device)
                z_list.append(model(x, x_raw)["z_last"])
        z_cat = torch.cat(z_list, dim=0)
        with torch.no_grad():
            Q = torch.einsum("bcl,kld->bkcd", z_cat, attn_mod.W_q)
            K_ = torch.einsum("bcl,kld->bkcd", z_cat, attn_mod.W_k)
            per_mode = (torch.einsum("bkid,bkjd->bkij", Q, K_) / (d_head ** 0.5))
            A_modes = per_mode.mean(dim=0).cpu().numpy()
        np.save(os.path.join(out_dir, f"{spatial_kind}_A_modes.npy"), A_modes)
        print(f"  [{spatial_kind}] A_modes saved: shape={A_modes.shape}, "
              f"mean weights={w_all.mean(0).round(3)}")

    return A_mean


def generate_data(args):
    print("=== Генерация drift_ring данных ===")
    x, v_arr = simulate(
        C=args.C, T=args.T, regime_len=args.regime_len,
        v_amp=args.v_amp, ell=args.ell, alpha=args.alpha,
        sigma=args.sigma, burn_in=200, seed=args.seed,
    )
    print(f"  x shape: {x.shape}, v_arr: {v_arr.shape}")
    return x, v_arr


def compute_takens(x, max_lag=40, m_max=8, fnn_thresh=0.02):
    print("=== Takens-вложение ===")
    T, C = x.shape
    amis = np.stack([ami_one_sensor(x[:, c], max_lag) for c in range(C)], axis=0)
    ami_avg = amis.mean(axis=0)
    tau = first_local_min(ami_avg)
    print(f"  global tau = {tau}")

    fnns = []
    m_sel = None
    for m in range(1, m_max + 1):
        frac = fnn_fraction_global(x, tau, m)
        fnns.append(frac)
        print(f"  m={m}: FNN={frac:.4f}")
        if frac < fnn_thresh:
            m_sel = m
            break

    if m_sel is None:
        m_sel = int(np.argmin(fnns)) + 1
        print(f"  [!] FNN не упал ниже {fnn_thresh} -- "
              f"выбран m при минимуме FNN={fnns[m_sel-1]:.4f}")

    print(f"  выбрано: tau={tau}, m={m_sel}")
    emb = embed_field(x, tau, m_sel)
    print(f"  emb shape: {emb.shape}")
    return emb, tau, m_sel




def run_one_seed(seed: int, args, base_dir: str) -> dict:

    seed_dir = os.path.join(base_dir, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)


    print(f"\n{'#'*60}")
    print(f"### SEED {seed} ###")
    print(f"{'#'*60}")
    x, v_arr = generate_data(args)
    T, C = x.shape

    emb, tau, m_sel = compute_takens(x, args.max_lag, args.m_max, args.fnn_thresh)

    if args.latent_dim == 0:
        latent_dim = max(2, round(m_sel * 2 / 3))
    else:
        latent_dim = args.latent_dim
    if latent_dim >= m_sel:
        latent_dim = max(1, m_sel - 1)

    print(f"\n  Сжатие: m={m_sel} -> latent_dim={latent_dim} "
          f"({latent_dim/m_sel:.1%} от исходной размерности)")
    print(f"  n_modes={args.n_modes}, d_head={latent_dim} "
          f"-> max rank(A)={args.n_modes * latent_dim}\n")

    ds = WindowDataset(emb, x, args.window, args.horizon, args.warmup, v_arr=v_arr)
    cut = int(len(ds) * 0.8)
    pin = args.device == "cuda"
    tr_ld = DataLoader(
        Subset(ds, list(range(cut))),
        batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=0, pin_memory=pin,
    )
    va_ld = DataLoader(
        Subset(ds, list(range(cut, len(ds)))),
        batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=pin,
    )
    val_idx = np.array(list(range(cut, len(ds))))

    raw_train = x[: int(T * 0.8)]
    corr_train = empirical_correlation(raw_train)

    T_pos = make_T(+args.v_amp, C, args.ell, args.alpha)
    T_neg = make_T(-args.v_amp, C, args.ell, args.alpha)

    mat_dir = os.path.join(seed_dir, "matrices")
    os.makedirs(mat_dir, exist_ok=True)
    np.save(os.path.join(mat_dir, "T_pos.npy"), T_pos)
    np.save(os.path.join(mat_dir, "T_neg.npy"), T_neg)

    seed_results = {}
    for kind in args.spatials:
        print(f"\n{'='*60}")
        print(f"=== Seed {seed} | Обучение: {kind} ===")
        print(f"{'='*60}")
        out_path = os.path.join(seed_dir, f"{kind}.pt")
        model, best_fc, mse_h, history = train_one(
            kind, m_sel, latent_dim, C,
            tr_ld, va_ld, corr_train, args, out_path,
        )
        A_mean = extract_matrices(
            model, va_ld, args.device, mat_dir,
            kind, v_arr, val_idx, args.window, args.warmup,
        )
        seed_results[kind] = dict(
            seed=seed,
            best_val_fc=best_fc,
            per_horizon_mse=mse_h.tolist(),
            A_asym=float(
                np.linalg.norm(A_mean - A_mean.T) / (np.linalg.norm(A_mean) + 1e-12)
            ),
            history=history,
            tau=tau,
            m_embed=m_sel,
            latent_dim=latent_dim,
        )

    seed_report_path = os.path.join(seed_dir, "seed_summary.json")
    with open(seed_report_path, "w") as f:
        json.dump(seed_results, f, indent=2)
    print(f"  [seed {seed}] Сохранено: {seed_report_path}")

    return seed_results


def aggregate_seeds(all_results: list[dict], spatials: list[str]) -> dict:

    agg = {}
    for kind in spatials:
        kind_results = [r[kind] for r in all_results if kind in r]
        if not kind_results:
            continue

        val_fcs = np.array([r["best_val_fc"] for r in kind_results])
        asyms = np.array([r["A_asym"] for r in kind_results])
        mse_h_list = np.array([r["per_horizon_mse"] for r in kind_results])  # [S, H]

        max_ep = max(len(r["history"]) for r in kind_results)
        hist_keys = [k for k in kind_results[0]["history"][0].keys() if k != "epoch"]
        history_agg = {}
        for hk in hist_keys:
            mat = np.full((len(kind_results), max_ep), np.nan)
            for s_idx, r in enumerate(kind_results):
                for ep_rec in r["history"]:
                    ep = ep_rec["epoch"]
                    mat[s_idx, ep] = ep_rec[hk]
            history_agg[hk] = {
                "mean": np.nanmean(mat, axis=0).tolist(),
                "std":  np.nanstd(mat,  axis=0).tolist(),
            }

        agg[kind] = dict(
            n_seeds=len(kind_results),
            val_fc_mean=float(val_fcs.mean()),
            val_fc_std=float(val_fcs.std()),
            val_fc_seeds=val_fcs.tolist(),
            A_asym_mean=float(asyms.mean()),
            A_asym_std=float(asyms.std()),
            per_horizon_mse_mean=mse_h_list.mean(0).tolist(),
            per_horizon_mse_std=mse_h_list.std(0).tolist(),
            per_horizon_mse_seeds=mse_h_list.tolist(),
            history=history_agg,
        )
    return agg


def main():
    ap = argparse.ArgumentParser(
        description="Эксперимент с реальным сжатием VAE (latent_dim < m), multi-seed"
    )
    ap.add_argument("--C", type=int, default=16)
    ap.add_argument("--T", type=int, default=3000)
    ap.add_argument("--regime_len", type=int, default=500)
    ap.add_argument("--v_amp", type=float, default=2.0)
    ap.add_argument("--ell", type=float, default=0.7)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--sigma", type=float, default=0.30)

    ap.add_argument("--max_lag", type=int, default=40)
    ap.add_argument("--m_max", type=int, default=25,
                    help="Максимальная размерность вложения Такенса для FNN")
    ap.add_argument("--fnn_thresh", type=float, default=0.005)

    ap.add_argument(
        "--latent_dim", type=int, default=0,
        help="0 = авто round(m*2/3); иначе фиксированное значение"
    )

    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--window", type=int, default=24)
    ap.add_argument("--warmup", type=int, default=6)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--vae_dim", type=int, default=32)
    ap.add_argument("--vae_blocks", type=int, default=2)
    ap.add_argument("--vae_spatial_blocks", type=int, default=1)
    ap.add_argument("--temporal_blocks", type=int, default=1)
    ap.add_argument("--deterministic_encoder", action="store_true", default=True)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--lam_rec", type=float, default=0.5)
    ap.add_argument("--lam_kl", type=float, default=0.0)
    ap.add_argument("--lam_aux", type=float, default=0.05)
    ap.add_argument("--lam_in", type=float, default=1.0)
    ap.add_argument("--lam_phase", type=float, default=0.3)
    ap.add_argument("--n_modes", type=int, default=6)

    ap.add_argument("--n_seeds", type=int, default=5,
                    help="Число сидов (0, 1, ..., n_seeds-1)")
    ap.add_argument("--seed", type=int, default=None,
                    help="Запустить только один сид (legacy-совместимость)")
    ap.add_argument(
        "--spatials", nargs="+",
        default=["correlation", "grand_diff", "grand_full"],
    )
    ap.add_argument("--results_dir", default="results/compressed")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)


    if args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = list(range(args.n_seeds))

    print(f"Запуск эксперимента: seeds={seeds}, "
          f"spatials={args.spatials}, epochs={args.epochs}")

    all_results = []
    for seed in seeds:
        seed_res = run_one_seed(seed, args, args.results_dir)
        all_results.append(seed_res)

    agg = aggregate_seeds(all_results, args.spatials)


    print(f"\n{'='*60}")
    print("=== Итог (все сиды) ===")
    for kind, s in agg.items():
        print(f"  [{kind:14s}]  "
              f"val_fc = {s['val_fc_mean']:.5f} +- {s['val_fc_std']:.5f}  "
              f"asym(A) = {s['A_asym_mean']:.3f} +- {s['A_asym_std']:.3f}")


    report = dict(
        seeds=seeds,
        n_seeds=len(seeds),
        spatials=args.spatials,
        epochs=args.epochs,
        C=args.C,
        T=args.T,
        models=agg,
    )
    report_path = os.path.join(args.results_dir, "summary.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n  Результаты: {args.results_dir}/")
    print(f"  Отчёт:      {report_path}")
    print(f"\n  Для построения графиков:")
    print(f"    python plot_training.py --results_dir {args.results_dir}")
    print(f"    python visualize_matrices.py --mat_dir {args.results_dir}/seed_0/matrices "
          f"--out_dir {args.results_dir}")


if __name__ == "__main__":
    main()
