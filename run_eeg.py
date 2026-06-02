
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

sys.path.insert(0, os.path.dirname(__file__))

from models import LatentForecaster
from takens_global import (
    ami_one_sensor,
    embed_field,
    first_local_min,
    fnn_fraction_global,
)


def load_data(path: str, sfreq_hint: float | None = None) -> tuple[np.ndarray, float, list[str]]:
    ext = os.path.splitext(path)[1].lower()

    if ext == ".npy":
        x = np.load(path).astype(np.float32)
        if x.ndim == 1:
            x = x[:, None]
        assert x.ndim == 2, f"Ожидается (T, C), получено {x.shape}"
        sfreq = sfreq_hint or 1.0
        ch_names = [f"ch{i}" for i in range(x.shape[1])]
        return x, sfreq, ch_names

    if ext == ".npz":
        d = np.load(path, allow_pickle=True)
        key = "x" if "x" in d else ("data" if "data" in d else list(d.keys())[0])
        x = d[key].astype(np.float32)
        if x.ndim == 1:
            x = x[:, None]
        if "sfreq" in d:
            sfreq = float(d["sfreq"])
        elif "dt" in d:
            sfreq = 1.0 / float(d["dt"])
        else:
            sfreq = sfreq_hint or 1.0
        ch_names = list(d["ch_names"]) if "ch_names" in d else [f"ch{i}" for i in range(x.shape[1])]
        return x, sfreq, ch_names

    try:
        import mne
        mne.set_log_level("WARNING")
    except ImportError:
        raise ImportError(
            f"Файл '{path}' требует MNE: pip install mne"
        )

    if ext in (".edf", ".bdf"):
        raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    elif ext == ".fif":
        raw = mne.io.read_raw_fif(path, preload=True, verbose=False)
    elif ext == ".set":
        raw = mne.io.read_raw_eeglab(path, preload=True, verbose=False)
    else:
        raise ValueError(f"Неизвестный формат файла: {ext}")

    sfreq = raw.info["sfreq"]
    ch_names = raw.ch_names
    x = raw.get_data().T.astype(np.float32)  # (T, C)
    return x, sfreq, ch_names


def bandpass_filter(x: np.ndarray, sfreq: float,
                    l_freq: float = 0.5, h_freq: float = 80.0,
                    notch_freq: float = 50.0) -> np.ndarray:
    try:
        from scipy import signal as sp_signal
    except ImportError:
        warnings.warn("scipy не найден -- фильтрация пропущена.")
        return x

    nyq = sfreq / 2.0
    if h_freq >= nyq:
        h_freq = nyq * 0.95
    b, a = sp_signal.butter(4, [l_freq / nyq, h_freq / nyq], btype="band")
    x = sp_signal.filtfilt(b, a, x, axis=0).astype(np.float32)

    if 0 < notch_freq < nyq:
        b, a = sp_signal.iirnotch(notch_freq / nyq, Q=30.0)
        x = sp_signal.filtfilt(b, a, x, axis=0).astype(np.float32)

    return x


def normalize(x: np.ndarray, x_train: np.ndarray | None = None) -> np.ndarray:
    ref = x_train if x_train is not None else x
    mu = ref.mean(0, keepdims=True)
    sd = ref.std(0, keepdims=True) + 1e-8
    return ((x - mu) / sd).astype(np.float32)


def estimate_tau(x: np.ndarray, max_lag: int = 60) -> tuple[int, np.ndarray]:
    C = x.shape[1]
    amis = np.stack([ami_one_sensor(x[:, c], max_lag) for c in range(C)])
    ami_avg = amis.mean(0)
    tau = first_local_min(ami_avg)
    return tau, ami_avg


def estimate_m(x: np.ndarray, tau: int,
               fnn_thresh: float = 0.005, m_max: int = 10) -> tuple[int, list[float]]:
    fnns = []
    m_sel = None
    for m in range(1, m_max + 1):
        frac = fnn_fraction_global(x, tau, m)
        fnns.append(frac)
        if frac < fnn_thresh:
            m_sel = m
            break
    if m_sel is None:
        m_sel = int(np.argmin(fnns)) + 1
    return m_sel, fnns


def estimate_n_modes_mp(x_train: np.ndarray) -> tuple[int, np.ndarray, float]:
    T, C = x_train.shape
    xc = x_train - x_train.mean(0)
    xc /= xc.std(0) + 1e-8
    R = (xc.T @ xc) / T  # (C, C)
    eigvals = np.sort(np.linalg.eigvalsh(R))[::-1]

    ac1 = float(np.mean([np.corrcoef(xc[:-1, c], xc[1:, c])[0, 1] for c in range(min(C, 16))]))
    ac1 = np.clip(ac1, 0.0, 0.99)
    T_eff = max(C + 1, int(T * (1 - ac1) / (1 + ac1)))

    gamma = C / T_eff
    lambda_plus = (1 + np.sqrt(gamma)) ** 2
    n_signal = int((eigvals > lambda_plus).sum())
    n_mp = max(2, min(20, n_signal))
    return n_mp, eigvals, lambda_plus


def estimate_n_modes_bands(x_train: np.ndarray, sfreq: float) -> tuple[int, dict[str, float]]:
    try:
        from scipy import signal as sp_signal
    except ImportError:
        return 5, {"unknown": 1.0}

    nyq = sfreq / 2.0
    nperseg = min(int(sfreq * 4), len(x_train) // 4, 1024)
    nperseg = max(nperseg, 32)
    f, psd = sp_signal.welch(x_train.T, fs=sfreq, nperseg=nperseg)
    psd_mean = psd.mean(0)

    bands = {
        "delta":  (0.5,  4.0),
        "theta":  (4.0,  8.0),
        "alpha":  (8.0, 13.0),
        "beta":  (13.0, 30.0),
        "gamma": (30.0, 80.0),
    }
    powers = {}
    for name, (lo, hi) in bands.items():
        if hi > nyq:
            continue
        mask = (f >= lo) & (f <= hi)
        if mask.sum() > 0:
            powers[name] = float(psd_mean[mask].mean())

    if not powers:
        return 2, {}
    max_p = max(powers.values())
    sig = {k: v for k, v in powers.items() if v > 0.1 * max_p}
    n_bands = max(2, min(8, len(sig)))
    return n_bands, sig


def choose_n_modes(C: int, latent_dim: int,
                   n_mp: int, n_bands: int,
                   eigvals: np.ndarray, lambda_plus: float,
                   sig_bands: dict[str, float]) -> tuple[int, str]:
    n_signal = max(n_mp, n_bands)
    n_rank = max(2, int(np.ceil(n_signal / latent_dim)))

    chosen = max(n_mp, n_bands, n_rank)
    chosen = min(chosen, 20)

    reason = (
        f"max(MP={n_mp}, bands={n_bands}, rank_coverage={n_rank}) = {chosen}"
    )
    return chosen, reason


def print_analysis(
    T: int, C: int, sfreq: float,
    tau: int, m: int, latent_dim: int,
    n_mp: int, lambda_plus: float, eigvals: np.ndarray,
    n_bands: int, sig_bands: dict[str, float],
    n_modes: int, reason: str,
    fnns: list[float], fnn_thresh: float,
):
    print("\n" + "=" * 62)
    print("=== Анализ гиперпараметров ===")
    print("=" * 62)
    print(f"  Данные   : T={T}, C={C}, sfreq={sfreq:.1f} Гц")
    print()
    print(f"  Takens:")
    print(f"    tau = {tau}  (первый лок. мин. AMI)")
    for i, fnn in enumerate(fnns, 1):
        mark = " <- выбрано" if i == m else ""
        flag = " < thresh" if fnn < fnn_thresh else ""
        print(f"    m={i}: FNN={fnn:.4f}{flag}{mark}")
    print(f"  Сжатие: m={m} -> latent_dim={latent_dim} "
          f"({latent_dim/m:.0%} от m)")
    print()
    print(f"  n_modes -- оценки:")
    print(f"    Марченко-Пастур  : {n_mp:2d}  "
          f"({int((eigvals > lambda_plus).sum())} собств. зн. > lambda+={lambda_plus:.3f})")
    if sig_bands:
        bands_str = ", ".join(f"{k}({v:.2e})" for k, v in sig_bands.items())
        print(f"    Частотные полосы : {n_bands:2d}  ({bands_str})")
    else:
        print(f"    Частотные полосы : {n_bands:2d}  (scipy не найден)")
    n_rank = max(2, int(np.ceil(max(n_mp, n_bands) / latent_dim)))
    print(f"    Покрытие ранга   : {n_rank:2d}  "
          f"(ceil(signal={max(n_mp,n_bands)}/{latent_dim}) -- покрыть сигн. подпространство)")
    print(f"  -> Выбрано: n_modes={n_modes}  ({reason})")
    print(f"  -> max rank(A) = {n_modes} * {latent_dim} = {n_modes * latent_dim}"
          f"  {'>=' if n_modes * latent_dim >= C else '<'} C={C}")
    if n_modes * latent_dim < C:
        print(f"  [!] rank(A) < C: связь будет аппроксимирована низкоранговой матрицей.")
        print(f"    Для полного покрытия нужно n_modes >= {n_rank}.")
    print("=" * 62)
    print(f"\n  Итоговые параметры:")
    print(f"    tau={tau}, m={m}, latent_dim={latent_dim}, "
          f"n_modes={n_modes}, d_head={latent_dim}")
    print()


class WindowDataset(Dataset):
    def __init__(self, emb, raw, window, horizon, warmup, labels=None):
        T_emb = emb.shape[0]
        offset = raw.shape[0] - T_emb
        self.emb = emb.astype(np.float32)
        self.raw = raw[offset:].astype(np.float32)
        self.window = window
        self.horizon = horizon
        self.warmup = warmup
        self.N = T_emb - window - horizon + 1
        if labels is not None:
            lab = labels[offset:]
            self.regime = (lab > 0).astype(np.float32)
        else:
            self.regime = None

    def __len__(self):
        return self.N

    def __getitem__(self, i):
        x = self.emb[i: i + self.window]
        x_raw = self.raw[i: i + self.window]
        y = self.raw[i + self.window: i + self.window + self.horizon]
        regime = float(self.regime[i + self.window - 1]) if self.regime is not None else -1.0
        return (
            torch.from_numpy(x),
            torch.from_numpy(x_raw),
            torch.from_numpy(y),
            torch.tensor(regime),
        )


def empirical_correlation(x):
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x, dtype=torch.float32)
    x = x - x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True) + 1e-8
    return (x.T @ x) / x.shape[0] / (std.T @ std)


def _base(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def run_epoch(model, loader, opt, device,
              lam_rec, lam_kl, lam_aux, lam_in, lam_phase, train,
              has_labels: bool = False):
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
            if has_labels and out["phase_logit"] is not None:
                valid = regime >= 0
                if valid.any():
                    phase_loss = F.binary_cross_entropy_with_logits(
                        out["phase_logit"][valid], regime[valid]
                    )

            total = (fc + lam_in * inw + lam_rec * rec + lam_kl * kl
                     + lam_aux * aux + lam_phase * phase_loss)
            if train:
                if not torch.isfinite(total):
                    continue
                opt.zero_grad()
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            bs = x.size(0)
            tot["L"] += float(total.detach()) * bs
            tot["fc"] += float(fc.detach()) * bs
            tot["inw"] += float(inw.detach()) * bs
            tot["rec"] += float(rec.detach()) * bs
            tot["kl"] += float(kl.detach()) * bs
            tot["aux"] += float(aux.detach()) * bs
            tot["phase"] += float(phase_loss.detach()) * bs
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


def make_model(spatial_kind, m_embed, latent_dim, C, n_modes, args,
               corr_train, has_labels):
    sk = {}
    if spatial_kind in ("grand_diff", "grand_full"):
        sk = dict(d_head=latent_dim, n_modes=n_modes, untied=True)
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
        use_phase_loss=has_labels,
    ).to(args.device)
    if args.gpu_ids:
        model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)
    return model


def train_one(spatial_kind, m_embed, latent_dim, C, n_modes,
              tr_ld, va_ld, corr_train, args, out_path, has_labels):
    import math
    model = make_model(spatial_kind, m_embed, latent_dim, C, n_modes,
                       args, corr_train, has_labels)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)

    best = float("inf")
    history = []
    nan_epochs = 0

    def _save():
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        torch.save({
            "state_dict": _base(model).state_dict(),
            "args": vars(args), "m": m_embed, "latent": latent_dim,
            "num_channels": C, "spatial_kind": spatial_kind,
        }, out_path)

    _save()

    for ep in range(args.epochs):
        tau_g = max(0.2, 1.0 - (ep / max(args.epochs - 1, 1)) * 0.8)
        for mod in model.modules():
            if hasattr(mod, "temperature"):
                mod.temperature = tau_g

        tr = run_epoch(model, tr_ld, opt, args.device,
                       args.lam_rec, args.lam_kl, args.lam_aux,
                       args.lam_in, args.lam_phase, True, has_labels)
        va = run_epoch(model, va_ld, opt, args.device,
                       args.lam_rec, args.lam_kl, args.lam_aux,
                       args.lam_in, args.lam_phase, False, has_labels)
        sched.step()

        val_fc = va["fc"]
        if not math.isfinite(val_fc):
            nan_epochs += 1
            print(f"  [{spatial_kind}] ep {ep:3d}  [!] NaN/Inf потери (эпох NaN: {nan_epochs})",
                  flush=True)
            if nan_epochs >= 5:
                print(f"  [{spatial_kind}] Прерывание: 5 NaN-эпох подряд.", flush=True)
                break
            continue
        nan_epochs = 0

        history.append({
            "epoch": ep, "gumbel_tau": tau_g,
            "train_fc": tr["fc"],    "val_fc":    val_fc,
            "train_inw": tr["inw"],  "val_inw":   va["inw"],
            "train_rec": tr["rec"],  "val_rec":   va["rec"],
            "train_kl":  tr["kl"],   "val_kl":    va["kl"],
            "train_aux": tr["aux"],  "val_aux":   va["aux"],
            "train_phase": tr["phase"], "val_phase": va["phase"],
            "train_total": tr["L"],  "val_total": va["L"],
        })
        phase_str = f" phase={tr['phase']:.4f}" if has_labels else ""
        print(f"  [{spatial_kind}] ep {ep:3d}  tau={tau_g:.2f}  "
              f"train fc={tr['fc']:.5f}{phase_str}  |  val fc={val_fc:.5f}",
              flush=True)
        if val_fc < best:
            best = val_fc
            _save()

    ck = torch.load(out_path, map_location=args.device, weights_only=False)
    _base(model).load_state_dict(ck["state_dict"])
    mse_h = per_horizon_mse(model, va_ld, args.device, args.horizon)

    metrics = dict(spatial=spatial_kind, best_val_fc=float(best),
                   latent_dim=latent_dim, m_embed=m_embed, history=history)
    with open(out_path.replace(".pt", "_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"  [{spatial_kind}] best val fc={best:.5f} | "
          f"latent_dim={latent_dim}/{m_embed} (сжатие {latent_dim/m_embed:.0%})")
    return model, best, mse_h, history


def extract_matrices(model, va_ld, device, out_dir, spatial_kind, labels, val_idx):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    A_all, w_all = [], []
    with torch.no_grad():
        for x, x_raw, _, _ in va_ld:
            x, x_raw = x.to(device), x_raw.to(device)
            out = model(x, x_raw)
            A_all.append(out["A_last"].cpu().numpy())
            if hasattr(_base(model).spatial, "attn_mod"):
                w = _base(model).spatial.attn_mod.phase_weights(out["z_last"])
                w_all.append(w.cpu().numpy())

    A_all = np.concatenate(A_all, axis=0)
    A_mean = A_all.mean(0)
    np.save(os.path.join(out_dir, f"{spatial_kind}_A_mean.npy"), A_mean)
    np.save(os.path.join(out_dir, f"{spatial_kind}_A_all.npy"), A_all)
    print(f"  [{spatial_kind}] A_mean saved: shape={A_mean.shape}")

    if labels is not None and len(val_idx) == len(A_all):
        unique_classes = np.unique(labels[val_idx])
        for cls in unique_classes:
            mask = labels[val_idx] == cls
            if mask.sum() > 0:
                A_cls = A_all[mask].mean(0)
                np.save(os.path.join(out_dir, f"{spatial_kind}_A_class{cls}.npy"), A_cls)
        print(f"  [{spatial_kind}] A per class: {list(unique_classes)}")

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
            A_modes = (torch.einsum("bkid,bkjd->bkij", Q, K_) / d_head ** 0.5
                       ).mean(0).cpu().numpy()
        np.save(os.path.join(out_dir, f"{spatial_kind}_A_modes.npy"), A_modes)
        print(f"  [{spatial_kind}] A_modes shape={A_modes.shape}, "
              f"mean weights={w_all.mean(0).round(3)}")

    return A_mean


def scan_n_modes(m_embed, latent_dim, C, tr_ld, va_ld, corr_train,
                 args, has_labels, candidates=(2, 4, 6, 8, 12)):
    print("\n=== Сканирование n_modes (grand_full, 10 эпох) ===")
    results = {}
    orig_epochs = args.epochs
    args.epochs = 10
    for nm in candidates:
        model = make_model("grand_full", m_embed, latent_dim, C, nm,
                           args, corr_train, has_labels)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        best = float("inf")
        for ep in range(args.epochs):
            tau_g = max(0.2, 1.0 - (ep / max(args.epochs - 1, 1)) * 0.8)
            for mod in model.modules():
                if hasattr(mod, "temperature"):
                    mod.temperature = tau_g
            va = run_epoch(model, va_ld, opt, args.device,
                           args.lam_rec, args.lam_kl, args.lam_aux,
                           args.lam_in, args.lam_phase, False, has_labels)
            run_epoch(model, tr_ld, opt, args.device,
                      args.lam_rec, args.lam_kl, args.lam_aux,
                      args.lam_in, args.lam_phase, True, has_labels)
            best = min(best, va["fc"])
        results[nm] = best
        print(f"  n_modes={nm:2d} -> val_fc={best:.5f}")

    args.epochs = orig_epochs
    best_nm = min(results, key=results.get)
    print(f"\n  -> Лучший n_modes={best_nm} (val_fc={results[best_nm]:.5f})")
    print("=" * 42)
    return best_nm


def main():
    ap = argparse.ArgumentParser(description="ЭЭГ: обнаружение матриц связности")

    ap.add_argument("--data", required=True, help="Путь к данным (.npy/.npz/.edf/.fif/.set)")
    ap.add_argument("--labels", default=None, help="Метки режима (T,) int .npy, опционально")
    ap.add_argument("--sfreq", type=float, default=None,
                    help="Частота дискретизации (обязательно для .npy/.npz без sfreq)")
    ap.add_argument("--l_freq",    type=float, default=0.5,  help="Нижняя частота полосы, Гц")
    ap.add_argument("--h_freq",    type=float, default=80.0, help="Верхняя частота полосы, Гц")
    ap.add_argument("--notch_freq",type=float, default=50.0, help="Notch-фильтр, Гц (0 = откл.)")
    ap.add_argument("--no_filter", action="store_true",      help="Пропустить фильтрацию")
    ap.add_argument("--max_lag",   type=int,   default=60,   help="Максимальный лаг AMI")
    ap.add_argument("--m_max",     type=int,   default=25,   help="Максимальное m для FNN")
    ap.add_argument("--fnn_thresh",type=float, default=0.005,help="Порог FNN")
    ap.add_argument("--n_modes",   type=int,   default=0,    help="0 = авто из анализа")
    ap.add_argument("--scan_modes",action="store_true",      help="Сканировать n_modes перед обучением")
    ap.add_argument("--latent_dim",type=int,   default=0,    help="0 = round(m*2/3)")
    ap.add_argument("--window",     type=int,   default=0,   help="Длина окна, отсчёты (0 = из window_sec)")
    ap.add_argument("--warmup",     type=int,   default=0,   help="Прогрев, отсчёты (0 = из warmup_sec)")
    ap.add_argument("--horizon",    type=int,   default=0,   help="Горизонт, отсчёты (0 = из horizon_sec)")
    ap.add_argument("--window_sec", type=float, default=2.0, help="Длина окна, с (если --window=0)")
    ap.add_argument("--warmup_sec", type=float, default=0.5, help="Прогрев, с (если --warmup=0)")
    ap.add_argument("--horizon_sec",type=float, default=0.5, help="Горизонт, с (если --horizon=0)")
    ap.add_argument("--epochs",    type=int,   default=50)
    ap.add_argument("--batch_size",type=int,   default=64)
    ap.add_argument("--vae_dim",   type=int,   default=32)
    ap.add_argument("--vae_blocks",type=int,   default=2)
    ap.add_argument("--vae_spatial_blocks", type=int, default=1)
    ap.add_argument("--temporal_blocks",    type=int, default=1)
    ap.add_argument("--deterministic_encoder", action="store_true", default=True)
    ap.add_argument("--lr",        type=float, default=3e-3)
    ap.add_argument("--lam_rec",   type=float, default=0.5)
    ap.add_argument("--lam_kl",    type=float, default=0.0)
    ap.add_argument("--lam_aux",   type=float, default=0.05)
    ap.add_argument("--lam_in",    type=float, default=1.0)
    ap.add_argument("--lam_phase", type=float, default=0.3)
    ap.add_argument("--spatials", nargs="+",
                    default=["grand_full", "grand_diff", "correlation"])
    ap.add_argument("--results_dir", default="results/eeg")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--gpus", type=str, default="",
                    help="GPU IDs для DataParallel, через запятую: '0,1,2'. "
                         "Если задано - device устанавливается на cuda:gpus[0]")
    ap.add_argument("--n_seeds", type=int, default=5,
                    help="Число сидов для оценки разброса (только init модели)")
    ap.add_argument("--seed", type=int, default=None,
                    help="Запустить один сид (legacy)")
    ap.add_argument("--val_frac", type=float, default=0.2, help="Доля данных для валидации")
    ap.add_argument("--tau", type=int, default=0, help="Задать tau вручную (0 = авто из AMI)")
    ap.add_argument("--m",   type=int, default=0, help="Задать m вручную (0 = авто из FNN)")
    ap.add_argument("--max_samples", type=int, default=0,
                    help="Обрезать данные до N отсчётов (0 = все)")

    args = ap.parse_args()

    if args.gpus:
        args.gpu_ids = [int(g) for g in args.gpus.split(",")]
        args.device  = f"cuda:{args.gpu_ids[0]}"
        print(f"  Multi-GPU: DataParallel на {args.gpu_ids}, device={args.device}")
    else:
        args.gpu_ids = []

    seeds = [args.seed] if args.seed is not None else list(range(args.n_seeds))

    print(f"\n=== Загрузка данных: {args.data} ===")
    x, sfreq, ch_names = load_data(args.data, args.sfreq)
    T, C = x.shape
    print(f"  shape={x.shape}, sfreq={sfreq:.1f} Гц, каналов={C}")
    if args.max_samples > 0 and T > args.max_samples:
        x = x[:args.max_samples]
        T = args.max_samples
        print(f"  Обрезано до {T} отсчётов")

    if not args.no_filter and sfreq > 1.0:
        print("  Фильтрация...")
        x = bandpass_filter(x, sfreq, args.l_freq, args.h_freq, args.notch_freq)

    cut_t = int(T * (1 - args.val_frac))
    x = normalize(x, x_train=x[:cut_t])

    labels = None
    has_labels = False
    if args.labels is not None:
        labels = np.load(args.labels).astype(np.int64)
        assert len(labels) == T, f"labels длина {len(labels)} != T={T}"
        has_labels = True
        print(f"  Метки: {np.unique(labels).tolist()}, has_labels=True")

    print("\n=== AMI + FNN ===")
    if args.tau > 0 and args.m > 0:
        tau, ami_avg = args.tau, None
        m, fnns = args.m, []
        print(f"  tau={tau}, m={m}  (заданы вручную -- анализ пропущен)")
    else:
        x_sub = x[::max(1, T // 10000)]
        tau, ami_avg = estimate_tau(x_sub, args.max_lag)
        print(f"  tau = {tau}")
        m, fnns = estimate_m(x_sub, tau, args.fnn_thresh, args.m_max)
    latent_dim = args.latent_dim if args.latent_dim > 0 else max(2, round(m * 2 / 3))
    if latent_dim >= m:
        latent_dim = max(1, m - 1)

    print("\n=== Оценка n_modes ===")
    x_train = x[:cut_t]
    n_mp, eigvals, lambda_plus = estimate_n_modes_mp(x_train)
    n_bands, sig_bands = estimate_n_modes_bands(x_train, sfreq)

    if args.n_modes > 0:
        n_modes = args.n_modes
        reason = f"задано вручную (--n_modes {args.n_modes})"
    else:
        n_modes, reason = choose_n_modes(C, latent_dim, n_mp, n_bands,
                                         eigvals, lambda_plus, sig_bands)

    if fnns:
        print_analysis(T, C, sfreq, tau, m, latent_dim,
                       n_mp, lambda_plus, eigvals,
                       n_bands, sig_bands, n_modes, reason, fnns, args.fnn_thresh)
    else:
        print(f"  latent_dim={latent_dim}, n_modes={n_modes}  ({reason})")

    print("=== Такенс-вложение ===")
    emb = embed_field(x, tau, m)
    print(f"  emb shape: {emb.shape}  (T_emb, C={C}, m={m})")

    window  = args.window  if args.window  > 0 else max(m + 1, round(args.window_sec  * sfreq))
    warmup  = args.warmup  if args.warmup  > 0 else max(1,     round(args.warmup_sec  * sfreq))
    horizon = args.horizon if args.horizon > 0 else max(1,     round(args.horizon_sec * sfreq))
    args.window  = window
    args.warmup  = warmup
    args.horizon = horizon
    print(f"  window={window} ({window/sfreq:.3f}с), warmup={warmup} ({warmup/sfreq:.3f}с), "
          f"horizon={horizon} ({horizon/sfreq:.3f}с)")

    ds = WindowDataset(emb, x, window, horizon, warmup, labels=labels)
    cut = int(len(ds) * (1 - args.val_frac))
    pin = args.device == "cuda"
    tr_ld = DataLoader(Subset(ds, list(range(cut))),
                       batch_size=args.batch_size, shuffle=True, drop_last=True,
                       num_workers=0, pin_memory=pin)
    va_ld = DataLoader(Subset(ds, list(range(cut, len(ds)))),
                       batch_size=args.batch_size, shuffle=False,
                       num_workers=0, pin_memory=pin)
    val_idx = np.array(list(range(cut, len(ds))))
    corr_train = empirical_correlation(torch.tensor(x_train))

    if args.scan_modes and args.n_modes == 0:
        n_modes = scan_n_modes(m, latent_dim, C, tr_ld, va_ld,
                               corr_train, args, has_labels)

    print(f"\n  n_modes={n_modes}, d_head={latent_dim}, "
          f"max rank(A)={n_modes * latent_dim}")
    print(f"  Сидов для обучения: {seeds}")

    all_results = []
    for seed in seeds:
        print(f"\n{'#'*60}")
        print(f"### SEED {seed} ###")
        print(f"{'#'*60}")
        torch.manual_seed(seed)
        np.random.seed(seed)

        seed_dir = os.path.join(args.results_dir, f"seed_{seed}")
        mat_dir  = os.path.join(seed_dir, "matrices")
        os.makedirs(mat_dir, exist_ok=True)

        seed_res = {}
        for kind in args.spatials:
            print(f"\n{'='*60}")
            print(f"=== Seed {seed} | Обучение: {kind} ===")
            print(f"{'='*60}")
            out_path = os.path.join(seed_dir, f"{kind}.pt")
            model, best_fc, mse_h, history = train_one(
                kind, m, latent_dim, C, n_modes,
                tr_ld, va_ld, corr_train, args, out_path, has_labels,
            )
            A_mean = extract_matrices(model, va_ld, args.device, mat_dir,
                                      kind, labels, val_idx)
            seed_res[kind] = dict(
                seed=seed, best_val_fc=best_fc,
                per_horizon_mse=mse_h.tolist(),
                A_asym=float(np.linalg.norm(A_mean - A_mean.T)
                             / (np.linalg.norm(A_mean) + 1e-12)),
                history=history,
            )

        seed_report_path = os.path.join(seed_dir, "seed_summary.json")
        with open(seed_report_path, "w") as f:
            json.dump(seed_res, f, indent=2, ensure_ascii=False)
        print(f"  [seed {seed}] Сохранено: {seed_report_path}")
        all_results.append(seed_res)

    agg = {}
    for kind in args.spatials:
        kind_results = [r[kind] for r in all_results if kind in r]
        if not kind_results:
            continue
        val_fcs  = np.array([r["best_val_fc"] for r in kind_results])
        asyms    = np.array([r["A_asym"] for r in kind_results])
        mse_h_list = np.array([r["per_horizon_mse"] for r in kind_results])

        max_ep = max(len(r["history"]) for r in kind_results)
        hist_keys = [k for k in kind_results[0]["history"][0] if k != "epoch"]
        history_agg = {}
        for hk in hist_keys:
            mat = np.full((len(kind_results), max_ep), np.nan)
            for s_idx, r in enumerate(kind_results):
                for ep_rec in r["history"]:
                    mat[s_idx, ep_rec["epoch"]] = ep_rec[hk]
            history_agg[hk] = {"mean": np.nanmean(mat, 0).tolist(),
                                "std":  np.nanstd(mat,  0).tolist()}

        agg[kind] = dict(
            n_seeds=len(kind_results),
            val_fc_mean=float(val_fcs.mean()), val_fc_std=float(val_fcs.std()),
            val_fc_seeds=val_fcs.tolist(),
            A_asym_mean=float(asyms.mean()),   A_asym_std=float(asyms.std()),
            per_horizon_mse_mean=mse_h_list.mean(0).tolist(),
            per_horizon_mse_std=mse_h_list.std(0).tolist(),
            per_horizon_mse_seeds=mse_h_list.tolist(),
            history=history_agg,
        )

    print(f"\n{'='*60}")
    print("=== Итог (все сиды) ===")
    print(f"  T={T}, C={C}, sfreq={sfreq:.1f} Гц, seeds={seeds}")
    print(f"  tau={tau}, m={m}, latent_dim={latent_dim}, n_modes={n_modes}")
    for kind, s in agg.items():
        print(f"  [{kind:14s}]  "
              f"val_fc = {s['val_fc_mean']:.5f} +- {s['val_fc_std']:.5f}  "
              f"asym(A) = {s['A_asym_mean']:.3f} +- {s['A_asym_std']:.3f}")

    report = dict(
        data=args.data, T=T, C=C, sfreq=sfreq,
        tau=tau, m_embed=m, latent_dim=latent_dim,
        n_modes=n_modes, d_head=latent_dim,
        compression_ratio=float(latent_dim / m),
        has_labels=has_labels, seeds=seeds, n_seeds=len(seeds),
        epochs=args.epochs,
        n_modes_analysis=dict(mp=n_mp, bands=n_bands,
                               rank_coverage=int(np.ceil(C / latent_dim)),
                               chosen=n_modes, reason=reason),
        models=agg,
    )
    report_path = os.path.join(args.results_dir, "summary.json")
    os.makedirs(args.results_dir, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n  Результаты: {args.results_dir}/")
    print(f"  Отчёт:      {report_path}")
    print(f"\n  Для построения графиков:")
    print(f"    python plot_training.py --results_dir {args.results_dir}")


if __name__ == "__main__":
    main()
