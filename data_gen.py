from __future__ import annotations

import argparse
import os

import numpy as np


def signed_ring_dist(C: int) -> np.ndarray:
    j = np.arange(C)
    return np.where(j <= C // 2, j, j - C).astype(np.float64)


def make_T(v: float, C: int, ell: float, alpha: float) -> np.ndarray:
    sd = signed_ring_dist(C)
    g = np.exp(-((sd - v) ** 2) / (2 * ell**2))
    g = g / g.sum() * (1 - alpha)
    r = g.copy()
    r[0] += alpha
    return np.stack([np.roll(r, i) for i in range(C)])


def simulate(
    C: int,
    T: int,
    regime_len: int,
    v_amp: float,
    ell: float,
    alpha: float,
    sigma: float,
    burn_in: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    total = T + burn_in
    regime = (np.arange(total) // regime_len) % 2
    v_arr = np.where(regime == 0, +v_amp, -v_amp).astype(np.float64)

    T_pos = make_T(+v_amp, C, ell, alpha)
    T_neg = make_T(-v_amp, C, ell, alpha)

    h = np.zeros((total, C), dtype=np.float64)
    h[0] = rng.standard_normal(C)
    for t in range(1, total):
        T_t = T_pos if regime[t - 1] == 0 else T_neg
        h[t] = T_t @ h[t - 1] + sigma * rng.standard_normal(C)

    return h[burn_in:].astype(np.float32), v_arr[burn_in:].astype(np.float32)


def load_eeg(
    subject: int = 1,
    runs=(4, 8, 12),
    sfreq_target: float = 80.0,
    l_freq: float = 1.0,
    h_freq: float = 40.0,
    max_seconds: float | None = None,
    verbose: bool = False,
) -> tuple[np.ndarray, float, list[str]]:
    import mne
    from mne.datasets import eegbci
    from mne.io import concatenate_raws, read_raw_edf

    if not verbose:
        mne.set_log_level("WARNING")

    fnames = eegbci.load_data(subjects=[subject], runs=list(runs), update_path=True)
    raws = [read_raw_edf(f, preload=True) for f in fnames]
    raw = concatenate_raws(raws)
    eegbci.standardize(raw)

    raw.pick(picks="eeg")

    raw.filter(l_freq, h_freq, fir_design="firwin")
    raw.resample(sfreq_target)

    data = raw.get_data().T.astype(np.float32)
    if max_seconds is not None:
        n_max = int(max_seconds * sfreq_target)
        data = data[:n_max]

    mu = data.mean(axis=0, keepdims=True)
    sd = data.std(axis=0, keepdims=True) + 1e-8
    data = (data - mu) / sd

    return data, 1.0 / sfreq_target, raw.ch_names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=["drift_ring", "eeg"], default="drift_ring")

    ap.add_argument("--C", type=int, default=32)
    ap.add_argument("--T", type=int, default=8000)
    ap.add_argument("--regime_len", type=int, default=400)
    ap.add_argument("--v_amp", type=float, default=2.0)
    ap.add_argument("--ell", type=float, default=0.7)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--sigma", type=float, default=0.30)
    ap.add_argument("--burn_in", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--eeg_subject", type=int, default=1)
    ap.add_argument("--eeg_runs", type=int, nargs="+", default=[4, 8, 12])
    ap.add_argument("--eeg_sfreq", type=float, default=80.0)
    ap.add_argument("--eeg_l_freq", type=float, default=1.0)
    ap.add_argument("--eeg_h_freq", type=float, default=40.0)
    ap.add_argument("--eeg_max_seconds", type=float, default=None)
    ap.add_argument(
        "--eeg_n_channels",
        type=int,
        default=None,
        help="if set, keep only the first N channels (uniform decimation)",
    )

    ap.add_argument("--out", default="data/series.npz")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    if args.system == "drift_ring":
        x, v_arr = simulate(
            args.C,
            args.T,
            args.regime_len,
            args.v_amp,
            args.ell,
            args.alpha,
            args.sigma,
            args.burn_in,
            args.seed,
        )
        x = (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + 1e-8)
        np.savez_compressed(
            args.out,
            x=x,
            v=v_arr,
            dt=1.0,
            system="drift_ring",
            params=dict(
                C=args.C,
                T=args.T,
                regime_len=args.regime_len,
                v_amp=args.v_amp,
                ell=args.ell,
                alpha=args.alpha,
                sigma=args.sigma,
            ),
        )

        T_pos = make_T(+args.v_amp, args.C, args.ell, args.alpha)
        T_neg = make_T(-args.v_amp, args.C, args.ell, args.alpha)
        np.savez_compressed(
            args.out.replace(".npz", "_truth.npz"), T_pos=T_pos, T_neg=T_neg
        )
        print(f"saved {args.out}: x shape {x.shape}, system=drift_ring")
        print(f"     true T saved in {args.out.replace('.npz', '_truth.npz')}")

    elif args.system == "eeg":
        print(
            f"Loading EEG: subject={args.eeg_subject}, runs={args.eeg_runs}, "
            f"target sfreq={args.eeg_sfreq} Hz"
        )
        x, dt, ch_names = load_eeg(
            subject=args.eeg_subject,
            runs=args.eeg_runs,
            sfreq_target=args.eeg_sfreq,
            l_freq=args.eeg_l_freq,
            h_freq=args.eeg_h_freq,
            max_seconds=args.eeg_max_seconds,
        )
        if args.eeg_n_channels is not None and args.eeg_n_channels < x.shape[1]:

            idx = np.linspace(0, x.shape[1] - 1, args.eeg_n_channels).astype(int)
            x = x[:, idx]
            ch_names = [ch_names[i] for i in idx]
        np.savez_compressed(
            args.out,
            x=x,
            dt=dt,
            system="eeg",
            ch_names=np.array(ch_names),
        )
        print(
            f"saved {args.out}: x shape {x.shape}, dt={dt:.5f}s, "
            f"sfreq={1/dt:.1f}Hz, channels={x.shape[1]}"
        )


if __name__ == "__main__":
    main()
