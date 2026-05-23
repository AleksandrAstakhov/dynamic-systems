from __future__ import annotations

import argparse
import os

import numpy as np
from scipy.spatial import cKDTree


def ami_one_sensor(x: np.ndarray, max_lag: int, bins: int = 32) -> np.ndarray:
    x = (x - x.min()) / (x.max() - x.min() + 1e-12)
    dig = np.minimum((x * bins).astype(int), bins - 1)
    amis = []
    for lag in range(1, max_lag + 1):
        a, b = dig[:-lag], dig[lag:]
        joint = np.zeros((bins, bins), np.float64)
        np.add.at(joint, (a, b), 1.0)
        joint /= joint.sum()
        pa = joint.sum(1, keepdims=True)
        pb = joint.sum(0, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            mi = np.nansum(
                joint
                * (np.log(joint + 1e-12) - np.log(pa + 1e-12) - np.log(pb + 1e-12))
            )
        amis.append(mi)
    return np.array(amis)


def first_local_min(a: np.ndarray) -> int:
    for i in range(1, len(a) - 1):
        if a[i] < a[i - 1] and a[i] < a[i + 1]:
            return i + 1
    return int(np.argmin(a)) + 1


def embed_one(x: np.ndarray, tau: int, m: int) -> np.ndarray:
    T = len(x)
    L = T - (m - 1) * tau
    return np.stack([x[i * tau : i * tau + L] for i in range(m)], axis=-1)


def embed_field(x: np.ndarray, tau: int, m: int) -> np.ndarray:

    embs = [embed_one(x[:, c], tau, m) for c in range(x.shape[1])]
    return np.stack(embs, axis=1).astype(np.float32)


def fnn_fraction_sensor(
    x_c: np.ndarray, tau: int, m: int, Rtol: float = 15.0, Atol: float = 2.0
) -> float:
    """FNN on one sensor's delay embedding (standard per-channel FNN)."""
    e_m = embed_one(x_c, tau, m)
    e_m1 = embed_one(x_c, tau, m + 1)
    n = min(len(e_m), len(e_m1))
    e_m, e_m1 = e_m[:n], e_m1[:n]
    tree = cKDTree(e_m)
    d, idx = tree.query(e_m, k=2)
    d1, nn = d[:, 1], idx[:, 1]
    extra = np.abs(e_m1[:, -1] - e_m1[nn, -1])
    sigma = x_c.std()
    fnn1 = extra / np.maximum(d1, 1e-12) > Rtol
    fnn2 = np.sqrt(d1**2 + extra**2) / max(sigma, 1e-12) > Atol
    return float(np.mean(fnn1 | fnn2))


def fnn_fraction_global(
    x: np.ndarray, tau: int, m: int, Rtol: float = 15.0, Atol: float = 2.0
) -> float:
    """Average per-sensor FNN over all sensors. Robust replacement for the
    joint-state FNN, which suffers from curse of dimensionality in
    distributed systems with many sensors."""
    fracs = [
        fnn_fraction_sensor(x[:, c], tau, m, Rtol, Atol) for c in range(x.shape[1])
    ]
    return float(np.mean(fracs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", default="data/series.npz")
    ap.add_argument("--out", default="data/takens.npz")
    ap.add_argument("--max_lag", type=int, default=40)
    ap.add_argument("--m_max", type=int, default=8)
    ap.add_argument("--fnn_thresh", type=float, default=0.02)
    args = ap.parse_args()

    data = np.load(args.series, allow_pickle=True)
    x = data["x"]
    T, C = x.shape

    amis = np.stack([ami_one_sensor(x[:, c], args.max_lag) for c in range(C)], axis=0)
    ami_avg = amis.mean(axis=0)
    tau = first_local_min(ami_avg)
    print(f"global tau (1st local min of mean AMI) = {tau}")

    m_sel = args.m_max
    for m in range(1, args.m_max + 1):
        frac = fnn_fraction_global(x, tau, m)
        print(f"  m={m}: mean per-sensor FNN = {frac:.4f}")
        if frac < args.fnn_thresh:
            m_sel = m
            break

    print(f"selected (global) tau={tau}, m={m_sel}")
    emb = embed_field(x, tau, m_sel)
    print(f"emb shape: {emb.shape}  (T_emb, C, m)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(
        args.out, emb=emb, tau=int(tau), m=int(m_sel), amis=ami_avg.astype(np.float32)
    )
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
