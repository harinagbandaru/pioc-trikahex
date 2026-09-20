"""Adaptive umbrella Metropolis + 1-D WHAM with bootstrap errors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .potential import PairPotential
from .system import MOBILE


@dataclass
class UmbrellaConfig:
    n_windows: int = 9
    n_steps: int = 900
    k_umb: float = 7.0
    r_min: float = 2.3
    r_max: float = 5.6
    step0: float = 0.05
    target_accept: float = 0.30
    n_bins: int = 36
    n_boot: int = 40
    burn_frac: float = 0.35


def _metropolis(pot, r0, r_target, k_umb, n_steps, beta, step_xyz, target_accept, burn_frac):
    r = r0.clone()
    v = pot.energy(r) + 0.5 * k_umb * (pot.al_c(r) - r_target) ** 2
    rs = []
    acc = 0
    step = step_xyz
    for i in range(n_steps):
        prop = r.clone()
        prop[list(MOBILE)] = prop[list(MOBILE)] + torch.randn(len(MOBILE), 3, dtype=r.dtype, device=r.device) * step
        vp = pot.energy(prop) + 0.5 * k_umb * (pot.al_c(prop) - r_target) ** 2
        dbeta = float((-beta * (vp - v)).clamp(max=20.0).detach())
        if np.log(np.random.random()) < dbeta:
            r, v = prop, vp
            acc += 1
        if i > 0 and i % 80 == 0:
            rate = acc / i
            if rate < target_accept - 0.08:
                step *= 0.85
            elif rate > target_accept + 0.08:
                step *= 1.12
            step = float(np.clip(step, 0.008, 0.18))
        if i >= int(n_steps * burn_frac) and i % 8 == 0:
            rs.append(float(pot.al_c(r).detach()))
    return np.asarray(rs, dtype=float), acc / max(n_steps, 1), step


def _wham(window_rs, centers, k_umb, beta, n_bins, rmin, rmax):
    edges = np.linspace(rmin, rmax, n_bins + 1)
    mid = 0.5 * (edges[:-1] + edges[1:])
    n_w = len(window_rs)
    hist = np.zeros((n_w, n_bins))
    n_i = np.zeros(n_w)
    for i, rs in enumerate(window_rs):
        h, _ = np.histogram(rs, bins=edges)
        hist[i] = h
        n_i[i] = h.sum()
    bias = np.stack([beta * 0.5 * k_umb * (mid - rc) ** 2 for rc in centers])
    f = np.zeros(n_w)
    p = np.zeros(n_bins)
    for _ in range(500):
        num = hist.sum(0)
        den = np.zeros(n_bins)
        for i in range(n_w):
            den += n_i[i] * np.exp(f[i] - bias[i])
        p = np.divide(num, den, out=np.zeros(n_bins), where=den > 0)
        f_new = np.array([-np.log(max(np.sum(p * np.exp(-bias[i])), 1e-30)) for i in range(n_w)])
        f_new -= f_new[0]
        if np.max(np.abs(f_new - f)) < 1e-7:
            f = f_new
            break
        f = f_new
    p = p / max(p.sum(), 1e-30)
    w = np.full_like(p, np.nan)
    live = p > 0
    w[live] = -np.log(p[live])
    w = w - np.nanmin(w)
    w = w / beta
    return mid, w, p, n_i


def _bootstrap(window_rs, centers, k_umb, beta, n_bins, rmin, rmax, n_boot, seed=0):
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_boot):
        boot = [rng.choice(rs, size=len(rs), replace=True) if len(rs) else rs for rs in window_rs]
        _, w, _, _ = _wham(boot, centers, k_umb, beta, n_bins, rmin, rmax)
        samples.append(w)
    arr = np.stack(samples)
    return np.nanstd(arr, axis=0)


def run_umbrellas(pot, xa, xb, beta, cfg: UmbrellaConfig, seed: int = 42) -> dict:
    np.random.seed(seed)
    r_centers = np.linspace(cfg.r_min, cfg.r_max, cfg.n_windows)
    window_rs, accs, steps = [], [], []
    for rc in r_centers:
        t = float(np.clip((rc - cfg.r_min) / max(cfg.r_max - cfg.r_min, 1e-9), 0.0, 1.0))
        r0 = xa * t + xb * (1.0 - t)
        rs, acc, step = _metropolis(
            pot, r0, float(rc), cfg.k_umb, cfg.n_steps, beta, cfg.step0, cfg.target_accept, cfg.burn_frac
        )
        window_rs.append(rs)
        accs.append(acc)
        steps.append(step)
    mid, w, p, n_i = _wham(window_rs, r_centers, cfg.k_umb, beta, cfg.n_bins, cfg.r_min - 0.3, cfg.r_max + 0.2)
    w_err = _bootstrap(window_rs, r_centers, cfg.k_umb, beta, cfg.n_bins, cfg.r_min - 0.3, cfg.r_max + 0.2, cfg.n_boot, seed)
    bound = (mid >= 2.2) & (mid <= 3.4) & np.isfinite(w)
    unbound = (mid >= 4.8) & (mid <= 5.6) & np.isfinite(w)
    dA = float("nan")
    if bound.any() and unbound.any():
        dA = float(np.nanmin(w[bound]) - np.nanmin(w[unbound]))
    return {
        "r": mid, "W": w, "W_err": w_err, "p": p,
        "window_centers": r_centers, "window_accept": accs, "window_step": steps,
        "window_n": [int(len(x)) for x in window_rs], "hist_n": n_i.tolist(),
        "dA_bound_unbound": dA,
    }
