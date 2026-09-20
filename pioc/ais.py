"""Path-space annealed importance sampling on Fourier modes."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .paths import FourierBridge
from .potential import PairPotential


@dataclass
class AISConfig:
    n_paths: int = 48
    n_stages: int = 8
    eta: float = 2.0
    ess_trigger: float = 0.5
    mala_steps: int = 6
    mala_eps: float = 0.03
    mode_clip: float = 1.2


def string_energy(pot: PairPotential, paths: torch.Tensor) -> torch.Tensor:
    return pot.energy(paths).mean(dim=-1)


def systematic_resample(modes: torch.Tensor, log_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n = modes.shape[0]
    w = torch.softmax(log_w.detach(), dim=0)
    cdf = torch.cumsum(w, dim=0)
    u0 = torch.rand((), device=modes.device) / n
    pos = u0 + torch.arange(n, device=modes.device, dtype=w.dtype) / n
    idx = torch.searchsorted(cdf, pos).clamp(max=n - 1)
    return modes[idx], torch.full_like(log_w, -math.log(n))


def mala_modes(
    modes, mu, sig, pot, xa, xb, gen, lam, beta, n_steps, eps, mode_clip,
):
    mask = gen.mobile_mask.view(1, 1, -1, 1)
    accepted = 0
    total = 0
    for _ in range(n_steps):
        a = modes.detach().requires_grad_(True)
        e = string_energy(pot, gen.synthesize(xa, xb, a))
        log_q = FourierBridge.log_prob(a, mu, sig)
        log_pi = (1.0 - lam) * log_q - lam * beta * e
        grad = torch.autograd.grad(log_pi.sum(), a)[0].detach()
        grad = torch.nan_to_num(grad) * mask
        noise = torch.randn_like(a) * mask
        prop = (a.detach() + 0.5 * eps * eps * grad + eps * noise).clamp(-mode_clip, mode_clip) * mask
        a_p = prop.detach().requires_grad_(True)
        e_p = string_energy(pot, gen.synthesize(xa, xb, a_p))
        log_q_p = FourierBridge.log_prob(a_p, mu, sig)
        log_pi_p = (1.0 - lam) * log_q_p - lam * beta * e_p
        grad_p = torch.autograd.grad(log_pi_p.sum(), a_p)[0].detach()
        grad_p = torch.nan_to_num(grad_p) * mask
        fwd = -0.5 * (((prop - a.detach() - 0.5 * eps * eps * grad) / eps) ** 2).sum((1, 2, 3))
        bwd = -0.5 * (((a.detach() - prop - 0.5 * eps * eps * grad_p) / eps) ** 2).sum((1, 2, 3))
        log_acc = (log_pi_p.detach() - log_pi.detach()) + (bwd - fwd)
        accept = torch.rand_like(log_acc).log() < log_acc.clamp(max=20)
        modes = torch.where(accept.view(-1, 1, 1, 1), prop.detach(), a.detach())
        accepted += int(accept.sum().item())
        total += int(accept.numel())
    return modes, accepted / max(total, 1)


def path_ais(gen, pot, xa, xb, beta, cfg: AISConfig) -> dict:
    _, modes, _, mu, sig = gen.sample(xa, xb, cfg.n_paths, mode_clip=cfg.mode_clip)
    log_w = torch.zeros(cfg.n_paths, device=xa.device)
    ess_hist = []
    mala_acc = []
    lambdas = [(m / cfg.n_stages) ** cfg.eta for m in range(cfg.n_stages + 1)]
    n_resample = 0
    for m in range(1, cfg.n_stages + 1):
        dlam = lambdas[m] - lambdas[m - 1]
        paths = gen.synthesize(xa, xb, modes)
        e = string_energy(pot, paths)
        log_q = FourierBridge.log_prob(modes, mu, sig)
        log_w = log_w + dlam * (-beta * e - log_q)
        w = torch.softmax(log_w.detach(), dim=0)
        ess = float((1.0 / (w ** 2).sum()).detach())
        ess_hist.append(ess)
        if ess < cfg.ess_trigger * cfg.n_paths and m < cfg.n_stages:
            modes, log_w = systematic_resample(modes, log_w)
            modes, acc = mala_modes(
                modes, mu, sig, pot, xa, xb, gen, lambdas[m], beta,
                cfg.mala_steps, cfg.mala_eps, cfg.mode_clip,
            )
            mala_acc.append(acc)
            n_resample += 1
    paths = gen.synthesize(xa, xb, modes)
    e = string_energy(pot, paths).detach()
    w = torch.softmax(log_w.detach(), dim=0)
    ess = float((1.0 / (w ** 2).sum()).detach())
    e0, e1 = gen.endpoint_error(xa, xb, paths)
    r_mid = pot.al_c(paths[:, paths.shape[1] // 2]).detach()
    return {
        "paths": paths.detach(),
        "modes": modes.detach(),
        "weights": w,
        "energy": e,
        "ess": ess,
        "ess_hist": ess_hist,
        "n_resample": n_resample,
        "mala_accept": mala_acc,
        "endpoint_err_A": e0,
        "endpoint_err_B": e1,
        "r_mid": r_mid,
        "r_mid_wmean": float((w * r_mid).sum().detach()),
        "e_wmean": float((w * e).sum().detach()),
    }
