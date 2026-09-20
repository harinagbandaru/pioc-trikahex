"""ELBO-style training of the Fourier proposal."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .ais import string_energy
from .paths import FourierBridge
from .potential import PairPotential


@dataclass
class TrainConfig:
    steps: int = 80
    batch: int = 24
    lr: float = 3e-4
    kl_coef: float = 0.15
    prior_sig: float = 0.25
    grad_clip: float = 2.0


def train_proposal(
    gen: FourierBridge,
    pot: PairPotential,
    xa: torch.Tensor,
    xb: torch.Tensor,
    beta: float,
    cfg: TrainConfig,
) -> dict[str, float]:
    opt = torch.optim.AdamW(gen.parameters(), lr=cfg.lr, betas=(0.9, 0.999), weight_decay=1e-4)
    last = {"loss": float("nan"), "E": float("nan"), "kl": float("nan")}
    gen.train()
    for _ in range(cfg.steps):
        opt.zero_grad(set_to_none=True)
        paths, modes, _, mu, sig = gen.sample(xa, xb, cfg.batch)
        e = string_energy(pot, paths)
        live = sig > 1e-8
        ps2 = cfg.prior_sig ** 2
        kl_el = torch.log(torch.full_like(sig, cfg.prior_sig).clamp_min(1e-6) / sig.clamp_min(1e-4))
        kl_el = kl_el + (sig ** 2 + mu ** 2) / (2 * ps2) - 0.5
        kl_el = torch.where(live, kl_el, torch.zeros_like(kl_el))
        kl = kl_el.sum() / cfg.batch
        loss = beta * e.mean() + cfg.kl_coef * kl
        if not torch.isfinite(loss):
            continue
        loss.backward()
        nn.utils.clip_grad_norm_(gen.parameters(), cfg.grad_clip)
        opt.step()
        last = {
            "loss": float(loss.detach()),
            "E": float(e.mean().detach()),
            "kl": float(kl.detach()),
        }
    gen.eval()
    return last
