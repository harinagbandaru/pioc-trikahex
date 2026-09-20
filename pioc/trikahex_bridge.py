"""Fourier open bridges whose (mu, sigma) come from TrikaHex."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .system import MOBILE
from .trikahex import TrikaHex, TrikaHexConfig


class TrikaHexBridge(nn.Module):
    def __init__(self, n_atoms: int = 6, n_modes: int = 8, n_beads: int = 32, cfg: TrikaHexConfig | None = None):
        super().__init__()
        cfg = cfg or TrikaHexConfig(n_modes=n_modes)
        self.net = TrikaHex(n_atoms=n_atoms, cfg=cfg)
        self.n_atoms = n_atoms
        self.n_modes = n_modes
        self.n_beads = n_beads
        s = torch.linspace(0.0, 1.0, n_beads)
        k = torch.arange(1, n_modes + 1, dtype=torch.float32)
        self.register_buffer("s", s.view(1, n_beads, 1, 1))
        self.register_buffer("k", k.view(1, 1, n_modes, 1, 1))
        mask = torch.zeros(n_atoms)
        mask[list(MOBILE)] = 1.0
        self.register_buffer("mobile_mask", mask)

    def synthesize(self, xa, xb, modes):
        xa, xb = xa.float(), xb.float()
        linear = xa.view(1, 1, self.n_atoms, 3) + (xb - xa).view(1, 1, self.n_atoms, 3) * self.s
        basis = torch.sin(self.k * math.pi * self.s.unsqueeze(2))
        fluct = (modes.float().unsqueeze(1) * basis).sum(dim=2)
        return linear + fluct

    @staticmethod
    def log_prob(modes, mu, sig):
        live = sig > 1e-8
        sig_s = torch.where(live, sig, torch.ones_like(sig))
        z = (modes - mu.unsqueeze(0)) / sig_s.unsqueeze(0)
        term = 0.5 * z ** 2 + torch.log(sig_s.unsqueeze(0)) + 0.5 * math.log(2 * math.pi)
        term = torch.where(live.unsqueeze(0), term, torch.zeros_like(term))
        return -term.sum(dim=(1, 2, 3))

    def sample(self, matter, xa, xb, n: int, mode_clip: float = 1.2):
        mu, sig = self.net(matter, xa, xb)
        mask = self.mobile_mask.view(1, -1, 1)
        mu = mu * mask
        sig = sig * mask
        eps = torch.randn(n, self.n_modes, self.n_atoms, 3, device=xa.device)
        modes = (mu.unsqueeze(0) + sig.unsqueeze(0) * eps).clamp(-mode_clip, mode_clip)
        modes = modes * self.mobile_mask.view(1, 1, -1, 1)
        log_q = self.log_prob(modes, mu, sig)
        paths = self.synthesize(xa, xb, modes)
        return paths, modes, log_q, mu, sig

    def endpoint_error(self, xa, xb, paths):
        e0 = float((paths[:, 0] - xa.float()).abs().max().detach())
        e1 = float((paths[:, -1] - xb.float()).abs().max().detach())
        return e0, e1
