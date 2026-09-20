"""Open Fourier sine bridges with analytic Gaussian log-density."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .system import MOBILE


class FourierBridge(nn.Module):
    def __init__(self, n_atoms: int = 6, n_modes: int = 8, n_beads: int = 32, hidden: int = 128):
        super().__init__()
        self.n_atoms = n_atoms
        self.n_modes = n_modes
        self.n_beads = n_beads
        self.net = nn.Sequential(
            nn.Linear(n_atoms * 3 * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * n_modes * n_atoms * 3),
        )
        s = torch.linspace(0.0, 1.0, n_beads)
        k = torch.arange(1, n_modes + 1, dtype=torch.float32)
        self.register_buffer("s", s.view(1, n_beads, 1, 1))
        self.register_buffer("k", k.view(1, 1, n_modes, 1, 1))
        mask = torch.zeros(n_atoms)
        mask[list(MOBILE)] = 1.0
        self.register_buffer("mobile_mask", mask)

    def _head(self, xa: torch.Tensor, xb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inp = torch.cat([xa.float().flatten(), xb.float().flatten()])
        out = self.net(inp).view(self.n_modes, self.n_atoms, 3, 2)
        mask = self.mobile_mask.view(1, -1, 1)
        mu = out[..., 0] * mask
        log_sig = out[..., 1].clamp(-4.0, 0.5)
        sig = torch.exp(log_sig) * mask
        sig = torch.where(mask > 0.5, sig.clamp_min(1e-3), torch.zeros_like(sig))
        return mu, sig

    def synthesize(self, xa: torch.Tensor, xb: torch.Tensor, modes: torch.Tensor) -> torch.Tensor:
        xa = xa.float()
        xb = xb.float()
        linear = xa.view(1, 1, self.n_atoms, 3) + (xb - xa).view(1, 1, self.n_atoms, 3) * self.s
        basis = torch.sin(self.k * math.pi * self.s.unsqueeze(2))
        fluct = (modes.float().unsqueeze(1) * basis).sum(dim=2)
        return linear + fluct

    @staticmethod
    def log_prob(modes: torch.Tensor, mu: torch.Tensor, sig: torch.Tensor) -> torch.Tensor:
        live = sig > 1e-8
        sig_s = torch.where(live, sig, torch.ones_like(sig))
        z = (modes - mu.unsqueeze(0)) / sig_s.unsqueeze(0)
        term = 0.5 * z ** 2 + torch.log(sig_s.unsqueeze(0)) + 0.5 * math.log(2 * math.pi)
        term = torch.where(live.unsqueeze(0), term, torch.zeros_like(term))
        return -term.sum(dim=(1, 2, 3))

    def sample(
        self, xa: torch.Tensor, xb: torch.Tensor, n: int, mode_clip: float = 1.2
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, sig = self._head(xa, xb)
        eps = torch.randn(n, self.n_modes, self.n_atoms, 3, device=xa.device)
        modes = mu.unsqueeze(0) + sig.unsqueeze(0) * eps
        modes = modes.clamp(-mode_clip, mode_clip) * self.mobile_mask.view(1, 1, -1, 1)
        log_q = self.log_prob(modes, mu, sig)
        paths = self.synthesize(xa, xb, modes)
        return paths, modes, log_q, mu, sig

    def endpoint_error(self, xa: torch.Tensor, xb: torch.Tensor, paths: torch.Tensor) -> tuple[float, float]:
        e0 = float((paths[:, 0] - xa.float()).abs().max().detach())
        e1 = float((paths[:, -1] - xb.float()).abs().max().detach())
        return e0, e1
