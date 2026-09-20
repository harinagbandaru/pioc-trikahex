"""Non-Markovian GLE bath I."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class GLEConfig:
    tau_fs: tuple[float, ...] = (10.0, 50.0, 250.0)
    c_eV_A2: tuple[float, ...] = (0.05, 0.02, 0.005)
    n_atoms: int = 6


class GLEBath:
    def __init__(self, cfg: GLEConfig, device="cpu"):
        self.tau = torch.tensor(cfg.tau_fs, dtype=torch.float64, device=device)
        self.c = torch.tensor(cfg.c_eV_A2, dtype=torch.float64, device=device)
        self.n_atoms = cfg.n_atoms
        self.L = len(cfg.tau_fs)

    def kernel(self, dt: torch.Tensor) -> torch.Tensor:
        decay = torch.exp(-dt.abs().unsqueeze(-1) / self.tau)
        return (self.c * decay).sum(dim=-1)

    def influence(self, velocity: torch.Tensor, dtau: float) -> torch.Tensor:
        B, P, N, _ = velocity.shape
        s = torch.zeros(B, device=velocity.device, dtype=velocity.dtype)
        for p in range(P):
            for q in range(p + 1):
                dt = torch.tensor((p - q) * dtau, dtype=velocity.dtype, device=velocity.device)
                k = self.kernel(dt)
                dot = (velocity[:, p] * velocity[:, q]).sum(dim=(-2, -1))
                s = s + 0.5 * k * dot * (dtau ** 2)
        return s
