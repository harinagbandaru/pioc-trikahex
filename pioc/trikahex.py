"""TrikaHex Hold / Turn graph network."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TrikaHexConfig:
    d_h: int = 256
    K_rbf: int = 64
    r_cut: float = 6.0
    d_t: int = 128
    d_c: int = 64
    n_layers: int = 6
    n_heads: int = 8
    p_drop: float = 0.05
    n_modes: int = 8
    matter_dim: int = 4


def rbf_expand(dist: torch.Tensor, k: int, r_cut: float) -> torch.Tensor:
    d = dist.unsqueeze(-1)
    mu = torch.linspace(0.0, r_cut, k, device=dist.device, dtype=dist.dtype)
    gamma = 1.0 / ((r_cut / max(k - 1, 1)) ** 2 + 1e-8)
    return torch.exp(-gamma * (d - mu) ** 2)


class HoldTurnBlock(nn.Module):
    def __init__(self, cfg: TrikaHexConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_h
        self.w_e = nn.Linear(2 * d + cfg.K_rbf, d)
        self.w_t = nn.Linear(2 * d + 2, cfg.d_t)
        self.w_v = nn.Linear(d, d)
        self.w_u = nn.Linear(cfg.d_t, d)
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.hex = nn.Linear(cfg.d_c + d, d)
        self.mlp = nn.Sequential(nn.Linear(2 * d, d), nn.SiLU(), nn.Dropout(cfg.p_drop), nn.Linear(d, d))
        self.norm = nn.LayerNorm(d)

    def forward(self, h, e_rbf, cosA, cosB, adj):
        N, d = h.shape
        hi = h.unsqueeze(1).expand(N, N, d)
        hj = h.unsqueeze(0).expand(N, N, d)
        mij = F.silu(self.w_e(torch.cat([hi, hj, e_rbf], dim=-1)))
        qi, kj = self.q(h), self.k(h)
        score = (qi.unsqueeze(1) * kj.unsqueeze(0)).sum(-1) / (d ** 0.5)
        score = score.masked_fill(adj < 0.5, -1e9)
        alpha = torch.softmax(score, dim=-1)
        hold = (alpha.unsqueeze(-1) * self.w_v(mij)).sum(dim=1)
        h_mid = self.norm(h + hold)
        g = torch.zeros_like(h_mid)
        return h_mid + self.mlp(torch.cat([h_mid, g], dim=-1))


class TrikaHex(nn.Module):
    def __init__(self, n_atoms: int = 6, cfg: TrikaHexConfig | None = None):
        super().__init__()
        self.cfg = cfg or TrikaHexConfig()
        self.n_atoms = n_atoms
        c = self.cfg
        self.w_m = nn.Linear(c.matter_dim, c.d_h)
        self.w_pos = nn.Linear(6, c.d_h)
        self.blocks = nn.ModuleList([HoldTurnBlock(c) for _ in range(c.n_layers)])
        self.head = nn.Linear(c.d_h, 2 * c.n_modes * 3)

    def graph(self, RA: torch.Tensor, RB: torch.Tensor):
        dA = torch.cdist(RA, RA)
        dB = torch.cdist(RB, RB)
        adj = ((dA < self.cfg.r_cut) | (dB < self.cfg.r_cut)).float()
        adj.fill_diagonal_(0.0)
        eA = rbf_expand(dA, self.cfg.K_rbf, self.cfg.r_cut)
        return eA, adj, dA, dB

    def forward(self, matter: torch.Tensor, RA: torch.Tensor, RB: torch.Tensor):
        RA = RA.float()
        RB = RB.float()
        matter = matter.float()
        h = self.w_m(matter) + self.w_pos(torch.cat([RA, RB], dim=-1))
        eA, adj, _, _ = self.graph(RA, RB)
        zeros = torch.zeros(self.n_atoms, self.n_atoms, self.n_atoms, device=h.device)
        for blk in self.blocks:
            h = blk(h, eA, zeros, zeros, adj)
        out = self.head(h).view(self.n_atoms, self.cfg.n_modes, 3, 2)
        out = out.permute(1, 0, 2, 3).contiguous()
        mu = out[..., 0]
        sig = torch.exp(out[..., 1].clamp(-4.0, 0.5)).clamp_min(1e-3)
        return mu, sig
