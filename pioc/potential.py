"""Coulomb + LJ + bonded CO2 + lattice restraints. Energy in eV."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .constants import KE2
from .system import Fragment


@dataclass
class PotentialConfig:
    k_lattice: float = 8.0
    k_co2_bond: float = 40.0
    r_co2_bond: float = 1.16
    k_co2_angle: float = 12.0
    coulomb_floor: float = 1.10
    lj_floor_frac: float = 0.65


class PairPotential:
    def __init__(self, frag: Fragment, lattice: torch.Tensor, cfg: PotentialConfig | None = None):
        self.frag = frag
        self.cfg = cfg or PotentialConfig()
        if lattice.shape[-2] != len(frag.framework):
            raise ValueError("lattice coords must match framework atom count")
        self.lattice = lattice.to(dtype=torch.float64)
        n = frag.n_atoms
        q = frag.q
        self.qij = q[:, None] * q[None, :]
        self.epsij = torch.sqrt(frag.eps[:, None] * frag.eps[None, :])
        self.sigij = 0.5 * (frag.sig[:, None] + frag.sig[None, :])
        eye = torch.eye(n, dtype=torch.bool, device=q.device)
        lj_off = eye.clone()
        for i, j in frag.bond_exclude:
            lj_off[i, j] = lj_off[j, i] = True
        self.coulomb_mask = ~eye
        self.lj_mask = ~lj_off

    def distances(self, R: torch.Tensor) -> torch.Tensor:
        d = R.unsqueeze(-2) - R.unsqueeze(-3)
        return torch.linalg.norm(d, dim=-1)

    def energy(self, R: torch.Tensor) -> torch.Tensor:
        R = R.to(dtype=torch.float64)
        dist = self.distances(R)
        inv_c = 1.0 / dist.clamp_min(self.cfg.coulomb_floor)
        vc = KE2 * self.qij * inv_c
        floor = self.cfg.lj_floor_frac * self.sigij
        inv_lj = 1.0 / torch.maximum(dist, floor)
        sr = self.sigij * inv_lj
        sr6 = sr ** 6
        vlj = 4.0 * self.epsij * (sr6 * sr6 - sr6)
        pair = torch.where(self.coulomb_mask, vc, torch.zeros_like(vc))
        pair = pair + torch.where(self.lj_mask, vlj, torch.zeros_like(vlj))
        v_pair = 0.5 * pair.sum(dim=(-2, -1))
        return v_pair + self._bonded(R) + self._lattice(R)

    def _bonded(self, R: torch.Tensor) -> torch.Tensor:
        c, oa, ob = R[..., 3, :], R[..., 4, :], R[..., 5, :]
        r1 = torch.linalg.norm(c - oa, dim=-1)
        r2 = torch.linalg.norm(c - ob, dim=-1)
        v_bond = 0.5 * self.cfg.k_co2_bond * (
            (r1 - self.cfg.r_co2_bond) ** 2 + (r2 - self.cfg.r_co2_bond) ** 2
        )
        u1 = F.normalize(oa - c, dim=-1)
        u2 = F.normalize(ob - c, dim=-1)
        cosang = (u1 * u2).sum(-1).clamp(-1.0, 1.0)
        v_ang = 0.5 * self.cfg.k_co2_angle * (cosang + 1.0) ** 2
        return v_bond + v_ang

    def _lattice(self, R: torch.Tensor) -> torch.Tensor:
        d = R[..., list(self.frag.framework), :] - self.lattice
        return 0.5 * self.cfg.k_lattice * (d ** 2).sum(dim=(-2, -1))

    def al_c(self, R: torch.Tensor) -> torch.Tensor:
        return torch.linalg.norm(R[..., 3, :] - R[..., 0, :], dim=-1)

    def components(self, R: torch.Tensor) -> dict[str, float]:
        R = R.to(dtype=torch.float64)
        total = self.energy(R)
        bonded = self._bonded(R)
        lat = self._lattice(R)
        return {
            "total": float(total.detach()),
            "bonded": float(bonded.detach()),
            "lattice": float(lat.detach()),
            "nonbonded": float((total - bonded - lat).detach()),
            "r_alc": float(self.al_c(R).detach()),
        }
