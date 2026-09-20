"""Charge-neutral 6-site Al-OH + CO2 fragment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

ATOM_NAMES: tuple[str, ...] = ("Al", "O_pore", "H", "C", "O_a", "O_b")
FRAMEWORK: tuple[int, ...] = (0, 1, 2)
MOBILE: tuple[int, ...] = (3, 4, 5)
BOND_EXCLUDE: tuple[tuple[int, int], ...] = ((0, 1), (1, 2), (3, 4), (3, 5))

Z = (13.0, 8.0, 1.0, 6.0, 8.0, 8.0)
MASSES_AMU = (26.9815, 15.999, 1.008, 12.011, 15.999, 15.999)
CHI = (1.61, 3.44, 2.20, 2.55, 3.44, 3.44)
VALENCE = (3.0, 2.0, 1.0, 4.0, 2.0, 2.0)
Q_EFF = (1.20, -0.70, 0.30, 0.70, -0.75, -0.75)
LJ_EPS = (0.08, 0.06, 0.01, 0.04, 0.05, 0.05)
LJ_SIG = (2.70, 3.00, 2.20, 3.30, 3.05, 3.05)
R_CO2 = 1.16


@dataclass(frozen=True)
class Fragment:
    names: tuple[str, ...]
    z: torch.Tensor
    masses: torch.Tensor
    chi: torch.Tensor
    valence: torch.Tensor
    q: torch.Tensor
    eps: torch.Tensor
    sig: torch.Tensor
    framework: tuple[int, ...]
    mobile: tuple[int, ...]
    bond_exclude: tuple[tuple[int, int], ...]

    @property
    def n_atoms(self) -> int:
        return len(self.names)

    def charge_sum(self) -> float:
        return float(self.q.sum().item())


def build_fragment(device: torch.device | str = "cpu") -> Fragment:
    dev = torch.device(device)
    q = torch.tensor(Q_EFF, dtype=torch.float64, device=dev)
    if abs(float(q.sum())) > 1e-6:
        raise RuntimeError(f"fragment is not charge-neutral: {float(q.sum())}")
    return Fragment(
        names=ATOM_NAMES,
        z=torch.tensor(Z, dtype=torch.float64, device=dev),
        masses=torch.tensor(MASSES_AMU, dtype=torch.float64, device=dev),
        chi=torch.tensor(CHI, dtype=torch.float64, device=dev),
        valence=torch.tensor(VALENCE, dtype=torch.float64, device=dev),
        q=q,
        eps=torch.tensor(LJ_EPS, dtype=torch.float64, device=dev),
        sig=torch.tensor(LJ_SIG, dtype=torch.float64, device=dev),
        framework=FRAMEWORK,
        mobile=MOBILE,
        bond_exclude=BOND_EXCLUDE,
    )


def _co2(c_xyz: Sequence[float], axis: str) -> list[list[float]]:
    cx, cy, cz = c_xyz
    if axis == "z":
        oa, ob = [cx, cy, cz - R_CO2], [cx, cy, cz + R_CO2]
    elif axis == "x":
        oa, ob = [cx + R_CO2, cy, cz], [cx - R_CO2, cy, cz]
    else:
        raise ValueError(axis)
    return [list(c_xyz), oa, ob]


def endpoints(device: torch.device | str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """A: CO2 at pore mouth (axis z). B: side-on physisorption pose (axis x)."""
    dev = torch.device(device)
    al, op, h = [0.00, 0.00, 0.00], [1.80, 0.00, 0.00], [2.15, 0.92, 0.10]
    cA, oaA, obA = _co2([0.20, 0.05, 5.50], "z")
    cB, oaB, obB = _co2([0.25, 0.05, 3.05], "x")
    xa = torch.tensor([al, op, h, cA, oaA, obA], dtype=torch.float64, device=dev)
    xb = torch.tensor([al, op, h, cB, oaB, obB], dtype=torch.float64, device=dev)
    return xa, xb
