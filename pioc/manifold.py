"""H_chem = M x X x G x I container."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .bath import GLEBath, GLEConfig
from .system import Fragment, build_fragment, endpoints
from .symmetry import center


@dataclass
class ChemicalState:
    frag: Fragment
    R: torch.Tensor
    bath: GLEBath

    def distance(self) -> torch.Tensor:
        d = self.R.unsqueeze(-2) - self.R.unsqueeze(-3)
        return torch.linalg.norm(d, dim=-1)

    def matter_table(self) -> torch.Tensor:
        return torch.stack([self.frag.z, self.frag.masses, self.frag.chi, self.frag.valence], dim=-1)


def build_state(which: str = "A", device="cpu") -> ChemicalState:
    frag = build_fragment(device)
    xa, xb = endpoints(device)
    R = xa if which.upper() == "A" else xb
    bath = GLEBath(GLEConfig(n_atoms=frag.n_atoms), device=device)
    return ChemicalState(frag=frag, R=center(R), bath=bath)
