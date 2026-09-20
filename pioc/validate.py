"""Hard invariants. Fail the run if the calculator is not well-posed."""

from __future__ import annotations

import math

import torch

from .potential import PairPotential
from .system import Fragment


class InvariantError(RuntimeError):
    pass


def require_finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise InvariantError(f"{name} is not finite: {value}")


def check_fragment(frag: Fragment, tol: float = 1e-6) -> None:
    if abs(frag.charge_sum()) > tol:
        raise InvariantError(f"charge sum {frag.charge_sum()} exceeds {tol}")
    if frag.n_atoms != 6:
        raise InvariantError("this build expects the 6-site fragment")
    if set(frag.framework) & set(frag.mobile):
        raise InvariantError("framework and mobile index sets overlap")


def check_endpoints(pot: PairPotential, xa: torch.Tensor, xb: torch.Tensor) -> dict[str, float]:
    va = pot.components(xa)
    vb = pot.components(xb)
    for label, comp in (("A", va), ("B", vb)):
        require_finite(f"V({label})", comp["total"])
        if abs(comp["total"]) > 200:
            raise InvariantError(f"V({label})={comp['total']} eV is outside the toy range")
        if comp["r_alc"] < 1.5 or comp["r_alc"] > 8.0:
            raise InvariantError(f"r_AlC({label})={comp['r_alc']} A is implausible")
    return {"V_A": va["total"], "V_B": vb["total"], "dV": vb["total"] - va["total"],
            "r_A": va["r_alc"], "r_B": vb["r_alc"]}


def check_paths(err_a: float, err_b: float, tol: float = 1e-4) -> None:
    if err_a > tol or err_b > tol:
        raise InvariantError(f"open-path endpoints drifted: dA={err_a}, dB={err_b}")
