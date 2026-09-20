"""Internal units: eV, Angstrom, fs, amu."""

from __future__ import annotations

K_B = 8.617333262e-5
HBAR = 0.6582119569
U_MASS = 103.6427
KE2 = 14.399645
EV_TO_KJ_MOL = 96.485
DEFAULT_T = 313.15


def beta(temperature_k: float) -> float:
    if temperature_k <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature_k}")
    return 1.0 / (K_B * temperature_k)
