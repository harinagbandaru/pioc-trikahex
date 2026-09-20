"""1-D overdamped committor on a tabulated PMF."""

from __future__ import annotations

import numpy as np


def committor_1d(r: np.ndarray, w: np.ndarray, beta: float, r_unbound: float = 5.3, r_bound: float = 2.8):
    order = np.argsort(r)
    r = r[order]
    w = w[order].astype(float)
    ok = np.isfinite(w)
    r, w = r[ok], w[ok]
    if len(r) < 4:
        raise RuntimeError("PMF too sparse for a committor")
    w = w - np.nanmin(w)
    integ_dens = np.exp(np.clip(beta * w, None, 40.0))
    integ = np.zeros_like(r)
    for i in range(1, len(r)):
        integ[i] = integ[i - 1] + 0.5 * (integ_dens[i] + integ_dens[i - 1]) * abs(r[i] - r[i - 1])

    def q_of(x: float) -> float:
        i = int(np.clip(np.searchsorted(r, x), 0, len(r) - 1))
        iB = int(np.clip(np.searchsorted(r, min(r_bound, r_unbound)), 0, len(r) - 1))
        iA = int(np.clip(np.searchsorted(r, max(r_bound, r_unbound)), 0, len(r) - 1))
        denom = integ[iA] - integ[iB]
        if abs(denom) < 1e-30:
            return 0.5
        return float(np.clip((integ[iA] - integ[i]) / denom, 0.0, 1.0))

    return q_of, {"q_unbound": q_of(r_unbound), "q_mid": q_of(0.5 * (r_bound + r_unbound)), "q_bound": q_of(r_bound)}
