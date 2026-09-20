"""Save energy-curve figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_energy_png(path: Path, x: np.ndarray, y: np.ndarray, ylabel: str, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok = np.isfinite(y)
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=140)
    ax.plot(x[ok], y[ok], color="#1f4e79", lw=2.0, marker="o", ms=4)
    ax.axhline(0.0, color="0.4", lw=0.6)
    ax.set_xlabel(r"$r(\mathrm{Al-C})$ / Angstrom")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def save_pmf_png(path: Path, r: np.ndarray, w: np.ndarray, w_err: np.ndarray, dA: float) -> None:
    save_energy_png(path, r, w, ylabel=r"$W(r)$ / eV", title=f"Toy PMF  dA = {dA:.3f} eV")
