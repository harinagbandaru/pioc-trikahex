"""End-to-end runners for the empirical toy and the DFT cluster model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .ais import AISConfig, path_ais
from .committor import committor_1d
from .constants import DEFAULT_T, EV_TO_KJ_MOL, beta
from .dft_pes import DFTConfig, run_dft_scan
from .paths import FourierBridge
from .plotting import save_energy_png
from .potential import PairPotential, PotentialConfig
from .system import build_fragment, endpoints
from .train import TrainConfig, train_proposal
from .validate import check_endpoints, check_fragment, check_paths
from .wham import UmbrellaConfig, run_umbrellas


@dataclass
class RunConfig:
    seed: int = 42
    temperature_k: float = DEFAULT_T
    engine: str = "dft"
    outdir: str = "out"


def seed_all(seed: int) -> torch.device:
    np.random.seed(seed)
    torch.manual_seed(seed)
    return torch.device("cpu")


def run_empirical(cfg: RunConfig) -> dict:
    device = seed_all(cfg.seed)
    frag = build_fragment(device)
    check_fragment(frag)
    xa, xb = endpoints(device)
    pot = PairPotential(frag, xa[list(frag.framework)], PotentialConfig())
    geom = check_endpoints(pot, xa, xb)
    b = beta(cfg.temperature_k)
    gen = FourierBridge().to(device)
    train = train_proposal(gen, pot, xa, xb, b, TrainConfig())
    ais = path_ais(gen, pot, xa, xb, b, AISConfig())
    check_paths(ais["endpoint_err_A"], ais["endpoint_err_B"], tol=5e-4)
    umb = run_umbrellas(pot, xa, xb, b, UmbrellaConfig(), seed=cfg.seed)
    qfun, qvals = committor_1d(umb["r"], umb["W"], b)
    dA = umb["dA_bound_unbound"]
    return {
        "engine": "empirical_lj_coulomb",
        "geometry": geom,
        "train": train,
        "ais": {
            "ess": ais["ess"],
            "ess_hist": ais["ess_hist"],
            "n_resample": ais["n_resample"],
            "mala_accept": ais["mala_accept"],
            "r_mid_wmean": ais["r_mid_wmean"],
            "e_wmean": ais["e_wmean"],
            "endpoint_err_A": ais["endpoint_err_A"],
            "endpoint_err_B": ais["endpoint_err_B"],
        },
        "pmf": {
            "r": umb["r"].tolist(),
            "W": umb["W"].tolist(),
            "W_err": umb["W_err"].tolist(),
            "dA_eV": dA,
            "dA_kj": None if dA != dA else dA * EV_TO_KJ_MOL,
            "accept": umb["window_accept"],
        },
        "committor": qvals,
        "disclaimer": "Empirical Coulomb+LJ fragment. Not DFT.",
    }


def run_dft(cfg: RunConfig, dft_cfg: DFTConfig | None = None) -> dict:
    seed_all(cfg.seed)
    dft_cfg = dft_cfg or DFTConfig(temperature_k=cfg.temperature_k)
    scan = run_dft_scan(dft_cfg)
    return {"engine": "pyscf_dft_cluster", **scan}


def write_outputs(payload: dict, outdir: str, engine: str) -> dict[str, str]:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / f"report_{engine}.json"
    npz_path = out / f"curve_{engine}.npz"

    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items() if k not in {"paths", "modes", "weights", "energy", "r_mid"}}
        if isinstance(obj, (list, tuple)):
            return [_clean(v) for v in obj]
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    json_path.write_text(json.dumps(_clean(payload), indent=2))
    if engine == "dft":
        np.savez(npz_path, r=np.array(payload["r_A"]), E=np.array(payload["E_int_eV"]))
        png = out / "dft_binding_curve.png"
        save_energy_png(png, np.array(payload["r_A"]), np.array(payload["E_int_eV"]),
                        ylabel="E_int / eV",
                        title=f"{payload['level']}  dE_CP = {payload['dE_cp_eV']:.3f} eV")
    else:
        pmf = payload["pmf"]
        np.savez(npz_path, r=np.array(pmf["r"]), W=np.array(pmf["W"]))
        png = out / "empirical_pmf.png"
        save_energy_png(png, np.array(pmf["r"]), np.array(pmf["W"]),
                        ylabel="W(r) / eV", title="Empirical WHAM PMF")
    return {"json": str(json_path), "npz": str(npz_path), "png": str(png)}
