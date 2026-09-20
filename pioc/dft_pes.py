"""Gas-phase DFT for Al(OH)3 + CO2. PBE0/def2-SVP, rigid O-down, Boys-Bernardi CP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .constants import EV_TO_KJ_MOL, K_B

HARTREE_EV = 27.211386245981
R_CO2 = 1.16


@dataclass
class DFTConfig:
    xc: str = "PBE0"
    basis: str = "def2-SVP"
    charge: int = 0
    spin: int = 0
    conv_tol: float = 1e-6
    n_scan: int = 10
    r_min: float = 1.90
    r_max: float = 4.80
    temperature_k: float = 313.15
    pressure_pa: float = 101325.0
    do_thermo: bool = False
    verbose: int = 0


def _aloh3() -> str:
    return """
Al  0.000000  0.000000  0.000000
O   1.720000  0.000000  0.300000
H   2.480000  0.000000 -0.150000
O  -0.860000  1.489000  0.300000
H  -1.240000  2.150000 -0.150000
O  -0.860000 -1.489000  0.300000
H  -1.240000 -2.150000 -0.150000
"""


def _co2_isolated() -> str:
    return f"O 0 0 0\nC 0 0 {R_CO2}\nO 0 0 {2 * R_CO2}"


def _complex(r_alo: float) -> str:
    return _aloh3() + (
        f"O  0.000000  0.000000  {r_alo:.6f}\n"
        f"C  0.000000  0.000000  {r_alo + R_CO2:.6f}\n"
        f"O  0.000000  0.000000  {r_alo + 2 * R_CO2:.6f}\n"
    )


def _complex_ghost_aloh3(r_alo: float) -> str:
    body = []
    for line in _aloh3().strip().splitlines():
        el, x, y, z = line.split()
        body.append(f"{el} {x} {y} {z}")
    body.append(f"ghost:O  0.000000  0.000000  {r_alo:.6f}")
    body.append(f"ghost:C  0.000000  0.000000  {r_alo + R_CO2:.6f}")
    body.append(f"ghost:O  0.000000  0.000000  {r_alo + 2 * R_CO2:.6f}")
    return "\n".join(body)


def _complex_ghost_co2(r_alo: float) -> str:
    body = []
    for line in _aloh3().strip().splitlines():
        el, x, y, z = line.split()
        body.append(f"ghost:{el} {x} {y} {z}")
    body.append(f"O  0.000000  0.000000  {r_alo:.6f}")
    body.append(f"C  0.000000  0.000000  {r_alo + R_CO2:.6f}")
    body.append(f"O  0.000000  0.000000  {r_alo + 2 * R_CO2:.6f}")
    return "\n".join(body)


def _scf(atom: str, cfg: DFTConfig, dm0=None):
    from pyscf import dft, gto
    mol = gto.M(atom=atom, basis=cfg.basis, charge=cfg.charge, spin=cfg.spin, unit="Angstrom", verbose=cfg.verbose)
    mf = dft.RKS(mol)
    mf.xc = cfg.xc
    mf.conv_tol = cfg.conv_tol
    e = mf.kernel(dm0=dm0)
    if not mf.converged:
        mf.level_shift = 0.4
        mf.max_cycle = 100
        e = mf.kernel(dm0=mf.make_rdm1())
    if not mf.converged:
        raise RuntimeError("SCF failed to converge")
    return float(e), mf


def rrho_correction_ev(mf, temperature_k: float, pressure_pa: float) -> dict[str, Any]:
    from pyscf.hessian import thermo
    hess = mf.Hessian().kernel()
    ha = thermo.harmonic_analysis(mf.mol, hess)
    th = thermo.thermo(mf, ha["freq_au"], temperature=temperature_k, pressure=pressure_pa)
    e0 = float(th["E0"][0]) * HARTREE_EV
    return {
        "zpe_ev": float(th["ZPE"][0]) * HARTREE_EV,
        "G_minus_E0_ev": float(th["G_tot"][0]) * HARTREE_EV - e0,
        "H_minus_E0_ev": float(th["H_tot"][0]) * HARTREE_EV - e0,
        "freq_cm": [float(x) for x in ha["freq_wavenumber"]],
    }


def run_dft_scan(cfg: DFTConfig) -> dict:
    rs = np.linspace(cfg.r_min, cfg.r_max, cfg.n_scan)
    e_tot = []
    dm = None
    for r in rs:
        e, mf = _scf(_complex(float(r)), cfg, dm0=dm)
        dm = mf.make_rdm1()
        e_tot.append(e)
    e_tot = np.asarray(e_tot, dtype=float)
    e_aloh3, mf_aloh3 = _scf(_aloh3(), cfg)
    e_co2, mf_co2 = _scf(_co2_isolated(), cfg)
    e_int = (e_tot - e_aloh3 - e_co2) * HARTREE_EV
    i_min = int(np.argmin(e_int))
    r_min = float(rs[i_min])
    dE = float(e_int[i_min])
    e_aloh_gh, _ = _scf(_complex_ghost_aloh3(r_min), cfg)
    e_co2_gh, _ = _scf(_complex_ghost_co2(r_min), cfg)
    dE_cp = float((e_tot[i_min] - e_aloh_gh - e_co2_gh) * HARTREE_EV)
    thermo_out: dict[str, Any] = {}
    if cfg.do_thermo:
        _, mf_min = _scf(_complex(r_min), cfg)
        t, p = cfg.temperature_k, cfg.pressure_pa
        th_c = rrho_correction_ev(mf_min, t, p)
        th_a = rrho_correction_ev(mf_aloh3, t, p)
        th_b = rrho_correction_ev(mf_co2, t, p)
        dG = dE_cp + th_c["G_minus_E0_ev"] - th_a["G_minus_E0_ev"] - th_b["G_minus_E0_ev"]
        thermo_out = {"complex": th_c, "aloh3": th_a, "co2": th_b,
                     "dG_rrho_cp_ev": dG, "dG_rrho_cp_kj": dG * EV_TO_KJ_MOL}
    return {
        "level": f"{cfg.xc}/{cfg.basis}",
        "cluster": "Al(OH)3 + CO2 (O-down, rigid)",
        "r_coord": "Al-O(CO2) / A",
        "r_A": rs.tolist(),
        "E_tot_Eh": e_tot.tolist(),
        "E_int_eV": e_int.tolist(),
        "E_AlOH3_Eh": e_aloh3,
        "E_CO2_Eh": e_co2,
        "r_min_A": r_min,
        "dE_raw_eV": dE,
        "dE_cp_eV": dE_cp,
        "bsse_eV": dE - dE_cp,
        "dE_cp_kj": dE_cp * EV_TO_KJ_MOL,
        "kT_eV": K_B * cfg.temperature_k,
        "thermo": thermo_out,
        "disclaimer": "Gas-phase Al(OH)3-CO2 cluster DFT, rigid scan. PBE0 lacks long-range dispersion. Not periodic MOF DFT.",
    }
