"""Framework orchestrator. Runs specialized builders and writes a stack report."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .ais import AISConfig, path_ais
from .bath import GLEBath, GLEConfig
from .constants import DEFAULT_T, beta
from .manifold import build_state
from .potential import PairPotential, PotentialConfig
from .symmetry import center, random_so3, rotate
from .system import build_fragment, endpoints
from .train import TrainConfig, train_proposal
from .trikahex import TrikaHex
from .trikahex_bridge import TrikaHexBridge
from .validate import check_endpoints, check_fragment, check_paths
from .wham import UmbrellaConfig, run_umbrellas
from .paths import FourierBridge


def _agent_manifold() -> dict:
    st = build_state("A")
    q = st.frag.charge_sum()
    D = st.distance()
    return {"agent": "manifold", "ok": abs(q) < 1e-6 and torch.isfinite(D).all().item(),
            "N": st.frag.n_atoms, "charge": q, "d_max": float(D.max()),
            "matter_shape": list(st.matter_table().shape)}


def _agent_symmetry() -> dict:
    st = build_state("A")
    Q = random_so3(device=st.R.device, dtype=st.R.dtype)
    Rp = rotate(center(st.R), Q)
    err = float((torch.cdist(st.R, st.R) - torch.cdist(Rp, Rp)).abs().max())
    return {"agent": "symmetry", "ok": err < 1e-8, "so3_dist_err": err, "detQ": float(torch.det(Q))}


def _agent_bath() -> dict:
    bath = GLEBath(GLEConfig())
    v = torch.zeros(2, 8, 6, 3, dtype=torch.float64)
    v[:, 1] = 0.01
    s = bath.influence(v, dtau=0.5)
    return {"agent": "bath", "ok": bool(torch.isfinite(s).all()), "S_sample": [float(x) for x in s]}


def _agent_trikahex() -> dict:
    frag = build_fragment()
    xa, xb = endpoints()
    net = TrikaHex(n_atoms=6)
    matter = torch.stack([frag.z, frag.masses, frag.chi, frag.valence], dim=-1)
    mu, sig = net(matter, xa, xb)
    return {"agent": "trikahex", "ok": mu.shape == (8, 6, 3) and torch.isfinite(mu).all().item(),
            "n_param": sum(p.numel() for p in net.parameters()),
            "mu_abs_max": float(mu.abs().max().detach()), "sig_mean": float(sig.mean().detach())}


def _agent_bridge_ais() -> dict:
    device = torch.device("cpu")
    frag = build_fragment(device)
    xa, xb = endpoints(device)
    pot = PairPotential(frag, xa[list(frag.framework)], PotentialConfig())
    check_fragment(frag)
    geom = check_endpoints(pot, xa, xb)
    matter = torch.stack([frag.z, frag.masses, frag.chi, frag.valence], dim=-1)
    gen = TrikaHexBridge(n_atoms=6)
    paths, modes, log_q, mu, sig = gen.sample(matter, xa, xb, n=8)
    e0, e1 = gen.endpoint_error(xa, xb, paths)
    check_paths(e0, e1, tol=5e-4)
    return {"agent": "bridge", "ok": e0 < 5e-4 and e1 < 5e-4, "endpoint_err": [e0, e1],
            "log_q_mean": float(log_q.mean().detach()), "V_A": geom["V_A"], "V_B": geom["V_B"]}


def _agent_empirical_short() -> dict:
    device = torch.device("cpu")
    torch.manual_seed(42)
    frag = build_fragment(device)
    xa, xb = endpoints(device)
    pot = PairPotential(frag, xa[list(frag.framework)], PotentialConfig())
    gen = FourierBridge().to(device)
    b = beta(DEFAULT_T)
    train = train_proposal(gen, pot, xa, xb, b, TrainConfig(steps=15, batch=12))
    ais = path_ais(gen, pot, xa, xb, b, AISConfig(n_paths=16, n_stages=4, mala_steps=2))
    umb = run_umbrellas(pot, xa, xb, b, UmbrellaConfig(n_windows=5, n_steps=180, n_boot=8, n_bins=20), seed=42)
    return {"agent": "empirical_pipeline", "ok": ais["ess"] > 0,
            "train_E": train["E"], "ess": ais["ess"], "dA_eV": umb["dA_bound_unbound"]}


AGENTS = (_agent_manifold, _agent_symmetry, _agent_bath, _agent_trikahex, _agent_bridge_ais, _agent_empirical_short)


def launch(outdir: str = "out") -> dict:
    reports, failed = [], []
    for fn in AGENTS:
        try:
            rep = fn()
        except Exception as exc:
            rep = {"agent": fn.__name__, "ok": False, "error": repr(exc)}
        reports.append(rep)
        if not rep.get("ok"):
            failed.append(rep.get("agent", fn.__name__))
    payload = {"framework": "PIOC-TrikaHex", "agents_run": len(reports),
               "agents_failed": failed, "all_ok": len(failed) == 0, "reports": reports}
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "framework_launch.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    payload["report_path"] = str(path)
    return payload
