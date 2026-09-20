import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pioc.potential import PairPotential, PotentialConfig
from pioc.system import build_fragment, endpoints
from pioc.validate import check_endpoints, check_fragment
from pioc.paths import FourierBridge


def test_charge_neutral():
    frag = build_fragment("cpu")
    check_fragment(frag)
    assert abs(frag.charge_sum()) < 1e-6


def test_endpoint_energies_sane():
    frag = build_fragment("cpu")
    xa, xb = endpoints("cpu")
    pot = PairPotential(frag, xa[list(frag.framework)], PotentialConfig())
    g = check_endpoints(pot, xa, xb)
    assert 2.0 < g["r_B"] < 4.5
    assert 4.5 < g["r_A"] < 7.0


def test_open_path_endpoints():
    xa, xb = endpoints("cpu")
    gen = FourierBridge()
    paths, _, _, _, _ = gen.sample(xa, xb, n=4)
    e0, e1 = gen.endpoint_error(xa, xb, paths)
    assert e0 < 1e-5 and e1 < 1e-5
