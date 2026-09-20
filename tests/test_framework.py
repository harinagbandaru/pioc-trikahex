import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from pioc.bath import GLEBath, GLEConfig
from pioc.manifold import build_state
from pioc.symmetry import center, random_so3, rotate
from pioc.system import build_fragment, endpoints
from pioc.trikahex import TrikaHex
from pioc.trikahex_bridge import TrikaHexBridge


def test_so3_preserves_distances():
    st = build_state("A")
    Q = random_so3(dtype=st.R.dtype)
    Rp = rotate(center(st.R), Q)
    d0 = torch.cdist(center(st.R), center(st.R))
    d1 = torch.cdist(Rp, Rp)
    assert torch.allclose(d0, d1, atol=1e-8)
    assert abs(float(torch.det(Q)) - 1.0) < 1e-6


def test_trikahex_shapes_and_bridge_endpoints():
    frag = build_fragment()
    xa, xb = endpoints()
    matter = torch.stack([frag.z, frag.masses, frag.chi, frag.valence], dim=-1)
    net = TrikaHex(n_atoms=6)
    mu, sig = net(matter, xa, xb)
    assert mu.shape == (8, 6, 3)
    assert torch.isfinite(mu).all()
    gen = TrikaHexBridge(n_atoms=6)
    paths, _, _, _, _ = gen.sample(matter, xa, xb, n=3)
    e0, e1 = gen.endpoint_error(xa, xb, paths)
    assert e0 < 1e-4 and e1 < 1e-4


def test_gle_finite():
    bath = GLEBath(GLEConfig())
    v = torch.randn(2, 4, 6, 3, dtype=torch.float64) * 0.01
    s = bath.influence(v, 0.4)
    assert s.shape == (2,)
    assert torch.isfinite(s).all()
