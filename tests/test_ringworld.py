import numpy as np
from persystems.gm import GenerativeModel, softmax
from persystems.efe import efe_one_step


def test_stochasticity():
    gm = GenerativeModel.make_ring_world(N=5)
    # A columns sum to 1
    assert np.allclose(gm.A.sum(axis=0), 1.0)
    # B columns sum to 1
    for B_a in gm.B:
        assert np.allclose(B_a.sum(axis=0), 1.0)


def test_efe_decompositions_close():
    gm = GenerativeModel.make_ring_world(N=5)
    qs = np.ones(5) / 5
    P = softmax(gm.C)
    for B_a in gm.B:
        G, comp = efe_one_step(qs, gm.A, B_a, P)
        # non-negativity sanity
        assert comp["risk"] >= -1e-9
        assert comp["expected_cost"] >= -1e-9
        # equivalence up to constants (here equals within tolerance)
        G_alt = comp["expected_cost"] - comp["info_gain"]
        assert abs(G - G_alt) < 1e-6

