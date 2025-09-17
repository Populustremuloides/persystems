import numpy as np
from persystems.gm import GenerativeModel
from persystems.planning import choose_action_planner, choose_action_beam

def test_depth_planner_vs_beam_reasonable():
    gm = GenerativeModel.make_ring_world(N=5, A_eps=0.15, target_idx=3)
    qs = np.ones(5)/5
    a1, comp1, Gs1, d1 = choose_action_planner(qs, gm.A, gm.B, gm.C, horizon=2, obs_prune_eps=1e-6)
    a2, comp2, Gs2, d2 = choose_action_beam(qs, gm.A, gm.B, gm.C, horizon=2, beam_width=8, obs_prune_eps=1e-6)
    assert d1.nodes_expanded > 0 and d2.nodes_expanded > 0
    # They should both return finite vectors and a valid action index
    assert np.all(np.isfinite(Gs1)) and np.all(np.isfinite(Gs2))
    assert 0 <= a1 < len(gm.B)
    assert 0 <= a2 < len(gm.B)

