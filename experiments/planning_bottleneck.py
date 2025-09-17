"""
Run depth-H planning with pruning/beam search and print diagnostics.
"""
import argparse, numpy as np

if __package__ is None or __package__ == "":
    import os, sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from persystems.gm import GenerativeModel
from persystems.planning import choose_action_planner, choose_action_beam

def main(N=5, A_eps=0.15, target=3, H=3, beam=16, prune=1e-4, trials=20, seed=0):
    rng = np.random.default_rng(seed)
    gm = GenerativeModel.make_ring_world(N=N, A_eps=A_eps, target_idx=target)
    qs = np.ones(N) / N
    print(f"Depth={H}, beam={beam}, prune_eps={prune}")
    # Planner with pruning
    a_idx, comp, Gs, diag = choose_action_planner(qs, gm.A, gm.B, gm.C, horizon=H, obs_prune_eps=prune)
    print("Planner:", dict(best_action=a_idx, nodes=diag.nodes_expanded, pruned=diag.pruned_obs))
    # Beam
    a_idx2, comp2, Gs2, diag2 = choose_action_beam(qs, gm.A, gm.B, gm.C, horizon=H, beam_width=beam, obs_prune_eps=prune)
    print("Beam   :", dict(best_action=a_idx2, nodes=diag2.nodes_expanded, pruned=diag2.pruned_obs))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=5)
    ap.add_argument("--A_eps", type=float, default=0.15)
    ap.add_argument("--target", type=int, default=3)
    ap.add_argument("--H", type=int, default=3)
    ap.add_argument("--beam", type=int, default=16)
    ap.add_argument("--prune", type=float, default=1e-4)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    main(**vars(args))

