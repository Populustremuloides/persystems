"""
CLI to run the ring-world active inference demo and plot results.
"""
import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path when running as a script
if __package__ is None or __package__ == '':
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from persystems.gm import GenerativeModel
from persystems.inference import bayes_predict, bayes_update, belief_entropy
from persystems.planning import choose_action
from persystems.viz import (
    plot_efe_trace, plot_cost_info, plot_entropy, plot_final_posterior
)


def run(N=5, T=30, target=3, A_eps=0.15, horizon=1, seed=7):
    np.random.seed(seed)
    gm = GenerativeModel.make_ring_world(N=N, A_eps=A_eps, target_idx=target)
    qs = np.ones(N) / N
    true_s = np.random.randint(0, N)

    hist = {k: [] for k in
            ["t", "true_s", "obs", "a", "G", "risk", "ambiguity", "exp_cost", "info_gain", "Hq"]}

    for t in range(T):
        a_idx, comp, Gs = choose_action(qs, gm.A, gm.B, gm.C, horizon=horizon)
        a = gm.actions[a_idx]

        # environment transition (deterministic ring for clarity)
        true_s = (true_s + a) % N

        # sample observation
        o = int(np.random.choice(np.arange(N), p=gm.A[:, true_s]))

        # Bayes filter
        q_pred = bayes_predict(qs, gm.B[a_idx])
        qs = bayes_update(q_pred, gm.A, o)

        hist["t"].append(t)
        hist["true_s"].append(true_s)
        hist["obs"].append(o)
        hist["a"].append(a)
        hist["G"].append(Gs[a_idx])
        hist["risk"].append(comp["risk"])
        hist["ambiguity"].append(comp["ambiguity"])
        hist["exp_cost"].append(comp["expected_cost"])
        hist["info_gain"].append(comp["info_gain"])
        hist["Hq"].append(belief_entropy(qs))

    # to arrays
    for k in hist:
        hist[k] = np.array(hist[k])

    # plots
    plot_efe_trace(hist["t"], hist["G"], hist["risk"], hist["ambiguity"])
    plot_cost_info(hist["t"], hist["exp_cost"], hist["info_gain"])
    plot_entropy(hist["t"], hist["Hq"])
    plot_final_posterior(qs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=5)
    ap.add_argument("--T", type=int, default=30)
    ap.add_argument("--target", type=int, default=3)
    ap.add_argument("--A_eps", type=float, default=0.15)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    run(**vars(args))

