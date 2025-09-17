"""
Expected Free Energy computations (risk + ambiguity) and (cost − information_gain).

Notation (one-step lookahead under action a):
- Current belief over states: qs
- Transition under action a: B_a (S x S)
- Likelihood: A (O x S) with columns P(o|s)
- Preferred outcomes: P_o_pref (O,)

Two equivalent decompositions (equal up to constants):
  G = D_KL(Q(o) || P(o)) + E_{Q(s')}[ H(P(o|s')) ]                    (risk + ambiguity)
  G = E_{Q(o)}[ -log P(o) ] - I_Q(S'; O)                              (expected cost − information gain)

Depth-H version enumerates observation branches exactly (for small problems).
"""
from __future__ import annotations
import numpy as np
from typing import Tuple, Dict, List
from .gm import entropy
from .inference import bayes_predict, bayes_update


def efe_one_step(
    qs: np.ndarray,
    A: np.ndarray,
    B_a: np.ndarray,
    P_o_pref: np.ndarray
) -> Tuple[float, Dict[str, float]]:
    """
    Compute one-step Expected Free Energy for a fixed action a.

    Args:
        qs: (S,) current belief over states
        A:  (O,S) likelihood P(o|s)
        B_a:(S,S) transition for action a
        P_o_pref: (O,) preferred outcome distribution P(o) = softmax(C)

    Returns:
        (G, components)
        G: scalar expected free energy
        components: dict with keys: 'risk', 'ambiguity', 'expected_cost', 'info_gain'
    """
    # Predictive prior over next state
    q_pred = bayes_predict(qs, B_a)            # q(s'|a)
    # Predictive outcomes
    Q_o = A @ q_pred                           # Q(o|a) = sum_s' A[o,s'] q_pred[s']
    eps = 1e-12

    # --- Decomposition 1: risk + ambiguity ---
    # risk = D_KL(Q(o) || P(o))
    risk = float(
        np.sum(
            Q_o * (np.log(np.clip(Q_o, eps, 1.0)) - np.log(np.clip(P_o_pref, eps, 1.0)))
        )
    )
    # ambiguity = E_{q(s')}[ H(P(o|s')) ]
    H_cols = np.array([entropy(A[:, s]) for s in range(A.shape[1])])
    ambiguity = float(np.dot(q_pred, H_cols))

    # --- Decomposition 2: expected cost − information gain ---
    expected_cost = float(-np.sum(Q_o * np.log(np.clip(P_o_pref, eps, 1.0))))
    # info gain = H(q_pred) − E_{Q(o)}[ H(q_post(o)) ]
    H_prior = entropy(q_pred)
    H_post = 0.0
    for o in range(A.shape[0]):
        q_post = bayes_update(q_pred, A, o)
        H_post += Q_o[o] * entropy(q_post)
    info_gain = float(H_prior - H_post)

    G = risk + ambiguity
    return G, {
        "risk": risk,
        "ambiguity": ambiguity,
        "expected_cost": expected_cost,
        "info_gain": info_gain,
    }


def efe_depth_H(
    qs: np.ndarray,
    A: np.ndarray,
    B: List[np.ndarray],
    P_o_pref: np.ndarray,
    H: int,
    gamma: float = 1.0
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    """
    Depth-H EFE for each immediate action (exact enumeration over future observations).
    This explodes as (|A||O|)^H; keep H small for discrete demos.

    Args:
        qs: (S,)
        A: (O,S)
        B: list[(S,S)]
        P_o_pref: (O,)
        H: search depth (H=1 reduces to efe_one_step per action)
        gamma: discount on tail (default 1.0)

    Returns:
        Gs:   (A,) EFE per *immediate* action
        comps:list of component dicts (risk, ambiguity, expected_cost, info_gain) for step 1
    """
    nA = len(B)
    Gs = np.zeros(nA)
    comps: List[Dict[str, float]] = []

    for a_idx, B_a in enumerate(B):
        # First-step contribution
        G1, comp = efe_one_step(qs, A, B_a, P_o_pref)
        comps.append(comp)

        if H == 1:
            Gs[a_idx] = G1
            continue

        # Recurse over possible observations at next step
        q_pred = bayes_predict(qs, B_a)
        Q_o = A @ q_pred

        tail = 0.0
        for o in range(A.shape[0]):
            q_post = bayes_update(q_pred, A, o)
            G_next, _ = efe_depth_H(q_post, A, B, P_o_pref, H=H - 1, gamma=gamma)
            tail += Q_o[o] * np.min(G_next)  # optimal continuation

        Gs[a_idx] = G1 + gamma * tail

    return Gs, comps

