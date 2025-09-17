"""
Planning utilities for action selection via Expected Free Energy (EFE).
"""
from __future__ import annotations
import numpy as np
from typing import Tuple, Dict, List
from .efe import efe_one_step, efe_depth_H
from .gm import softmax


def choose_action(
    qs: np.ndarray,
    A: np.ndarray,
    B: List[np.ndarray],
    C: np.ndarray,
    horizon: int = 1,
    gamma: float = 1.0
) -> Tuple[int, Dict[str, float], np.ndarray]:
    """
    Choose the action index that minimizes Expected Free Energy.

    Args:
        qs: (S,) current belief over states
        A:  (O,S) likelihood
        B:  list[(S,S)] transition matrices per action
        C:  (O,) log-preferences (P(o) = softmax(C))
        horizon: lookahead depth (H=1 is myopic; H>1 enumerates observation branches)
        gamma: discount on future terms

    Returns:
        a_idx: index of chosen action in B
        comp:  dict of components for the chosen action at step 1
               {'risk','ambiguity','expected_cost','info_gain'}
        Gs:    (A,) vector of EFE values for each immediate action
    """
    P_o_pref = softmax(C)

    if horizon == 1:
        Gs_list = []
        comps = []
        for B_a in B:
            G, comp = efe_one_step(qs, A, B_a, P_o_pref)
            Gs_list.append(G)
            comps.append(comp)
        Gs = np.array(Gs_list)
        a_idx = int(np.argmin(Gs))
        return a_idx, comps[a_idx], Gs

    # depth-H evaluation
    Gs, comps = efe_depth_H(qs, A, B, P_o_pref, H=horizon, gamma=gamma)
    a_idx = int(np.argmin(Gs))
    return a_idx, comps[a_idx], Gs

