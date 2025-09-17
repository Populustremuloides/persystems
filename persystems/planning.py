"""
Planning utilities for action selection via Expected Free Energy (EFE).
"""
from __future__ import annotations
import numpy as np
from typing import Tuple, Dict, List
from .efe import efe_one_step, efe_depth_H
from .gm import softmax
from .inference import bayes_predict, bayes_update

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

# --- Additions to persystems/planning.py ---

from dataclasses import dataclass

@dataclass
class PlanDiagnostics:
    horizon: int
    nodes_expanded: int
    pruned_obs: int
    beam_width: int | None


def _efe_tail_recursive(qs, A, B, P_o_pref, H, obs_prune_eps, counter):
    """
    Internal: returns min_a [ G1(a) + E_{o}[ min_a' tail(...) ] ] with
    observation-branch pruning for very small Q(o).
    """
    # Base: evaluate each immediate action 1-step
    Gs = []
    tails = []
    comps = []
    for a_idx, B_a in enumerate(B):
        # step-1 EFE
        G1, comp = efe_one_step(qs, A, B_a, P_o_pref)
        comps.append(comp)
        if H == 1:
            Gs.append(G1)
            tails.append(0.0)
            counter["nodes"] += 1
            continue

        # recurse over observations with pruning
        q_pred = bayes_predict(qs, B_a)
        Qo = A @ q_pred
        tail = 0.0
        kept = 0
        for o in range(A.shape[0]):
            p_o = Qo[o]
            if p_o <= obs_prune_eps:
                counter["pruned"] += 1
                continue
            q_post = bayes_update(q_pred, A, o)
            G_next, _, _ = _efe_tail_recursive(q_post, A, B, P_o_pref, H-1, obs_prune_eps, counter)
            tail += p_o * np.min(G_next)
            kept += 1
            counter["nodes"] += 1
        if kept == 0:
            # if all pruned (very unlikely), fall back to unpruned average
            for o in range(A.shape[0]):
                q_post = bayes_update(q_pred, A, o)
                G_next, _, _ = _efe_tail_recursive(q_post, A, B, P_o_pref, H-1, 0.0, counter)
                tail += Qo[o] * np.min(G_next)
        Gs.append(G1 + tail)
        tails.append(tail)
    return np.array(Gs), comps, tails


def choose_action_planner(
    qs: np.ndarray,
    A: np.ndarray,
    B: List[np.ndarray],
    C: np.ndarray,
    horizon: int = 2,
    obs_prune_eps: float = 1e-4
) -> Tuple[int, Dict[str, float], np.ndarray, PlanDiagnostics]:
    """
    Depth-H planner with observation-branch pruning (keeps correctness in expectation,
    but drops negligible-probability observations).
    """
    P_o_pref = softmax(C)
    counter = {"nodes": 0, "pruned": 0}
    Gs, comps, _ = _efe_tail_recursive(qs, A, B, P_o_pref, horizon, obs_prune_eps, counter)
    a_idx = int(np.argmin(Gs))
    diags = PlanDiagnostics(horizon=horizon, nodes_expanded=counter["nodes"],
                            pruned_obs=counter["pruned"], beam_width=None)
    return a_idx, comps[a_idx], Gs, diags


def choose_action_beam(
    qs: np.ndarray,
    A: np.ndarray,
    B: List[np.ndarray],
    C: np.ndarray,
    horizon: int = 3,
    beam_width: int = 16,
    obs_prune_eps: float = 1e-4,
    gamma: float = 1.0
) -> Tuple[int, Dict[str, float], np.ndarray, PlanDiagnostics]:
    """
    Beam search over action sequences of length H.
    Evaluates expected EFE per partial policy; keeps top-K at each depth.

    Returns:
      a_idx (int): first action of the best beam
      comp (dict): components for the chosen first action at step 1
      Gs (np.ndarray): EFE per immediate action (from best-beam continuation)
      diags (PlanDiagnostics)
    """
    P_o_pref = softmax(C)
    # Beam: list of tuples (expected_G, first_action_idx, qs_after_prefix, depth)
    # Start with all 1-step actions scored by one-step EFE
    beams = []
    comps_first = []
    for a0, B_a0 in enumerate(B):
        G1, comp = efe_one_step(qs, A, B_a0, P_o_pref)
        comps_first.append(comp)
        q_pred = bayes_predict(qs, B_a0)
        Qo = A @ q_pred
        # expected continuation value initialized to 0; we'll expand later
        beams.append((G1, a0, q_pred, 1, Qo))
    # Keep the best beams at depth 1
    beams.sort(key=lambda t: t[0])
    beams = beams[:beam_width]

    nodes = len(beams)
    pruned = 0

    # Expand to depth H
    while beams and beams[0][3] < horizon:
        depth = beams[0][3]
        new_beams = []
        for G_so_far, a0, q_pred, d, Qo in beams:
            # branch on observations (prune tiny)
            for o in range(A.shape[0]):
                p_o = Qo[o]
                if p_o <= obs_prune_eps:
                    pruned += 1
                    continue
                q_post = bayes_update(q_pred, A, o)
                # choose next action by 1-step EFE (myopic inside beam)
                # (cheap & works well on small toys; could also branch over actions)
                G1_list = []
                for a_next, B_next in enumerate(B):
                    G1n, _ = efe_one_step(q_post, A, B_next, P_o_pref)
                    G1n = float(G1n)
                    G1_list.append(G1n)
                # expected next-step value = p_o * min_a' G1
                G_ext = G_so_far + gamma * p_o * (min(G1_list) if d + 1 < horizon else 0.0)
                # prepare state for next depth
                # advance with action that minimized one-step continuation (greedy inside beam)
                best_a_next = int(np.argmin(G1_list))
                q_next_pred = bayes_predict(q_post, B[best_a_next])
                Qo_next = A @ q_next_pred
                new_beams.append((G_ext, a0, q_next_pred, d + 1, Qo_next))
                nodes += 1

        # prune to beam width
        new_beams.sort(key=lambda t: t[0])
        beams = new_beams[:beam_width]

    # Aggregate expected G per immediate action from beams’ best
    if not beams:
        # fallback to depth-1
        Gs = np.array([efe_one_step(qs, A, B_a, softmax(C))[0] for B_a in B])
        a_idx = int(np.argmin(Gs))
        diags = PlanDiagnostics(horizon=horizon, nodes_expanded=nodes, pruned_obs=pruned, beam_width=beam_width)
        return a_idx, comps_first[a_idx], Gs, diags

    best = min(beams, key=lambda t: t[0])
    a_best = best[1]
    # Build Gs by taking the best beam per starting action (if present)
    per_first = {}
    for Gexp, a0, *_ in beams:
        per_first[a0] = min(per_first.get(a0, np.inf), Gexp)
    # fill any missing with 1-step
    Gs = []
    for a0, B_a0 in enumerate(B):
        if a0 in per_first:
            Gs.append(per_first[a0])
        else:
            Gs.append(efe_one_step(qs, A, B_a0, softmax(C))[0])
    Gs = np.array(Gs)
    diags = PlanDiagnostics(horizon=horizon, nodes_expanded=nodes, pruned_obs=pruned, beam_width=beam_width)
    return int(a_best), comps_first[a_best], Gs, diags


