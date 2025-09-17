"""
blankets.py — Empirical Markov-blanket diagnostics from samples.

Given samples Z = [x_int, x_act, x_sens, x_ext] ∈ R^{N x d},
we compute covariance Σ and precision K = Σ^{-1}, and report norms of off-diagonal
blocks that should be ~0 if conditional independencies hold (Gaussian proxy).

This is an *approximate*, scale-dependent diagnostic: blankets emerge under coarse-graining
and timescale separation. We provide numeric summaries to visualize how close blocks are to 0.
"""
from __future__ import annotations
import numpy as np
from typing import Dict, Tuple, List


Array = np.ndarray


def _block(M: Array, rows: slice | List[int], cols: slice | List[int]) -> Array:
    return M[rows, :][:, cols]


def blanket_block_norms(
    Z: Array,
    dims: Dict[str, int],
    reg: float = 1e-6
) -> Dict[str, float]:
    """
    Compute precision-matrix block norms implied by a (I, A, S, E) partition.

    Args:
        Z:   (N, d) samples; columns concatenated as [I | A | S | E]
        dims: dict with keys 'I','A','S','E' and integer sizes summing to d
        reg: ridge added to the covariance diagonal before inversion

    Returns:
        dict of Frobenius norms of off-diagonal precision blocks:
           K_IA, K_IS, K_IE, K_AS, K_AE, K_SE
        Key 'K_IE_cond_AS' is the block of the Schur-complement precision for (I,E) given (A,S),
        which is the most direct blanket test for I ⟂ E | (A,S) under Gaussian assumptions.
    """
    d_I, d_A, d_S, d_E = dims["I"], dims["A"], dims["S"], dims["E"]
    assert Z.shape[1] == d_I + d_A + d_S + d_E, "dims must sum to Z.shape[1]"

    # indices
    I = slice(0, d_I)
    A = slice(d_I, d_I + d_A)
    S = slice(d_I + d_A, d_I + d_A + d_S)
    E = slice(d_I + d_A + d_S, d_I + d_A + d_S + d_E)

    # covariance + ridge
    Zc = Z - Z.mean(axis=0, keepdims=True)
    Sigma = (Zc.T @ Zc) / max(1, Zc.shape[0] - 1)
    Sigma += reg * np.eye(Sigma.shape[0])

    # precision
    K = np.linalg.inv(Sigma)

    out = {}
    out["K_IA"] = float(np.linalg.norm(_block(K, I, A), ord="fro"))
    out["K_IS"] = float(np.linalg.norm(_block(K, I, S), ord="fro"))
    out["K_IE"] = float(np.linalg.norm(_block(K, I, E), ord="fro"))
    out["K_AS"] = float(np.linalg.norm(_block(K, A, S), ord="fro"))
    out["K_AE"] = float(np.linalg.norm(_block(K, A, E), ord="fro"))
    out["K_SE"] = float(np.linalg.norm(_block(K, S, E), ord="fro"))

    # Schur complement precision for [I,E] conditioned on [A,S]
    # Partition variables as v = [u, w] with u=(I,E) and w=(A,S)
    # K_cond = K_uu - K_uw * K_ww^{-1} * K_wu  (equivalently via Sigma blocks)
    # Here we compute via Sigma blocks for numerical stability.
    idx_u = list(range(d_I)) + list(range(d_I + d_A + d_S, d_I + d_A + d_S + d_E))
    idx_w = list(range(d_I, d_I + d_A + d_S))
    Sigma_uu = Sigma[np.ix_(idx_u, idx_u)]
    Sigma_uw = Sigma[np.ix_(idx_u, idx_w)]
    Sigma_wu = Sigma[np.ix_(idx_w, idx_u)]
    Sigma_ww = Sigma[np.ix_(idx_w, idx_w)]
    # Conditional precision for u given w:
    # K_u|w = (Sigma_uu - Sigma_uw Sigma_ww^{-1} Sigma_wu)^{-1}
    Sw_inv = np.linalg.inv(Sigma_ww)
    Schur = Sigma_uu - Sigma_uw @ Sw_inv @ Sigma_wu
    K_u_given_w = np.linalg.inv(Schur + reg * np.eye(Schur.shape[0]))
    # Off-diagonal block between I and E inside K_u|w:
    K_uI_E = K_u_given_w[0:d_I, d_I:d_I + d_E]
    out["K_IE_cond_AS"] = float(np.linalg.norm(K_uI_E, ord="fro"))

    return out

