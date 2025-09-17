"""
Inference utilities (state-belief updates) for discrete active inference.
"""
import numpy as np
from .gm import entropy


def bayes_predict(qs: np.ndarray, B_a: np.ndarray) -> np.ndarray:
    """
    Predictive prior over next state.
    q_{t+1}^-(s') = (B[a] @ q_t)(s')
    """
    return B_a @ qs


def bayes_update(q_pred: np.ndarray, A: np.ndarray, o: int) -> np.ndarray:
    """
    Posterior q(s|o) ∝ A[o,s] * q_pred(s).
    """
    post = A[o, :] * q_pred
    ssum = post.sum()
    if ssum <= 0:
        # fallback to uniform if likelihood wiped things out
        return np.ones_like(post) / post.size
    return post / ssum


def belief_entropy(qs: np.ndarray) -> float:
    """Shannon entropy of belief distribution."""
    return entropy(qs)

