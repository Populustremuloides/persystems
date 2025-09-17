"""
Generative model utilities for discrete active inference.
Implements A (likelihood), B (transition per action), and C (log-preferences).
"""
import numpy as np
from dataclasses import dataclass
from typing import List


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    e = np.exp(z)
    return e / e.sum()


def entropy(p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))


def shift_matrix(shift: int, N: int) -> np.ndarray:
    M = np.zeros((N, N))
    for s in range(N):
        M[(s + shift) % N, s] = 1.0
    return M


@dataclass
class GenerativeModel:
    """
    Discrete generative model for ring world.
    - A: (O,S) observation likelihood P(o|s)
    - B: list of (S,S) transition matrices per action a: P(s'|s,a)
    - C: (O,) log-preferences over outcomes; P(o) = softmax(C)
    - actions: list of integer shifts [-1,0,+1] on the ring
    """
    A: np.ndarray
    B: List[np.ndarray]
    C: np.ndarray
    actions: List[int]

    def __post_init__(self):
        O, S = self.A.shape
        assert self.C.shape == (O,), f"C must be shape ({O},), got {self.C.shape}"
        for s in range(S):
            col = self.A[:, s]
            assert np.all(col >= -1e-9), "A has negative entries"
            assert np.isclose(col.sum(), 1.0, atol=1e-8)
        for i, B_a in enumerate(self.B):
            Ss, Ss2 = B_a.shape
            assert Ss == Ss2 == S, "B[a] must be square SxS"
            for s in range(S):
                col = B_a[:, s]
                assert np.all(col >= -1e-9), f"B[{i}] has negative entries"
                assert np.isclose(col.sum(), 1.0, atol=1e-8)

    @property
    def P_outcomes(self) -> np.ndarray:
        """Preferred outcome distribution P(o) from log-preferences C."""
        return softmax(self.C)

    @staticmethod
    def make_ring_world(
        N: int = 5, actions=(-1, 0, +1), A_eps: float = 0.15, target_idx: int = 3
    ):
        A = (1 - A_eps) * np.eye(N) + (A_eps / (N - 1)) * (np.ones((N, N)) - np.eye(N))
        B = [shift_matrix(a, N) for a in actions]
        C = np.full(N, -4.0)
        C[target_idx] = 0.0
        return GenerativeModel(A=A, B=B, C=C, actions=list(actions))

