"""
renorm.py — Coarse-graining / renormalization utilities for the ring world.

We support mapping a fine-state ring (size N) to a coarser ring (size M),
given a partition of fine states into consecutive blocks. We then derive
coarse-grained observation (A') and transition (B') operators and provide
lifting/restriction maps to move beliefs between levels.
"""
from __future__ import annotations
import numpy as np
from typing import List, Sequence, Tuple

Array = np.ndarray


# ---------- Helpers ----------

def make_hard_partition_R(N: int, blocks: List[Sequence[int]]) -> Array:
    """
    Build a hard partition matrix R (M x N) from explicit blocks (list of index lists).
    Each fine index appears exactly once across blocks.
    Columns are one-hot → column-stochastic.
    """
    M = len(blocks)
    R = np.zeros((M, N), dtype=float)
    seen = set()
    for i, J in enumerate(blocks):
        for j in J:
            if j in seen:
                raise ValueError(f"Fine index {j} appears in multiple blocks.")
            seen.add(j)
            R[i, j] = 1.0
    if len(seen) != N:
        missing = set(range(N)) - seen
        raise ValueError(f"Partition missing fine indices: {sorted(missing)}")
    assert np.allclose(R.sum(axis=0), 1.0)
    return R


def uniform_lift_L(R: Array) -> Array:
    """
    Construct a simple right-inverse L (N x M) for a hard R (M x N) with RL = I_M,
    by uniformly spreading mass within each block.
    """
    M, N = R.shape
    L = np.zeros((N, M), dtype=float)
    # For each fine state j, find its block i and assign uniform weight within that block.
    for j in range(N):
        i = int(np.argmax(R[:, j]))   # block index
        block_mask = (R[i, :] == 1.0)
        block_size = int(block_mask.sum())
        L[j, i] = 1.0 / block_size
    RL = R @ L
    if not np.allclose(RL, np.eye(M), atol=1e-12):
        raise ValueError("Uniform lift failed to satisfy RL=I.")
    return L


def normalize_columns(M_: Array, eps: float = 1e-12) -> Array:
    """Normalize columns to sum to 1 (stochastic form)."""
    M_ = np.asarray(M_, dtype=float).copy()
    colsum = M_.sum(axis=0, keepdims=True)
    colsum = np.clip(colsum, eps, None)
    M_ /= colsum
    return M_


# ---------- Coarse-graining operators ----------

def coarse_A(A: Array, blocks: List[Sequence[int]], weights: List[Array] | None = None) -> Array:
    """
    Coarse observation model A' (O x M) from fine A (O x N), given blocks.
    weights[i] is a |J_i|-vector of convex weights for block i (default: uniform).
    """
    O, N = A.shape
    M = len(blocks)
    Acoarse = np.zeros((O, M), dtype=float)
    for i, J in enumerate(blocks):
        J = list(J)
        if weights is not None:
            w = np.asarray(weights[i], dtype=float)
            if w.size != len(J) or np.any(w < 0) or not np.isclose(w.sum(), 1.0):
                raise ValueError(f"weights[{i}] must be nonnegative, length {len(J)}, and sum to 1.")
        else:
            w = np.ones(len(J), dtype=float) / len(J)
        Acoarse[:, i] = A[:, J] @ w
    # Each column is convex combination of columns of A → sums to 1
    assert np.allclose(Acoarse.sum(axis=0), 1.0, atol=1e-10)
    return Acoarse


def coarse_B(B: List[Array], R: Array, L: Array) -> List[Array]:
    """
    Coarse transition models B'^a (M x M) from fine B^a (N x N) via:
        B'^a = R B^a L
    Renormalize columns afterwards for numerical hygiene.
    """
    M, N = R.shape
    Bcoarse = []
    for Ba in B:
        if Ba.shape != (N, N):
            raise ValueError("Each B[a] must be N x N.")
        Bc = R @ Ba @ L
        Bc = normalize_columns(Bc)
        Bcoarse.append(Bc)
    return Bcoarse


# ---------- Belief transfer ----------

def restrict_belief(R: Array, q_fine: Array) -> Array:
    """Restrict a fine belief q_fine (N,) to coarse Q = R q_fine (M,)."""
    q_fine = np.asarray(q_fine, dtype=float)
    Q = R @ q_fine
    Q = np.clip(Q, 0.0, 1.0)
    if Q.sum() > 0:
        Q /= Q.sum()
    return Q


def lift_belief(L: Array, Q: Array) -> Array:
    """Lift a coarse belief Q (M,) to fine q = L Q (N,)."""
    Q = np.asarray(Q, dtype=float)
    q = L @ Q
    q = np.clip(q, 0.0, 1.0)
    if q.sum() > 0:
        q /= q.sum()
    return q


# ---------- Ring-world convenience ----------

def contiguous_blocks_ring(N: int, M: int) -> List[Sequence[int]]:
    """
    Partition N fine states into M contiguous blocks on a ring.
    Last block absorbs remainder if N % M != 0.
    Example: N=10, M=4 → blocks [[0,1,2],[3,4],[5,6],[7,8,9]]
    """
    if M <= 0 or M > N:
        raise ValueError("Require 1 <= M <= N.")
    base = N // M
    rem = N % M
    blocks: List[Sequence[int]] = []
    start = 0
    for i in range(M):
        size = base + (1 if i < rem else 0)
        idxs = [(start + k) % N for k in range(size)]
        blocks.append(idxs)
        start += size
    flat = sorted([j for J in blocks for j in J])
    assert flat == list(range(N)), "Blocks must cover all fine indices exactly once."
    return blocks


def coarse_grain_ringworld(
    A: Array, B: List[Array], N: int, M: int
) -> Tuple[Array, List[Array], Array, Array, List[Sequence[int]]]:
    """
    Convenience wrapper for ring world:
      - Build contiguous ring blocks (N → M),
      - Construct R (M x N) and uniform right-inverse L (N x M),
      - Produce coarse A' and B' via convex averaging and R B L.
    Returns: A_coarse, B_coarse, R, L, blocks
    """
    blocks = contiguous_blocks_ring(N, M)
    R = make_hard_partition_R(N, blocks)
    L = uniform_lift_L(R)
    A_coarse = coarse_A(A, blocks)
    B_coarse = coarse_B(B, R, L)
    return A_coarse, B_coarse, R, L, blocks

