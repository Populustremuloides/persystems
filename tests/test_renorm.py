import numpy as np
from persystems.gm import GenerativeModel
from persystems.renorm import (
    contiguous_blocks_ring, make_hard_partition_R, uniform_lift_L,
    coarse_A, coarse_B, restrict_belief, lift_belief, coarse_grain_ringworld
)


def test_partition_and_lift_roundtrip():
    N, M = 10, 4
    blocks = contiguous_blocks_ring(N, M)
    R = make_hard_partition_R(N, blocks)
    L = uniform_lift_L(R)

    # RL = I_M exactly for a hard partition with uniform lift
    RL = R @ L
    assert np.allclose(RL, np.eye(M), atol=1e-12)

    # Belief restriction/lifting keeps us on simplex and is a left-inverse on the coarse level
    rng = np.random.default_rng(0)
    q_fine = rng.random(N); q_fine /= q_fine.sum()
    Q = restrict_belief(R, q_fine)
    assert np.isclose(Q.sum(), 1.0, atol=1e-12)
    assert np.all(Q >= 0.0)
    q_up = lift_belief(L, Q)
    assert np.isclose(q_up.sum(), 1.0, atol=1e-12)
    assert np.all(q_up >= 0.0)

    # Restrict(lift(Q)) = Q
    Q2 = restrict_belief(R, q_up)
    assert np.allclose(Q2, Q, atol=1e-12)


def test_coarse_A_B_are_stochastic_and_dimensions_match():
    N = 9
    gm = GenerativeModel.make_ring_world(N=N, A_eps=0.15, target_idx=3)
    # Coarsen to M=3
    A_coarse, B_coarse, R, L, blocks = coarse_grain_ringworld(gm.A, gm.B, N=N, M=3)

    # A' columns sum to 1
    assert np.allclose(A_coarse.sum(axis=0), 1.0, atol=1e-10)
    # Each B'^a columns sum to 1 (stochastic)
    for Bc in B_coarse:
        colsum = Bc.sum(axis=0)
        assert np.allclose(colsum, 1.0, atol=1e-10)

    # Shapes
    assert A_coarse.shape == (N, 3)   # O=N outcomes, M coarse states
    for Bc in B_coarse:
        assert Bc.shape == (3, 3)


def test_coarse_restrict_then_plan_belief_is_valid():
    N, M = 10, 4
    gm = GenerativeModel.make_ring_world(N=N, A_eps=0.15, target_idx=3)
    A_coarse, B_coarse, R, L, blocks = coarse_grain_ringworld(gm.A, gm.B, N=N, M=M)

    # Uniform fine belief -> coarse belief
    q_f = np.ones(N) / N
    Q = restrict_belief(R, q_f)
    assert np.isclose(Q.sum(), 1.0)
    assert np.all(Q >= 0.0)

    # After one coarse transition (choose, e.g., first action), belief remains valid
    Bc0 = B_coarse[0]
    Q_pred = Bc0 @ Q
    assert np.isclose(Q_pred.sum(), 1.0, atol=1e-12)
    assert np.all(Q_pred >= -1e-12)  # allow tiny numerical noise


def test_manual_coarse_A_matches_block_average():
    # Build a tiny ring where we can verify A' by hand
    N, M = 6, 3
    gm = GenerativeModel.make_ring_world(N=N, A_eps=0.2, target_idx=2)
    blocks = contiguous_blocks_ring(N, M)  # [[0,1], [2,3], [4,5]]
    R = make_hard_partition_R(N, blocks)
    # Coarse A by our function
    A_coarse = coarse_A(gm.A, blocks)

    # Manual average inside each block should equal A' column
    for i, J in enumerate(blocks):
        J = list(J)
        manual = gm.A[:, J].mean(axis=1)
        assert np.allclose(A_coarse[:, i], manual, atol=1e-12)

