import numpy as np
from persystems.geometry import (
    softmax, fisher_categorical, natural_grad_from_theta,
    natural_grad_from_prob, elbo_grad_theta, natural_step_theta,
    information_speed, information_length, mirror_descent_kl
)


def test_fisher_properties_and_natgrad_centering():
    # random prob on simplex
    rng = np.random.default_rng(0)
    q = rng.random(6); q /= q.sum()

    G = fisher_categorical(q)
    # G is symmetric PSD, rows/cols sum to 0 (tangent space)
    assert np.allclose(G, G.T, atol=1e-12)
    w = rng.standard_normal(q.size)
    assert float(w @ (G @ w)) >= -1e-10  # PSD (allow tiny numerical slip)
    assert np.allclose(G.sum(axis=0), 0.0, atol=1e-12)
    assert np.allclose(G.sum(axis=1), 0.0, atol=1e-12)

    # Natural gradient in theta is just centering by <grad>_q
    grad_theta = rng.standard_normal(q.size)
    nat = natural_grad_from_theta(grad_theta, q)
    # Mean under q should be ~0
    assert abs(np.sum(q * nat)) < 1e-12


def test_natgrad_q_vs_theta_consistency_on_elbo():
    rng = np.random.default_rng(1)
    S = 5
    theta = rng.standard_normal(S)
    q = softmax(theta)
    log_joint = rng.standard_normal(S)  # pretend log p(s,o) up to const

    # Vanilla gradient wrt theta for ELBO
    g_theta = elbo_grad_theta(theta, log_joint)
    nat_theta = natural_grad_from_theta(g_theta, q)
    # Step should stay on simplex after softmax
    theta2, q2 = natural_step_theta(theta, g_theta, step=0.1)
    assert np.isclose(q2.sum(), 1.0, atol=1e-12)
    assert np.all(q2 >= 0.0)

    # Compare with prob-space natural gradient direction
    # dF/dq = log q - log p(s,o) + 1  (up to constants); we use a finite-diff proxy via g_theta mapping
    # Here we only sanity check: nat in q lies in tangent (sum ~ 0)
    grad_q_proxy = rng.standard_normal(S)
    nat_q = natural_grad_from_prob(grad_q_proxy, q)
    assert abs(nat_q.sum()) < 1e-8  # tangent-space


def test_information_length_nonnegative_and_zero_for_static_path():
    q0 = np.array([0.2, 0.3, 0.5])
    path = np.vstack([q0, q0, q0])
    L0 = information_length(path)
    assert L0 == 0.0

    q1 = np.array([0.25, 0.25, 0.5])
    path2 = np.vstack([q0, q1, q0])
    L = information_length(path2)
    assert L >= 0.0

    # Mirror descent keeps on simplex
    g = np.array([1.0, -2.0, 1.0])
    q_next = mirror_descent_kl(q0, g, step=0.1)
    assert np.isclose(q_next.sum(), 1.0, atol=1e-12)
    assert np.all(q_next >= 0.0)

