"""
geometry.py — Information geometry utilities for discrete (categorical) beliefs.

Focus:
- Fisher information metric on the probability simplex (categorical).
- Natural-gradient updates for functionals F(q) where q is categorical.
- Simple mirror-descent (KL) and natural-step helpers in logits space.
- Information length of a curve q(t) := sum_t sqrt( dq^T G(q) dq ).

Conventions
-----------
We represent a categorical distribution by:
- probabilities q ∈ Δ^{S-1} (on the simplex),
- or unconstrained logits θ ∈ R^S with q = softmax(θ).

For the categorical family, the Fisher information in θ-coordinates equals
the covariance of sufficient statistics (one-hot indicators), which yields
the well-known expression in probability-coordinates:
    G(q) = diag(q) − q q^T
This is the pull-back of the Fisher metric to the simplex.

Natural gradient
----------------
Given ∂F/∂θ (gradient w.r.t. logits), the natural gradient is:
    ∇_nat F = G(θ)^{-1} ∂F/∂θ
With the categorical exponential family, this simplifies to:
    ∇_nat F in θ = ∂F/∂θ − (1^T ∂F/∂θ) * q
i.e. just subtract the q-weighted mean of the vanilla gradient, because the
Fisher metric in θ is the softmax Fisher (a.k.a. "center the gradient").
"""
from __future__ import annotations
import numpy as np
from typing import Tuple

Array = np.ndarray


# ---------- Basic transforms ----------

def softmax(theta: Array) -> Array:
    z = theta - np.max(theta)
    e = np.exp(z)
    return e / np.sum(e)


def logit_from_prob(q: Array, eps: float = 1e-12) -> Array:
    """Return logits proportional to log(q) (shift-invariant)."""
    q = np.clip(q, eps, 1.0)
    return np.log(q)


def project_to_simplex(v: Array) -> Array:
    """
    Euclidean projection of v ∈ R^S onto the probability simplex.
    (Sorting-based projection.)
    """
    v = np.asarray(v, dtype=float).copy()
    n = v.size
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1.0))[0][-1]
    theta = (cssv[rho] - 1.0) / (rho + 1)
    w = np.maximum(v - theta, 0.0)
    return w / w.sum()


# ---------- Fisher metric and natural gradient ----------

def fisher_categorical(q: Array) -> Array:
    """
    Fisher information matrix for categorical probabilities q on the simplex.
    G(q) = diag(q) - q q^T
    """
    q = np.asarray(q, dtype=float)
    G = np.diag(q) - np.outer(q, q)
    return G


def natural_grad_from_theta(grad_theta: Array, q: Array) -> Array:
    """
    Natural gradient in θ-coordinates for the categorical softmax family.
    For softmax, ∇_nat F = grad_theta - (q^T grad_theta) * 1  (center the gradient).
    """
    grad_theta = np.asarray(grad_theta, dtype=float)
    q = np.asarray(q, dtype=float)
    centered = grad_theta - np.sum(q * grad_theta)
    return centered


def natural_grad_from_prob(grad_q: Array, q: Array, eps: float = 1e-12) -> Array:
    """
    Natural gradient in probability-coordinates:
      ∇_nat F (in q) = G(q)^{+} grad_q
    where G(q)=diag(q)-q q^T and + denotes the Moore–Penrose pseudoinverse
    on the tangent space (sum of components is zero).
    """
    q = np.clip(np.asarray(q, dtype=float), eps, 1.0)
    grad_q = np.asarray(grad_q, dtype=float)
    G = fisher_categorical(q)
    G_pinv = np.linalg.pinv(G, rcond=1e-12)
    nat = G_pinv @ grad_q
    # Project to tangent space (sum ~ 0)
    nat = nat - np.mean(nat)
    return nat


# ---------- Free-energy examples & updates ----------

def elbo_grad_theta(theta: Array, log_joint: Array) -> Array:
    """
    Vanilla gradient ∂F/∂θ for F(q)=E_q[log q - log_joint] with q=softmax(θ),
    and 'log_joint[s]' a fixed vector (proportional to log p(s, o)).

    A convenient and numerically stable form is:
      grad_theta = q - softmax(log_joint)
    """
    q = softmax(theta)
    p_tilde = softmax(np.asarray(log_joint, dtype=float))
    return q - p_tilde


def natural_step_theta(
    theta: Array,
    grad_theta: Array,
    step: float
) -> Tuple[Array, Array]:
    """
    One natural-gradient step in θ:
      θ_new = θ - step * ∇_nat F
    For softmax, ∇_nat F = grad_theta - <grad_theta>_q.
    Returns (theta_new, q_new).
    """
    q = softmax(theta)
    nat = natural_grad_from_theta(grad_theta, q)
    theta_new = theta - step * nat
    q_new = softmax(theta_new)
    return theta_new, q_new


def mirror_descent_kl(
    q: Array,
    grad_q: Array,
    step: float,
    eps: float = 1e-12
) -> Array:
    """
    KL-mirror descent step on the simplex (a.k.a. exponentiated gradient):
      q_new ∝ q * exp(-step * grad_q), renormalized to the simplex.
    """
    q = np.clip(np.asarray(q, dtype=float), eps, 1.0)
    g = np.asarray(grad_q, dtype=float)
    z = np.log(q) - step * g
    z -= np.max(z)
    q_new = np.exp(z)
    q_new /= q_new.sum()
    return q_new


# ---------- Information length ----------

def information_speed(q: Array, dq: Array, eps: float = 1e-12) -> float:
    """
    Instantaneous information speed:
      ||dq||_G = sqrt( dq^T G(q) dq )
    """
    q = np.clip(np.asarray(q, dtype=float), eps, 1.0)
    dq = np.asarray(dq, dtype=float)
    G = fisher_categorical(q)
    s = float(np.sqrt(max(0.0, dq @ (G @ dq))))
    return s


def information_length(path_q: Array) -> float:
    """
    Discrete information length along q_0, ..., q_T:
      L = sum_t sqrt( (q_{t+1}-q_t)^T G(q_t) (q_{t+1}-q_t) )
    """
    path_q = np.asarray(path_q, dtype=float)
    T, S = path_q.shape
    L = 0.0
    for t in range(T - 1):
        L += information_speed(path_q[t], path_q[t + 1] - path_q[t])
    return float(L)

