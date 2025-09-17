"""
sde.py — Simple 2D Langevin SDE simulator (Euler–Maruyama).

dx = f(x) dt + sqrt(2D) dW_t, with isotropic diffusion D (scalar) or diag(Dx, Dy).

Provides:
- euler_maruyama(...): simulate trajectories
- preset drifts: drift_doublewell(...), drift_doublewell_with_swirl(...)
"""
from __future__ import annotations
import numpy as np
from typing import Callable, Tuple


Array = np.ndarray
DriftFn = Callable[[Array], Array]


def euler_maruyama(
    x0: Array,
    f: DriftFn,
    dt: float,
    T: int,
    D: Tuple[float, float] | float = 0.05,
    rng: np.random.Generator | None = None,
) -> Array:
    """
    Simulate a 2D SDE path with Euler–Maruyama.
    Args:
        x0: (2,) initial state
        f:  drift function R^2 -> R^2
        dt: time step
        T:  number of steps
        D:  diffusion; scalar or (Dx, Dy)
        rng: numpy Generator (optional)
    Returns:
        X: (T+1, 2) states
    """
    if rng is None:
        rng = np.random.default_rng()

    x = np.array(x0, dtype=float).reshape(2)
    X = np.zeros((T + 1, 2), dtype=float)
    X[0] = x

    if isinstance(D, tuple) or isinstance(D, list) or (hasattr(D, "__len__") and len(D) == 2):
        Dx, Dy = float(D[0]), float(D[1])
    else:
        Dx = Dy = float(D)

    for t in range(T):
        drift = np.asarray(f(x), dtype=float).reshape(2)
        noise = np.array([np.sqrt(2 * Dx * dt), np.sqrt(2 * Dy * dt)]) * rng.standard_normal(2)
        x = x + drift * dt + noise
        X[t + 1] = x
    return X


# ------------------------
# Preset drifts (for demos)
# ------------------------

def drift_doublewell(a: float = 1.0, b: float = 1.0) -> DriftFn:
    """
    Gradient flow of a 2D double-well potential:
      Phi(x,y) = (x^2 - a)^2/4 + (y^2 - b)^2/4
    Drift = -grad Phi
    """
    def f(xy: Array) -> Array:
        x, y = xy
        return np.array([-(x**3 - a*x), -(y**3 - b*y)], dtype=float)
    return f


def drift_doublewell_with_swirl(a: float = 1.0, b: float = 1.0, k: float = 1.0) -> DriftFn:
    """
    Double-well gradient flow + solenoidal swirl field:
      f = -grad Phi + Q grad Phi  (heuristically)
    We implement: grad part from drift_doublewell + rotational component
      v_rot = k * R @ [x, y], where R = [[0, -1],[1, 0]]
    """
    base = drift_doublewell(a=a, b=b)
    R = np.array([[0.0, -1.0], [1.0, 0.0]])
    def f(xy: Array) -> Array:
        x, y = xy
        grad_part = base(xy)
        swirl = k * (R @ np.array([x, y], dtype=float))
        return grad_part + swirl
    return f

