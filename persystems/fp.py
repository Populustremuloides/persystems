"""
fp.py — Empirical steady-state (NESS) estimation and smoothing utilities.

We estimate p*(x) by long-run histogram of SDE samples (ergodic approximation).
Also provides simple Gaussian smoothing (separable) implemented with NumPy only.
"""
from __future__ import annotations
import numpy as np
from typing import Tuple


Array = np.ndarray


def histogram2d_density(samples: Array, xbins: np.ndarray, ybins: np.ndarray, eps: float = 1e-12) -> Tuple[Array, np.ndarray, np.ndarray]:
    """
    Estimate a 2D density via normalized histogram.
    Args:
        samples: (N,2)
        xbins, ybins: bin edges (1D arrays)
    Returns:
        p: (len(xbins)-1, len(ybins)-1) estimated density (sum p ~ 1 over the grid)
        xc, yc: bin centers
    """
    H, xe, ye = np.histogram2d(samples[:,0], samples[:,1], bins=[xbins, ybins], density=False)
    H = H.astype(float)
    H /= H.sum() + eps
    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    return H.T, xc, yc  # transpose so p[i,j] maps to yc[i], xc[j]


def gaussian_kernel_1d(sigma: float, radius: int | None = None) -> np.ndarray:
    """
    Create 1D Gaussian kernel with unit L1 norm.
    radius defaults to ceil(3*sigma).
    """
    if sigma <= 0:
        return np.array([1.0])
    if radius is None:
        radius = int(np.ceil(3 * sigma))
    x = np.arange(-radius, radius + 1, dtype=float)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    return k


def separable_gaussian_blur(img: Array, sigma_x: float, sigma_y: float) -> Array:
    """
    Separable Gaussian blur via 1D convs (reflect padding).
    """
    def conv1d_along_axis(M: Array, k: Array, axis: int) -> Array:
        r = (len(k) - 1) // 2
        pad = [(0,0)] * M.ndim
        pad[axis] = (r, r)
        P = np.pad(M, pad, mode="reflect")
        out = np.zeros_like(M)
        it = np.nditer(out, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = list(it.multi_index)
            s = 0.0
            for i, w in enumerate(k):
                idx_pad = idx.copy()
                idx_pad[axis] = idx[axis] + i
                s += w * P[tuple(idx_pad)]
            out[tuple(idx)] = s
            it.iternext()
        return out

    kx = gaussian_kernel_1d(sigma_x)
    ky = gaussian_kernel_1d(sigma_y)
    tmp = conv1d_along_axis(img, kx, axis=1)
    out = conv1d_along_axis(tmp, ky, axis=0)
    return out


def potential_from_density(p: Array, eps: float = 1e-12) -> Array:
    """
    Phi = -log p (up to a constant). For visualization we just compute elementwise.
    """
    return -np.log(p + eps)

