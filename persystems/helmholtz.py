"""
helmholtz.py — 2D Helmholtz/Hodge decomposition on a periodic grid using FFT.

Given a vector field v = (vx, vy) on a 2D periodic grid:
  v = v_grad + v_sol, with
  v_grad   = curl-free  (longitudinal)   component
  v_sol    = div-free   (transverse)     component

We compute this via Fourier-space projection:
  P_L = (k k^T) / |k|^2   (longitudinal projector)
  P_T = I - P_L           (transverse  projector)

Returned arrays reconstruct v to numerical precision: v ≈ v_grad + v_sol.
"""
from __future__ import annotations
import numpy as np
from numpy.fft import fft2, ifft2
from typing import Tuple

Array = np.ndarray


def helmholtz_decompose_periodic(
    vx: Array, vy: Array, Lx: float = 1.0, Ly: float = 1.0
) -> Tuple[Array, Array, Array, Array]:
    """
    Periodic 2D Helmholtz decomposition via full complex FFT.

    Args:
        vx, vy: (Ny, Nx) components sampled on a uniform periodic grid.
        Lx, Ly: physical domain lengths (set to 1.0 if units are arbitrary)

    Returns:
        v_grad_x, v_grad_y, v_sol_x, v_sol_y
    """
    Ny, Nx = vx.shape
    # Fourier wavenumbers (match fft2 conventions)
    kx = 2.0 * np.pi * np.fft.fftfreq(Nx, d=Lx / Nx)  # shape (Nx,)
    ky = 2.0 * np.pi * np.fft.fftfreq(Ny, d=Ly / Ny)  # shape (Ny,)

    Vx = fft2(vx)
    Vy = fft2(vy)

    Vgrad_x = np.zeros_like(Vx, dtype=np.complex128)
    Vgrad_y = np.zeros_like(Vy, dtype=np.complex128)
    Vsol_x  = np.zeros_like(Vx, dtype=np.complex128)
    Vsol_y  = np.zeros_like(Vy, dtype=np.complex128)

    eps = 1e-18
    for j in range(Ny):
        ky_j = ky[j]
        for i in range(Nx):
            kx_i = kx[i]
            k2 = kx_i * kx_i + ky_j * ky_j
            vx_hat = Vx[j, i]
            vy_hat = Vy[j, i]

            if k2 < eps:  # zero mode: no direction, all transverse/longitudinal undefined
                # put everything in solenoidal (or zero both; choice doesn’t affect reconstruction)
                Vgrad_x[j, i] = 0.0
                Vgrad_y[j, i] = 0.0
                Vsol_x[j, i]  = vx_hat
                Vsol_y[j, i]  = vy_hat
                continue

            # Projection matrices applied to vector [vx_hat, vy_hat]
            kx_over_k = kx_i / k2
            ky_over_k = ky_j / k2
            # Longitudinal component: (k k^T / |k|^2) v_hat
            vLx = (kx_i * (kx_over_k * vx_hat + ky_over_k * vy_hat))
            vLy = (ky_j * (kx_over_k * vx_hat + ky_over_k * vy_hat))
            # Transverse component: v - v_L
            vTx = vx_hat - vLx
            vTy = vy_hat - vLy

            Vgrad_x[j, i] = vLx
            Vgrad_y[j, i] = vLy
            Vsol_x[j, i]  = vTx
            Vsol_y[j, i]  = vTy

    v_grad_x = ifft2(Vgrad_x).real
    v_grad_y = ifft2(Vgrad_y).real
    v_sol_x  = ifft2(Vsol_x).real
    v_sol_y  = ifft2(Vsol_y).real

    return v_grad_x, v_grad_y, v_sol_x, v_sol_y

