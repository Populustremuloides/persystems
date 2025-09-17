"""
Chemotaxis-like SDE demo:
1) Simulate 2D Langevin SDE with a double-well + swirl drift.
2) Estimate steady-state density via histogram; compute Phi = -log p*.
3) On a grid, evaluate the drift and do Helmholtz (gradient vs. solenoidal) split.
4) Build simple (I, A, S, E) variables from the trajectory and run blanket diagnostics.

Run:
  python -m experiments.chemotaxis_sde --T 200000 --dt 0.002 --D 0.05 --sigma 1.0
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt

# --- Path bootstrapping (only if run as a script) ---
if __package__ is None or __package__ == "":
    import os, sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from persystems.sde import euler_maruyama, drift_doublewell_with_swirl
from persystems.fp import histogram2d_density, separable_gaussian_blur, potential_from_density
from persystems.helmholtz import helmholtz_decompose_periodic
from persystems.blankets import blanket_block_norms


def main(T=200_000, dt=0.002, D=0.05, a=1.0, b=1.0, k=1.0,
         x0=(0.0, 0.5), sigma_blur=1.0, seed=7):
    rng = np.random.default_rng(seed)

    # 1) Simulate SDE
    f = drift_doublewell_with_swirl(a=a, b=b, k=k)
    X = euler_maruyama(np.array(x0, dtype=float), f, dt=dt, T=T, D=D, rng=rng)
    # Drop burn-in
    burn = int(0.2 * T)
    Xs = X[burn:]

    # 2) Empirical steady-state (histogram)
    # Domain bounds: a few std devs around data
    pad = 0.5
    xmin, xmax = Xs[:, 0].min() - pad, Xs[:, 0].max() + pad
    ymin, ymax = Xs[:, 1].min() - pad, Xs[:, 1].max() + pad

    Nx = 100
    Ny = 100
    xbins = np.linspace(xmin, xmax, Nx + 1)
    ybins = np.linspace(ymin, ymax, Ny + 1)

    p, xc, yc = histogram2d_density(Xs, xbins, ybins)
    # Smooth density for better viz; sigma in pixels
    p_s = separable_gaussian_blur(p, sigma_x=sigma_blur, sigma_y=sigma_blur)
    Phi = potential_from_density(p_s)

    # 3) Drift field on the grid + Helmholtz split (periodic assumption for demo)
    XX, YY = np.meshgrid(xc, yc)  # shape (Ny, Nx)
    Vx = np.zeros_like(XX)
    Vy = np.zeros_like(YY)
    for i in range(Ny):
        for j in range(Nx):
            v = f(np.array([XX[i, j], YY[i, j]]))
            Vx[i, j], Vy[i, j] = v[0], v[1]

    vgx, vgy, vsx, vsy = helmholtz_decompose_periodic(Vx, Vy, Lx=(xmax - xmin), Ly=(ymax - ymin))
    # Reconstruction error
    rec_err = np.sqrt(np.mean((Vx - (vgx + vsx)) ** 2 + (Vy - (vgy + vsy)) ** 2))

    # 4) Blanket diagnostics from trajectory
    # Build variables:
    #   I (internal): x_t
    #   A (active):   v_x,t ≈ (x_{t+1} - x_t)/dt
    #   S (sensory):  y_t
    #   E (external): y_{t+1}   (what internal cannot access directly without S/A)
    x = Xs[:-1, 0]
    y = Xs[:-1, 1]
    vx = (Xs[1:, 0] - Xs[:-1, 0]) / dt
    y_next = Xs[1:, 1]
    # Stack as columns [I | A | S | E]
    Z = np.stack([x, vx, y, y_next], axis=1)
    norms = blanket_block_norms(Z, dims={"I": 1, "A": 1, "S": 1, "E": 1}, reg=1e-4)

    # ---- Plots ----
    plt.figure(figsize=(6, 5))
    plt.imshow(p_s, origin="lower", extent=[xmin, xmax, ymin, ymax], aspect="auto")
    plt.colorbar(label="p*(x)")
    plt.contour(XX, YY, Phi, levels=20, colors="k", alpha=0.5, linewidths=0.5)
    plt.title("Empirical steady-state density and potential Φ = -log p*")
    plt.xlabel("x"); plt.ylabel("y")
    plt.tight_layout()

    plt.figure(figsize=(6, 5))
    skip = 4
    plt.quiver(XX[::skip, ::skip], YY[::skip, ::skip],
               Vx[::skip, ::skip], Vy[::skip, ::skip], scale=60, width=0.003)
    plt.title("Drift field f(x) = -∇Φ + swirl (double-well + rotation)")
    plt.xlabel("x"); plt.ylabel("y")
    plt.tight_layout()

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.quiver(XX[::skip, ::skip], YY[::skip, ::skip],
               vgx[::skip, ::skip], vgy[::skip, ::skip], scale=60, width=0.003)
    plt.title("Helmholtz: gradient (curl-free) component")
    plt.xlabel("x"); plt.ylabel("y")

    plt.subplot(1, 2, 2)
    plt.quiver(XX[::skip, ::skip], YY[::skip, ::skip],
               vsx[::skip, ::skip], vsy[::skip, ::skip], scale=60, width=0.003)
    plt.title("Helmholtz: solenoidal (divergence-free) component")
    plt.xlabel("x"); plt.ylabel("y")
    plt.suptitle(f"Reconstruction RMSE ≈ {rec_err:.3e}")
    plt.tight_layout()

    # Blanket diagnostics readout
    print("\n=== Blanket diagnostics (precision-matrix block norms) ===")
    for k, v in norms.items():
        print(f"{k:>14s}: {v:.3e}")
    print("Heuristic check: K_IE_cond_AS should be smaller than (unconditioned) K_IE "
          "if a blanket-like separation holds (approximate).")

    plt.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=200_000)
    ap.add_argument("--dt", type=float, default=0.002)
    ap.add_argument("--D", type=float, default=0.05)
    ap.add_argument("--a", type=float, default=1.0)
    ap.add_argument("--b", type=float, default=1.0)
    ap.add_argument("--k", type=float, default=1.0)
    ap.add_argument("--x0x", type=float, default=0.0)
    ap.add_argument("--x0y", type=float, default=0.5)
    ap.add_argument("--sigma", type=float, default=1.0, help="Gaussian blur sigma (pixels)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    main(T=args.T, dt=args.dt, D=args.D, a=args.a, b=args.b, k=args.k,
         x0=(args.x0x, args.x0y), sigma_blur=args.sigma, seed=args.seed)

