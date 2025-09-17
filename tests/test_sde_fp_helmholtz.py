import numpy as np
from persystems.sde import euler_maruyama, drift_doublewell_with_swirl
from persystems.fp import histogram2d_density, potential_from_density
from persystems.helmholtz import helmholtz_decompose_periodic


def test_histogram_density_normalizes():
    rng = np.random.default_rng(0)
    f = drift_doublewell_with_swirl(k=0.5)
    X = euler_maruyama(np.array([0.0, 0.0]), f, dt=0.003, T=50_000, D=0.05, rng=rng)
    Xs = X[10_000:]

    xbins = np.linspace(Xs[:,0].min()-0.5, Xs[:,0].max()+0.5, 60)
    ybins = np.linspace(Xs[:,1].min()-0.5, Xs[:,1].max()+0.5, 60)
    p, xc, yc = histogram2d_density(Xs, xbins, ybins)
    assert p.min() >= 0.0
    tot = float(p.sum())
    # Histogram is normalized to ~1
    assert abs(tot - 1.0) < 1e-6
    Phi = potential_from_density(p)
    assert np.isfinite(Phi).all()


def test_helmholtz_reconstruction_small_error():
    # Synthetic periodic vector field on a grid
    Nx, Ny = 64, 48
    x = np.linspace(0, 2*np.pi, Nx, endpoint=False)
    y = np.linspace(0, 2*np.pi, Ny, endpoint=False)
    XX, YY = np.meshgrid(x, y)
    Vx =  2.0*np.cos(XX) - 0.8*np.sin(YY)
    Vy = -2.0*np.sin(XX) + 0.8*np.cos(YY)

    vgx, vgy, vsx, vsy = helmholtz_decompose_periodic(Vx, Vy, Lx=2*np.pi, Ly=2*np.pi)
    rec = np.sqrt(np.mean((Vx - (vgx + vsx))**2 + (Vy - (vgy + vsy))**2))
    assert rec < 1e-6

