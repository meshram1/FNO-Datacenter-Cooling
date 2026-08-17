"""
2D incompressible Navier-Stokes data generator (vorticity form, pseudo-spectral).

Periodic domain [0,1]^2:
    dw/dt + (u . grad) w = nu * lap(w) + f ,   w = curl(u)   (scalar vorticity)
    u = d(psi)/dy ,  v = -d(psi)/dx ,          lap(psi) = -w  (streamfunction)
    f = 0.1*( sin(2pi(x+y)) + cos(2pi(x+y)) )                 (fixed forcing)

Scheme: everything linear (Laplacian, streamfunction) is algebraic in Fourier
space; the viscous term is advanced with Crank-Nicolson (implicit, unconditionally
stable) and the nonlinear advection explicitly, evaluated pseudo-spectrally
(products in physical space, transform back) with 2/3-rule dealiasing.
"""
import argparse
import numpy as np


# ----------------------------------------------------------------------
# initial vorticity: periodic Gaussian random field  ~ N(0,(-lap+tau^2)^-alpha)
# ----------------------------------------------------------------------
def gaussian_rf(n, s, alpha=2.5, tau=7.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    sigma = tau ** (0.5 * (2 * alpha - 2.0))
    k = np.fft.fftfreq(s, d=1.0 / s)                 # integer wavenumbers
    kx, ky = np.meshgrid(k, k, indexing="ij")
    sqrt_eig = (s ** 2) * np.sqrt(2.0) * sigma * \
        ((4.0 * np.pi ** 2 * (kx ** 2 + ky ** 2) + tau ** 2) ** (-alpha / 2.0))
    sqrt_eig[0, 0] = 0.0                              # zero mean
    xi = rng.standard_normal((n, s, s)) + 1j * rng.standard_normal((n, s, s))
    return np.fft.ifft2(sqrt_eig[None] * xi, axes=(1, 2)).real


# ----------------------------------------------------------------------
# spectral operators for the solver (angular wavenumbers, 2/3 dealias mask)
# ----------------------------------------------------------------------
def build_operators(s):
    k = 2.0 * np.pi * np.fft.fftfreq(s, d=1.0 / s)   # angular wavenumbers
    kx, ky = np.meshgrid(k, k, indexing="ij")
    ksq = kx ** 2 + ky ** 2
    ksq_safe = ksq.copy()
    ksq_safe[0, 0] = 1.0                             # avoid /0 at the mean mode
    absf = np.abs(np.fft.fftfreq(s, d=1.0 / s))
    keep = absf < (s / 3.0)                          # 2/3 rule
    dealias = (keep[:, None] & keep[None, :]).astype(np.float64)
    return kx, ky, ksq, ksq_safe, dealias


def forcing(s):
    x = np.linspace(0.0, 1.0, s, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="ij")
    f = 0.1 * (np.sin(2 * np.pi * (X + Y)) + np.cos(2 * np.pi * (X + Y)))
    return np.fft.fft2(f)                            # f_hat (steady forcing)


def velocity(w_hat, kx, ky, ksq_safe):
    """Physical (u, v) from the vorticity spectrum via the streamfunction."""
    psi_hat = w_hat / ksq_safe                       # lap(psi) = -w  ->  psi_hat = w_hat/ksq
    psi_hat[:, 0, 0] = 0.0
    u = np.fft.ifft2(1j * ky * psi_hat, axes=(1, 2)).real
    v = np.fft.ifft2(-1j * kx * psi_hat, axes=(1, 2)).real
    return u, v


def advection_hat(w_hat, kx, ky, ksq_safe, dealias):
    """Spectrum of N = u.grad(w), dealiased."""
    u, v = velocity(w_hat, kx, ky, ksq_safe)
    wx = np.fft.ifft2(1j * kx * w_hat, axes=(1, 2)).real
    wy = np.fft.ifft2(1j * ky * w_hat, axes=(1, 2)).real
    n_hat = np.fft.fft2(u * wx + v * wy, axes=(1, 2))
    return n_hat * dealias


def step(w_hat, dt, nu, kx, ky, ksq, ksq_safe, f_hat, dealias):
    n_hat = advection_hat(w_hat, kx, ky, ksq_safe, dealias)
    num = (1.0 - 0.5 * dt * nu * ksq) * w_hat - dt * n_hat + dt * f_hat
    den = 1.0 + 0.5 * dt * nu * ksq
    return num / den


# ----------------------------------------------------------------------
def simulate(n, s, T, dt, nu, record_every, rng):
    kx, ky, ksq, ksq_safe, dealias = build_operators(s)
    f_hat = forcing(s)[None]                          # (1,s,s) broadcasts over batch
    w0 = gaussian_rf(n, s, rng=rng)
    w_hat = np.fft.fft2(w0, axes=(1, 2))
    w_hat[:, 0, 0] = 0.0

    n_steps = int(round(T / dt))
    frames = []
    for it in range(n_steps):
        w_hat = step(w_hat, dt, nu, kx, ky, ksq, ksq_safe, f_hat, dealias)
        if (it + 1) % record_every == 0:
            w = np.fft.ifft2(w_hat, axes=(1, 2)).real
            if not np.isfinite(w).all():
                raise FloatingPointError(
                    f"blew up at step {it+1}: reduce dt (CFL) or nu.")
            frames.append(w.astype(np.float32))
    traj = np.stack(frames, axis=1)                   # (n, n_frames, s, s)
    print(f"  mean|w|={np.abs(traj).mean():.4e}  mean(w)={traj.mean():.2e} (~0 expected)  "
          f"range=[{traj.min():.2f},{traj.max():.2f}]  frames={traj.shape[1]}")
    return traj


def save_preview(traj, path="ns_preview.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nf = traj.shape[1]
    idx = np.linspace(0, nf - 1, min(6, nf)).astype(int)
    fig, ax = plt.subplots(1, len(idx), figsize=(3 * len(idx), 3))
    for a, i in zip(np.atleast_1d(ax), idx):
        a.imshow(traj[0, i], cmap="RdBu_r"); a.set_title(f"frame {i}"); a.axis("off")
    fig.suptitle("trajectory 0: vorticity over time")
    fig.tight_layout(); fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"  wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=64)
    ap.add_argument("--ntraj", type=int, default=200)
    ap.add_argument("--T", type=float, default=20.0)
    ap.add_argument("--dt", type=float, default=1e-3)
    ap.add_argument("--nu", type=float, default=1e-3, help="viscosity; Re ~ 1/nu")
    ap.add_argument("--record_every", type=int, default=500)
    ap.add_argument("--out", type=str, default="ns_64.npz")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    print(f"simulating {args.ntraj} trajectories  grid={args.grid}  T={args.T}  "
          f"dt={args.dt}  nu={args.nu}")
    traj = simulate(args.ntraj, args.grid, args.T, args.dt, args.nu,
                    args.record_every, rng)
    np.savez_compressed(args.out, w=traj, dt=args.dt,
                        record_every=args.record_every, nu=args.nu)
    print(f"saved -> {args.out}   shape={traj.shape}")
    if args.preview:
        save_preview(traj)


if __name__ == "__main__":
    main()
