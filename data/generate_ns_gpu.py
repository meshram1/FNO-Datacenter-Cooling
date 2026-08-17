"""
2D incompressible Navier-Stokes data generator -- GPU version (torch.fft on CUDA).

Identical pseudo-spectral vorticity-form solver as generate_ns.py, but every
array is a CUDA tensor and every FFT runs on the GPU (cuFFT). ~20-40x faster:
the full 200-trajectory set generates in ~1-2 min instead of ~40.

Runs in float32 (fine for training data), so results are equally valid but not
bit-identical to the float64 numpy version. Same output format (.npz with w,
dt, record_every, nu), so the rest of the pipeline is unchanged.
"""
import argparse
import time
import numpy as np
import torch


def gaussian_rf(n, s, device, gen, alpha=2.5, tau=7.0):
    sigma = tau ** (0.5 * (2 * alpha - 2.0))
    k = torch.fft.fftfreq(s, d=1.0 / s, device=device)          # integer wavenumbers
    kx, ky = torch.meshgrid(k, k, indexing="ij")
    sqrt_eig = (s ** 2) * (2.0 ** 0.5) * sigma * \
        ((4.0 * (torch.pi ** 2) * (kx ** 2 + ky ** 2) + tau ** 2) ** (-alpha / 2.0))
    sqrt_eig[0, 0] = 0.0
    re = torch.randn(n, s, s, device=device, generator=gen)
    im = torch.randn(n, s, s, device=device, generator=gen)
    xi = torch.complex(re, im)
    return torch.fft.ifft2(sqrt_eig[None] * xi, dim=(1, 2)).real


def build_operators(s, device):
    k = 2.0 * torch.pi * torch.fft.fftfreq(s, d=1.0 / s, device=device)  # angular
    kx, ky = torch.meshgrid(k, k, indexing="ij")
    ksq = kx ** 2 + ky ** 2
    ksq_safe = ksq.clone(); ksq_safe[0, 0] = 1.0                 # avoid /0 at mean mode
    ikx = torch.complex(torch.zeros_like(kx), kx)               # i*kx  (d/dx)
    iky = torch.complex(torch.zeros_like(ky), ky)               # i*ky  (d/dy)
    absf = torch.fft.fftfreq(s, d=1.0 / s, device=device).abs()
    keep = absf < (s / 3.0)                                     # 2/3 dealias rule
    dealias = (keep[:, None] & keep[None, :]).to(torch.float32)
    return ikx, iky, ksq, ksq_safe, dealias


def forcing(s, device):
    x = torch.arange(s, device=device) / s                     # 0..(s-1)/s (endpoint=False)
    X, Y = torch.meshgrid(x, x, indexing="ij")
    f = 0.1 * (torch.sin(2 * torch.pi * (X + Y)) + torch.cos(2 * torch.pi * (X + Y)))
    return torch.fft.fft2(f)                                    # f_hat


def velocity(w_hat, ikx, iky, ksq_safe):
    psi_hat = w_hat / ksq_safe                                 # lap(psi)=-w -> psi_hat=w_hat/ksq
    psi_hat[:, 0, 0] = 0.0
    u = torch.fft.ifft2(iky * psi_hat, dim=(1, 2)).real        #  d(psi)/dy
    v = torch.fft.ifft2(-ikx * psi_hat, dim=(1, 2)).real       # -d(psi)/dx
    return u, v


def advection_hat(w_hat, ikx, iky, ksq_safe, dealias):
    u, v = velocity(w_hat, ikx, iky, ksq_safe)
    wx = torch.fft.ifft2(ikx * w_hat, dim=(1, 2)).real
    wy = torch.fft.ifft2(iky * w_hat, dim=(1, 2)).real
    n_hat = torch.fft.fft2(u * wx + v * wy, dim=(1, 2))         # N = u.grad(w)
    return n_hat * dealias


def step(w_hat, dt, nu, ksq, ksq_safe, ikx, iky, f_hat, dealias):
    n_hat = advection_hat(w_hat, ikx, iky, ksq_safe, dealias)
    num = (1.0 - 0.5 * dt * nu * ksq) * w_hat - dt * n_hat + dt * f_hat
    den = 1.0 + 0.5 * dt * nu * ksq
    return num / den


@torch.no_grad()
def simulate(n, s, T, dt, nu, record_every, device, gen):
    ikx, iky, ksq, ksq_safe, dealias = build_operators(s, device)
    f_hat = forcing(s, device)[None]                           # (1,s,s) broadcasts over batch
    w0 = gaussian_rf(n, s, device, gen)
    w_hat = torch.fft.fft2(w0, dim=(1, 2)); w_hat[:, 0, 0] = 0.0

    n_steps = int(round(T / dt))
    frames = []
    for it in range(n_steps):
        w_hat = step(w_hat, dt, nu, ksq, ksq_safe, ikx, iky, f_hat, dealias)
        if (it + 1) % record_every == 0:
            w = torch.fft.ifft2(w_hat, dim=(1, 2)).real
            if not torch.isfinite(w).all():
                raise FloatingPointError(f"blew up at step {it+1}: reduce dt or nu.")
            frames.append(w.float().cpu())
    traj = torch.stack(frames, dim=1).numpy()                  # (n, n_frames, s, s)
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = torch.Generator(device=device).manual_seed(args.seed)
    print(f"[{device}] simulating {args.ntraj} trajectories  grid={args.grid}  "
          f"T={args.T}  dt={args.dt}  nu={args.nu}")
    t0 = time.time()
    traj = simulate(args.ntraj, args.grid, args.T, args.dt, args.nu,
                    args.record_every, device, gen)
    print(f"  done in {time.time()-t0:.1f}s")
    np.savez_compressed(args.out, w=traj, dt=args.dt,
                        record_every=args.record_every, nu=args.nu)
    print(f"saved -> {args.out}   shape={traj.shape}")
    if args.preview:
        save_preview(traj)


if __name__ == "__main__":
    main()
