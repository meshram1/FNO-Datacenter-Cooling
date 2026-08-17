"""
2D data-center cooling data generator -- Boussinesq + heated racks (GPU spectral).

Extends the buoyancy solver with SUSTAINED heat sources (racks) and cooling:
    dw/dt + (u.grad)w = nu*lap(w) + buoy*dT/dx
    dT/dt + (u.grad)T = kappa*lap(T) + S(x) - alpha*T
                                        \_____/   \____/
                                    rack heating   ambient cooling (AC)

Racks = Gaussian hot patches on the floor, at random x-positions per trajectory
(the design parameter). Buoyancy lifts the heated fluid -> rising plumes off the
racks. Sustained forcing -> statistically steady state (should also stabilize
rollout, unlike the freely-decaying v1).

Output: w, T (both (N, frames, H, W)) AND S (N, H, W) = the rack layout map, so
the FNO can learn  rack-layout -> thermal field.
"""
import argparse
import time
import numpy as np
import torch


def gaussian_rf(n, s, device, gen, alpha=2.5, tau=7.0):
    sigma = tau ** (0.5 * (2 * alpha - 2.0))
    k = torch.fft.fftfreq(s, d=1.0 / s, device=device)
    kx, ky = torch.meshgrid(k, k, indexing="ij")
    sqrt_eig = (s ** 2) * (2.0 ** 0.5) * sigma * \
        ((4.0 * (torch.pi ** 2) * (kx ** 2 + ky ** 2) + tau ** 2) ** (-alpha / 2.0))
    sqrt_eig[0, 0] = 0.0
    re = torch.randn(n, s, s, device=device, generator=gen)
    im = torch.randn(n, s, s, device=device, generator=gen)
    return torch.fft.ifft2(sqrt_eig[None] * torch.complex(re, im), dim=(1, 2)).real


def build_operators(s, device):
    k = 2.0 * torch.pi * torch.fft.fftfreq(s, d=1.0 / s, device=device)
    kx, ky = torch.meshgrid(k, k, indexing="ij")
    ksq = kx ** 2 + ky ** 2
    ksq_safe = ksq.clone(); ksq_safe[0, 0] = 1.0
    ikx = torch.complex(torch.zeros_like(kx), kx)
    iky = torch.complex(torch.zeros_like(ky), ky)
    absf = torch.fft.fftfreq(s, d=1.0 / s, device=device).abs()
    keep = absf < (s / 3.0)
    dealias = (keep[:, None] & keep[None, :]).to(torch.float32)
    return ikx, iky, ksq, ksq_safe, dealias


def sample_racks(n, s, device, gen, n_racks=3, power=6.0, sigma=0.045, floor=0.12):
    """Random rack layouts -> heat-source field S, shape (n, s, s)."""
    x = torch.arange(s, device=device) / s
    X, Y = torch.meshgrid(x, x, indexing="ij")            # gravity along Y (axis 1)
    S = torch.zeros(n, s, s, device=device)
    for i in range(n):
        for _ in range(n_racks):
            x0 = 0.15 + 0.70 * torch.rand(1, generator=gen, device=device).item()
            S[i] += power * torch.exp(-((X - x0) ** 2 + (Y - floor) ** 2) / (2 * sigma ** 2))
    return S


def velocity(w_hat, ikx, iky, ksq_safe):
    psi_hat = w_hat / ksq_safe
    psi_hat[:, 0, 0] = 0.0
    u = torch.fft.ifft2(iky * psi_hat, dim=(1, 2)).real
    v = torch.fft.ifft2(-ikx * psi_hat, dim=(1, 2)).real
    return u, v


def adv_hat(field_hat, u, v, ikx, iky, dealias):
    fx = torch.fft.ifft2(ikx * field_hat, dim=(1, 2)).real
    fy = torch.fft.ifft2(iky * field_hat, dim=(1, 2)).real
    return torch.fft.fft2(u * fx + v * fy, dim=(1, 2)) * dealias


def step(w_hat, T_hat, S_hat, dt, nu, kappa, alpha, buoy, ikx, iky, ksq, ksq_safe, dealias):
    u, v = velocity(w_hat, ikx, iky, ksq_safe)
    Nw = adv_hat(w_hat, u, v, ikx, iky, dealias)
    NT = adv_hat(T_hat, u, v, ikx, iky, dealias)
    buoy_src = buoy * (ikx * T_hat) * dealias
    w_new = ((1 - 0.5 * dt * nu * ksq) * w_hat - dt * Nw + dt * buoy_src) / (1 + 0.5 * dt * nu * ksq)
    Tc = kappa * ksq + alpha                               # diffusion + cooling, implicit
    T_new = ((1 - 0.5 * dt * Tc) * T_hat - dt * NT + dt * S_hat) / (1 + 0.5 * dt * Tc)
    return w_new, T_new


@torch.no_grad()
def simulate(n, s, T, dt, nu, kappa, alpha, buoy, n_racks, record_every, device, gen):
    ikx, iky, ksq, ksq_safe, dealias = build_operators(s, device)
    S = sample_racks(n, s, device, gen, n_racks=n_racks)  # rack layout (n,s,s)
    S_hat = torch.fft.fft2(S, dim=(1, 2))
    T_hat = torch.fft.fft2(0.01 * gaussian_rf(n, s, device, gen), dim=(1, 2))  # tiny seed
    w_hat = torch.zeros_like(T_hat)
    n_steps = int(round(T / dt))
    wf, Tf = [], []
    for it in range(n_steps):
        w_hat, T_hat = step(w_hat, T_hat, S_hat, dt, nu, kappa, alpha, buoy,
                            ikx, iky, ksq, ksq_safe, dealias)
        if (it + 1) % record_every == 0:
            w = torch.fft.ifft2(w_hat, dim=(1, 2)).real
            Tt = torch.fft.ifft2(T_hat, dim=(1, 2)).real
            if not (torch.isfinite(w).all() and torch.isfinite(Tt).all()):
                raise FloatingPointError(f"blew up at step {it+1}: reduce dt/buoy/power.")
            wf.append(w.float().cpu()); Tf.append(Tt.float().cpu())
    w = torch.stack(wf, dim=1).numpy()
    Tt = torch.stack(Tf, dim=1).numpy()
    print(f"  w range [{w.min():.2f},{w.max():.2f}]  T range [{Tt.min():.2f},{Tt.max():.2f}]  "
          f"frames={w.shape[1]}")
    return w, Tt, S.cpu().numpy()


def save_preview(Tt, S, path="cooling_preview.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nf = Tt.shape[1]
    idx = np.linspace(0, nf - 1, 5).astype(int)
    fig, ax = plt.subplots(1, 6, figsize=(18, 3))
    ax[0].imshow(S[0].T, cmap="viridis", origin="lower"); ax[0].set_title("rack layout S"); ax[0].axis("off")
    for a, i in zip(ax[1:], idx):
        a.imshow(Tt[0, i].T, cmap="inferno", origin="lower")
        a.set_title(f"T, frame {i}"); a.axis("off")
    fig.suptitle("trajectory 0: rack layout + temperature (plumes rise from racks)")
    fig.tight_layout(); fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"  wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=64)
    ap.add_argument("--ntraj", type=int, default=200)
    ap.add_argument("--T", type=float, default=15.0)
    ap.add_argument("--dt", type=float, default=1e-3)
    ap.add_argument("--nu", type=float, default=1e-3)
    ap.add_argument("--kappa", type=float, default=1.4e-3)
    ap.add_argument("--alpha", type=float, default=1.0, help="ambient cooling rate")
    ap.add_argument("--buoy", type=float, default=0.5)
    ap.add_argument("--n_racks", type=int, default=3)
    ap.add_argument("--record_every", type=int, default=375)
    ap.add_argument("--out", type=str, default="cooling_64.npz")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = torch.Generator(device=device).manual_seed(args.seed)
    print(f"[{device}] cooling  ntraj={args.ntraj}  grid={args.grid}  T={args.T}  "
          f"nu={args.nu} kappa={args.kappa} alpha={args.alpha} buoy={args.buoy} racks={args.n_racks}")
    t0 = time.time()
    w, Tt, S = simulate(args.ntraj, args.grid, args.T, args.dt, args.nu, args.kappa,
                        args.alpha, args.buoy, args.n_racks, args.record_every, device, gen)
    print(f"  done in {time.time()-t0:.1f}s")
    np.savez_compressed(args.out, w=w, T=Tt, S=S, dt=args.dt, record_every=args.record_every,
                        nu=args.nu, kappa=args.kappa, alpha=args.alpha, buoy=args.buoy)
    print(f"saved -> {args.out}   w{w.shape}  T{Tt.shape}  S{S.shape}")
    if args.preview:
        save_preview(Tt, S)


if __name__ == "__main__":
    main()
