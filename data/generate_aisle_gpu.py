"""
Hot-aisle / cold-aisle data-center cooling generator (Boussinesq + vents, GPU).

Fixed racks (hot sources) + movable cold-aisle vents (cold sources) on the floor:
    dw/dt + (u.grad)w = nu*lap(w) + buoy*dT/dx
    dT/dt + (u.grad)T = kappa*lap(T) + S(x) - alpha*T
    S(x) = sum(rack hot bumps)  -  sum(vent cold bumps)      (both on the floor)

Buoyancy sorts it out: cold (dense) air pools in the cold aisles over the vents;
hot rack exhaust rises into the hot aisles. Racks are FIXED; the vent positions
vary per trajectory = the design parameter (where to place cooling).

Output: w, T (each (N,frames,H,W)) AND S (N,H,W) = the racks+vents source map.
"""
import argparse
import time
import numpy as np
import torch

RACK_X = [0.25, 0.50, 0.75]          # fixed installed rack rows


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


def sample_sources(n, s, device, gen, n_vents=2, rack_power=6.0, vent_power=6.0,
                   sigma=0.045, floor=0.12):
    """Fixed racks (hot) + random cold vents (cold) -> source map S, shape (n,s,s)."""
    x = torch.arange(s, device=device) / s
    X, Y = torch.meshgrid(x, x, indexing="ij")           # gravity along Y (axis 1)
    S = torch.zeros(n, s, s, device=device)
    for xr in RACK_X:                                    # fixed hot racks
        S += rack_power * torch.exp(-((X - xr) ** 2 + (Y - floor) ** 2) / (2 * sigma ** 2))
    for i in range(n):                                   # random cold vents
        for _ in range(n_vents):
            xv = 0.12 + 0.76 * torch.rand(1, generator=gen, device=device).item()
            S[i] -= vent_power * torch.exp(-((X - xv) ** 2 + (Y - floor) ** 2) / (2 * sigma ** 2))
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
    Tc = kappa * ksq + alpha
    T_new = ((1 - 0.5 * dt * Tc) * T_hat - dt * NT + dt * S_hat) / (1 + 0.5 * dt * Tc)
    return w_new, T_new


@torch.no_grad()
def simulate(n, s, T, dt, nu, kappa, alpha, buoy, n_vents, record_every, device, gen):
    ikx, iky, ksq, ksq_safe, dealias = build_operators(s, device)
    S = sample_sources(n, s, device, gen, n_vents=n_vents)
    S_hat = torch.fft.fft2(S, dim=(1, 2))
    T_hat = torch.fft.fft2(0.01 * gaussian_rf(n, s, device, gen), dim=(1, 2))
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


def save_preview(Tt, S, path="aisle_preview.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nf = Tt.shape[1]
    idx = np.linspace(0, nf - 1, 5).astype(int)
    fig, ax = plt.subplots(1, 6, figsize=(18, 3))
    ax[0].imshow(S[0].T, cmap="coolwarm", origin="lower"); ax[0].set_title("sources S (red=rack, blue=vent)"); ax[0].axis("off")
    tv = float(np.abs(Tt[0]).max())
    for a, i in zip(ax[1:], idx):
        a.imshow(Tt[0, i].T, cmap="RdBu_r", origin="lower", vmin=-tv, vmax=tv)
        a.set_title(f"T, frame {i}"); a.axis("off")
    fig.suptitle("hot aisle / cold aisle: sources + temperature (hot=red over racks, cold=blue over vents)")
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
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--buoy", type=float, default=0.5)
    ap.add_argument("--n_vents", type=int, default=2)
    ap.add_argument("--record_every", type=int, default=375)
    ap.add_argument("--out", type=str, default="aisle_64.npz")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = torch.Generator(device=device).manual_seed(args.seed)
    print(f"[{device}] aisle  ntraj={args.ntraj}  grid={args.grid}  T={args.T}  "
          f"nu={args.nu} kappa={args.kappa} alpha={args.alpha} buoy={args.buoy} vents={args.n_vents}")
    t0 = time.time()
    w, Tt, S = simulate(args.ntraj, args.grid, args.T, args.dt, args.nu, args.kappa,
                        args.alpha, args.buoy, args.n_vents, args.record_every, device, gen)
    print(f"  done in {time.time()-t0:.1f}s")
    np.savez_compressed(args.out, w=w, T=Tt, S=S, dt=args.dt, record_every=args.record_every,
                        nu=args.nu, kappa=args.kappa, alpha=args.alpha, buoy=args.buoy,
                        rack_x=np.array(RACK_X))
    print(f"saved -> {args.out}   w{w.shape}  T{Tt.shape}  S{S.shape}")
    if args.preview:
        save_preview(Tt, S)


if __name__ == "__main__":
    main()
