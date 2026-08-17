"""Gradient-based rack-placement optimizer using the trained cooling FNO.

The FNO is a fast, DIFFERENTIABLE simulator: rack layout -> thermal field. We
parameterise the layout by rack x-positions, build the heat-source map from them
(differentiable Gaussians), predict a short thermal rollout with the FNO, and
minimise the hot-spot (peak) temperature by gradient descent on the positions.

    python optimize_cool.py --ckpt cool_fno.pt --data ../data/cooling_64.npz

This is the "optimise cooling" step: find where to place racks so plumes don't
merge into one big hot spot.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from fno2d_cool import FNO2d

FLOOR, POWER, SIGMA, GRID = 0.12, 6.0, 0.045, 64   # match the data generator


def load_model(ckpt, device):
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    c = ck["config"]
    model = FNO2d(c["t_in"], c["n_fields"], c["n_static"], c["modes"], c["modes"],
                  c["width"], c["layers"]).to(device)
    model.load_state_dict(ck["model_state"]); model.eval()
    for p in model.parameters():
        p.requires_grad_(False)               # freeze weights; we optimise the INPUT
    return model, c


def build_S(pos, s, device):
    """Differentiable rack heat-source map from x-positions -> (s, s)."""
    x = torch.arange(s, device=device) / s
    X, Y = torch.meshgrid(x, x, indexing="ij")
    S = torch.zeros(s, s, device=device)
    for x0 in pos:
        S = S + POWER * torch.exp(-((X - x0) ** 2 + (Y - FLOOR) ** 2) / (2 * SIGMA ** 2))
    return S


def predict_T(model, pos, stats, cfg, device, k_steps=8):
    """Build rack map from positions, seed from rest, roll k steps, return
    (time-averaged physical temperature field, physical rack map)."""
    fmean, fstd, smean, sstd = stats
    s, t_in, C = GRID, cfg["t_in"], cfg["n_fields"]
    S = build_S(pos, s, device)
    Sc = ((S - smean) / (sstd + 1e-8)).view(1, s, s, 1)
    rest = ((torch.zeros(C, device=device) - fmean) / (fstd + 1e-8))          # normalized rest
    window = rest.view(1, 1, C, 1, 1).expand(1, t_in, C, s, s).contiguous()
    T_frames = []
    for _ in range(k_steps):
        x = window.permute(0, 3, 4, 1, 2).reshape(1, s, s, t_in * C)
        x = torch.cat([x, Sc], dim=-1)
        pf = model(x).permute(0, 3, 1, 2)                                     # (1, C, H, W)
        window = torch.cat([window[:, 1:], pf.unsqueeze(1)], dim=1)
        T_frames.append(pf[:, 1])                                             # temperature channel
    T_n = torch.stack(T_frames).mean(0)                                       # (1,H,W) time-avg
    T_phys = T_n * (fstd[1] + 1e-8) + fmean[1]
    return T_phys[0], S


def soft_peak(T, beta=8.0):
    return (1.0 / beta) * torch.logsumexp(beta * T.reshape(-1), dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="cool_fno.pt")
    ap.add_argument("--data", default="../data/cooling_64.npz")
    ap.add_argument("--n_racks", type=int, default=3)
    ap.add_argument("--iters", type=int, default=80)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--k_steps", type=int, default=8)
    ap.add_argument("--figdir", default="../figures")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.figdir, exist_ok=True)
    model, cfg = load_model(args.ckpt, device)

    # normalization stats from the training split (must match training)
    d = np.load(args.data)
    field = torch.stack([torch.from_numpy(d["w"]).float(),
                         torch.from_numpy(d["T"]).float()], dim=2)
    S_all = torch.from_numpy(d["S"]).float()
    nt = 40
    fmean = field[:-nt].mean(dim=(0, 1, 3, 4)).to(device)
    fstd = field[:-nt].std(dim=(0, 1, 3, 4)).to(device)
    smean = S_all[:-nt].mean().to(device); sstd = S_all[:-nt].std().to(device)
    stats = (fmean, fstd, smean, sstd)

    # start from a CLUSTERED layout (racks close together -> plumes merge)
    pos = torch.tensor([0.40, 0.50, 0.60][:args.n_racks], device=device, requires_grad=True)
    pos0 = pos.detach().clone()

    with torch.no_grad():
        T0, S0 = predict_T(model, pos0, stats, cfg, device, args.k_steps)
        peak0 = T0.max().item()

    opt = torch.optim.Adam([pos], lr=args.lr)
    for it in range(args.iters):
        opt.zero_grad()
        T, _ = predict_T(model, pos, stats, cfg, device, args.k_steps)
        loss = soft_peak(T)
        loss.backward()
        opt.step()
        with torch.no_grad():
            pos.clamp_(0.12, 0.88)
        if it % 20 == 0 or it == args.iters - 1:
            print(f"  iter {it:3d}  soft-peak {loss.item():.4f}  peak {T.max().item():.4f}  "
                  f"pos {[round(p,3) for p in pos.detach().tolist()]}")

    with torch.no_grad():
        T1, S1 = predict_T(model, pos, stats, cfg, device, args.k_steps)
        peak1 = T1.max().item()

    print(f"\n  initial layout {[round(p,3) for p in pos0.tolist()]}  peak T = {peak0:.3f}")
    print(f"  optimised layout {[round(p,3) for p in pos.detach().tolist()]}  peak T = {peak1:.3f}")
    print(f"  peak temperature reduced {100*(peak0-peak1)/peak0:.1f}%")

    # before/after figure
    S0, S1 = S0.cpu().numpy(), S1.cpu().numpy()
    T0n, T1n = T0.cpu().numpy(), T1.cpu().numpy()
    vmax = float(max(T0n.max(), T1n.max()))
    fig, ax = plt.subplots(2, 2, figsize=(8, 8))
    ax[0, 0].imshow(S0.T, cmap="viridis", origin="lower"); ax[0, 0].set_title("BEFORE: rack layout"); ax[0, 0].axis("off")
    ax[0, 1].imshow(T0n.T, cmap="inferno", origin="lower", vmin=0, vmax=vmax); ax[0, 1].set_title(f"BEFORE: T  (peak {peak0:.2f})"); ax[0, 1].axis("off")
    ax[1, 0].imshow(S1.T, cmap="viridis", origin="lower"); ax[1, 0].set_title("AFTER: rack layout"); ax[1, 0].axis("off")
    ax[1, 1].imshow(T1n.T, cmap="inferno", origin="lower", vmin=0, vmax=vmax); ax[1, 1].set_title(f"AFTER: T  (peak {peak1:.2f})"); ax[1, 1].axis("off")
    fig.suptitle("Rack-placement optimization: minimize hot-spot temperature")
    fig.tight_layout(); fig.savefig(f"{args.figdir}/cool_optimize.png", dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {args.figdir}/cool_optimize.png")


if __name__ == "__main__":
    main()
