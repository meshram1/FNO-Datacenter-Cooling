"""Gradient-based COLD-VENT placement optimizer (fixed racks) using the aisle FNO.

Racks are FIXED (installed); the design variable is where to put the cold-aisle
vents. We build the source map from (fixed racks) - (vents at optimizable
positions), predict a short thermal rollout with the FNO, and minimise the
hot-spot (peak) temperature by gradient descent on the vent x-positions.

    python optimize_aisle.py --ckpt aisle_fno.pt --data ../data/aisle_64.npz
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from fno2d_cool import FNO2d

RACK_X = [0.25, 0.50, 0.75]                       # fixed installed racks
RACK_POWER, VENT_POWER, SIGMA, FLOOR, GRID = 6.0, 6.0, 0.045, 0.12, 64


def load_model(ckpt, device):
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    c = ck["config"]
    model = FNO2d(c["t_in"], c["n_fields"], c["n_static"], c["modes"], c["modes"],
                  c["width"], c["layers"]).to(device)
    model.load_state_dict(ck["model_state"]); model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, c


def build_S(vent_pos, s, device):
    """Fixed hot racks minus cold vents (vents = the design variable)."""
    x = torch.arange(s, device=device) / s
    X, Y = torch.meshgrid(x, x, indexing="ij")
    S = torch.zeros(s, s, device=device)
    for xr in RACK_X:
        S = S + RACK_POWER * torch.exp(-((X - xr) ** 2 + (Y - FLOOR) ** 2) / (2 * SIGMA ** 2))
    for xv in vent_pos:
        S = S - VENT_POWER * torch.exp(-((X - xv) ** 2 + (Y - FLOOR) ** 2) / (2 * SIGMA ** 2))
    return S


def predict_T(model, vent_pos, stats, cfg, device, k_steps=8):
    fmean, fstd, smean, sstd = stats
    s, t_in, C = GRID, cfg["t_in"], cfg["n_fields"]
    S = build_S(vent_pos, s, device)
    Sc = ((S - smean) / (sstd + 1e-8)).view(1, s, s, 1)
    rest = ((torch.zeros(C, device=device) - fmean) / (fstd + 1e-8))
    window = rest.view(1, 1, C, 1, 1).expand(1, t_in, C, s, s).contiguous()
    T_frames = []
    for _ in range(k_steps):
        x = window.permute(0, 3, 4, 1, 2).reshape(1, s, s, t_in * C)
        x = torch.cat([x, Sc], dim=-1)
        pf = model(x).permute(0, 3, 1, 2)
        window = torch.cat([window[:, 1:], pf.unsqueeze(1)], dim=1)
        T_frames.append(pf[:, 1])
    T_n = torch.stack(T_frames).mean(0)
    return (T_n * (fstd[1] + 1e-8) + fmean[1])[0], S


def soft_peak(T, beta=8.0):
    return (1.0 / beta) * torch.logsumexp(beta * T.reshape(-1), dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="aisle_fno.pt")
    ap.add_argument("--data", default="../data/aisle_64.npz")
    ap.add_argument("--n_vents", type=int, default=2)
    ap.add_argument("--n_starts", type=int, default=6)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--k_steps", type=int, default=8)
    ap.add_argument("--figdir", default="../figures")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.figdir, exist_ok=True)
    model, cfg = load_model(args.ckpt, device)

    d = np.load(args.data)
    field = torch.stack([torch.from_numpy(d["w"]).float(),
                         torch.from_numpy(d["T"]).float()], dim=2)
    S_all = torch.from_numpy(d["S"]).float()
    nt = 40
    fmean = field[:-nt].mean(dim=(0, 1, 3, 4)).to(device)
    fstd = field[:-nt].std(dim=(0, 1, 3, 4)).to(device)
    smean = S_all[:-nt].mean().to(device); sstd = S_all[:-nt].std().to(device)
    stats = (fmean, fstd, smean, sstd)

    torch.manual_seed(0)
    # baseline: a naive placement (vents at the edges, far from the racks)
    vent0 = torch.tensor([0.12, 0.88][:args.n_vents], device=device)
    with torch.no_grad():
        T0, S0 = predict_T(model, vent0, stats, cfg, device, args.k_steps)
        peak0 = T0.max().item()

    # MULTI-START gradient descent: the design space is low-dim and non-convex
    # (single-start gets stuck), so restart from several random inits and keep the best.
    best_peak, best_vent = float("inf"), None
    for st in range(args.n_starts):
        vent = (0.12 + 0.76 * torch.rand(args.n_vents, device=device)).requires_grad_(True)
        opt = torch.optim.Adam([vent], lr=args.lr)
        for _ in range(args.iters):
            opt.zero_grad()
            T, _ = predict_T(model, vent, stats, cfg, device, args.k_steps)
            soft_peak(T).backward(); opt.step()
            with torch.no_grad():
                vent.clamp_(0.06, 0.94)
        with torch.no_grad():
            pk = predict_T(model, vent, stats, cfg, device, args.k_steps)[0].max().item()
        print(f"  start {st}:  vents {[round(p,3) for p in vent.detach().tolist()]}  peak {pk:.3f}")
        if pk < best_peak:
            best_peak, best_vent = pk, vent.detach().clone()

    vent = best_vent
    with torch.no_grad():
        T1, S1 = predict_T(model, vent, stats, cfg, device, args.k_steps)
        peak1 = T1.max().item()
    print(f"\n  racks fixed at {RACK_X}")
    print(f"  vents  {[round(p,3) for p in vent0.tolist()]}  ->  {[round(p,3) for p in vent.detach().tolist()]}")
    print(f"  peak T  {peak0:.3f}  ->  {peak1:.3f}   ({100*(peak0-peak1)/peak0:.1f}% reduction)")

    S0, S1 = S0.cpu().numpy(), S1.cpu().numpy()
    T0n, T1n = T0.cpu().numpy(), T1.cpu().numpy()
    sv = float(max(np.abs(S0).max(), np.abs(S1).max()))
    tv = float(max(np.abs(T0n).max(), np.abs(T1n).max()))
    fig, ax = plt.subplots(2, 2, figsize=(8, 8))
    ax[0, 0].imshow(S0.T, cmap="coolwarm", origin="lower", vmin=-sv, vmax=sv); ax[0, 0].set_title("BEFORE: racks+vents"); ax[0, 0].axis("off")
    ax[0, 1].imshow(T0n.T, cmap="RdBu_r", origin="lower", vmin=-tv, vmax=tv); ax[0, 1].set_title(f"BEFORE: T  (peak {peak0:.2f})"); ax[0, 1].axis("off")
    ax[1, 0].imshow(S1.T, cmap="coolwarm", origin="lower", vmin=-sv, vmax=sv); ax[1, 0].set_title("AFTER: racks+vents"); ax[1, 0].axis("off")
    ax[1, 1].imshow(T1n.T, cmap="RdBu_r", origin="lower", vmin=-tv, vmax=tv); ax[1, 1].set_title(f"AFTER: T  (peak {peak1:.2f})"); ax[1, 1].axis("off")
    fig.suptitle("Cold-vent placement optimization (racks fixed): minimize hot-spot temperature")
    fig.tight_layout(); fig.savefig(f"{args.figdir}/aisle_optimize.png", dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {args.figdir}/aisle_optimize.png")


if __name__ == "__main__":
    main()
