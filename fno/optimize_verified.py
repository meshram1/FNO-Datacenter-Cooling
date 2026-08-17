"""CFD-verified cooling-layout optimization loop (racks fixed, optimize vents).

Naive surrogate optimization exploits the FNO's errors (the "optimizer's curse"):
its predicted 15.7% reduction verified at only 2.8% in CFD. The fix is a
surrogate + ground-truth loop:

    1. PROPOSE : the FNO cheaply generates/ranks candidate vent layouts
    2. VERIFY  : run the real spectral CFD solver on each candidate
    3. SELECT  : keep the candidate with the lowest CFD (ground-truth) hot spot

The reported reduction is CFD-verified, so it holds in the real solver.

    python optimize_verified.py --ckpt aisle_fno.pt --data ../data/aisle_64.npz
"""
import argparse
import math

import numpy as np
import torch

from optimize_aisle import load_model, build_S, predict_T, soft_peak, RACK_X, GRID
from verify_optimize import cfd_peak                       # the ground-truth evaluator


def surrogate_peak(model, vent, stats, cfg, device):
    v = torch.as_tensor(vent, dtype=torch.float32, device=device)
    with torch.no_grad():
        return predict_T(model, v, stats, cfg, device, 8)[0].max().item()


def propose(model, stats, cfg, device, n_starts=4, iters=100, lr=0.02):
    """Surrogate proposals: multi-start gradient optima + a coarse grid (coverage)."""
    cands = []
    torch.manual_seed(0)
    for _ in range(n_starts):                              # surrogate-optimized proposals
        vent = (0.12 + 0.76 * torch.rand(2, device=device)).requires_grad_(True)
        opt = torch.optim.Adam([vent], lr=lr)
        for _ in range(iters):
            opt.zero_grad()
            soft_peak(predict_T(model, vent, stats, cfg, device, 8)[0]).backward()
            opt.step()
            with torch.no_grad():
                vent.clamp_(0.06, 0.94)
        cands.append([round(x, 3) for x in vent.detach().tolist()])
    grid = [0.25, 0.40, 0.55, 0.70]                        # coarse coverage
    for i in range(len(grid)):
        for j in range(i, len(grid)):
            cands.append([grid[i], grid[j]])
    # de-dup
    uniq = []
    for c in cands:
        if not any(abs(c[0]-u[0]) < 1e-3 and abs(c[1]-u[1]) < 1e-3 for u in uniq):
            uniq.append(c)
    return uniq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="aisle_fno.pt")
    ap.add_argument("--data", default="../data/aisle_64.npz")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, cfg = load_model(args.ckpt, device)
    d = np.load(args.data)
    field = torch.stack([torch.from_numpy(d["w"]).float(),
                         torch.from_numpy(d["T"]).float()], dim=2)
    S_all = torch.from_numpy(d["S"]).float(); nt = 40
    fmean = field[:-nt].mean(dim=(0, 1, 3, 4)).to(device); fstd = field[:-nt].std(dim=(0, 1, 3, 4)).to(device)
    smean = S_all[:-nt].mean().to(device); sstd = S_all[:-nt].std().to(device)
    stats = (fmean, fstd, smean, sstd)

    def cfd(v):
        return cfd_peak(build_S(torch.as_tensor(v, dtype=torch.float32, device=device), GRID, device), device)

    before = [0.12, 0.88]
    cfd_before = cfd(before)
    print(f"racks fixed {RACK_X}   baseline vents {before}   CFD peak = {cfd_before:.3f}\n")

    cands = propose(model, stats, cfg, device)
    print(f"proposing + CFD-verifying {len(cands)} candidate layouts:\n")
    print(f"  {'vents':>14} {'surrogate':>10} {'CFD':>8}")
    best = None
    for v in cands:
        s = surrogate_peak(model, v, stats, cfg, device)
        c = cfd(v)
        flag = ""
        if not math.isnan(c) and (best is None or c < best[1]):
            best = (v, c); flag = "  <-- best CFD"
        print(f"  {str([round(x,2) for x in v]):>14} {s:>10.3f} {c:>8.3f}{flag}")

    print(f"\n  === CFD-VERIFIED optimum ===")
    print(f"  verified-best vents : {best[0]}")
    print(f"  CFD peak            : {before}->{cfd_before:.3f}   optimized->{best[1]:.3f}")
    print(f"  CFD-VERIFIED reduction : {100*(cfd_before-best[1])/cfd_before:.1f}%")
    print(f"  (naive surrogate-only optimization verified at only 2.8%)")


if __name__ == "__main__":
    main()
