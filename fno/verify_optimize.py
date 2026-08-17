"""Close the loop: verify the FNO-optimized cooling layout against ground-truth CFD.

The optimizer minimizes a SURROGATE (FNO) prediction of the hot spot. That surrogate
carries ~8-10% error, so the honest question is: does the predicted improvement hold
when you run the REAL solver? Here we take the optimizer's before/after cold-vent
layouts (racks fixed) and re-run the actual spectral Boussinesq solver on each,
reporting surrogate-predicted vs. CFD-verified peak-temperature reduction.

    python verify_optimize.py --ckpt aisle_fno.pt --data ../data/aisle_64.npz
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
import generate_aisle_gpu as G                                   # the real CFD solver
from optimize_aisle import load_model, build_S, predict_T, soft_peak, RACK_X, GRID


@torch.no_grad()
def cfd_peak(S_phys, device, Ttot=3.0, dt=5e-4, rec=750,
             nu=1e-3, kappa=1.4e-3, alpha=0.5, buoy=0.5, skip=0):
    """Run the ground-truth solver with source map S; return peak of the
    time-averaged temperature over the SAME 8-frame-from-rest window the surrogate
    optimizes (dt halved vs. the data gen for numerical stability)."""
    s = S_phys.shape[-1]
    ikx, iky, ksq, ksq_safe, dealias = G.build_operators(s, device)
    S_hat = torch.fft.fft2(S_phys.unsqueeze(0), dim=(1, 2))
    gen = torch.Generator(device=device).manual_seed(0)
    T_hat = torch.fft.fft2(0.01 * G.gaussian_rf(1, s, device, gen), dim=(1, 2))
    w_hat = torch.zeros_like(T_hat)
    frames = []
    for it in range(int(round(Ttot / dt))):
        w_hat, T_hat = G.step(w_hat, T_hat, S_hat, dt, nu, kappa, alpha, buoy,
                              ikx, iky, ksq, ksq_safe, dealias)
        if (it + 1) % rec == 0:
            frames.append(torch.fft.ifft2(T_hat, dim=(1, 2)).real)
    Tavg = torch.stack(frames)[skip:].mean(0)[0]                 # drop initial transient
    return Tavg.max().item()


def optimize_vents(model, stats, cfg, device, n_vents=2, n_starts=6, iters=100, lr=0.02, k=8):
    torch.manual_seed(0)
    best = (None, 1e9)
    for _ in range(n_starts):
        vent = (0.12 + 0.76 * torch.rand(n_vents, device=device)).requires_grad_(True)
        opt = torch.optim.Adam([vent], lr=lr)
        for _ in range(iters):
            opt.zero_grad()
            soft_peak(predict_T(model, vent, stats, cfg, device, k)[0]).backward()
            opt.step()
            with torch.no_grad():
                vent.clamp_(0.06, 0.94)
        with torch.no_grad():
            pk = predict_T(model, vent, stats, cfg, device, k)[0].max().item()
        if pk < best[1]:
            best = (vent.detach().clone(), pk)
    return best[0]


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

    before = torch.tensor([0.12, 0.88], device=device)          # naive layout
    after = optimize_vents(model, stats, cfg, device)           # FNO-optimized layout
    print(f"racks fixed at {RACK_X}")
    print(f"  before vents: {[round(x,3) for x in before.tolist()]}")
    print(f"  after  vents: {[round(x,3) for x in after.tolist()]}\n")

    print(f"  {'layout':>7} {'surrogate peak':>15} {'CFD peak':>12}")
    res = {}
    for name, vent in (("before", before), ("after", after)):
        with torch.no_grad():
            surr = predict_T(model, vent, stats, cfg, device, 8)[0].max().item()
            cfd = cfd_peak(build_S(vent, GRID, device), device)
        res[name] = (surr, cfd)
        print(f"  {name:>7} {surr:>15.3f} {cfd:>12.3f}")

    (sb, cb), (sa, ca) = res["before"], res["after"]
    print(f"\n  surrogate-predicted reduction : {100*(sb-sa)/sb:5.1f}%")
    print(f"  CFD-VERIFIED reduction        : {100*(cb-ca)/cb:5.1f}%")


if __name__ == "__main__":
    main()
