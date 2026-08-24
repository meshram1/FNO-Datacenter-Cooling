"""CFD-verified vent optimization driven by the PhysicsNeMo FNO.
Same loop as optimize_verified.py, but the surrogate is the Modulus model
(channels-first), proving the ported model runs the full design pipeline.

    python optimize_modulus.py --ckpt aisle_modulus.pt
"""
import argparse
import math
import torch

from modulus_model import build_model, T_IN
from optimize_aisle import build_S, soft_peak, RACK_X, GRID
from verify_optimize import cfd_peak                      # ground-truth CFD


def load_modulus(ckpt, device):
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    m = build_model(ck["config"]["modes"], ck["config"]["width"]).to(device)
    m.load_state_dict(ck["model_state"]); m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m, [torch.tensor(float(s), device=device) for s in ck["stats"]]


def predict_T(model, vent_pos, stats, device, k_steps=8):
    """Rest-seeded k-step rollout -> time-averaged physical temperature (differentiable)."""
    wm, ws, Tm, Ts, Sm, Ss = stats
    Sn = (build_S(vent_pos, GRID, device) - Sm) / (Ss + 1e-8)
    w = torch.full((GRID, GRID), float(-wm / ws), device=device)   # rest state, normalized
    T = torch.full((GRID, GRID), float(-Tm / Ts), device=device)
    hist = [t for _ in range(T_IN) for t in (w, T)]                # 10 identical rest frames
    frames = []
    for _ in range(k_steps):
        x = torch.stack(hist + [Sn], 0).unsqueeze(0)               # (1, 21, H, W)
        out = model(x)[0]                                          # (2, H, W)
        hist = hist[2:] + [out[0], out[1]]                         # slide the window
        frames.append(out[1] * Ts + Tm)                            # physical T
    return torch.stack(frames).mean(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="aisle_modulus.pt")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, stats = load_modulus(args.ckpt, device)

    def cfd(v):
        return cfd_peak(build_S(torch.as_tensor(v, dtype=torch.float32, device=device),
                                GRID, device), device)

    before = [0.12, 0.88]
    cfd_before = cfd(before)
    print(f"racks fixed {RACK_X}   baseline vents {before}   CFD peak = {cfd_before:.3f}\n")

    cands, torch_seed = [], torch.manual_seed(0)
    for _ in range(4):                                    # multi-start surrogate proposals
        vent = (0.12 + 0.76 * torch.rand(2, device=device)).requires_grad_(True)
        opt = torch.optim.Adam([vent], lr=0.02)
        for _ in range(100):
            opt.zero_grad()
            soft_peak(predict_T(model, vent, stats, device)).backward()
            opt.step()
            with torch.no_grad():
                vent.clamp_(0.06, 0.94)
        cands.append([round(x, 3) for x in vent.detach().tolist()])
    grid = [0.25, 0.40, 0.55, 0.70]                       # coarse coverage
    cands += [[grid[i], grid[j]] for i in range(4) for j in range(i, 4)]

    print(f"  {'vents':>14} {'surrogate':>10} {'CFD':>8}")
    best = None
    for v in cands:
        with torch.no_grad():
            s = predict_T(model, torch.as_tensor(v, dtype=torch.float32, device=device),
                          stats, device).max().item()
        c = cfd(v)
        flag = ""
        if not math.isnan(c) and (best is None or c < best[1]):
            best, flag = (v, c), "  <-- best CFD"
        print(f"  {str([round(x,2) for x in v]):>14} {s:>10.3f} {c:>8.3f}{flag}")

    print(f"\n  === CFD-VERIFIED optimum (PhysicsNeMo surrogate) ===")
    print(f"  verified-best vents    : {best[0]}")
    print(f"  CFD peak               : {cfd_before:.3f} -> {best[1]:.3f}")
    print(f"  CFD-VERIFIED reduction : {100*(cfd_before-best[1])/cfd_before:.1f}%")


if __name__ == "__main__":
    main()
