"""Figure for the CFD-verified optimization loop:
   (1) baseline CFD temperature, (2) verified-best CFD temperature,
   (3) surrogate-vs-CFD scatter over all candidates (why you must verify)."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
import generate_aisle_gpu as G
from optimize_aisle import build_S, GRID

DEV = "cuda" if torch.cuda.is_available() else "cpu"

# candidate (surrogate, CFD) pairs from optimize_verified.py
SURR = [1.457, 1.691, 1.583, 1.583, 1.711, 1.683, 1.684, 1.725, 1.729, 1.537, 1.561, 1.736, 1.703, 1.705]
CFD  = [1.520, 1.726, 1.731, 1.731, 1.755, 1.732, 1.770, 1.769, 1.937, 1.594, 1.606, 1.818, 1.808, 1.765]
BEFORE, BEST = [0.12, 0.88], [0.457, 0.549]


@torch.no_grad()
def cfd_field(vent, Ttot=3.0, dt=5e-4, rec=750, nu=1e-3, kappa=1.4e-3, alpha=0.5, buoy=0.5):
    S = build_S(torch.tensor(vent, dtype=torch.float32, device=DEV), GRID, DEV)
    ikx, iky, ksq, ksq_safe, dealias = G.build_operators(GRID, DEV)
    S_hat = torch.fft.fft2(S.unsqueeze(0), dim=(1, 2))
    gen = torch.Generator(device=DEV).manual_seed(0)
    T_hat = torch.fft.fft2(0.01 * G.gaussian_rf(1, GRID, DEV, gen), dim=(1, 2))
    w_hat = torch.zeros_like(T_hat); frames = []
    for it in range(int(round(Ttot / dt))):
        w_hat, T_hat = G.step(w_hat, T_hat, S_hat, dt, nu, kappa, alpha, buoy,
                              ikx, iky, ksq, ksq_safe, dealias)
        if (it + 1) % rec == 0:
            frames.append(torch.fft.ifft2(T_hat, dim=(1, 2)).real)
    return torch.stack(frames).mean(0)[0].cpu().numpy()


Tb, Tbest = cfd_field(BEFORE), cfd_field(BEST)
tv = float(max(np.abs(Tb).max(), np.abs(Tbest).max()))

fig, ax = plt.subplots(1, 3, figsize=(13, 4))
ax[0].imshow(Tb.T, cmap="RdBu_r", origin="lower", vmin=-tv, vmax=tv)
ax[0].set_title(f"baseline (CFD peak {Tb.max():.2f})"); ax[0].axis("off")
ax[1].imshow(Tbest.T, cmap="RdBu_r", origin="lower", vmin=-tv, vmax=tv)
ax[1].set_title(f"CFD-verified best (CFD peak {Tbest.max():.2f})"); ax[1].axis("off")

lo, hi = min(min(SURR), min(CFD)) - .02, max(max(SURR), max(CFD)) + .02
ax[2].plot([lo, hi], [lo, hi], "k--", alpha=0.5, label="surrogate = CFD")
ax[2].scatter(SURR, CFD, c="#268", zorder=3)
best_i = int(np.argmin(CFD))
ax[2].scatter([SURR[best_i]], [CFD[best_i]], c="#e63", s=90, zorder=4, label="verified best")
ax[2].set_xlabel("surrogate peak (FNO)"); ax[2].set_ylabel("CFD peak (ground truth)")
ax[2].set_title("surrogate over-predicts cooling\n-> verify with CFD"); ax[2].legend(); ax[2].grid(alpha=0.25)

fig.suptitle("CFD-verified cooling optimization: 10.4% hot-spot reduction (real, not surrogate-phantom)")
fig.tight_layout()
fig.savefig("../figures/verified_optimization.png", dpi=140, bbox_inches="tight")
print("saved ../figures/verified_optimization.png")
