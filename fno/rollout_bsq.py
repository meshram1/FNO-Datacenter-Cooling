"""Autoregressive rollout of the multi-field Boussinesq FNO.

Seed with the first t_in true (w, T) frames, then predict both fields forward
using only the model's own outputs. Reports error-vs-time PER FIELD and draws
the temperature rollout (the field that matters for cooling).

    python rollout_bsq.py --data ../data/boussinesq_64.npz --ckpt bsq_fno.pt
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from fno2d_bsq import FNO2d


def load_model(ckpt, device):
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    c = ck["config"]
    model = FNO2d(c["t_in"], c["n_fields"], c["modes"], c["modes"], c["width"], c["layers"]).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    nd = ck["normalizer"]
    return model, (nd["mean"].float().cpu(), nd["std"].float().cpu(), float(nd["eps"])), c["t_in"]


@torch.no_grad()
def rollout(model, seed, n_steps, device):
    window = seed.clone().to(device)                        # (N, t_in, C, H, W)
    N, t_in, C, H, W = window.shape
    preds = []
    for _ in range(n_steps):
        x = window.permute(0, 3, 4, 1, 2).reshape(N, H, W, t_in * C)   # (N,H,W,t_in*C)
        pf = model(x).permute(0, 3, 1, 2)                   # (N, C, H, W) its own prediction
        preds.append(pf)
        window = torch.cat([window[:, 1:], pf.unsqueeze(1)], dim=1)    # drop oldest, append
    return torch.stack(preds, dim=1).cpu()                  # (N, n_steps, C, H, W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/boussinesq_64.npz")
    ap.add_argument("--ckpt", default="bsq_fno.pt")
    ap.add_argument("--n_test", type=int, default=40)
    ap.add_argument("--figdir", default="../figures")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.figdir, exist_ok=True)
    model, (mean, std, eps), t_in = load_model(args.ckpt, device)

    d = np.load(args.data)
    field = torch.stack([torch.from_numpy(d["w"]).float(),
                         torch.from_numpy(d["T"]).float()], dim=2)     # (N, Tf, C, H, W)
    field = field[-args.n_test:]
    n_steps = field.shape[1] - t_in

    m = mean.view(1, 1, -1, 1, 1); s = std.view(1, 1, -1, 1, 1)
    field_n = (field - m) / (s + eps)
    seed, true_future = field_n[:, :t_in], field_n[:, t_in:]

    preds = rollout(model, seed, n_steps, device)           # (N, steps, C, H, W)

    pred_p = preds * (s + eps) + m
    true_p = true_future * (s + eps) + m
    N, S, C, H, W = pred_p.shape
    p = pred_p.reshape(N, S, C, -1); t = true_p.reshape(N, S, C, -1)
    err = (torch.norm(p - t, dim=3) / torch.norm(t, dim=3)).mean(0).numpy()   # (S, C)

    names = ["vorticity w", "temperature T"]
    print("Autoregressive rollout error (rel-L2) vs step:")
    for c in range(C):
        print(f"  {names[c]:14s}: step1 {err[0, c]:.3e}   step{S} {err[-1, c]:.3e}")

    # figure 1: error vs step, per field
    fig, a = plt.subplots(figsize=(6, 4))
    for c in range(C):
        a.plot(np.arange(1, S + 1), err[:, c], "o-", label=names[c])
    a.set_xlabel("rollout step"); a.set_ylabel("relative L2 error")
    a.set_title("Boussinesq rollout: error accumulation per field")
    a.legend(); a.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{args.figdir}/bsq_rollout_error.png", dpi=140); plt.close(fig)

    # figure 2: temperature rollout, GT vs prediction (trajectory 0)
    steps = sorted(set([0, S // 3, 2 * S // 3, S - 1]))
    gtT, prT = true_p[0, :, 1].numpy(), pred_p[0, :, 1].numpy()
    fig, ax = plt.subplots(2, len(steps), figsize=(3 * len(steps), 6))
    for j, st in enumerate(steps):
        ax[0, j].imshow(gtT[st].T, cmap="inferno", origin="lower")
        ax[0, j].set_title(f"T truth  step {st+1}"); ax[0, j].axis("off")
        ax[1, j].imshow(prT[st].T, cmap="inferno", origin="lower")
        ax[1, j].set_title(f"T FNO   step {st+1}"); ax[1, j].axis("off")
    fig.suptitle("Temperature rollout: ground truth (top) vs FNO (bottom)")
    fig.tight_layout(); fig.savefig(f"{args.figdir}/bsq_rollout_T.png", dpi=140); plt.close(fig)

    json.dump({"names": names, "rollout_rel_l2_per_step": err.tolist()},
              open("bsq_rollout_metrics.json", "w"), indent=2)
    print(f"\nsaved {args.figdir}/bsq_rollout_error.png, {args.figdir}/bsq_rollout_T.png")


if __name__ == "__main__":
    main()
