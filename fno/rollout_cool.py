"""Autoregressive rollout of the rack-aware cooling FNO.

Seed with the first t_in true (w,T) frames + the fixed rack map, then predict
both fields forward using only the model's own outputs (rack map stays fixed).
Reports per-field error-vs-time and draws the temperature rollout.

    python rollout_cool.py --data ../data/cooling_64.npz --ckpt cool_fno.pt
"""
import argparse, json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from fno2d_cool import FNO2d


def load_model(ckpt, device):
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    c = ck["config"]
    model = FNO2d(c["t_in"], c["n_fields"], c["n_static"], c["modes"], c["modes"],
                  c["width"], c["layers"]).to(device)
    model.load_state_dict(ck["model_state"]); model.eval()
    return model, c["t_in"]


@torch.no_grad()
def rollout(model, seed, S, n_steps, device):
    window = seed.clone().to(device)                        # (N, t_in, C, H, W)
    Sc = S.to(device).unsqueeze(-1)                         # (N, H, W, 1) fixed rack map
    N, t_in, C, H, W = window.shape
    preds = []
    for _ in range(n_steps):
        x = window.permute(0, 3, 4, 1, 2).reshape(N, H, W, t_in * C)
        x = torch.cat([x, Sc], dim=-1)                      # append fixed rack channel
        pf = model(x).permute(0, 3, 1, 2)                   # (N, C, H, W)
        preds.append(pf)
        window = torch.cat([window[:, 1:], pf.unsqueeze(1)], dim=1)   # slide dynamic fields
    return torch.stack(preds, dim=1).cpu()                  # (N, n_steps, C, H, W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/cooling_64.npz")
    ap.add_argument("--ckpt", default="cool_fno.pt")
    ap.add_argument("--n_test", type=int, default=40)
    ap.add_argument("--figdir", default="../figures")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.figdir, exist_ok=True)
    model, t_in = load_model(args.ckpt, device)

    d = np.load(args.data)
    field = torch.stack([torch.from_numpy(d["w"]).float(),
                         torch.from_numpy(d["T"]).float()], dim=2)   # (N, Tf, C, H, W)
    S = torch.from_numpy(d["S"]).float()                             # (N, H, W)

    # normalization recomputed from the TRAIN split (matches training exactly)
    nt = args.n_test
    fmean = field[:-nt].mean(dim=(0, 1, 3, 4)); fstd = field[:-nt].std(dim=(0, 1, 3, 4))
    smean, sstd = S[:-nt].mean(), S[:-nt].std(); eps = 1e-8
    m = fmean.view(1, 1, -1, 1, 1); s = fstd.view(1, 1, -1, 1, 1)

    field_te = (field[-nt:] - m) / (s + eps)
    S_te = (S[-nt:] - smean) / (sstd + eps)
    seed, true_future = field_te[:, :t_in], field_te[:, t_in:]
    n_steps = field_te.shape[1] - t_in

    preds = rollout(model, seed, S_te, n_steps, device)

    pred_p = preds * (s + eps) + m
    true_p = true_future * (s + eps) + m
    N, Sn, C, H, W = pred_p.shape
    p = pred_p.reshape(N, Sn, C, -1); t = true_p.reshape(N, Sn, C, -1)
    err = (torch.norm(p - t, dim=3) / torch.norm(t, dim=3)).mean(0).numpy()   # (Sn, C)

    names = ["vorticity w", "temperature T"]
    print("Cooling rollout error (rel-L2) vs step:")
    for c in range(C):
        print(f"  {names[c]:14s}: step1 {err[0, c]:.3e}   step{Sn} {err[-1, c]:.3e}")

    fig, a = plt.subplots(figsize=(6, 4))
    for c in range(C):
        a.plot(np.arange(1, Sn + 1), err[:, c], "o-", label=names[c])
    a.set_xlabel("rollout step"); a.set_ylabel("relative L2 error")
    a.set_title("Cooling rollout: error accumulation per field")
    a.legend(); a.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{args.figdir}/cool_rollout_error.png", dpi=140); plt.close(fig)

    steps = sorted(set([0, Sn // 3, 2 * Sn // 3, Sn - 1]))
    gtT, prT = true_p[0, :, 1].numpy(), pred_p[0, :, 1].numpy()
    vmax = float(max(gtT.max(), prT.max()))
    fig, ax = plt.subplots(2, len(steps), figsize=(3 * len(steps), 6))
    for j, st in enumerate(steps):
        ax[0, j].imshow(gtT[st].T, cmap="inferno", origin="lower", vmin=0, vmax=vmax)
        ax[0, j].set_title(f"T truth  step {st+1}"); ax[0, j].axis("off")
        ax[1, j].imshow(prT[st].T, cmap="inferno", origin="lower", vmin=0, vmax=vmax)
        ax[1, j].set_title(f"T FNO   step {st+1}"); ax[1, j].axis("off")
    fig.suptitle("Cooling temperature rollout: ground truth (top) vs FNO (bottom)")
    fig.tight_layout(); fig.savefig(f"{args.figdir}/cool_rollout_T.png", dpi=140); plt.close(fig)

    json.dump({"names": names, "rollout_rel_l2_per_step": err.tolist()},
              open("cool_rollout_metrics.json", "w"), indent=2)
    print(f"\nsaved {args.figdir}/cool_rollout_error.png, {args.figdir}/cool_rollout_T.png")


if __name__ == "__main__":
    main()
