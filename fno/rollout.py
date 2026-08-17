import argparse, json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from fno2d import FNO2d


def load_model(ckpt, device):
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    c = ck["config"]
    model = FNO2d(c["t_in"], c["modes"], c["modes"], c["width"], c["layers"]).to(device)
    model.load_state_dict(ck["model_state"]); model.eval()
    sd = ck["normalizer"]
    return model, (float(sd["mean"]), float(sd["std"]), float(sd["eps"])), c["t_in"]


@torch.no_grad()
def rollout(model, seed, n_steps, device):
    window = seed.clone().to(device)                         # (N, t_in, H, W)
    preds = []
    for _ in range(n_steps):
        x = window.permute(0, 2, 3, 1)                       # (N, H, W, t_in)
        p = model(x).squeeze(-1)                             # (N, H, W) its own prediction
        preds.append(p)
        window = torch.cat([window[:, 1:], p.unsqueeze(1)], dim=1)   # drop oldest, append
    return torch.stack(preds, dim=1).cpu()                   # (N, n_steps, H, W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/ns_64.npz")
    ap.add_argument("--ckpt", default="ns_fno.pt")
    ap.add_argument("--n_test", type=int, default=40)
    ap.add_argument("--figdir", default="../figures")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.figdir, exist_ok=True)
    model, (mean, std, eps), t_in = load_model(args.ckpt, device)

    w = torch.from_numpy(np.load(args.data)["w"]).float()
    w_test = w[-args.n_test:]
    n_steps = w_test.shape[1] - t_in
    w_norm = (w_test - mean) / (std + eps)
    seed, true_future = w_norm[:, :t_in], w_norm[:, t_in:]

    preds = rollout(model, seed, n_steps, device)
    pred_p = preds * (std + eps) + mean
    true_p = true_future * (std + eps) + mean
    N = pred_p.shape[0]
    err = (torch.norm((pred_p - true_p).reshape(N, n_steps, -1), dim=2)
           / torch.norm(true_p.reshape(N, n_steps, -1), dim=2))
    err_curve = err.mean(0).numpy()

    print("Autoregressive rollout error (rel-L2), averaged over test trajectories:")
    print(f"  step  1 (one-step): {err_curve[0]:.4e}")
    for k in (4, 9, 14, 19):
        if k < n_steps: print(f"  step {k+1:2d}          : {err_curve[k]:.4e}")
    print(f"  step {n_steps:2d} (final)   : {err_curve[-1]:.4e}")

    fig, a = plt.subplots(figsize=(6, 4))
    a.plot(np.arange(1, n_steps + 1), err_curve, "o-")
    a.set_xlabel("rollout step"); a.set_ylabel("relative L2 error")
    a.set_title("Autoregressive rollout: error accumulation"); a.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{args.figdir}/rollout_error.png", dpi=140); plt.close(fig)

    steps = sorted(set([0, n_steps // 3, 2 * n_steps // 3, n_steps - 1]))
    gt, pr = true_p[0].numpy(), pred_p[0].numpy()
    vmax = float(np.abs(gt).max())
    fig, ax = plt.subplots(2, len(steps), figsize=(3 * len(steps), 6))
    for j, s in enumerate(steps):
        ax[0, j].imshow(gt[s], cmap="RdBu_r", vmin=-vmax, vmax=vmax); ax[0, j].set_title(f"truth  step {s+1}"); ax[0, j].axis("off")
        ax[1, j].imshow(pr[s], cmap="RdBu_r", vmin=-vmax, vmax=vmax); ax[1, j].set_title(f"FNO   step {s+1}"); ax[1, j].axis("off")
    fig.suptitle("Rollout: ground truth (top) vs FNO prediction (bottom)")
    fig.tight_layout(); fig.savefig(f"{args.figdir}/rollout_compare.png", dpi=140); plt.close(fig)

    json.dump({"rollout_rel_l2_per_step": err_curve.tolist(),
               "one_step": float(err_curve[0]), "final": float(err_curve[-1])},
              open("rollout_metrics.json", "w"), indent=2)
    print(f"\nsaved {args.figdir}/rollout_error.png, {args.figdir}/rollout_compare.png, rollout_metrics.json")


if __name__ == "__main__":
    main()
    
