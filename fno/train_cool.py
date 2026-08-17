"""Train the rack-aware cooling FNO on one-step (w, T) prediction.

    python train_cool.py --data ../data/cooling_64.npz --epochs 50

Reports per-field relative L2 (w and T) so you can watch temperature learn.
"""
import argparse
import json
import time

import torch
from torch.utils.data import DataLoader

from fno2d_cool import FNO2d
from dataset_cool import load_cooling


def field_rel_l2(pred, true):
    B, _, _, C = pred.shape
    p = pred.permute(0, 3, 1, 2).reshape(B, C, -1)
    t = true.permute(0, 3, 1, 2).reshape(B, C, -1)
    return (torch.norm(p - t, dim=2) / torch.norm(t, dim=2)).mean(0)   # (C,)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    tot, n = None, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        w = x.shape[0]
        rel = field_rel_l2(model(x), y) * w
        tot = rel if tot is None else tot + rel
        n += w
    return (tot / n).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/cooling_64.npz")
    ap.add_argument("--t_in", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--modes", type=int, default=12)
    ap.add_argument("--width", type=int, default=20)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--out", default="cool_fno.pt")
    ap.add_argument("--metrics", default="cool_metrics.json")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    train_ds, test_ds, norm = load_cooling(args.data, args.t_in)
    C = train_ds.C
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False)

    model = FNO2d(args.t_in, C, 1, args.modes, args.modes, args.width, args.layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    print(f"device={device}  params={model.num_params():,}  C={C}  "
          f"train={len(train_ds)}  test={len(test_ds)}")

    history, best, t0 = [], float("inf"), time.time()
    for ep in range(args.epochs):
        model.train()
        run, n, te = None, 0, time.time()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            fl = field_rel_l2(model(x), y)
            loss = fl.mean()
            loss.backward()
            opt.step()
            w = x.shape[0]
            run = fl.detach() * w if run is None else run + fl.detach() * w
            n += w
        sched.step()
        tr = (run / n).tolist()
        te_err = evaluate(model, test_loader, device)
        combined = sum(te_err) / len(te_err)
        best = min(best, combined)
        history.append({"epoch": ep, "train_w": tr[0], "train_T": tr[1],
                        "test_w": te_err[0], "test_T": te_err[1]})
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"ep {ep:3d}  train[w {tr[0]:.3e}  T {tr[1]:.3e}]  "
                  f"test[w {te_err[0]:.3e}  T {te_err[1]:.3e}]  ({time.time()-te:.1f}s)")

    torch.save({"model_state": model.state_dict(),
                "config": {"t_in": args.t_in, "n_fields": C, "n_static": 1,
                           "modes": args.modes, "width": args.width, "layers": args.layers},
                "normalizer": norm.state_dict()}, args.out)
    json.dump({"history": history, "best_test_mean_rel_l2": best,
               "total_time_s": time.time() - t0, "args": vars(args)},
              open(args.metrics, "w"), indent=2)
    print(f"\nBest test mean rel-L2 = {best:.4e}   saved {args.out}")


if __name__ == "__main__":
    main()
