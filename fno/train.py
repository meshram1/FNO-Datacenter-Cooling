"""Train the recurrent FNO2d on one-step Navier-Stokes prediction.

    python train.py --data ../data/ns_64.npz --epochs 50
"""
import argparse
import json
import time

import torch
from torch.utils.data import DataLoader

from fno2d import FNO2d
from dataset import load_ns
from losses import LpLoss


def evaluate(model, loader, lploss, device):
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).squeeze(-1)
            tot += lploss(pred, y).item() * x.shape[0]
            n += x.shape[0]
    return tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/ns_64.npz")
    ap.add_argument("--t_in", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--modes", type=int, default=12)
    ap.add_argument("--width", type=int, default=20)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--out", default="ns_fno.pt")
    ap.add_argument("--metrics", default="ns_metrics.json")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    train_ds, test_ds, normalizer = load_ns(args.data, args.t_in)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False)

    model = FNO2d(args.t_in, args.modes, args.modes, args.width, args.layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lploss = LpLoss()

    print(f"device={device}  params={model.num_params():,}  "
          f"train_samples={len(train_ds)}  test_samples={len(test_ds)}")

    history, best, t0 = [], float("inf"), time.time()
    for ep in range(args.epochs):
        model.train()
        ep_loss, n, te = 0.0, 0, time.time()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(x).squeeze(-1)
            loss = lploss(pred, y)
            loss.backward()
            opt.step()
            ep_loss += loss.item() * x.shape[0]; n += x.shape[0]
        sched.step()
        tr = ep_loss / n
        te_rel = evaluate(model, test_loader, lploss, device)
        best = min(best, te_rel)
        history.append({"epoch": ep, "train_rel_l2": tr, "test_rel_l2": te_rel,
                        "epoch_time_s": time.time() - te})
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"ep {ep:3d}  train {tr:.4e}  test {te_rel:.4e}  ({time.time()-te:.1f}s)")

    torch.save({"model_state": model.state_dict(),
                "config": {"t_in": args.t_in, "modes": args.modes,
                           "width": args.width, "layers": args.layers},
                "normalizer": normalizer.state_dict()}, args.out)
    json.dump({"history": history, "best_test_rel_l2": best,
               "total_time_s": time.time() - t0, "args": vars(args)},
              open(args.metrics, "w"), indent=2)
    print(f"\nBest test rel-L2 = {best:.4e}   saved {args.out}")


if __name__ == "__main__":
    main()
