"""Distributed training with PhysicsNeMo's DistributedManager.
    single GPU : python train_modulus.py --data ../data/aisle_64.npz
    multi  GPU : torchrun --standalone --nproc_per_node=2 train_modulus.py --data ../data/aisle_64.npz
"""
import argparse
import torch
import torch.distributed as tdist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    from physicsnemo.distributed import DistributedManager
except ImportError:
    from modulus.distributed import DistributedManager

from modulus_model import build_model
from modulus_dataset import CoolingOneStep


def rel_l2(pred, tgt):                        # relative L2, our usual metric
    p = pred.reshape(pred.shape[0], -1)
    t = tgt.reshape(tgt.shape[0], -1)
    return (torch.linalg.norm(p - t, dim=1) / (torch.linalg.norm(t, dim=1) + 1e-8)).mean()


def all_mean(val, dist):                      # average a scalar across ranks (for logging)
    t = torch.tensor([val], device=dist.device)
    if dist.distributed:
        tdist.all_reduce(t, op=tdist.ReduceOp.SUM); t /= dist.world_size
    return t.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/aisle_64.npz")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--modes", type=int, default=12)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--out", default="aisle_modulus.pt")
    args = ap.parse_args()

    DistributedManager.initialize()          # reads WORLD_SIZE/RANK/LOCAL_RANK from torchrun
    dist = DistributedManager()
    dev = dist.device

    tr = CoolingOneStep(args.data, "train")
    va = CoolingOneStep(args.data, "val", stats=tr.stats)

    tr_samp = DistributedSampler(tr, dist.world_size, dist.rank, shuffle=True) if dist.distributed else None
    tr_ld = DataLoader(tr, args.bs, sampler=tr_samp, shuffle=(tr_samp is None),
                       num_workers=4, pin_memory=True, drop_last=True)
    va_ld = DataLoader(va, args.bs, shuffle=False, num_workers=2, pin_memory=True)

    model = build_model(args.modes, args.width).to(dev)
    if dist.distributed:
        model = DDP(model, device_ids=[dist.local_rank], output_device=dev,
                    broadcast_buffers=dist.broadcast_buffers,
                    find_unused_parameters=dist.find_unused_parameters)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    for ep in range(args.epochs):
        if tr_samp: tr_samp.set_epoch(ep)     # reshuffle differently each epoch
        model.train(); run = 0.0
        for x, y in tr_ld:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            loss = rel_l2(model(x), y)
            loss.backward(); opt.step()
            run += loss.item()
        sched.step()

        model.eval(); vl = 0.0
        with torch.no_grad():
            for x, y in va_ld:
                x, y = x.to(dev), y.to(dev)
                vl += rel_l2(model(x), y).item()
        tr_l = all_mean(run / len(tr_ld), dist)
        vl_l = all_mean(vl / len(va_ld), dist)
        if dist.rank == 0:
            print(f"epoch {ep:3d}  train {tr_l:.4f}  val {vl_l:.4f}")

    if dist.rank == 0:                        # only rank 0 writes the checkpoint
        core = model.module if dist.distributed else model
        torch.save({"model_state": core.state_dict(),
                    "config": {"modes": args.modes, "width": args.width},
                    "stats": tuple(float(s) for s in tr.stats)}, args.out)
        print("saved", args.out)


if __name__ == "__main__":
    main()
