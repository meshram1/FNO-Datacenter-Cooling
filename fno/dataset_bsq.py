"""Multi-field dataset loader for the Boussinesq (cooling) FNO.

Raw data: boussinesq_64.npz holds two fields, w and T, each (N, Tframes, H, W).
We stack them into a 2-channel field (N, Tframes, C=2, H, W) and window it:
    input  = frames [t : t+t_in], both channels -> (H, W, t_in*C)
    target = frame   t+t_in,      both channels -> (H, W, C)

Key difference vs the NS loader: PER-CHANNEL normalization. w (~+/-4) and
T (~+/-0.6) have different scales, so each channel gets its own mean/std -- else
the larger field dominates the loss and the other barely trains.
"""
import numpy as np
import torch
from torch.utils.data import Dataset


class Normalizer:
    """Per-channel global z-score. Operates on channel-LAST tensors (..., C)."""

    def __init__(self, mean, std, eps=1e-8):
        self.mean = mean          # (C,)
        self.std = std            # (C,)
        self.eps = eps

    def encode(self, x):          # x: (..., C)
        return (x - self.mean.to(x.device)) / (self.std.to(x.device) + self.eps)

    def decode(self, x):
        return x * (self.std.to(x.device) + self.eps) + self.mean.to(x.device)

    def state_dict(self):
        return {"mean": self.mean.cpu(), "std": self.std.cpu(), "eps": self.eps}


class BoussinesqWindowed(Dataset):
    def __init__(self, field, t_in):
        self.field = field                        # (N, T, C, H, W), already normalized
        self.t_in = t_in
        N, T, C, H, W = field.shape
        self.C, self.H, self.W = C, H, W
        self.index = [(n, t) for n in range(N) for t in range(T - t_in)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        n, t = self.index[i]
        x = self.field[n, t:t + self.t_in]                       # (t_in, C, H, W)
        y = self.field[n, t + self.t_in]                        # (C, H, W)
        x = x.permute(2, 3, 0, 1).reshape(self.H, self.W, -1)   # (H, W, t_in*C)
        y = y.permute(1, 2, 0)                                  # (H, W, C)
        return x, y


def load_boussinesq(path, t_in, n_test=40):
    d = np.load(path)
    w = torch.from_numpy(d["w"]).float()          # (N, T, H, W)
    T = torch.from_numpy(d["T"]).float()          # (N, T, H, W)
    field = torch.stack([w, T], dim=2)            # (N, T, C=2, H, W)  ch0=w, ch1=T
    field_train, field_test = field[:-n_test], field[-n_test:]

    # per-channel stats from TRAIN only -> (C,)
    mean = field_train.mean(dim=(0, 1, 3, 4))
    std = field_train.std(dim=(0, 1, 3, 4))
    norm = Normalizer(mean, std)

    def enc(f):                                   # broadcast (C,) over (N,T,C,H,W)
        return (f - mean[None, None, :, None, None]) / (std[None, None, :, None, None] + norm.eps)

    return (BoussinesqWindowed(enc(field_train), t_in),
            BoussinesqWindowed(enc(field_test), t_in),
            norm)
