"""Rack-aware dataset loader for the data-center cooling FNO.

cooling_64.npz holds w, T (each (N,Tf,H,W)) and S (N,H,W) = the static rack map.
We stack (w,T) into a 2-channel dynamic field and window it, then attach the
NORMALIZED rack map S as an extra static input channel -- so the model learns
    (rack layout S, recent (w,T) history)  ->  next (w,T).

    input  = [t_in frames of (w,T)]  ++  [rack map S]   -> (H, W, t_in*C + 1)
    target = next (w,T)                                 -> (H, W, C)
"""
import numpy as np
import torch
from torch.utils.data import Dataset


class Normalizer:
    """Per-channel z-score for the dynamic fields (used to decode outputs)."""

    def __init__(self, mean, std, eps=1e-8):
        self.mean, self.std, self.eps = mean, std, eps

    def encode(self, x):
        return (x - self.mean.to(x.device)) / (self.std.to(x.device) + self.eps)

    def decode(self, x):
        return x * (self.std.to(x.device) + self.eps) + self.mean.to(x.device)

    def state_dict(self):
        return {"mean": self.mean.cpu(), "std": self.std.cpu(), "eps": self.eps}


class CoolingWindowed(Dataset):
    def __init__(self, field, S, t_in):
        self.field = field          # (N, T, C, H, W) normalized dynamic fields
        self.S = S                  # (N, H, W) normalized rack map
        self.t_in = t_in
        N, T, C, H, W = field.shape
        self.C, self.H, self.W = C, H, W
        self.index = [(n, t) for n in range(N) for t in range(T - t_in)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        n, t = self.index[i]
        x = self.field[n, t:t + self.t_in]                     # (t_in, C, H, W)
        y = self.field[n, t + self.t_in]                       # (C, H, W)
        x = x.permute(2, 3, 0, 1).reshape(self.H, self.W, -1)  # (H, W, t_in*C)
        s = self.S[n].unsqueeze(-1)                            # (H, W, 1) rack map
        x = torch.cat([x, s], dim=-1)                          # (H, W, t_in*C + 1)
        y = y.permute(1, 2, 0)                                 # (H, W, C)
        return x, y


def load_cooling(path, t_in, n_test=40):
    d = np.load(path)
    w = torch.from_numpy(d["w"]).float()
    T = torch.from_numpy(d["T"]).float()
    S = torch.from_numpy(d["S"]).float()          # (N, H, W)
    field = torch.stack([w, T], dim=2)            # (N, T, C=2, H, W)
    field_tr, field_te = field[:-n_test], field[-n_test:]
    S_tr, S_te = S[:-n_test], S[-n_test:]

    mean = field_tr.mean(dim=(0, 1, 3, 4))        # per-channel (C,)
    std = field_tr.std(dim=(0, 1, 3, 4))
    s_mean, s_std = S_tr.mean(), S_tr.std()       # rack-map scalar stats
    norm = Normalizer(mean, std)

    def encf(f):
        return (f - mean[None, None, :, None, None]) / (std[None, None, :, None, None] + norm.eps)

    def encs(s):
        return (s - s_mean) / (s_std + norm.eps)

    return (CoolingWindowed(encf(field_tr), encs(S_tr), t_in),
            CoolingWindowed(encf(field_te), encs(S_te), t_in),
            norm)
