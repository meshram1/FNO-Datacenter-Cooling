"""Windowed (t_in frames -> next frame) pairs from aisle_64.npz, channels-first,
per-field normalized. Stats come from the train split and are reused for val."""
import numpy as np
import torch
from torch.utils.data import Dataset

from modulus_model import T_IN


class CoolingOneStep(Dataset):
    def __init__(self, path, split="train", train_frac=0.9, stats=None):
        d = np.load(path)
        w = torch.from_numpy(d["w"]).float()     # (N, Nt, H, W)
        T = torch.from_numpy(d["T"]).float()
        S = torch.from_numpy(d["S"]).float()     # (N, H, W)
        N = w.shape[0]; ntr = int(N * train_frac)
        sl = slice(0, ntr) if split == "train" else slice(ntr, N)

        self.stats = stats or (w[:ntr].mean(), w[:ntr].std(),
                               T[:ntr].mean(), T[:ntr].std(),
                               S[:ntr].mean(), S[:ntr].std())
        self.w, self.T, self.S = w[sl], T[sl], S[sl]
        self.per = self.w.shape[1] - T_IN        # windows per trajectory

    def __len__(self):
        return self.w.shape[0] * self.per

    def __getitem__(self, i):
        wm, ws, Tm, Ts, Sm, Ss = self.stats
        traj, t = divmod(i, self.per)
        wn = (self.w[traj] - wm) / ws
        Tn = (self.T[traj] - Tm) / Ts
        Sn = (self.S[traj] - Sm) / Ss
        hist = [f for k in range(t, t + T_IN) for f in (wn[k], Tn[k])]  # time-major, field-minor
        x = torch.stack(hist + [Sn], 0)                        # (21, H, W)
        y = torch.stack([wn[t + T_IN], Tn[t + T_IN]], 0)       # (2, H, W)
        return x, y
