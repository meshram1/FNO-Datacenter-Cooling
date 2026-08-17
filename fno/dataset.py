"""Turn NS trajectories into (input window -> next frame) training pairs.

Raw data: w of shape (N, T, H, W). Windowing (one-step prediction):
    input  = w[n, t : t+t_in]  -> (H, W, t_in)   (t_in past frames as channels)
    target = w[n, t+t_in]      -> (H, W)         (the next frame)
"""
import numpy as np
import torch
from torch.utils.data import Dataset


class Normalizer:
    """Global scalar z-score, fit on the training set (resolution-invariant)."""

    def __init__(self, x, eps=1e-8):
        self.mean = x.mean()
        self.std = x.std()
        self.eps = eps

    def encode(self, x):
        return (x - self.mean) / (self.std + self.eps)

    def decode(self, x):
        return x * (self.std + self.eps) + self.mean

    def state_dict(self):
        return {"mean": self.mean, "std": self.std, "eps": self.eps}


class NSWindowed(Dataset):
    def __init__(self, w, t_in, normalizer):
        self.w = normalizer.encode(w)                 # (N, T, H, W), normalized once
        self.t_in = t_in
        N, T, H, W = w.shape
        self.index = [(n, t) for n in range(N) for t in range(T - t_in)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        n, t = self.index[i]
        x = self.w[n, t:t + self.t_in]                # (t_in, H, W)
        y = self.w[n, t + self.t_in]                  # (H, W)
        return x.permute(1, 2, 0), y                  # x -> (H, W, t_in), channels-last


def load_ns(path, t_in, n_test=40):
    """Split by trajectory (last n_test held out), fit normalizer on train."""
    d = np.load(path)
    w = torch.from_numpy(d["w"]).float()              # (N, T, H, W)
    w_train, w_test = w[:-n_test], w[-n_test:]
    normalizer = Normalizer(w_train)
    return (NSWindowed(w_train, t_in, normalizer),
            NSWindowed(w_test, t_in, normalizer),
            normalizer)
            
