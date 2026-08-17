"""Relative Lp loss -- the standard FNO metric (scale-invariant error)."""
import torch


class LpLoss:
    def __init__(self, p=2):
        self.p = p

    def __call__(self, pred, true):
        B = pred.shape[0]
        diff = torch.norm(pred.reshape(B, -1) - true.reshape(B, -1), self.p, dim=1)
        base = torch.norm(true.reshape(B, -1), self.p, dim=1)
        return torch.mean(diff / base)
