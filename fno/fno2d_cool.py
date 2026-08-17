"""FNO2d for rack-aware data-center cooling prediction.

Input : t_in frames of C dynamic fields + n_static conditioning channels (the
        rack map)                         -> (B, H, W, t_in*C + n_static).
Output: next frame of C dynamic fields    -> (B, H, W, C).

Same as the Boussinesq FNO but fc0 also takes the n_static rack channel(s).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from spectral_conv import SpectralConv2d


class FNO2d(nn.Module):
    def __init__(self, t_in, n_fields=2, n_static=1, modes1=12, modes2=12, width=20, n_layers=4):
        super().__init__()
        self.width = width
        self.n_fields = n_fields
        self.n_static = n_static
        self.fc0 = nn.Linear(t_in * n_fields + n_static + 2, width)   # + rack + (x,y)
        self.spectral = nn.ModuleList(
            [SpectralConv2d(width, width, modes1, modes2) for _ in range(n_layers)])
        self.pointwise = nn.ModuleList(
            [nn.Conv2d(width, width, 1) for _ in range(n_layers)])
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, n_fields)

    @staticmethod
    def get_grid(shape, device):
        B, H, W = shape
        gx = torch.linspace(0, 1, H, device=device).reshape(1, H, 1, 1).expand(B, H, W, 1)
        gy = torch.linspace(0, 1, W, device=device).reshape(1, 1, W, 1).expand(B, H, W, 1)
        return torch.cat((gx, gy), dim=-1)

    def forward(self, x):
        B, H, W, _ = x.shape
        grid = self.get_grid((B, H, W), x.device)
        x = torch.cat((x, grid), dim=-1)                   # (B, H, W, t_in*C + n_static + 2)
        x = self.fc0(x).permute(0, 3, 1, 2)                # (B, width, H, W)
        for spec, pw in zip(self.spectral, self.pointwise):
            x = F.gelu(spec(x) + pw(x))
        x = x.permute(0, 2, 3, 1)                          # (B, H, W, width)
        x = F.gelu(self.fc1(x))
        return self.fc2(x)                                 # (B, H, W, C)

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
