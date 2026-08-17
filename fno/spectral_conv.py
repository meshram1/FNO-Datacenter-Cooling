import torch
import torch.nn as nn


class SpectralConv2d(nn.Module):
    """The core FNO layer: a global convolution done as multiplication in Fourier space.

    Only the lowest modes1 x modes2 frequencies are kept and mixed with learned
    complex weights; the rest are dropped. Few modes -> smooth global mixing, and
    -- because it acts on frequencies, not grid cells -- it works at ANY resolution.
    """

    def __init__(self, in_ch, out_ch, modes1, modes2):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.modes1 = modes1                 # kept Fourier modes along height
        self.modes2 = modes2                 # kept Fourier modes along width
        scale = 1.0 / (in_ch * out_ch)
        # two learned complex weight blocks: the low +freq and low -freq corners
        self.w1 = nn.Parameter(scale * torch.rand(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat))
        self.w2 = nn.Parameter(scale * torch.rand(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat))

    @staticmethod
    def _mul(x, w):
        # (B,in,H,W) x (in,out,H,W) -> (B,out,H,W): complex matmul over channels, per mode
        return torch.einsum("bixy,ioxy->boxy", x, w)

    def forward(self, x):
        B, _, H, W = x.shape
        # 1) to Fourier space (rfft2: width axis -> W//2+1 complex coeffs)
        x_ft = torch.fft.rfft2(x)
        # 2) empty output spectrum
        out_ft = torch.zeros(B, self.out_ch, H, W // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        # 3) mix ONLY the low modes with the learned weights (the two corners)
        out_ft[:, :, :self.modes1, :self.modes2] = self._mul(
            x_ft[:, :, :self.modes1, :self.modes2], self.w1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self._mul(
            x_ft[:, :, -self.modes1:, :self.modes2], self.w2)
        # 4) back to grid space
        return torch.fft.irfft2(out_ft, s=(H, W))
