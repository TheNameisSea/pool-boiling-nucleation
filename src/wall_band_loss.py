"""Wall-band-weighted relative-L2 loss ("wall-band upweighting" in the paper).

Multiplies the squared error of the signed-distance (SDF) channel in the wall band (rows < band_rows)
by `weight` and changes nothing else:

    err2[b, t, c] = sum over pixels of  w(pixel) * (pred - tgt)^2,   w = weight in the SDF wall band, else 1
    loss          = mean_b mean_t sum_c  sqrt(err2) / ||tgt||           (same reduction as LpLoss)

With weight = 1 it reproduces Bubbleformer's LpLoss exactly (checked in `__main__`).
Tensors are (B, T, C, H, W), row 0 = the heated wall.
"""
import torch
import torch.nn as nn


class WallBandWeightedLoss(nn.Module):
    def __init__(self, weight: float = 10.0, band_rows: int = 16, sdf_channel: int = 0):
        super().__init__()
        self.weight, self.band_rows, self.sdf = weight, band_rows, sdf_channel

    def forward(self, pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        e2 = (pred - tgt) ** 2
        err2 = e2.flatten(-2).sum(-1)                                     # (B, T, C)
        if self.weight != 1:
            band = e2[:, :, self.sdf, :self.band_rows].flatten(-2).sum(-1)  # (B, T)
            extra = torch.zeros_like(err2)
            extra[:, :, self.sdf] = (self.weight - 1) * band
            err2 = err2 + extra
        norm = tgt.flatten(-2).norm(dim=-1)
        return (err2.sqrt() / norm).mean(0).mean(0).sum()


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "external", "Bubbleformer"))
    from bubbleformer.utils.losses import LpLoss
    torch.manual_seed(0)
    ref = LpLoss(d=2, p=2, reduce_dims=[0, 1, 2], reductions=["mean", "mean", "sum"])
    p, t = torch.randn(2, 5, 4, 64, 64), torch.randn(2, 5, 4, 64, 64)
    a, b = ref(p, t).item(), WallBandWeightedLoss(weight=1, band_rows=8)(p, t).item()
    print(f"weight=1 reproduces LpLoss: {a:.6f} vs {b:.6f}")
    assert abs(a - b) < 1e-5 * max(1, abs(a))
    # a missed seed in the band costs `weight`x more squared error; an error outside the band does not
    t = -torch.ones(1, 5, 4, 64, 64)
    t[:, 2:, 0, :3, 30:34] = 0.1
    miss, far = t.clone(), t.clone()
    miss[:, 2:, 0, :3, 30:34] = -1.0
    far[:, 2:, 0, 40:43, 30:34] = 0.1
    L1, L10 = WallBandWeightedLoss(weight=1, band_rows=8), WallBandWeightedLoss(weight=10, band_rows=8)
    print(f"missed seed : w=1 {L1(miss, t).item():.5f}  w=10 {L10(miss, t).item():.5f}")
    print(f"far-field   : w=1 {L1(far, t).item():.5f}  w=10 {L10(far, t).item():.5f}")
    assert abs(L10(miss, t) / L1(miss, t) - 10 ** 0.5) < 1e-4 and abs(L10(far, t) - L1(far, t)) < 1e-7
    print("OK")
