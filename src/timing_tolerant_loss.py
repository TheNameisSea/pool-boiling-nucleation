"""Timing-tolerant relative-L2 loss ("timing-tolerant loss" in the paper).

    everywhere except the wall band of the SDF channel -> identical to Bubbleformer's LpLoss
    wall band (rows < band_rows) of the SDF channel    -> split into column tiles; each tile's
        squared error is the MINIMUM over matching the prediction at output frame t to the target
        at t-s, t, or t+s (|s| <= shift, within the output window)

A seed predicted one frame early or late therefore costs ~0. The minimum is taken per column tile,
not per pixel, so the prediction cannot assemble a field from different frames cell by cell.

Reduction is identical to `LpLoss(d=2, p=2, reduce_dims=[0,1,2], reductions=[mean, mean, sum])`:
per (batch, frame, field) relative L2 over H x W, mean over batch, mean over frames, sum over
fields. With shift = 0 it reproduces LpLoss exactly (checked in `__main__`).
Tensors are (B, T, C, H, W), row 0 = the heated wall.
"""
import torch
import torch.nn as nn


class TimingTolerantLoss(nn.Module):
    def __init__(self, shift: int = 1, band_rows: int = 16, tile: int = 16, sdf_channel: int = 0):
        super().__init__()
        self.shift, self.band_rows, self.tile, self.sdf = shift, band_rows, tile, sdf_channel

    def forward(self, pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = tgt.shape
        err2 = ((pred - tgt) ** 2).flatten(-2).sum(-1)                  # (B, T, C)
        if self.shift > 0:
            R, K = self.band_rows, self.tile
            nt = W // K
            pb = pred[:, :, self.sdf, :R, :nt * K].reshape(B, T, R, nt, K)
            tb = tgt[:, :, self.sdf, :R, :nt * K].reshape(B, T, R, nt, K)
            cands = []
            for s in range(-self.shift, self.shift + 1):
                e = torch.full((B, T, nt), float("inf"), device=pred.device, dtype=pred.dtype)
                lo, hi = max(0, -s), min(T, T - s)
                if hi > lo:
                    e[:, lo:hi] = ((pb[:, lo:hi] - tb[:, lo + s:hi + s]) ** 2).sum(dim=(2, 4))
                cands.append(e)
            band_tol = torch.stack(cands).min(dim=0).values.sum(-1)     # (B, T)
            band_exact = ((pb - tb) ** 2).sum(dim=(2, 4)).sum(-1)        # (B, T)
            sdf_err2 = err2[:, :, self.sdf] - band_exact + band_tol
            err2 = torch.cat([err2[:, :, :self.sdf], sdf_err2.unsqueeze(-1), err2[:, :, self.sdf + 1:]], dim=-1)
        norm = tgt.flatten(-2).norm(dim=-1)                              # (B, T, C)
        rel = err2.clamp_min(0).sqrt() / norm
        return rel.mean(0).mean(0).sum()


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "external", "Bubbleformer"))
    from bubbleformer.utils.losses import LpLoss
    torch.manual_seed(0)
    ref = LpLoss(d=2, p=2, reduce_dims=[0, 1, 2], reductions=["mean", "mean", "sum"])
    p, t = torch.randn(2, 5, 4, 64, 64), torch.randn(2, 5, 4, 64, 64)
    a, b = ref(p, t).item(), TimingTolerantLoss(shift=0, band_rows=8, tile=16)(p, t).item()
    print(f"shift=0 reproduces LpLoss: {a:.6f} vs {b:.6f}")
    assert abs(a - b) < 1e-5 * max(1, abs(a))

    # a seed at the wall: true at frames 2-4, predicted one frame early (1-4) vs half-strength 2-4
    t = -torch.ones(1, 5, 4, 64, 64)
    t[:, 2:, 0, :3, 30:34] = 0.1
    early, half = t.clone(), t.clone()
    early[:, 1, 0, :3, 30:34] = 0.1
    half[:, 2:, 0, :3, 30:34] = -0.45
    L0, L1 = TimingTolerantLoss(shift=0, band_rows=8, tile=16), TimingTolerantLoss(shift=1, band_rows=8, tile=16)
    print(f"early-by-one seed : strict {L0(early, t).item():.5f}  tolerant {L1(early, t).item():.5f}")
    print(f"half-strength seed: strict {L0(half, t).item():.5f}  tolerant {L1(half, t).item():.5f}")
    assert L1(early, t) < 1e-6 < L1(half, t), "tolerant loss must forgive timing, not hedging"
    print("OK")
