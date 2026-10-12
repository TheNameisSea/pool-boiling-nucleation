"""Wall-band phase-indicator loss: the paper's loss term (Eq. 3), optionally with wall-band upweighting.

Adds a binary cross-entropy on the phase indicator implied by the SDF, in the wall band:

    s = sigmoid(phi_pred / eps),   y = 1[phi_tgt > 0],   rows < band_rows of the SDF channel
    loss = WallBandWeightedLoss(pred, tgt) + ind_weight * mean_{b,t} mean_band BCE(s, y)

The defaults (band_weight = 1, ind_weight = 1) are Eq. 3 of the paper: the plain relative-L2 term plus the
indicator term. band_weight > 1 adds wall-band upweighting (the "+ upweighting" runs, e.g. 1055: 10).
eps is one cell (1/32 domain units; the configs use `normalize: none`, so predictions are in physical
units). The BCE is computed on logits for numerical stability.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from wall_band_loss import WallBandWeightedLoss


class IndicatorBandLoss(nn.Module):
    def __init__(self, band_weight: float = 1.0, band_rows: int = 16, ind_weight: float = 1.0,
                 eps: float = 1.0 / 32, sdf_channel: int = 0):
        super().__init__()
        self.base = WallBandWeightedLoss(weight=band_weight, band_rows=band_rows, sdf_channel=sdf_channel)
        self.band_rows, self.ind_weight, self.eps, self.sdf = band_rows, ind_weight, eps, sdf_channel

    def indicator(self, pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        logit = pred[:, :, self.sdf, :self.band_rows] / self.eps
        y = (tgt[:, :, self.sdf, :self.band_rows] > 0).to(pred.dtype)
        return F.binary_cross_entropy_with_logits(logit, y, reduction="none").flatten(-2).mean(-1).mean()

    def forward(self, pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        loss = self.base(pred, tgt)
        return loss + self.ind_weight * self.indicator(pred, tgt) if self.ind_weight else loss


if __name__ == "__main__":
    torch.manual_seed(0)
    p, t = torch.randn(2, 5, 4, 64, 64), torch.randn(2, 5, 4, 64, 64)
    a = IndicatorBandLoss(band_weight=10, band_rows=8, ind_weight=0)(p, t).item()
    b = WallBandWeightedLoss(weight=10, band_rows=8)(p, t).item()
    print(f"ind_weight=0 reproduces the wall-band-weighted loss: {a:.6f} vs {b:.6f}"); assert abs(a - b) < 1e-6
    # the defaults are Eq. 3: unweighted relative L2 (LpLoss, see wall_band_loss.py) plus the indicator at weight 1
    d = IndicatorBandLoss(band_rows=8)
    assert d.base.weight == 1 and d.ind_weight == 1
    assert abs(d(p, t).item() - (WallBandWeightedLoss(weight=1, band_rows=8)(p, t) + d.indicator(p, t)).item()) < 1e-6
    print("defaults = Eq. 3 (band_weight 1, ind_weight 1)")
    # check: with an uncertain target (probability p of vapour), the indicator term's optimum has phi > 0 iff p > 0.5
    L = IndicatorBandLoss(band_weight=1, band_rows=8, ind_weight=1)
    seed, noseed = torch.full((1, 1, 4, 8, 8), 0.06), torch.full((1, 1, 4, 8, 8), -0.5)
    for prob in (0.3, 0.7, 0.9):
        phi = torch.zeros(1, 1, 4, 8, 8, requires_grad=True)
        opt = torch.optim.Adam([phi], lr=1e-2)
        for _ in range(3000):
            opt.zero_grad()
            (prob * L.indicator(phi, seed) + (1 - prob) * L.indicator(phi, noseed)).backward(); opt.step()
        l2 = prob * 0.06 + (1 - prob) * -0.5
        print(f"p = {prob}: indicator optimum phi = {phi[0, 0, 0, 0, 0].item():+.4f} (L2 optimum {l2:+.3f})")
        assert (phi[0, 0, 0, 0, 0].item() > 0) == (prob > 0.5)
    print("OK")
