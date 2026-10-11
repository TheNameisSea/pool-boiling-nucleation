"""Local wall-feature path for the AViT.

Reads the bottom `patch_size` rows (one token row) of all T input frames at full resolution, and ADDS
a per-frame correction to the wall-row tokens right after the patch embedding:

    band (B, T*C, p, W) -> 3x conv3x3 (64) -> conv p x p, stride p (128) -> 1x1 conv (T*E), ZERO-INIT
                        -> (B, T, E, 1, W/p) added to tokens[:, :, :, 0:1, :]

The last layer starts at zero, so an untrained path leaves the model exactly as upstream (checked in
`__main__`). About 2.4 M parameters.

`install()` patches the upstream AViT class in place (upstream code is not edited). The training
wrapper (`train_wallpath.py`) and `scripts/inference.build_model` (for checkpoints holding
`wall_path.*` keys) both call it before the model is built.
"""
import torch
import torch.nn as nn
from einops import rearrange


class WallPath(nn.Module):
    def __init__(self, time_window: int, in_fields: int, embed_dim: int, patch_size: int, width: int = 64,
                 hidden: int = 128):
        super().__init__()
        self.t, self.e, self.p = time_window, embed_dim, patch_size
        cin = time_window * in_fields
        self.conv = nn.Sequential(
            nn.Conv2d(cin, width, 3, padding=1), nn.GroupNorm(8, width), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, hidden, patch_size, stride=patch_size), nn.GELU(),
        )
        self.out = nn.Conv2d(hidden, time_window * embed_dim, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x (B, T, C, H, W), row 0 = the heated wall -> (B, T, E, 1, W // p)."""
        band = rearrange(x[:, :, :, :self.p, :], "b t c h w -> b (t c) h w")
        y = self.out(self.conv(band))                                   # (B, T*E, 1, W/p)
        return rearrange(y, "b (t e) h w -> b t e h w", t=self.t)


def install():
    """Patch bubbleformer's AViT so every instance built afterwards carries a WallPath."""
    import torch.nn.functional as F
    from bubbleformer.models import axial_vit as A

    if getattr(A.AViT, "_wall_path_installed", False):
        return
    orig_init, orig_forward = A.AViT.__init__, A.AViT.forward

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        embed_dim = self.embed.embed_dim
        self.wall_path = WallPath(time_window=kwargs.get("time_window", 12),
                                  in_fields=kwargs.get("input_fields", 3),
                                  embed_dim=embed_dim, patch_size=self.patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upstream AViT.forward (axial_vit.py) with one addition after the embedding."""
        if not hasattr(self, "wall_path"):                              # built before install()
            return orig_forward(self, x)
        b, t, c, H, W = x.shape
        p = self.patch_size
        assert H % p == 0 and W % p == 0, "wall path assumes H, W divisible by the patch size"
        wall = self.wall_path(x)                                        # (B, T, E, 1, W/p)
        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = self.embed(x)
        x = rearrange(x, "(b t) c h w -> b t c h w", t=t)
        x = torch.cat([x[:, :, :, :1] + wall, x[:, :, :, 1:]], dim=3)   # wall token row += path
        for blk in self.blocks:
            x = blk(x)
        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = self.debed(x)
        return rearrange(x, "(b t) c h w -> b t c h w", t=t)

    A.AViT.__init__ = __init__
    A.AViT.forward = forward
    A.AViT._wall_path_installed = True
    print("WALL_PATH: installed on AViT (zero-init; wall token row += local 5-frame band features)", flush=True)


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "external", "Bubbleformer"))
    from bubbleformer.models import get_model
    torch.manual_seed(0)
    kw = dict(input_fields=4, output_fields=4, time_window=5, patch_size=16, embed_dim=384, num_heads=6,
              processor_blocks=12)
    base = get_model("avit", **kw).eval()
    install()
    mod = get_model("avit", **kw).eval()
    missing, unexpected = mod.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected and all(k.startswith("wall_path.") for k in missing), (missing, unexpected)
    x = torch.randn(1, 5, 4, 128, 128)
    with torch.no_grad():
        d = (base(x) - mod(x)).abs().max().item()
    n = sum(p.numel() for p in mod.wall_path.parameters())
    print(f"zero-init path reproduces upstream: max|diff| = {d:.2e}; wall path params = {n / 1e6:.2f} M")
    assert d < 1e-5
    mod.train(); mod(x).square().mean().backward()
    g = mod.wall_path.out.weight.grad.abs().sum().item()
    print(f"gradient reaches the zero-init output layer: {g:.3e}")
    assert g > 0
    print("OK")
