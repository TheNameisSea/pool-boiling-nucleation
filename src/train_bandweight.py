#!/usr/bin/env python3
"""Train with the wall-band-weighted loss (`wall_band_loss.py`), without editing upstream Bubbleformer.

Replaces the TRAINING criterion of Bubbleformer's ForecastModule / ConditionedForecastModule, then runs
upstream `scripts/train.py` as __main__. Validation keeps upstream's LpLoss, so `val_loss` is
comparable across all runs.

Configured by environment variables (Hydra rejects unknown keys):
    BAND_WEIGHT     multiplier on the SDF wall band's squared error   (default 10)
    BAND_ROWS       wall-band height in cells                          (default 16 = 0.5 domain units)

Example (see bin/train.sh):
    TRAIN_ENTRY=src/train_bandweight.py BAND_WEIGHT=10 bin/train.sh model_cfg=avit_small data_cfg=poolboiling_fc72_nocond
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import bubbleformer.modules as M  # noqa: E402
from wall_band_loss import WallBandWeightedLoss  # noqa: E402

CFG = dict(weight=float(os.getenv("BAND_WEIGHT", "10")), band_rows=int(os.getenv("BAND_ROWS", "16")))
print(f"WALL_BAND_WEIGHTED_LOSS: {CFG}  (training only; validation keeps upstream LpLoss)", flush=True)


def _criterion(self):
    """Created lazily rather than in a patched __init__: Lightning's save_hyperparameters inspects
    the __init__ call frames and fails on a (*args, **kwargs) wrapper."""
    if not hasattr(self, "_band_criterion"):
        self._band_criterion = WallBandWeightedLoss(**CFG)
    return self._band_criterion


def _log_and_return(self, loss):
    """Upstream training_step's logging, verbatim in effect (modules.py:90-111)."""
    self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    current_lr = self.optimizers().param_groups[0]["lr"]
    self.log("learning_rate", current_lr, on_step=True, on_epoch=False, prog_bar=True, logger=True)
    if self.log_wandb and self.trainer.is_global_zero:
        import wandb
        wandb.log({"train_loss": loss, "learning_rate": current_lr})
    return loss


def _step_uncond(self, batch, batch_idx):
    inp, tgt = batch
    return _log_and_return(self, _criterion(self)(self.model(inp), tgt))


def _step_cond(self, batch, batch_idx):
    inp, tgt, cond = batch
    return _log_and_return(self, _criterion(self)(self.model(inp, cond), tgt))


M.ForecastModule.training_step = _step_uncond
M.ConditionedForecastModule.training_step = _step_cond

if __name__ == "__main__":
    import runpy
    runpy.run_path(os.path.join(HERE, "..", "external", "Bubbleformer", "scripts", "train.py"),
                   run_name="__main__")
