#!/usr/bin/env python3
"""Train on the FIRST output frame only ("first output frame only" in the paper).

Upstream LpLoss applied to pred[:, :1] and tgt[:, :1]; output frames 2-5 receive no training signal.
Output frames 2-5 are therefore untrained, so a standard 5-frame rollout of this model is meaningless:
evaluate it one step ahead on the first frame (`rollout_diagnostics.py --teacher-forced`). Validation
keeps upstream's LpLoss over all 5 frames, so its val_loss is not comparable with the other runs.

Example:
    TRAIN_ENTRY=src/train_firstframe.py bin/train.sh model_cfg=avit_small data_cfg=poolboiling_fc72_nocond
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import bubbleformer.modules as M  # noqa: E402
from bubbleformer.utils.losses import LpLoss  # noqa: E402

print("FIRST_FRAME_LOSS: LpLoss on output frame 1 only  (training only; validation keeps all 5 frames)",
      flush=True)


def _criterion(self):
    """Created lazily rather than in a patched __init__ (see train_bandweight.py)."""
    if not hasattr(self, "_ff_criterion"):
        lp = LpLoss(d=2, p=2, reduce_dims=[0, 1, 2], reductions=["mean", "mean", "sum"])
        self._ff_criterion = lambda pred, tgt: lp(pred[:, :1], tgt[:, :1])
    return self._ff_criterion


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
