#!/usr/bin/env python3
"""The phase-indicator loss + pushforward training ("phase-indicator + pushforward" in the paper).

Pushforward (Brandstetter, Worrall & Welling 2022): on every PF_EVERY-th training step the model first
predicts the next window from the TRUE input without gradients, then predicts again from its OWN
prediction, and only that second step is trained, against the true window after next. Other steps
are the ordinary one-step loss. One gradient graph per step, so memory use is unchanged.

Loss on every step: `indicator_loss.IndicatorBandLoss`, configured as in `train_indicator.py`
(BAND_WEIGHT default 1 here, IND_WEIGHT, IND_EPS, BAND_ROWS), plus PF_EVERY (default 2).
The TRAINING dataset (the first BubbleForecast that upstream train.py builds) additionally returns the
window after the target; windows whose next window would run past the trajectory skip the pushforward
term. Validation is untouched (one step, upstream LpLoss). Unconditioned models only.

Example:
    TRAIN_ENTRY=src/train_pushforward.py PF_EVERY=2 bin/train.sh model_cfg=avit_small data_cfg=poolboiling_fc72_nocond
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import bubbleformer.modules as M  # noqa: E402
from bubbleformer.data import dataset as DS  # noqa: E402
from indicator_loss import IndicatorBandLoss  # noqa: E402

CFG = dict(band_weight=float(os.getenv("BAND_WEIGHT", "1")), band_rows=int(os.getenv("BAND_ROWS", "16")),
           ind_weight=float(os.getenv("IND_WEIGHT", "1")), eps=float(os.getenv("IND_EPS", str(1 / 32))))
PF_EVERY = int(os.getenv("PF_EVERY", "2"))
print(f"PUSHFORWARD: every {PF_EVERY} steps; loss INDICATOR_BAND_LOSS {CFG} (training only; validation "
      f"keeps upstream one-step LpLoss)", flush=True)

# --- the training dataset also returns the window after the target -------------------------------------
_orig_init, _orig_getitem = DS.BubbleForecast.__init__, DS.BubbleForecast.__getitem__
_instances = [0]


def _init(self, *args, **kwargs):
    _orig_init(self, *args, **kwargs)
    self._pushforward = _instances[0] == 0          # upstream train.py builds train first, then val
    _instances[0] += 1


def _next_window(self, idx):
    """Raw frames [start + 2T, start + 3T) of sample idx, normalised as upstream; (window, valid)."""
    T = self.time_window
    spt = [x * (y - self.start_time - 2 * T + 1) for x, y in zip(self.num_trajs, self.traj_lens)]
    cum = np.cumsum(spt)
    f = np.searchsorted(cum, idx, side="right")
    start = idx + self.start_time - (cum[f - 1] if f > 0 else 0)
    if start + 3 * T > self.traj_lens[f]:
        return None, False
    s = slice(start + 2 * T, start + 3 * T)
    out = [(torch.tensor(self.data[f][k][s]) - self.diff_terms[k]) / self.div_terms[k] for k in self.output_fields]
    return torch.stack(out).float().permute(1, 0, 2, 3), True


def _getitem(self, idx):
    item = _orig_getitem(self, idx)
    if not getattr(self, "_pushforward", False):
        return item
    nxt, ok = _next_window(self, idx)
    if not ok:
        nxt = item[1].clone()
    return (*item, nxt, torch.tensor(ok))


DS.BubbleForecast.__init__ = _init
DS.BubbleForecast.__getitem__ = _getitem


# --- training step --------------------------------------------------------------------------------------
def _criterion(self):
    if not hasattr(self, "_pf_criterion"):            # lazy, see train_bandweight.py
        self._pf_criterion = IndicatorBandLoss(**CFG)
    return self._pf_criterion


def _log_and_return(self, loss, pf):
    self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    self.log("pushforward_step", float(pf), on_step=True, on_epoch=False, logger=True)
    current_lr = self.optimizers().param_groups[0]["lr"]
    self.log("learning_rate", current_lr, on_step=True, on_epoch=False, prog_bar=True, logger=True)
    if self.log_wandb and self.trainer.is_global_zero:
        import wandb
        wandb.log({"train_loss": loss, "learning_rate": current_lr})
    return loss


def _step_uncond(self, batch, batch_idx):
    inp, tgt, nxt, ok = batch
    crit = _criterion(self)
    if batch_idx % PF_EVERY == PF_EVERY - 1 and bool(ok.any()):
        with torch.no_grad():
            first = self.model(inp[ok])
        return _log_and_return(self, crit(self.model(first), nxt[ok]), True)
    return _log_and_return(self, crit(self.model(inp), tgt), False)


M.ForecastModule.training_step = _step_uncond

if __name__ == "__main__":
    import runpy
    runpy.run_path(os.path.join(HERE, "..", "external", "Bubbleformer", "scripts", "train.py"),
                   run_name="__main__")
