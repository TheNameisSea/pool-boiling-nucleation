#!/usr/bin/env python3
"""Train the AViT with the local wall-feature path (`wall_path.py`) and the wall-band-weighted loss.

Identical to `train_bandweight.py` (BAND_WEIGHT / BAND_ROWS, default x10) except that the AViT gains
the zero-initialised wall path ("local wall path + upweighting" in the paper). Validation keeps
upstream LpLoss.

Example:
    TRAIN_ENTRY=src/train_wallpath.py BAND_WEIGHT=10 bin/train.sh model_cfg=avit_small data_cfg=poolboiling_fc72_nocond
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import wall_path  # noqa: E402

wall_path.install()
import train_bandweight  # noqa: E402,F401  — patches the training criterion; its __main__ does not run

if __name__ == "__main__":
    import runpy
    runpy.run_path(os.path.join(HERE, "..", "external", "Bubbleformer", "scripts", "train.py"),
                   run_name="__main__")
