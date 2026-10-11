"""Locations of the data and of large intermediate files, set by environment variables.

    BUBBLEML_ROOT  directory holding the BubbleML 2.0 studies, e.g. <root>/PoolBoiling-Saturated-FC72-2D/
                   (default: data/bubbleml2 in the repository)
    WORK_ROOT      directory for checkpoints and saved model predictions (several GB per model)
                   (default: work in the repository)
"""
import os

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.environ.get("BUBBLEML_ROOT", os.path.join(_REPO, "data", "bubbleml2"))
WORK_ROOT = os.environ.get("WORK_ROOT", os.path.join(_REPO, "work"))
POOL = os.path.join(DATA_ROOT, "PoolBoiling-Saturated-FC72-2D")
SINGLE = os.path.join(DATA_ROOT, "SingleBubble-Saturated-FC72-2D")
