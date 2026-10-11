"""Vapour volume and bubble anatomy in rollouts of the conditioned models (Section 5.4).

For each source (reference, the three conditioned FiLM-AViT runs 716/813/814, the AViT run 735) at
Twall 75/82/102/107: void fraction at frames 0/150/299, wall-band and bulk void fraction, and the
number and median size of bubbles (components >= 16 cells), at frame 150 and over every 10th frame.

Input: rollouts saved by `rollout_diagnostics.py --save-fields $WORK_ROOT/rollout_fields_<run>`.
Run from the repository root:
    python scripts/cond_anatomy.py
"""
import os, sys
import numpy as np, h5py
from scipy import ndimage
sys.path.insert(0, "scripts")
from diagnostics import DX, MIN_AREA_CELLS
from paths import POOL as DATA, WORK_ROOT

SRC = {"716 COND": f"{WORK_ROOT}/rollout_fields_716",
       "813 COND": f"{WORK_ROOT}/rollout_fields_813",
       "814 COND": f"{WORK_ROOT}/rollout_fields_814",
       "735 uncond": f"{WORK_ROOT}/rollout_fields_735"}
WALL_BAND = 0.5   # domain units above the heater, same as the nucleation counter
BLOB_FRAMES = range(0, 300, 10)   # frames for the time-resolved blob statistics


def blobs(vap2d):
    lab, n = ndimage.label(vap2d)
    sz = ndimage.sum(vap2d, lab, index=np.arange(1, n + 1)) if n else np.array([])
    return sz[sz >= MIN_AREA_CELLS]


for tw in ("75", "82", "102", "107"):
    with h5py.File(os.path.join(DATA, f"Twall_{tw}.hdf5"), "r") as h:
        ref = h["dfun"][100:400].astype(np.float32)
    series = [("reference", ref)] + [(k, np.load(os.path.join(v, f"dfun_pred_Twall_{tw}.npy")))
                                     for k, v in SRC.items()]
    rows = max(1, int(round(WALL_BAND / DX)))
    print(f"\n{'='*92}\nTwall_{tw}   (wall band = first {rows} rows = {WALL_BAND} domain units)\n{'='*92}")
    print(f"{'source':>12} {'void t=0':>9} {'void t=150':>11} {'void t=299':>11} "
          f"{'wall-band void':>15} {'bulk void':>10} {'med blob':>9} {'n blobs':>8}")
    for name, a in series:
        vap = a > 0
        v0, vm, vl = vap[0].mean(), vap[150].mean(), vap[-1].mean()
        wall = vap[:, :rows, :].mean()
        bulk = vap[:, rows:, :].mean()
        lab, n = ndimage.label(vap[150])
        sz = ndimage.sum(vap[150], lab, index=np.arange(1, n + 1)) if n else np.array([0])
        sz = sz[sz >= MIN_AREA_CELLS]
        print(f"{name:>12} {v0:>9.4f} {vm:>11.4f} {vl:>11.4f} {wall:>15.4f} {bulk:>10.4f} "
              f"{np.median(sz) if sz.size else 0:>9.0f} {sz.size:>8}")
    # Over every 10th frame: per-frame count (median and IQR) and the per-frame median size.
    print(f"\n  time-resolved blobs, {len(BLOB_FRAMES)} frames (0..299 step 10)")
    print(f"{'source':>12} {'n blobs med':>12} {'IQR':>9} {'med blob med':>13} {'IQR':>11}")
    for name, a in series:
        per = [blobs(a[t] > 0) for t in BLOB_FRAMES]
        cnt = np.array([s.size for s in per])
        med = np.array([np.median(s) for s in per if s.size])
        print(f"{name:>12} {np.median(cnt):>12.1f} {np.percentile(cnt, 25):>4.0f}-{np.percentile(cnt, 75):<4.0f}"
              f" {np.median(med):>13.0f} {np.percentile(med, 25):>5.0f}-{np.percentile(med, 75):<5.0f}")
