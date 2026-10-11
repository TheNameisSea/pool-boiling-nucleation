#!/usr/bin/env python3
"""Training-loss penalty for omitting young bubbles, next to the models' one-step error (Table 2).

The training loss is `LpLoss(d=2, p=2, reduce_dims=[0,1,2], reductions=[mean, mean, sum])` on
(B, T=5, C=4, H, W): a relative L2 per frame and field, ||ŷ − y|| / ||y|| over H x W, averaged over
batch and the 5 output frames, summed over the 4 fields. A prediction that is exact except that it
omits newly born bubbles therefore pays, in its dfun term,

    penalty(t) = || sdf(vapour_t minus new bubbles) − sdf(vapour_t) || / || φ_t ||

Both sides are reinitialised from their masks, so the difference isolates the removal. Two
versions of "new":
  birth : components born at frame t (the counter's rule: touch the wall band, no overlap with
          frame t−1, >= 16 cells)
  young : wall-band components with no overlap with the vapour of frame t−5, i.e. bubbles born
          inside the last output window (the "young bubbles" of Table 2)

Compared with the models' one-step relative L2 of dfun: the first 5 predicted frames of each saved
rollout, which are predicted from true inputs (runs 716, 735 at Twall 75/82/102/107; control 778).

Run from the repository root (~32 GB RAM):
    python scripts/loss_weighting.py
"""
import os, sys

import h5py
import numpy as np
from scipy import ndimage

sys.path.insert(0, "scripts")
from diagnostics import DX, MIN_AREA_CELLS, reinit_sdf
from paths import POOL, SINGLE, WORK_ROOT as F

ALIGN, FRAMES, LOOKBACK, WALL_BAND = 100, 300, 5, 0.5
POOL_MODELS = {"716 cond": f"{F}/rollout_fields_716", "735 uncond": f"{F}/rollout_fields_735"}
CASES = ([("pool", POOL, tw, POOL_MODELS if tw in ("75", "82", "102", "107") else {})
          for tw in ("75", "82", "90", "96", "102", "107")]
         + [("single", SINGLE, tw, {"778 control": f"{F}/rollout_fields_778"})
            for tw in ("90", "92", "94", "96", "98", "100")])
OUT = "results/diagnostics/loss_weighting.csv"


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def new_components(cur, ref_prev, rows, min_area=MIN_AREA_CELLS):
    """Mask of wall-band components of `cur` (>= min_area) that do not overlap `ref_prev`."""
    lab, n = ndimage.label(cur)
    if not n:
        return np.zeros_like(cur)
    near = set(np.unique(lab[:rows])) - {0}
    overl = set(np.unique(lab[ref_prev])) - {0}
    sizes = ndimage.sum(cur, lab, index=np.arange(1, n + 1))
    keep = [l for l in near - overl if sizes[l - 1] >= min_area]
    return np.isin(lab, keep) if keep else np.zeros_like(cur)


def analyse(root, tw, models):
    rows = max(1, int(round(WALL_BAND / DX)))
    with h5py.File(os.path.join(root, f"Twall_{tw}.hdf5"), "r") as h:
        phi = h["dfun"][ALIGN - LOOKBACK:ALIGN + FRAMES].astype(np.float32)   # 5 frames of lookback
    vap = phi > 0
    pen_birth, pen_young, n_birth_frames, n_young_frames, young_cells = [], [], 0, 0, []
    for i in range(LOOKBACK, phi.shape[0]):
        base = reinit_sdf(phi[i])
        norm = np.linalg.norm(phi[i])
        nb = new_components(vap[i], vap[i - 1], rows)
        ny = new_components(vap[i], vap[i - LOOKBACK], rows)
        pb = float(np.linalg.norm(reinit_sdf(np.where(nb, -1.0, phi[i])) - base) / norm) if nb.any() else 0.0
        py = float(np.linalg.norm(reinit_sdf(np.where(ny, -1.0, phi[i])) - base) / norm) if ny.any() else 0.0
        pen_birth.append(pb); pen_young.append(py)
        n_birth_frames += bool(nb.any()); n_young_frames += bool(ny.any())
        young_cells.append(int(ny.sum()))
    pen_birth, pen_young = np.array(pen_birth), np.array(pen_young)
    out = dict(frames=len(pen_young), cells=int(phi[0].size),
               frac_frames_with_birth=n_birth_frames / len(pen_birth),
               frac_frames_with_young=n_young_frames / len(pen_young),
               young_cells_mean=float(np.mean(young_cells)),
               pen_birth_mean=float(pen_birth.mean()),
               pen_birth_when_present=float(pen_birth[pen_birth > 0].mean()) if (pen_birth > 0).any() else 0.0,
               pen_young_mean=float(pen_young.mean()),
               pen_young_when_present=float(pen_young[pen_young > 0].mean()) if (pen_young > 0).any() else 0.0)
    # the models' one-step dfun error: first output window of each rollout, predicted from truth
    truth = phi[LOOKBACK:LOOKBACK + 5]
    for name, d in models.items():
        pred = np.load(os.path.join(d, f"dfun_pred_Twall_{tw}.npy"), mmap_mode="r")[:5]
        out[f"onestep_{name}"] = float(np.mean([rel(pred[k], truth[k]) for k in range(5)]))
    return out


def main():
    import pandas as pd
    rows = []
    for study, root, tw, models in CASES:
        r = analyse(root, tw, models)
        rows.append(dict(study=study, twall=float(tw), **r))
        ms = "  ".join(f"one-step {k.split()[0]} {v:.4f}" for k, v in r.items() if k.startswith("onestep_"))
        print(f"{study:>6} Twall {tw:>3} | frames w/ birth {r['frac_frames_with_birth']:.2f} "
              f"w/ young {r['frac_frames_with_young']:.2f} young cells {r['young_cells_mean']:.0f} "
              f"of {r['cells']} | penalty birth mean {r['pen_birth_mean']:.5f} (when present "
              f"{r['pen_birth_when_present']:.5f}) | young mean {r['pen_young_mean']:.5f} (when present "
              f"{r['pen_young_when_present']:.5f}) | {ms}", flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
