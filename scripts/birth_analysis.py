#!/usr/bin/env python3
"""Size, position and lifetime of the births counted in the single-bubble control model's rollouts,
at a 4-cell threshold (the event-size threshold check of Section 3.3).

For each counted birth: area, distance from the single true site (x = 0), lifetime (frames until its
footprint stops overlapping vapour). Reference and model are reported side by side.

Input: the control model's saved rollout fields (`rollout_diagnostics.py --save-fields
$WORK_ROOT/rollout_fields_778` on the single-bubble sims). Run from the repository root.
"""
import json, os, sys
import numpy as np
from scipy import ndimage

sys.path.insert(0, "scripts")
from diagnostics import DX
from paths import SINGLE as DATA, WORK_ROOT

FIELDS = os.path.join(WORK_ROOT, "rollout_fields_778")
SIMS = ["90", "92", "94", "96", "98", "100"]
ALIGN, FRAMES, MIN_AREA = 100, 300, 4


def births(phi_seq, wall_band=0.5, min_area=MIN_AREA):
    """Same rule as diagnostics.count_nucleation_events, but also return x, area, lifetime."""
    rows = max(1, int(round(wall_band / DX)))
    T = phi_seq.shape[0]
    out = []
    prev = phi_seq[0] > 0
    for t in range(1, T):
        cur = phi_seq[t] > 0
        lab, n = ndimage.label(cur)
        if n:
            near = set(np.unique(lab[:rows, :])) - {0}
            over = set(np.unique(lab[prev])) - {0}
            sizes = ndimage.sum(cur, lab, index=np.arange(1, n + 1))
            for l in near - over:
                if sizes[l - 1] < min_area:
                    continue
                ys, xs = np.where(lab == l)
                # lifetime: follow this blob's footprint forward until nothing overlaps
                foot = (lab == l)
                life = 1
                for u in range(t + 1, min(t + 60, T)):
                    nxt = phi_seq[u] > 0
                    if not (foot & nxt).any():
                        break
                    l2, _ = ndimage.label(nxt)
                    keep = set(np.unique(l2[foot])) - {0}
                    foot = np.isin(l2, list(keep))
                    life += 1
                out.append(dict(t=t, x_cell=float(xs.mean()), area=float(sizes[l - 1]),
                                life=life, peak=float(foot.sum())))
        prev = cur
    return out


def summarise(tag, ev, W, xsite_cell):
    if not ev:
        print(f"  {tag:<5} no births"); return
    a = np.array([e["area"] for e in ev]); L = np.array([e["life"] for e in ev])
    x = np.array([e["x_cell"] for e in ev]); pk = np.array([e["peak"] for e in ev])
    dx_from_site = np.abs(x - xsite_cell) * DX
    print(f"  {tag:<5} n={len(ev):<4} area: med={np.median(a):5.1f} p90={np.percentile(a,90):6.1f}  "
          f"life(frames): med={np.median(L):4.1f} frac<=2={np.mean(L<=2):.2f}  "
          f"peak/birth={np.median(pk/a):4.1f}  |x-site|: med={np.median(dx_from_site):.2f} "
          f"frac<0.5={np.mean(dx_from_site<0.5):.2f}")


for tw in SIMS:
    f = os.path.join(FIELDS, f"dfun_pred_Twall_{tw}.npy")
    if not os.path.exists(f):
        print(f"Twall_{tw}: fields not saved yet"); continue
    pred = np.load(f)
    import h5py
    with h5py.File(os.path.join(DATA, f"Twall_{tw}.hdf5"), "r") as h:
        ref = h["dfun"][ALIGN:ALIGN + FRAMES].astype(np.float32)
    meta = json.load(open(os.path.join(DATA, f"Twall_{tw}.json")))
    xs_phys = meta["nuc_sites_x"][0]
    xmin, xmax = float(meta["x_min"]), float(meta["x_max"])
    W = pred.shape[-1]
    xsite_cell = (xs_phys - xmin) / (xmax - xmin) * W
    print(f"\nTwall_{tw}  pred{pred.shape} ref{ref.shape}  site x={xs_phys} -> cell {xsite_cell:.1f}/{W}")
    summarise("ref", births(ref), W, xsite_cell)
    summarise("pred", births(pred), W, xsite_cell)
