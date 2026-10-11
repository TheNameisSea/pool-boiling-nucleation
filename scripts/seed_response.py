#!/usr/bin/env python3
"""How much of a real seed does a model predict one step ahead, at the right frame and at nearby frames?

For every true on-site birth at raw frame f (component N = its footprint, >= 16 cells), take the
mean of φ over N in:
    true[f-1]  the frame before the seed
    true[f]    the seed itself
    tf[f-2], tf[f-1], tf[f], tf[f+1], tf[f+2]
               the model's one-step predictions of those frames, each made from true inputs
The response at offset o is
    r(o) = (tf[f+o] - true[f-1]) / (true[f] - true[f-1])
i.e. the fraction of the seed's jump in φ over its footprint that the model predicts: 1 for a full
seed at the right frame, 0 for no change. Also r(0) by the birth's delay since its site freed, and
the fraction of births where the model puts any vapour on the footprint at f.

Input: the one-step predictions saved by `rollout_diagnostics.py --teacher-forced --save-fields
$WORK_ROOT/tf_fields_<run>`. Run from the repository root (~48 GB RAM):
    python scripts/seed_response.py
"""
import os, sys

import h5py
import numpy as np
from scipy import ndimage

sys.path.insert(0, "scripts")
from birth_predictability import PATCH_HALF, PATCH_ROWS, meta, transitions
from diagnostics import DX, MIN_AREA_CELLS
from paths import POOL, SINGLE, WORK_ROOT as F
ALIGN, FRAMES, WALL_BAND, SITE_TOL = 100, 300, 0.5, 3
OFFSETS = (-2, -1, 0, 1, 2)
CASES = ([("pool", POOL, j, tw) for j in ("735", "766", "773", "716", "813", "814", "739", "855", "942", "952", "974", "980", "1011", "1034", "1052", "1055", "1070", "1077", "1089", "1184", "1181", "1194", "1200")
          for tw in ("75", "82", "90", "96", "102", "107")]
         + [("single", SINGLE, "778", tw) for tw in ("90", "92", "94", "96", "98", "100")])
OUT = "results/diagnostics/seed_response.csv"
OUT_D = "results/diagnostics/seed_response_by_delay.csv"
MAX_D = 6


def births_with_footprint(ref, cols):
    """(ref index t, footprint mask) for every on-site birth, by the counter's rule."""
    rows = max(1, int(round(WALL_BAND / DX)))
    out, prev = [], ref[0] > 0
    for t in range(1, ref.shape[0]):
        cur = ref[t] > 0
        lab, n = ndimage.label(cur)
        if n:
            near = set(np.unique(lab[:rows])) - {0}
            overl = set(np.unique(lab[prev])) - {0}
            sizes = ndimage.sum(cur, lab, index=np.arange(1, n + 1))
            for l in near - overl:
                if sizes[l - 1] < MIN_AREA_CELLS:
                    continue
                xs = np.where((lab[:rows] == l).any(axis=0))[0]
                dist = np.abs(xs[:, None] - cols[None, :]).min(axis=0)
                if dist.min() <= SITE_TOL:
                    out.append((t, lab == l, int(dist.argmin())))
        prev = cur
    return out


def analyse(study, root, job, tw):
    m = meta(os.path.join(root, f"Twall_{tw}.json"))
    cols = np.round((m["sites"] - m["x_min"]) / DX - 0.5).astype(int)
    with h5py.File(os.path.join(root, f"Twall_{tw}.hdf5"), "r") as h:
        ref = h["dfun"][ALIGN - 1:ALIGN + FRAMES].astype(np.float32)   # ref[t] = raw 99+t
    tf = np.load(f"{F}/tf_fields_{job}/dfun_tf_first_Twall_{tw}.npy", mmap_mode="r")
    # tf[k] predicts raw 100+k  ->  predicts ref[k+1]
    # site occupancy (same patch as birth_predictability.py), to get each birth's delay since its site freed
    wall = ref[:, :PATCH_ROWS, :] > 0
    W = ref.shape[2]
    deps = [transitions(wall[:, :, max(0, c - PATCH_HALF):min(W, c + PATCH_HALF + 1)].any(axis=(1, 2)))[1]
            for c in cols]
    rs, any_vap, jumps, by_d = {o: [] for o in OFFSETS}, [], [], {}
    for t, fp, s in births_with_footprint(ref, cols):
        before, seed = float(ref[t - 1][fp].mean()), float(ref[t][fp].mean())
        jump = seed - before
        if jump <= 0:
            continue
        jumps.append(jump)
        for o in OFFSETS:
            k = t - 1 + o
            if 0 <= k < FRAMES:
                rs[o].append((float(tf[k][fp].mean()) - before) / jump)
        any_vap.append(bool((tf[t - 1][fp] > 0).any()))
        prev = deps[s][deps[s] <= t - 1]
        if prev.size:
            d = min(int(t - prev[-1]), MAX_D + 1)
            by_d.setdefault(d, []).append((float(tf[t - 1][fp].mean()) - before) / jump)
    row = dict(study=study, job=job, twall=float(tw), n_births=len(jumps),
               jump_mean=float(np.mean(jumps)) if jumps else np.nan,
               model_any_vapour_on_footprint=float(np.mean(any_vap)) if any_vap else np.nan)
    for o in OFFSETS:
        v = np.array(rs[o])
        row[f"r{o:+d}_mean"] = float(v.mean()) if v.size else np.nan
        row[f"r{o:+d}_median"] = float(np.median(v)) if v.size else np.nan
    drows = [dict(study=study, job=job, twall=float(tw), delay=d, n=len(v),
                  r0_mean=float(np.mean(v)), r0_median=float(np.median(v)))
             for d, v in sorted(by_d.items())]
    return row, drows


def main():
    import pandas as pd
    rows, drows = [], []
    for case in CASES:
        r, dr = analyse(*case)
        rows.append(r)
        drows.extend(dr)
        print(f"{r['study']:>6} {r['job']} Twall {r['twall']:>4.0f} n {r['n_births']:>4} jump {r['jump_mean']:.3f} "
              f"| response r(o) median: " + " ".join(f"{o:+d}:{r[f'r{o:+d}_median']:+.2f}" for o in OFFSETS)
              + f" | any vapour on footprint {r['model_any_vapour_on_footprint']:.2f}", flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    pd.DataFrame(drows).to_csv(OUT_D, index=False)
    print(f"\nwrote {OUT}\nwrote {OUT_D}")


if __name__ == "__main__":
    main()
