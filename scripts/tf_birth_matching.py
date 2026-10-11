#!/usr/bin/env python3
"""Matched recall and precision of one-step births: are a model's seeds the right ones?

Each predicted birth is matched one to one to a true birth at the same site:
  pred births at frame k : counter's on-site births in [true frame k, model's first prediction of k+1]
  ref births at frame k  : the same rule on [true frame k, true frame k+1]
  exact  : same site, same frame
  +-1    : same site, frame within one (a seed a frame early or late)
  recall = matched ref births / ref births;  precision = matched pred births / pred births
Uses the first-frame predictions saved by `rollout_diagnostics.py --teacher-forced --save-fields
$WORK_ROOT/tf_fields_<run>` (tf[k] predicts raw frame 100 + k).

Run from the repository root:
    python scripts/tf_birth_matching.py 735 766 773 716 813 814 739 855
    MATCH_STUDY=single MATCH_OUT=results/diagnostics/tf_birth_matching_control.csv python scripts/tf_birth_matching.py 778
Writes results/diagnostics/tf_birth_matching.csv (or MATCH_OUT).
"""
import os, sys

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from birth_predictability import counter_births_on_site, meta
from diagnostics import DX
from paths import POOL, SINGLE, WORK_ROOT as F

ALIGN, FRAMES = 100, 300
SWEEP = ("75", "82", "90", "96", "102", "107")
if os.getenv("MATCH_STUDY") == "single":        # the single-bubble control, same recipe
    POOL = SINGLE
    SWEEP = ("90", "92", "94", "96", "98", "100")
OUT = os.getenv("MATCH_OUT", "results/diagnostics/tf_birth_matching.csv")


def events(prev, cur, cols):
    _, _, ev = counter_births_on_site(np.stack([prev, cur]), cols)
    return {s for _, s in ev}


def match(ref_ev, pred_ev, tol):
    """Greedy one-to-one matching per site across frames, |frame difference| <= tol."""
    used, hit = set(), 0
    for k, s in sorted(ref_ev):
        for dk in sorted(range(-tol, tol + 1), key=abs):
            if (k + dk, s) in pred_ev and (k + dk, s) not in used:
                used.add((k + dk, s)); hit += 1
                break
    return hit


def main():
    jobs = sys.argv[1:] or ["735", "952", "1011", "1052", "1055", "739", "1034"]
    rows = []
    for tw in SWEEP:
        m = meta(os.path.join(POOL, f"Twall_{tw}.json"))
        cols = np.round((m["sites"] - m["x_min"]) / DX - 0.5).astype(int)
        with h5py.File(os.path.join(POOL, f"Twall_{tw}.hdf5"), "r") as h:
            ref = h["dfun"][ALIGN - 1:ALIGN + FRAMES].astype(np.float32)       # ref[k] = raw 99 + k
        ref_ev = {(k, s) for k in range(FRAMES) for s in events(ref[k], ref[k + 1], cols)}
        for job in jobs:
            tf = np.load(f"{F}/tf_fields_{job}/dfun_tf_first_Twall_{tw}.npy", mmap_mode="r")   # tf[k] -> ref[k+1]
            pred_ev = {(k, s) for k in range(FRAMES) for s in events(ref[k], np.asarray(tf[k]), cols)}
            r = dict(job=job, twall=int(tw), n_sites=len(cols), ref_births=len(ref_ev), pred_births=len(pred_ev))
            for tol in (0, 1):
                hit_r = match(ref_ev, pred_ev, tol)
                hit_p = match(pred_ev, ref_ev, tol)
                r[f"recall_tol{tol}"] = hit_r / len(ref_ev) if ref_ev else np.nan
                r[f"precision_tol{tol}"] = hit_p / len(pred_ev) if pred_ev else np.nan
            rows.append(r)
            print(f"Twall_{tw} ({len(cols):>2} sites) {job:>5}: ref {len(ref_ev):>3} pred {len(pred_ev):>3} | exact recall "
                  f"{r['recall_tol0']:.2f} precision {r['precision_tol0']:.2f} | +-1 recall {r['recall_tol1']:.2f} "
                  f"precision {r['precision_tol1']:.2f}", flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
