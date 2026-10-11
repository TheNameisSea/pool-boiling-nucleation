#!/usr/bin/env python3
"""Reseeding one step from true inputs (Section 6.6): P(birth at a site within 1-5 frames | site freed).

  frees  : the reference's debounced site-freeing events (site_freeing.debounced_frees), i.e. the
           true inputs the model sees
  births : for the reference, the counter's on-site births (as in site_freeing.measure); for a
           model, its one-step births: counter rule on [true frame r-1, model's first prediction
           of r] (as in tf_birth_matching.py)
The reference row reproduces the reference row of site_freeing_rollouts.csv (printed as a check).
Uses $WORK_ROOT/tf_fields_<run>/dfun_tf_first_Twall_<tw>.npy (tf[k] predicts raw frame 100 + k).

Run from the repository root:
    python scripts/tf_reseeding.py 735 766 773 739 855 952 980 1055 1077 1089 1184 1070 1181 1200 1194
Writes results/diagnostics/tf_reseeding.csv.
"""
import os, sys

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from birth_predictability import counter_births_on_site, meta
from diagnostics import DX
from site_freeing import ALIGN, FRAMES, POOL, RESPOND, debounced_frees, occupancy
from tf_birth_matching import events
from paths import WORK_ROOT as F
SWEEP = ("75", "82", "90", "96", "102", "107")
OUT = "results/diagnostics/tf_reseeding.csv"


def respond(frees, births, n):
    """P(birth at the site within 1..RESPOND frames | debounced free), pooled over sites."""
    hit = tot = 0
    for k in range(n):
        bk = np.array(births.get(k, []), dtype=int)
        for t in frees[k]:
            tot += 1
            hit += bool(np.any((bk > t) & (bk <= t + RESPOND)))
    return hit / tot if tot else float("nan"), tot


def main():
    jobs = sys.argv[1:]
    rows = []
    for tw in SWEEP:
        m = meta(os.path.join(POOL, f"Twall_{tw}.json"))
        cols = np.round((m["sites"] - m["x_min"]) / DX - 0.5).astype(int)
        with h5py.File(os.path.join(POOL, f"Twall_{tw}.hdf5"), "r") as h:
            full = h["dfun"][ALIGN - 1:ALIGN + FRAMES].astype(np.float32)    # full[i] = raw 99 + i
        ref = full[1:]                                                       # ref[i] = raw 100 + i
        n = len(cols)
        occ = occupancy(ref, cols)
        frees = [debounced_frees(occ[:, k]) for k in range(n)]
        births = {}
        for t, k in counter_births_on_site(ref, cols)[2]:
            births.setdefault(k, []).append(t)
        p, tot = respond(frees, births, n)
        rows.append(dict(twall=int(tw), n_sites=n, source="reference", n_frees_deb=tot, respond=p))
        print(f"Twall_{tw} ({n:>2} sites) reference : {tot:>4} frees, P(birth<=5f|freed) {p:.2f}", flush=True)
        for job in jobs:
            tf = np.load(f"{F}/tf_fields_{job}/dfun_tf_first_Twall_{tw}.npy", mmap_mode="r")  # tf[i] -> raw 100 + i
            assert tf.shape == ref.shape, (job, tf.shape, ref.shape)
            pb = {}
            for i in range(1, FRAMES):          # index 0 cannot follow a free (frees start at 1)
                for k in events(full[i], np.asarray(tf[i]), cols):
                    pb.setdefault(k, []).append(i)
            p, _ = respond(frees, pb, n)
            rows.append(dict(twall=int(tw), n_sites=n, source=job, n_frees_deb=tot, respond=p))
            print(f"Twall_{tw} ({n:>2} sites) {job:>9} : one-step P(birth<=5f|freed) {p:.2f}", flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
