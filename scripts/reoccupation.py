#!/usr/bin/env python3
"""What re-occupies a freed site, per delay since it freed (reference data only)?

For each site, each time it goes dry -> occupied (raw frames [99, 400)):
  counted   : the counter records an on-site birth for this site in that frame
  merged    : no counted birth, but a seed-like blob appears at the site: the wall cell under the
              site newly becomes vapour and >= SEED_MIN new vapour cells appear in the seed box
              (rows < 4, |x - site| <= 3 cells; the seed radius 0.1 is 3.2 cells)
  encroach  : anything else, i.e. vapour entering the patch without a new blob at the site centre
Reported per delay d (frames since the site freed): the counted-only hazard, the any-seed hazard
(counted + merged) / dry and the share of seeds merging on arrival. Feeds Fig. 1b.

Run from the repository root (~32 GB RAM):
    python scripts/reoccupation.py
"""
import os, sys
from collections import defaultdict

import h5py
import numpy as np

sys.path.insert(0, "scripts")
from birth_predictability import PATCH_HALF, PATCH_ROWS, counter_births_on_site, meta, transitions
from diagnostics import DX
from paths import POOL, SINGLE
ALIGN, FRAMES = 100, 300
SEED_ROWS, SEED_HALF, SEED_MIN = 4, 3, 10
MAX_D = 6
CASES = ([("pool", POOL, tw) for tw in ("75", "82", "90", "96", "102", "107")]
         + [("single", SINGLE, tw) for tw in ("90", "92", "94", "96", "98", "100")])
OUT = "results/diagnostics/reoccupation_by_delay.csv"


def analyse(study, root, tw):
    m = meta(os.path.join(root, f"Twall_{tw}.json"))
    cols = np.round((m["sites"] - m["x_min"]) / DX - 0.5).astype(int)
    with h5py.File(os.path.join(root, f"Twall_{tw}.hdf5"), "r") as h:
        ref = h["dfun"][ALIGN - 1:ALIGN + FRAMES].astype(np.float32)      # ref[t] = raw 99+t
    vap = ref > 0
    W = ref.shape[2]
    _, _, events = counter_births_on_site(ref, cols)
    counted = {(t, s) for t, s in events}
    occ = np.stack([vap[:, :PATCH_ROWS, max(0, c - PATCH_HALF):min(W, c + PATCH_HALF + 1)].any(axis=(1, 2))
                    for c in cols], axis=1)

    tally = defaultdict(lambda: defaultdict(int))   # tally[d][kind]
    for s, c in enumerate(cols):
        births, deps = transitions(occ[:, s])
        for t in births:                            # t: first occupied frame after a dry spell
            prev = deps[deps < t]
            if not prev.size:
                continue
            d = min(int(t - prev[-1]), MAX_D + 1)
            lo, hi = max(0, c - SEED_HALF), min(W, c + SEED_HALF + 1)
            new = vap[t, :SEED_ROWS, lo:hi] & ~vap[t - 1, :SEED_ROWS, lo:hi]
            centre_new = bool(vap[t, 0, c] and not vap[t - 1, 0, c])
            if (t, s) in counted:
                kind = "counted"
            elif centre_new and new.sum() >= SEED_MIN:
                kind = "merged"
            else:
                kind = "encroach"
            tally[d][kind] += 1
        # dry frames at each delay (the hazard's denominator)
        for t in range(1, occ.shape[0]):
            if occ[t - 1, s]:
                continue
            prev = deps[deps <= t - 1]
            if prev.size:
                tally[min(int(t - prev[-1]), MAX_D + 1)]["dry"] += 1
    return tally


def main():
    import pandas as pd
    rows = []
    for study, root, tw in CASES:
        t = analyse(study, root, tw)
        for d, k in sorted(t.items()):
            rows.append(dict(study=study, twall=float(tw), delay=d, dry=k["dry"], counted=k["counted"],
                             merged=k["merged"], encroach=k["encroach"]))
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    for name, sub in (("pool, 12+ sites", df[(df.study == "pool") & (df.twall >= 82)]),
                      ("pool, 5 sites", df[(df.study == "pool") & (df.twall == 75)]),
                      ("control", df[df.study == "single"])):
        g = sub.groupby("delay")[["dry", "counted", "merged", "encroach"]].sum()
        g["hazard_counted"] = g.counted / g.dry
        g["hazard_any_seed"] = (g.counted + g.merged) / g.dry
        g["hazard_any_occupation"] = (g.counted + g.merged + g.encroach) / g.dry
        print(f"\n== {name}\n{g.round(3).to_string()}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
