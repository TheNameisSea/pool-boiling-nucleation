#!/usr/bin/env python3
"""Recall reachable by a deterministic squared-error forecaster (Section 6.2, the "ceiling").

Under a squared error the optimum at a site with seed probability p is the blend
    phi_hat = p * phi_seed + (1 - p) * phi_noseed,      p = P(seed next frame | input window).
The counter only sees phi_hat > 0 regions of >= MIN_AREA_CELLS, so a blend can go uncounted.

Per true on-site birth (the counter's own events, held-out sims, raw frames [100, 400)):
  phi_seed   = the true frame f
  phi_noseed = the true frame f with the seed's component removed (true dfun + the EDT difference,
               so every cell the seed does not affect keeps its exact true value)
  p*         = the smallest p (grid 0.02) at which phi_hat still yields a counted on-site birth
               at that site, by count_nucleation_events' rule (wall band, no overlap with frame
               f-1's true vapour, >= MIN_AREA_CELLS)
A probability source q then "hits" a birth when q >= p*. Sources:
  cnn      : the CNN of timing_readability.py on the true 5-frame local patch (mean of 3 seeds,
             retrained here on the same split)
  hazard   : P(seed | delay since the site freed)
  sharp    : p = 1 on every true seed (check: must give 1.00)
False positives: a non-seed sample hedged at p counts with probability F(p), F = the empirical
CDF of p* in the same sim. The count ceiling (`count_*`) is (hits + expected false positives) / births.
Idealisation: the no-seed alternative is "no vapour at the site this frame"; other vapour is taken
as exactly predicted.

Needs all 20 pool-boiling simulations. Run from the repository root (~32 GB RAM):
    python scripts/l2opt_ceiling.py
Writes results/diagnostics/l2opt_ceiling.csv (per sim) and l2opt_ceiling_births.csv (per birth).
"""
import os, sys, time

import h5py
import numpy as np
import pandas as pd
from scipy import ndimage

sys.path.insert(0, "scripts")
from birth_predictability import PATCH_HALF, WALL_BAND, counter_births_on_site, meta
from diagnostics import DX, MIN_AREA_CELLS
from timing_readability import (ALIGN, FRAMES, MAX_D, POOL, SINGLE, SWEEP, WIN, fit, onehot, samples,
                                sims)

ROWS = max(1, int(round(WALL_BAND / DX)))       # the counter's wall band, 16 rows
CROP = 160                                       # rows kept when relabelling (births live at the wall)
PGRID = np.round(np.arange(0, 1.0001, 0.02), 2)
CNN_SEEDS = (0, 1, 2)
OUT, OUT_B = "results/diagnostics/l2opt_ceiling.csv", "results/diagnostics/l2opt_ceiling_births.csv"


def sdf(mask):
    return np.where(mask, ndimage.distance_transform_edt(mask), -ndimage.distance_transform_edt(~mask)) * DX


def counted_at(region, prev, cols, s):
    """Does `region` hold a birth assigned to site s by count_nucleation_events' rule?"""
    lab, n = ndimage.label(region)
    if not n:
        return False
    near = set(np.unique(lab[:ROWS])) - {0}
    overl = set(np.unique(lab[prev])) - {0}
    sizes = ndimage.sum(region, lab, index=np.arange(1, n + 1))
    for l in near - overl:
        if sizes[l - 1] < MIN_AREA_CELLS:
            continue
        xs = np.where((lab[:ROWS] == l).any(axis=0))[0]
        dist = np.abs(xs[:, None] - cols[None, :]).min(axis=0)
        if dist.min() <= PATCH_HALF and int(dist.argmin()) == s:
            return True
    return False


def seed_component(cur, prev, cols, s):
    lab, n = ndimage.label(cur)
    near = set(np.unique(lab[:ROWS])) - {0}
    overl = set(np.unique(lab[prev])) - {0}
    best, bd = None, np.inf
    for l in near - overl:
        if (lab == l).sum() < MIN_AREA_CELLS:
            continue
        xs = np.where((lab[:ROWS] == l).any(axis=0))[0]
        dist = np.abs(xs[:, None] - cols[None, :]).min(axis=0)
        if dist.min() <= PATCH_HALF and int(dist.argmin()) == s and dist[s] < bd:
            best, bd = l, dist[s]
    return None if best is None else lab == best


def pstar_for_births(root, tw):
    """[(f, site, p*, monotone)] for every on-site birth of sim tw, f indexing from raw ALIGN - WIN."""
    m = meta(os.path.join(root, f"Twall_{tw}.json"))
    cols = np.round((m["sites"] - m["x_min"]) / DX - 0.5).astype(int)
    with h5py.File(os.path.join(root, f"Twall_{tw}.hdf5"), "r") as h:
        dfun = h["dfun"][ALIGN - WIN:ALIGN + FRAMES].astype(np.float32)
    _, _, ev = counter_births_on_site(dfun[WIN - 1:], cols)
    out = []
    for t, s in ev:
        f = t + WIN - 1
        cur, prev = dfun[f] > 0, dfun[f - 1] > 0
        seed = seed_component(cur, prev, cols, s)
        if seed is None:                                  # recorded as NaN (not covered)
            out.append((f, s, np.nan, False)); continue
        phi_t = dfun[f]
        phi_n = phi_t + (sdf(cur & ~seed) - sdf(cur))
        pt, pn, pv = phi_t[:CROP], phi_n[:CROP], prev[:CROP]
        hit = np.array([counted_at(p * pt + (1 - p) * pn > 0, pv, cols, s) for p in PGRID])
        if not hit[-1]:
            out.append((f, s, np.nan, False)); continue
        first = int(np.argmax(hit))
        out.append((f, s, float(PGRID[first]), bool(hit[first:].all())))
    return out


def run(study, root, test):
    t0 = time.time()
    data = {tw: samples(root, tw, with_ids=True) for tw in sims(root)}
    tr_k = [k for k in data if k not in test]; te_k = [k for k in data if k in test]
    cat = lambda keys, i: np.concatenate([data[k][i] for k in keys])
    Xtr, dtr, ytr = cat(tr_k, 0), cat(tr_k, 1), cat(tr_k, 2)
    print(f"\n== {study}: {len(ytr)} train samples, test sims {te_k} ({time.time() - t0:.0f} s)", flush=True)
    haz = {dd: ytr[dtr == dd].mean() if (dtr == dd).any() else ytr.mean() for dd in range(1, MAX_D + 2)}
    Xte, dte = cat(te_k, 0), cat(te_k, 1)
    p_cnn = np.mean([fit(Xtr, ytr, Xte, onehot(dtr), onehot(dte), seed=sd) for sd in CNN_SEEDS], axis=0)
    del Xtr, Xte
    rows, births = [], []
    off = 0
    for tw in te_k:
        X, d, y, mg, twall, ids = data[tw]
        n = len(d); pc = p_cnn[off:off + n]; off += n
        look = {(int(s), int(f)): i for i, (s, f) in enumerate(ids)}
        ps = pstar_for_births(root, tw)
        ns = len(meta(os.path.join(root, f"Twall_{tw}.json"))["sites"])
        rec = []
        for f, s, pst, mono in ps:
            i = look.get((s, f))
            dd = int(d[i]) if i is not None else np.nan
            row = dict(study=study, twall=tw, n_sites=ns, f=f, site=s, pstar=pst, monotone=mono, delay=dd,
                       p_cnn=float(pc[i]) if i is not None else np.nan,
                       p_hazard=float(haz[dd]) if i is not None else np.nan)
            rec.append(row)
        births += rec
        B = pd.DataFrame(rec)
        ok = B.dropna(subset=["pstar", "p_cnn"])
        cdf = lambda q: (ok.pstar.values[None, :] <= np.asarray(q)[:, None]).mean(axis=1)
        neg = ~y & (mg == False)                                     # non-seed samples
        out = dict(study=study, twall=tw, n_sites=ns, n_births=len(B), covered=len(ok),
                   pstar_median=float(ok.pstar.median()), pstar_p25=float(ok.pstar.quantile(.25)),
                   pstar_p75=float(ok.pstar.quantile(.75)), monotone_share=float(ok.monotone.mean()))
        srcs = {"cnn": (ok.p_cnn, pc[neg]), "hazard": (ok.p_hazard, np.array([haz[x] for x in d[neg]])),
                "sharp": (np.ones(len(ok)), np.zeros(neg.sum()))}
        for k, (qp, qn) in srcs.items():
            hits = float((np.asarray(qp, float) >= ok.pstar.values).sum())
            fp = float(cdf(np.clip(qn, 0, 1)).sum()) if len(qn) else 0.0
            out[f"hit_{k}"] = hits / len(ok)
            out[f"count_{k}"] = (hits + fp) / len(ok)
        rows.append(out)
        print(f"  Twall_{tw} sites={ns:>2} births {len(B)} (covered {len(ok)}) p* median {out['pstar_median']:.2f} "
              f"[{out['pstar_p25']:.2f}, {out['pstar_p75']:.2f}] | hit/count: cnn {out['hit_cnn']:.2f}/{out['count_cnn']:.2f}  "
              f"hazard {out['hit_hazard']:.2f}/{out['count_hazard']:.2f}  sharp {out['hit_sharp']:.2f}", flush=True)
    return rows, births


def main():
    rows, births = run("single", SINGLE, {"98", "100"})
    r2, b2 = run("pool", POOL, SWEEP)
    pd.DataFrame(rows + r2).to_csv(OUT, index=False)
    pd.DataFrame(births + b2).to_csv(OUT_B, index=False)
    print(f"\nwrote {OUT} and {OUT_B}")


if __name__ == "__main__":
    main()
