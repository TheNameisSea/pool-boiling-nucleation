#!/usr/bin/env python3
"""Readability of seeds in later output frames, and cross-validation of the next-frame classifier.

1. Per output frame k = 1..5: sample (sim, site, f) with the site dry in the input's last frame f-1 and
   a known delay d; label = a seed (counted or merged) at the site in output frame k, i.e. raw frame
   f+k-1. Hazard P(y_k | d) vs the CNN of timing_readability.py on the same window, trained on the 14
   simulations outside the evaluation set and scored on the 6 evaluation sims.
2. Output frame 1 under 5-fold cross-validation grouped by simulation (4 sims per fold).

Needs all 20 pool-boiling simulations. Run from the repository root:
    python scripts/horizon_readability.py
"""
import os, sys, time

import h5py
import numpy as np

sys.path.insert(0, "scripts")
from birth_predictability import PATCH_HALF, PATCH_ROWS, counter_births_on_site, meta, transitions
from diagnostics import DX
from timing_readability import (ALIGN, FIELDS, FRAMES, MAX_D, P_HALF, P_ROWS, POOL, SEED_HALF, SEED_MIN,
                                SEED_ROWS, SWEEP, WIN, auc, fit, onehot, sims)

K = 5
OUT = "results/diagnostics/horizon_readability.csv"
OUT_CV = "results/diagnostics/timing_readability_groupcv.csv"


def samples(root, tw):
    """X (n, 20, P_ROWS, 2*P_HALF+1), d (n,), Y (n, K): seed at the site in output frame k."""
    m = meta(os.path.join(root, f"Twall_{tw}.json"))
    cols = np.round((m["sites"] - m["x_min"]) / DX - 0.5).astype(int)
    lo = ALIGN - WIN
    hi = ALIGN + FRAMES + K - 1                           # labels reach K-1 frames past the last f
    with h5py.File(os.path.join(root, f"Twall_{tw}.hdf5"), "r") as h:
        dfun = h["dfun"][lo:hi].astype(np.float32)
        band = {k: h[k][lo:hi, :P_ROWS].astype(np.float32) for k in FIELDS[1:]}
    band["dfun"] = dfun[:, :P_ROWS]
    W = dfun.shape[2]
    vap = dfun > 0
    _, _, ev = counter_births_on_site(dfun[WIN - 1:], cols)
    counted = {(t + WIN - 1, s) for t, s in ev}
    occ = np.stack([vap[:, :PATCH_ROWS, max(0, c - PATCH_HALF):min(W, c + PATCH_HALF + 1)].any(axis=(1, 2))
                    for c in cols], axis=1)
    # every seed event (frame, site): counted, or the merged-seed rule at a dry -> occupied transition
    seed = np.zeros(occ.shape, bool)
    for s, c in enumerate(cols):
        l2, h2 = max(0, c - SEED_HALF), min(W, c + SEED_HALF + 1)
        for t in range(1, occ.shape[0]):
            if occ[t, s] and not occ[t - 1, s]:
                new = vap[t, :SEED_ROWS, l2:h2] & ~vap[t - 1, :SEED_ROWS, l2:h2]
                seed[t, s] = ((t, s) in counted
                              or (vap[t, 0, c] and not vap[t - 1, 0, c] and new.sum() >= SEED_MIN))
    padded = {k: np.pad(v, ((0, 0), (0, 0), (P_HALF, P_HALF)), mode="edge") for k, v in band.items()}
    X, D, Y = [], [], []
    for s, c in enumerate(cols):
        _, deps = transitions(occ[:, s])
        for f in range(WIN, WIN + FRAMES):
            if occ[f - 1, s]:
                continue
            prev = deps[deps <= f - 1]
            if not prev.size:
                continue
            X.append(np.concatenate([padded[k][f - WIN:f, :, c:c + 2 * P_HALF + 1] for k in FIELDS]))
            D.append(min(int(f - prev[-1]), MAX_D + 1))
            Y.append(seed[f:f + K, s])
    return np.stack(X), np.array(D), np.array(Y)


def metrics(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    ll = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    base = y.mean()
    ll0 = float(-(base * np.log(base) + (1 - base) * np.log(1 - base)))
    return dict(pos_rate=float(base), auc=float(auc(y, p)), logloss=ll,
                info_frac=1 - ll / ll0,                   # share of the label's entropy explained
                brier=float(np.mean((p - y) ** 2)), seeds_gt90=float((p[y] > 0.9).mean()) if y.any() else np.nan)


def hazard(dtr, ytr, dte):
    h = {dd: ytr[dtr == dd].mean() if (dtr == dd).any() else ytr.mean() for dd in range(1, MAX_D + 2)}
    return np.array([h[x] for x in dte])


def main():
    import pandas as pd
    t0 = time.time()
    data = {tw: samples(POOL, tw) for tw in sims(POOL)}
    print(f"{sum(len(v[1]) for v in data.values())} samples from {len(data)} sims ({time.time() - t0:.0f} s)", flush=True)
    cat = lambda keys, i: np.concatenate([data[k][i] for k in keys])

    # ---- per output frame k, evaluation sims held out
    tr = [k for k in data if k not in SWEEP]; te = [k for k in data if k in SWEEP]
    Xtr, dtr, Ytr = (cat(tr, i) for i in range(3)); Xte, dte, Yte = (cat(te, i) for i in range(3))
    rows = []
    for k in range(K):
        ytr, yte = Ytr[:, k], Yte[:, k]
        for name, p in (("hazard", hazard(dtr, ytr, dte)),
                        ("cnn", fit(Xtr, ytr, Xte, onehot(dtr), onehot(dte)))):
            r = dict(output_frame=k + 1, model=name, n=len(yte), **metrics(yte, p))
            for dd in (3, 4):
                sel = dte == dd
                r[f"auc_d{dd}"] = float(auc(yte[sel], p[sel])) if sel.sum() > 10 else np.nan
            rows.append(r)
            print(f"  frame {k + 1} {name:>6}: pos {r['pos_rate']:.3f} | AUC {r['auc']:.3f} "
                  f"(d=3 {r['auc_d3']:.3f}, d=4 {r['auc_d4']:.3f}) | info {r['info_frac']:.2f} "
                  f"logloss {r['logloss']:.3f} | seeds >0.9 {r['seeds_gt90']:.2f}", flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)

    # ---- output frame 1, 5-fold CV grouped by simulation
    rng = np.random.default_rng(0)
    keys = list(rng.permutation(sorted(data, key=float)))
    cv = []
    for fold in range(5):
        te = keys[fold::5]; tr = [k for k in keys if k not in te]
        Xtr, dtr, Ytr = (cat(tr, i) for i in range(3)); Xte, dte, Yte = (cat(te, i) for i in range(3))
        for name, p in (("hazard", hazard(dtr, Ytr[:, 0], dte)),
                        ("cnn", fit(Xtr, Ytr[:, 0], Xte, onehot(dtr), onehot(dte)))):
            r = dict(fold=fold, test_sims=" ".join(sorted(te, key=float)), model=name, n=len(dte),
                     **metrics(Yte[:, 0], p))
            cv.append(r)
            print(f"  CV fold {fold} [{r['test_sims']}] {name:>6}: AUC {r['auc']:.3f} "
                  f"logloss {r['logloss']:.3f}", flush=True)
    cvd = pd.DataFrame(cv)
    cvd.to_csv(OUT_CV, index=False)
    print(cvd.groupby("model")[["auc", "logloss", "brier"]].agg(["mean", "min", "max"]).round(3).to_string())
    print(f"\nwrote {OUT}\nwrote {OUT_CV}")


if __name__ == "__main__":
    main()
