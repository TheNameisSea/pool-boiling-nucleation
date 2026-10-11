#!/usr/bin/env python3
"""Can the frame of the next seed be read from the forecaster's input window? (premise audit, Fig. 1c)

Sample: every (sim, site, frame f) with the site dry in the input's last frame f-1 and a known delay
d = f - (frame the site last freed), reference data, raw frames [100, 400).
  input : the TRUE 5-frame window f-5..f-1, all 4 model fields (dfun, temperature, velx, vely),
          a local patch rows [0, 16) x cols site ±12
  label : a seed appears at the site in frame f, counted or merged (merged-seed rule as reoccupation.py)
Models, trained on the 14 pool simulations outside the evaluation set, scored on the 6 evaluation sims:
  hazard      : P(seed | d), the delay-only baseline
  hazard+tw   : logistic regression on one-hot d + Twall
  cnn         : small CNN on the patch + one-hot d
  cnn_s1/s2   : the CNN with two more seeds; cnn_dfun / cnn_temp / cnn_vel: one field group only
Second task, on seeds only: will the seed merge with existing vapour on arrival? (cnn vs base rate)
Scored by log loss, Brier and AUC on held-out sims (also per delay), and by the share of true seeds
called with p > 0.9. The single-bubble study (6 FC-72 sims, split by sim 4/2) is run the same way.
Writes timing_readability.csv and the held-out predictions used by fig1_premise_audit.py.

Needs all 20 pool-boiling simulations. Run from the repository root:
    python scripts/timing_readability.py
"""
import os, sys, time

import h5py
import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage

sys.path.insert(0, "scripts")
from birth_predictability import PATCH_HALF, PATCH_ROWS, counter_births_on_site, meta, transitions
from diagnostics import DX
from paths import POOL, SINGLE
ALIGN, FRAMES, WIN = 100, 300, 5
FIELDS = ("dfun", "temperature", "velx", "vely")
P_ROWS, P_HALF = 16, 12
SEED_ROWS, SEED_HALF, SEED_MIN = 4, 3, 10          # merged-seed rule, as reoccupation.py
MAX_D = 8
SWEEP = {"75", "82", "90", "96", "102", "107"}
OUT = "results/diagnostics/timing_readability.csv"
OUT_PRED = "results/diagnostics/timing_readability_preds_{study}.npz"
torch.set_num_threads(int(os.getenv("SLURM_CPUS_PER_TASK", "8")))


def sims(root):
    return sorted(f[6:-5] for f in os.listdir(root) if f.startswith("Twall_") and f.endswith(".hdf5"))


def samples(root, tw, with_ids=False):
    """X (n, 20, P_ROWS, 2*P_HALF+1), d (n,), seed (n,), merged (n,), twall[, ids (n, 2) = (site, f)].

    f indexes frames from raw ALIGN - WIN (f = WIN is raw frame ALIGN)."""
    m = meta(os.path.join(root, f"Twall_{tw}.json"))
    cols = np.round((m["sites"] - m["x_min"]) / DX - 0.5).astype(int)
    lo = ALIGN - WIN                                   # need inputs from f-5 for f = ALIGN
    with h5py.File(os.path.join(root, f"Twall_{tw}.hdf5"), "r") as h:
        dfun = h["dfun"][lo:ALIGN + FRAMES].astype(np.float32)          # full frames: the counter
        band = {k: h[k][lo:ALIGN + FRAMES, :P_ROWS].astype(np.float32) for k in FIELDS[1:]}
    band["dfun"] = dfun[:, :P_ROWS]
    W = dfun.shape[2]
    vap = dfun > 0
    # counter's on-site births; events index into dfun[WIN-1:] (= raw ALIGN-1 ...)
    _, _, ev = counter_births_on_site(dfun[WIN - 1:], cols)
    counted = {(t + WIN - 1, s) for t, s in ev}
    occ = np.stack([vap[:, :PATCH_ROWS, max(0, c - PATCH_HALF):min(W, c + PATCH_HALF + 1)].any(axis=(1, 2))
                    for c in cols], axis=1)
    padded = {k: np.pad(v, ((0, 0), (0, 0), (P_HALF, P_HALF)), mode="edge") for k, v in band.items()}
    X, D, Y, M, I = [], [], [], [], []
    for s, c in enumerate(cols):
        _, deps = transitions(occ[:, s])
        for f in range(WIN, dfun.shape[0]):            # f: the predicted frame (raw lo + f)
            if occ[f - 1, s]:
                continue
            prev = deps[deps <= f - 1]
            if not prev.size:
                continue
            seed = merged = False
            if occ[f, s]:
                l2, h2 = max(0, c - SEED_HALF), min(W, c + SEED_HALF + 1)
                new = vap[f, :SEED_ROWS, l2:h2] & ~vap[f - 1, :SEED_ROWS, l2:h2]
                if (f, s) in counted:
                    seed = True
                elif vap[f, 0, c] and not vap[f - 1, 0, c] and new.sum() >= SEED_MIN:
                    seed = merged = True
            X.append(np.concatenate([padded[k][f - WIN:f, :, c:c + 2 * P_HALF + 1] for k in FIELDS]))
            D.append(min(int(f - prev[-1]), MAX_D + 1)); Y.append(seed); M.append(merged); I.append((s, f))
    n = len(D)
    out = (np.stack(X) if n else np.zeros((0, 4 * WIN, P_ROWS, 2 * P_HALF + 1), np.float32),
           np.array(D), np.array(Y), np.array(M), m["twall"])
    return out + (np.array(I, dtype=int).reshape(-1, 2),) if with_ids else out


class Net(nn.Module):
    def __init__(self, cin, nd):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(cin, 32, 3, padding=1), nn.GELU(),
                                  nn.Conv2d(32, 32, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),
                                  nn.Conv2d(32, 64, 3, padding=1), nn.GELU(), nn.AdaptiveAvgPool2d((2, 3)))
        self.head = nn.Sequential(nn.Linear(64 * 6 + nd, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x, d):
        return self.head(torch.cat([self.conv(x).flatten(1), d], 1)).squeeze(1)


def onehot(d):
    return np.eye(MAX_D + 1, dtype=np.float32)[np.clip(d, 1, MAX_D + 1) - 1]


def fit(Xtr, ytr, Xte, extra_tr, extra_te, epochs=30, seed=0):
    """CNN (if Xtr is not None) or logistic on the extras alone. Early-stops on 15 % of train."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    n = len(ytr); idx = rng.permutation(n); nv = max(1, int(0.15 * n))
    va, tr = idx[:nv], idx[nv:]
    if Xtr is not None:
        mu = Xtr[tr].mean(axis=(0, 2, 3), keepdims=True); sd = Xtr[tr].std(axis=(0, 2, 3), keepdims=True) + 1e-6
        norm = lambda a: torch.from_numpy((a - mu) / sd)
        net = Net(Xtr.shape[1], extra_tr.shape[1])
        fwd = lambda X, E: net(X, E)
    else:
        net = nn.Linear(extra_tr.shape[1], 1)
        fwd = lambda X, E: net(E).squeeze(1)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    Etr, Ete = torch.from_numpy(extra_tr), torch.from_numpy(extra_te)
    Y = torch.from_numpy(ytr.astype(np.float32))
    lossf = nn.BCEWithLogitsLoss()
    get = lambda X, i: norm(X[i]) if X is not None else None

    def predict(X, E, ids):
        net.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(ids), 4096):
                j = ids[i:i + 4096]
                out.append(torch.sigmoid(fwd(get(X, j), E[j])))
        net.train(); return torch.cat(out).numpy()

    best, best_state, bad = np.inf, None, 0
    for ep in range(epochs):
        for i in range(0, len(tr), 256):
            j = tr[rng.integers(0, len(tr), 256)]
            loss = lossf(fwd(get(Xtr, j), Etr[j]), Y[j])
            opt.zero_grad(); loss.backward(); opt.step()
        p = np.clip(predict(Xtr, Etr, va), 1e-6, 1 - 1e-6)
        vl = -np.mean(ytr[va] * np.log(p) + (1 - ytr[va]) * np.log(1 - p))
        if vl < best - 1e-4:
            best, bad = vl, 0; best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 5:
                break
    net.load_state_dict(best_state)
    return predict(Xte, Ete, np.arange(len(extra_te)))


def auc(y, p):
    y = y.astype(bool)
    if y.all() or (~y).all():
        return np.nan
    r = np.argsort(np.argsort(p)) + 1.0
    return (r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())


def score(name, task, study, y, p, d):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    row = dict(study=study, task=task, model=name, n=len(y), pos_rate=float(y.mean()),
               logloss=float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
               brier=float(np.mean((p - y) ** 2)), auc=float(auc(y, p)),
               pos_called_gt90=float((p[y] > 0.9).mean()) if y.any() else np.nan,
               neg_called_lt10=float((p[~y] < 0.1).mean()) if (~y).any() else np.nan)
    for dd in (2, 3, 4, 5):                             # per-delay: can it split the jittered cases?
        sel = d == dd
        row[f"auc_d{dd}"] = float(auc(y[sel], p[sel])) if sel.sum() > 10 else np.nan
        row[f"brier_d{dd}"] = float(np.mean((p[sel] - y[sel]) ** 2)) if sel.any() else np.nan
    print(f"  {study:>6} {task:>6} {name:>9}: n {row['n']:>6} pos {row['pos_rate']:.3f} | logloss {row['logloss']:.4f} "
          f"brier {row['brier']:.4f} auc {row['auc']:.3f} | seeds called >0.9 {row['pos_called_gt90']:.2f} | "
          f"auc at d=3 {row['auc_d3']:.3f} d=4 {row['auc_d4']:.3f}", flush=True)
    return row


def run(study, root, test_sims):
    t0 = time.time()
    data = {tw: samples(root, tw) for tw in sims(root)}
    print(f"\n== {study}: {sum(len(v[1]) for v in data.values())} samples from {len(data)} sims "
          f"({time.time() - t0:.0f} s)", flush=True)
    cat = lambda keys, i: np.concatenate([data[k][i] for k in keys])
    tr_k = [k for k in data if k not in test_sims]; te_k = [k for k in data if k in test_sims]
    Xtr, dtr, ytr, mtr = (cat(tr_k, i) for i in range(4)); Xte, dte, yte, mte = (cat(te_k, i) for i in range(4))
    twtr = np.concatenate([np.full(len(data[k][1]), data[k][4], np.float32) for k in tr_k])
    twte = np.concatenate([np.full(len(data[k][1]), data[k][4], np.float32) for k in te_k])
    rows = []
    # hazard: P(seed | d) from the training sims
    haz = {dd: ytr[dtr == dd].mean() if (dtr == dd).any() else ytr.mean() for dd in range(1, MAX_D + 2)}
    p_haz = np.array([haz[x] for x in dte])
    rows.append(score("hazard", "seed", study, yte, p_haz, dte))
    z = lambda a: ((a - twtr.mean()) / (twtr.std() + 1e-6))[:, None].astype(np.float32)
    Etr = np.concatenate([onehot(dtr), z(twtr)], 1); Ete = np.concatenate([onehot(dte), z(twte)], 1)
    rows.append(score("hazard+tw", "seed", study, yte, fit(None, ytr, None, Etr, Ete, epochs=60), dte))
    p_cnn = fit(Xtr, ytr, Xte, onehot(dtr), onehot(dte))
    rows.append(score("cnn", "seed", study, yte, p_cnn, dte))
    # held-out predictions, for the Fig. 1c ROC panel (scripts/fig1_premise_audit.py)
    np.savez_compressed(OUT_PRED.format(study=study), y=yte, d=dte, p_hazard=p_haz, p_cnn=p_cnn)
    if study == "pool":
        # two more seeds, and a field ablation (one field group at a time)
        for sd in (1, 2):
            rows.append(score(f"cnn_s{sd}", "seed", study, yte,
                              fit(Xtr, ytr, Xte, onehot(dtr), onehot(dte), seed=sd), dte))
        for name, fi in (("cnn_dfun", [0]), ("cnn_temp", [1]), ("cnn_vel", [2, 3])):
            ch = [f * WIN + w for f in fi for w in range(WIN)]          # X channels are field-major
            rows.append(score(name, "seed", study, yte,
                              fit(Xtr[:, ch], ytr, Xte[:, ch], onehot(dtr), onehot(dte)), dte))
    # merge-on-arrival, seeds only
    if mtr[ytr].any() and mte[yte].any():
        rows.append(score("base_rate", "merge", study, mte[yte], np.full(yte.sum(), mtr[ytr].mean()), dte[yte]))
        rows.append(score("cnn", "merge", study, mte[yte],
                          fit(Xtr[ytr], mtr[ytr], Xte[yte], onehot(dtr[ytr]), onehot(dte[yte])), dte[yte]))
    # calibration of the CNN by delay: mean p among true seeds vs non-seeds
    for dd in range(1, MAX_D + 2):
        sel = dte == dd
        if sel.sum() and yte[sel].any():
            print(f"    d={dd}: n {sel.sum():>5} hazard {haz[dd]:.2f} | cnn mean p on seeds {p_cnn[sel & yte].mean():.2f} "
                  f"non-seeds {p_cnn[sel & ~yte].mean() if (sel & ~yte).any() else np.nan:.2f}", flush=True)
    return rows


def main():
    import pandas as pd
    rows = run("single", SINGLE, {"98", "100"})
    rows += run("pool", POOL, SWEEP)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
