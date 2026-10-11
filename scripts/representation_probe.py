#!/usr/bin/env python3
"""Linear probe of the forecaster's internal features for the next-frame seed cue (Section 6.3).

Stage 1, --extract (GPU): run the AViT one step from true inputs over all 20 pool sims (frames
[100, 400), stride 1, as `rollout_diagnostics.py --teacher-forced`). Hook the patch embedding and
blocks 4, 8 and 12, and keep the wall row of tokens (patch row 0 = image rows 0-15, the wall band)
for all 5 time slots: (300, 4 layers, 5 slots, 32 cols, emb) float16 per sim. The max φ of the
model's one-step prediction over each site's seed patch is kept too.

Stage 2, --probe (CPU): the same site-frame samples and labels as timing_readability.py (site dry in
the last input frame, known delay d; label = a seed, counted or merged, in the first predicted frame).
Train on the 14 simulations outside the evaluation set, test on the 6 evaluation sims. AUC and log loss for:
  hazard                     P(seed | d), the delay-only baseline
  probe <layer> slot0 / all  logistic regression on the site's token (time slot 0, or all 5 slots
                             concatenated) + one-hot d + the site's offset inside its patch
  MLP probe <layer> all      two-layer MLP on the same features (embedding and last block)
  model output               the model's own one-step prediction at the site as a monotone score:
                             φ at the site's wall cells and its change from the last input frame
                             (from tf_fields_<run>); max φ over the seed patch is also reported but
                             includes neighbouring vapour.

Run from the repository root (needs all 20 pool-boiling simulations):
    python scripts/representation_probe.py --extract --job 735 --checkpoint <ckpt>
    python scripts/representation_probe.py --probe --job 735
"""
import argparse, json, os, sys, time

import h5py
import numpy as np
import torch

sys.path.insert(0, "scripts")
sys.path.insert(0, "external/Bubbleformer")
from birth_predictability import PATCH_HALF, PATCH_ROWS, counter_births_on_site, meta, transitions
from diagnostics import DX
from timing_readability import (ALIGN, FRAMES, MAX_D, POOL, SEED_HALF, SEED_MIN, SEED_ROWS, SWEEP, WIN,
                                auc, onehot, sims)
from paths import WORK_ROOT as F
LAYERS = ("embed", "block4", "block8", "block12")
OUT = "results/diagnostics/representation_probe_{job}.csv"
PATCH = 16
MODEL_SEED_ROWS, MODEL_SEED_HALF = 5, 4                 # seed patch for the max-phi output score


def feat_dir(job):
    return f"{F}/probe_feats_{job}"


# ------------------------------------------------------------------ stage 1: extract (GPU)
def extract(job, ckpt_path, device):
    from inference import build_model
    from bubbleformer.data import BubbleForecast
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model, name, hp = build_model(ckpt)
    assert name == "avit", f"probe hooks are written for the unconditioned AViT, got {name}"
    model = model.to(device).eval()
    data_cfg = hp["data_cfg"]
    grabbed = {}

    def hook(key):
        def f(_m, _i, out):                             # (b, t, emb, h, w) for blocks, ((b t), emb, h, w) for embed
            o = out if out.dim() == 5 else out.unsqueeze(0)
            grabbed[key] = o[0, :, :, 0, :].permute(0, 2, 1).half().cpu()   # (t, w, emb): wall token row
        return f
    model.embed.register_forward_hook(hook("embed"))
    for i, key in ((3, "block4"), (7, "block8"), (11, "block12")):
        model.blocks[i].register_forward_hook(hook(key))

    os.makedirs(feat_dir(job), exist_ok=True)
    for tw in sims(POOL):
        path = f"{POOL}/Twall_{tw}.hdf5"
        m = meta(f"{POOL}/Twall_{tw}.json")
        cols = np.round((m["sites"] - m["x_min"]) / DX - 0.5).astype(int)
        ds = BubbleForecast(filenames=[path], input_fields=list(data_cfg["input_fields"]),
                            output_fields=list(data_cfg["output_fields"]), norm=data_cfg["normalize"],
                            downsample_factor=data_cfg["downsample_factor"],
                            time_window=data_cfg["time_window"], start_time=ALIGN - WIN,
                            return_fluid_params=False)
        ds.normalize(*hp["normalization_constants"])
        assert data_cfg["normalize"] == "none", "model-output score below assumes physical units"
        feats = np.zeros((FRAMES, len(LAYERS), WIN, 512 // PATCH, 384), np.float16)
        outmax = np.zeros((FRAMES, len(cols)), np.float32)
        t0 = time.time()
        with torch.no_grad():
            for s in range(FRAMES):
                inp, _ = ds[s]
                pred = model(inp.to(device).float().unsqueeze(0))
                # the embed hook fires on (b t) frames; its t axis holds the 5 INPUT frames
                for li, key in enumerate(LAYERS):
                    feats[s, li] = grabbed[key].numpy()
                phi = pred[0, 0, 0, :MODEL_SEED_ROWS].cpu().numpy()          # first predicted frame, SDF
                outmax[s] = [phi[:, max(0, c - MODEL_SEED_HALF):c + MODEL_SEED_HALF + 1].max() for c in cols]
        np.save(f"{feat_dir(job)}/feats_Twall_{tw}.npy", feats)
        np.save(f"{feat_dir(job)}/outmax_Twall_{tw}.npy", outmax)
        print(f"  Twall_{tw}: {len(cols)} sites, feats {feats.shape} [{time.time() - t0:.0f} s]", flush=True)


# ------------------------------------------------------------------ stage 2: probe (CPU)
def labels(tw):
    """(s index into the 300 predicted frames, site, col, delay, seed) for timing_readability.py's samples."""
    m = meta(f"{POOL}/Twall_{tw}.json")
    cols = np.round((m["sites"] - m["x_min"]) / DX - 0.5).astype(int)
    lo = ALIGN - WIN
    with h5py.File(f"{POOL}/Twall_{tw}.hdf5", "r") as h:
        dfun = h["dfun"][lo:ALIGN + FRAMES].astype(np.float32)
    W = dfun.shape[2]
    vap = dfun > 0
    _, _, ev = counter_births_on_site(dfun[WIN - 1:], cols)
    counted = {(t + WIN - 1, s) for t, s in ev}
    occ = np.stack([vap[:, :PATCH_ROWS, max(0, c - PATCH_HALF):min(W, c + PATCH_HALF + 1)].any(axis=(1, 2))
                    for c in cols], axis=1)
    rows = []
    for s, c in enumerate(cols):
        _, deps = transitions(occ[:, s])
        l2, h2 = max(0, c - SEED_HALF), min(W, c + SEED_HALF + 1)
        for f in range(WIN, dfun.shape[0]):
            if occ[f - 1, s]:
                continue
            prev = deps[deps <= f - 1]
            if not prev.size:
                continue
            seed = False
            if occ[f, s]:
                new = vap[f, :SEED_ROWS, l2:h2] & ~vap[f - 1, :SEED_ROWS, l2:h2]
                seed = (f, s) in counted or (vap[f, 0, c] and not vap[f - 1, 0, c] and new.sum() >= SEED_MIN)
            rows.append((f - WIN, s, c, min(int(f - prev[-1]), MAX_D + 1), seed))
    return np.array(rows, dtype=np.int64)


def logistic(Xtr, ytr, Xte, wd=1e-3, epochs=200):
    """L2-regularised logistic regression, full-batch LBFGS on standardised features."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    A = torch.from_numpy(((Xtr - mu) / sd).astype(np.float32))
    B = torch.from_numpy(((Xte - mu) / sd).astype(np.float32))
    y = torch.from_numpy(ytr.astype(np.float32))
    lin = torch.nn.Linear(A.shape[1], 1)
    opt = torch.optim.LBFGS(lin.parameters(), lr=0.5, max_iter=epochs, history_size=20,
                            line_search_fn="strong_wolfe")
    lossf = torch.nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        l = lossf(lin(A).squeeze(1), y) + wd * lin.weight.pow(2).sum()
        l.backward()
        return l
    opt.step(closure)
    with torch.no_grad():
        return torch.sigmoid(lin(B).squeeze(1)).numpy()


def mlp(Xtr, ytr, Xte, seed=0, epochs=60):
    """Nonlinear probe: 2-layer MLP, early-stopped on 15 % of train."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    A = torch.from_numpy(((Xtr - mu) / sd).astype(np.float32)); B = torch.from_numpy(((Xte - mu) / sd).astype(np.float32))
    y = torch.from_numpy(ytr.astype(np.float32))
    idx = rng.permutation(len(y)); nv = int(0.15 * len(y)); va, tr = idx[:nv], idx[nv:]
    net = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(A.shape[1], 256), torch.nn.GELU(),
                              torch.nn.Dropout(0.2), torch.nn.Linear(256, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    lossf = torch.nn.BCEWithLogitsLoss(); best, state, bad = np.inf, None, 0
    for _ in range(epochs):
        net.train()
        for i in range(0, len(tr), 256):
            j = tr[rng.integers(0, len(tr), 256)]
            l = lossf(net(A[j]).squeeze(1), y[j]); opt.zero_grad(); l.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl = lossf(net(A[va]).squeeze(1), y[va]).item()
        if vl < best - 1e-4:
            best, bad, state = vl, 0, {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 6:
                break
    net.load_state_dict(state); net.eval()
    with torch.no_grad():
        return torch.sigmoid(net(B).squeeze(1)).numpy()


def probe(job):
    import pandas as pd
    torch.set_num_threads(int(os.getenv("SLURM_CPUS_PER_TASK", "8")))
    per = {}
    for tw in sims(POOL):
        L = labels(tw)
        fe = np.load(f"{feat_dir(job)}/feats_Twall_{tw}.npy", mmap_mode="r")
        om = np.load(f"{feat_dir(job)}/outmax_Twall_{tw}.npy")
        s, site, col, d, y = L.T
        tok = fe[s, :, :, col // PATCH].astype(np.float32)      # (n, layers, 5 slots, emb)
        off = ((col % PATCH) / PATCH - 0.5)[:, None].astype(np.float32)
        per[tw] = dict(tok=tok, off=off, d=d, y=y.astype(bool), out=om[s, site], L=L)
        print(f"  Twall_{tw}: {len(y)} samples, {y.mean():.3f} seeds", flush=True)
    tr = [k for k in per if k not in SWEEP]; te = [k for k in per if k in SWEEP]
    cat = lambda keys, f: np.concatenate([per[k][f] for k in keys])
    ytr, yte, dtr, dte = cat(tr, "y"), cat(te, "y"), cat(tr, "d"), cat(te, "d")
    Ttr, Tte = cat(tr, "tok"), cat(te, "tok")
    base_tr = np.concatenate([onehot(dtr), cat(tr, "off")], 1)
    base_te = np.concatenate([onehot(dte), cat(te, "off")], 1)
    rows = []

    def add(name, p):
        pc = np.clip(p, 1e-6, 1 - 1e-6)
        r = dict(job=job, model=name, n=len(yte), pos_rate=float(yte.mean()), auc=float(auc(yte, p)),
                 logloss=float(-np.mean(yte * np.log(pc) + (1 - yte) * np.log(1 - pc))),
                 auc_d3=float(auc(yte[dte == 3], p[dte == 3])), auc_d4=float(auc(yte[dte == 4], p[dte == 4])))
        rows.append(r)
        print(f"  {name:>22}: AUC {r['auc']:.3f} (d=3 {r['auc_d3']:.3f}, d=4 {r['auc_d4']:.3f})  "
              f"logloss {r['logloss']:.3f}", flush=True)

    haz = {dd: ytr[dtr == dd].mean() if (dtr == dd).any() else ytr.mean() for dd in range(1, MAX_D + 2)}
    add("hazard", np.array([haz[x] for x in dte]))
    out = cat(te, "out")
    add("model output (max phi)", 1 / (1 + np.exp(-out / 0.05)))   # monotone in max φ; AUC only
    # Site-centred scores from the saved one-step fields (`tf_fields_<job>`, evaluation sims only):
    # φ at the site's wall cells (rows 0-2, cols c-1..c+1), and its change from the last input frame.
    tfd = f"{F}/tf_fields_{job}"
    if all(os.path.exists(f"{tfd}/dfun_tf_first_Twall_{tw}.npy") for tw in te):
        at, dl = [], []
        for tw in te:
            tf = np.load(f"{tfd}/dfun_tf_first_Twall_{tw}.npy", mmap_mode="r")    # tf[s] predicts raw 100+s
            with h5py.File(f"{POOL}/Twall_{tw}.hdf5", "r") as h:
                last = h["dfun"][ALIGN - 1:ALIGN + FRAMES - 1, :3].astype(np.float32)  # raw 99+s
            L = per[tw]["L"]
            for s_, c in zip(L[:, 0], L[:, 2]):
                sl = slice(max(0, c - 1), c + 2)
                v = float(tf[s_, :3, sl].mean())
                at.append(v); dl.append(v - float(last[s_, :, sl].mean()))
        sig = lambda x: 1 / (1 + np.exp(-np.asarray(x) / 0.05))
        add("model output (site phi)", sig(at))                          # monotone; AUC only
        add("model output (site dphi)", sig(dl))
    for li, lname in enumerate(LAYERS):
        add(f"probe {lname} slot0", logistic(np.concatenate([Ttr[:, li, 0], base_tr], 1), ytr,
                                              np.concatenate([Tte[:, li, 0], base_te], 1)))
        add(f"probe {lname} all", logistic(np.concatenate([Ttr[:, li].reshape(len(ytr), -1), base_tr], 1), ytr,
                                            np.concatenate([Tte[:, li].reshape(len(yte), -1), base_te], 1)))
    for li in (0, 3):                                   # nonlinear probe: embedding and last block
        Xa = np.concatenate([Ttr[:, li].reshape(len(ytr), -1), base_tr], 1)
        Xb = np.concatenate([Tte[:, li].reshape(len(yte), -1), base_te], 1)
        add(f"MLP probe {LAYERS[li]} all", mlp(Xa, ytr, Xb))
    pd.DataFrame(rows).to_csv(OUT.format(job=job), index=False)
    print(f"\nwrote {OUT.format(job=job)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--job", required=True)
    ap.add_argument("--checkpoint")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if a.extract:
        extract(a.job, a.checkpoint, a.device)
    if a.probe:
        probe(a.job)


if __name__ == "__main__":
    main()
