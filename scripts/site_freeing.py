#!/usr/bin/env python3
"""Site freeing and reseeding in autoregressive rollouts (Fig. 3; the rollout reseeding of Section 6.6).

Measured per site on the same patch as `birth_predictability.py` (first 3 wall rows, ±3 cells), for
the reference and each model's saved rollout over the same frames [100, 400):
  duty      fraction of frames a site is occupied
  frees     site-freeing events (occupied -> dry), per site per 100 frames; also debounced
            (dry for >= 2 consecutive frames), since the reference flickers at the wall
  births    the counter's births on a listed site (min_area 16), per site per 100 frames
  respond   P(a counter birth at that site within 1-5 frames | the site freed, debounced)

Default sources: runs 716 (conditioned AViT) and 735 (AViT) at Twall 75/82/102/107, and the
single-bubble control model 778 at its six sims -> site_freeing.csv.
With run ids as arguments, measures the rollouts saved by `rollout_diagnostics.py --save-fields
$WORK_ROOT/rollout_fields_<run>` at all six evaluation sims -> site_freeing_rollouts.csv.

Run from the repository root (~32 GB RAM):
    python scripts/site_freeing.py
    python scripts/site_freeing.py 952 1055 1077 1089 1184 1070 1181 1200 1194
"""
import os, sys

import h5py
import numpy as np

sys.path.insert(0, "scripts")
from birth_predictability import PATCH_HALF, PATCH_ROWS, counter_births_on_site, meta, transitions
from diagnostics import DX
from paths import POOL, SINGLE, WORK_ROOT as F

ALIGN, FRAMES, RESPOND = 100, 300, 5
CASES = ([("pool", POOL, tw, {"716 cond": f"{F}/rollout_fields_716", "735 uncond": f"{F}/rollout_fields_735"})
          for tw in ("75", "82", "102", "107")]
         + [("single", SINGLE, tw, {"778 control": f"{F}/rollout_fields_778"})
            for tw in ("90", "92", "94", "96", "98", "100")])
OUT = "results/diagnostics/site_freeing.csv"


def rollout_cases(jobs):
    """Cases for the rollouts saved under $WORK_ROOT/rollout_fields_<run>, at all six evaluation sims."""
    return [("pool", POOL, tw, {j: f"{F}/rollout_fields_{j}" for j in jobs
                                if os.path.exists(f"{F}/rollout_fields_{j}/dfun_pred_Twall_{tw}.npy")}
             | ({"735 uncond": f"{F}/rollout_fields_735"} if tw in ("75", "82", "102", "107") else {}))
            for tw in ("75", "82", "90", "96", "102", "107")]


def occupancy(phi, cols):
    wall = phi[:, :PATCH_ROWS, :] > 0
    W = wall.shape[2]
    return np.stack([wall[:, :, max(0, c - PATCH_HALF):min(W, c + PATCH_HALF + 1)].any(axis=(1, 2))
                     for c in cols], axis=1)                      # (T, n_sites)


def debounced_frees(occ_k):
    """Occupied -> dry transitions where the site then stays dry for >= 2 frames."""
    _, d = transitions(occ_k)
    return np.array([t for t in d if t + 1 < len(occ_k) and not occ_k[t + 1]], dtype=int)


def measure(phi, cols):
    T, n = phi.shape[0], len(cols)
    occ = occupancy(phi, cols)
    _, _, events = counter_births_on_site(phi, cols)
    births = {}
    for t, k in events:
        births.setdefault(k, []).append(t)
    raw = sum(len(transitions(occ[:, k])[1]) for k in range(n))
    deb_all, resp = [], 0
    for k in range(n):
        fr = debounced_frees(occ[:, k])
        deb_all.append(fr)
        bk = np.array(births.get(k, []), dtype=int)
        resp += sum(bool(np.any((bk > t) & (bk <= t + RESPOND))) for t in fr)
    n_deb = int(sum(len(f) for f in deb_all))
    per100 = 100.0 / (T * n)
    return dict(duty=float(occ.mean()), frees_raw=raw * per100, frees_deb=n_deb * per100,
                births=len(events) * per100, n_births=len(events), n_frees_deb=n_deb,
                respond=resp / n_deb if n_deb else float("nan"),
                sites_never_freed=int(sum(len(f) == 0 for f in deb_all)))


def main():
    import pandas as pd
    cases, out = CASES, OUT
    if len(sys.argv) > 1:
        cases, out = rollout_cases(sys.argv[1:]), "results/diagnostics/site_freeing_rollouts.csv"
    rows = []
    for study, root, tw, models in cases:
        m = meta(os.path.join(root, f"Twall_{tw}.json"))
        cols = np.round((m["sites"] - m["x_min"]) / DX - 0.5).astype(int)
        with h5py.File(os.path.join(root, f"Twall_{tw}.hdf5"), "r") as h:
            ref = h["dfun"][ALIGN:ALIGN + FRAMES].astype(np.float32)
        srcs = [("reference", ref)] + [(name, np.load(os.path.join(d, f"dfun_pred_Twall_{tw}.npy")))
                                       for name, d in models.items()]
        print(f"\n{study} Twall_{tw}  ({len(cols)} sites, {FRAMES} frames from {ALIGN})")
        print(f"{'source':>12} {'duty':>6} {'frees/site/100f':>16} {'debounced':>10} "
              f"{'births/site/100f':>17} {'P(birth<=5f|freed)':>19} {'sites never freed':>18}")
        for name, phi in srcs:
            assert phi.shape == ref.shape, (name, phi.shape, ref.shape)
            r = measure(phi, cols)
            rows.append(dict(study=study, twall=float(tw), n_sites=len(cols), source=name, **r))
            print(f"{name:>12} {r['duty']:>6.3f} {r['frees_raw']:>16.2f} {r['frees_deb']:>10.2f} "
                  f"{r['births']:>17.2f} {r['respond']:>19.2f} "
                  f"{r['sites_never_freed']:>11} / {len(cols):<3}", flush=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
