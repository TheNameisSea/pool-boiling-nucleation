#!/usr/bin/env python3
"""Premise audit, reference data only: where and when do seeds appear (Table 1 of the paper)?

Flash-X re-seeds a bubble of radius `nuc_seed_radii` at a site `nucWaitTime` after the site becomes
free. Measured per simulation, on reference data (no model involved):

  A. Per-site occupancy time series (vapour in the first 3 wall rows within ±3 cells of the site).
     Births (0->1) and departures (1->0) give the dry-period and wet-period distributions per site.
  B. The nucleation counter's births on frames [100, 400) and how many land on a listed site.
  C. A clock-rule forecaster using only the last 5 frames: predict a birth at site k in frame t+1 iff
     the site is dry at t and its last departure (inside the window) was D frames before t+1, D within
     ±1 of the modal dry period.
  D. A and C again, scored against the counter's births only (the occupancy series also fires on
     wall flicker): the delay from site freeing to seed (`cb_delay_*`) and the share inside the window.

Also provides helpers used by the other analysis scripts (`meta`, `transitions`,
`counter_births_on_site`, PATCH_ROWS, PATCH_HALF, WALL_BAND).

Run from the repository root (reads full frames; ~48 GB RAM):
    python scripts/birth_predictability.py [--all]
"""
import glob, json, os, sys
from collections import Counter

import h5py
import numpy as np
from scipy import ndimage

sys.path.insert(0, "scripts")
from diagnostics import DX, MIN_AREA_CELLS, count_nucleation_events
from paths import POOL, SINGLE

STUDIES = {"pool": POOL, "single": SINGLE}
START = 100            # skip the initial transient, as every other measurement does
WINDOW = 5             # the model's input window, in frames
PATCH_ROWS = 3         # occupancy = vapour in the first 3 rows (seed radius 0.1 = 3.2 cells)
PATCH_HALF = 3         # ... within +-3 columns of the site
VALID_FRAMES = 300     # measurement B uses the reference window [100, 400)
WALL_BAND = 0.5        # same band as count_nucleation_events
OUT = "results/diagnostics/birth_predictability.csv"
BLOCK = 100            # frames per sequential read
# Default: the six pool-boiling evaluation sims and the six single-bubble control sims.
# --all runs every sim in both studies.
SUBSET = {"pool": {75, 82, 90, 96, 102, 107}, "single": {90, 92, 94, 96, 98, 100}}


def flat(d, p=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flat(v, f"{p}.{k}"))
    else:
        out[p] = d
    return out


def meta(json_path):
    F = flat(json.load(open(json_path)))

    def get(suffix, default=None):
        return next((v for k, v in F.items() if k.endswith(suffix)), default)

    return dict(sites=np.atleast_1d(np.array(get("nuc_sites_x"), float)),
                wait=float(get("nucWaitTime", np.nan)), dt=float(get(".dt", np.nan)),
                x_min=float(get("x_min")), twall=float(get("wallTemp")))


def transitions(occ):
    """Births (0->1) and departures (1->0), as the frame index of the new state."""
    d = np.diff(occ.astype(np.int8))
    return np.where(d == 1)[0] + 1, np.where(d == -1)[0] + 1


def counter_births_on_site(full, cols):
    """Replay count_nucleation_events' rule and record where each birth touches the wall."""
    rows = max(1, int(round(WALL_BAND / DX)))
    n_birth = on_site = 0
    events = []                                   # (frame, site index) of on-site births
    prev = full[0] > 0
    for t in range(1, full.shape[0]):
        cur = full[t] > 0
        lab, n = ndimage.label(cur)
        if n:
            near = set(np.unique(lab[:rows])) - {0}
            overl = set(np.unique(lab[prev])) - {0}
            sizes = ndimage.sum(cur, lab, index=np.arange(1, n + 1))
            for l in near - overl:
                if sizes[l - 1] < MIN_AREA_CELLS:
                    continue
                n_birth += 1
                xs = np.where((lab[:rows] == l).any(axis=0))[0]
                dist = np.abs(xs[:, None] - cols[None, :]).min(axis=0)
                if dist.min() <= PATCH_HALF:
                    on_site += 1
                    events.append((t, int(dist.argmin())))
        prev = cur
    return n_birth, on_site, events


def analyse(study, h5_path):
    m = meta(h5_path.replace(".hdf5", ".json"))
    cols = np.round((m["sites"] - m["x_min"]) / DX - 0.5).astype(int)
    with h5py.File(h5_path, "r") as h:
        # Read whole frames in sequential blocks and slice in memory: the files are contiguous and
        # uncompressed, so a row slice per frame becomes many small scattered reads.
        ds = h["dfun"]
        wall = np.concatenate([ds[t:min(t + BLOCK, ds.shape[0])][:, :PATCH_ROWS, :] > 0
                               for t in range(START, ds.shape[0], BLOCK)])  # (T, rows, W)
        full = h["dfun"][START:START + VALID_FRAMES + 1].astype(np.float32)
    T, W = wall.shape[0], wall.shape[2]
    occ = np.stack([wall[:, :, max(0, c - PATCH_HALF):min(W, c + PATCH_HALF + 1)].any(axis=(1, 2))
                    for c in cols], axis=1)                             # (T, n_sites)

    # ---- A. dry and wet periods per site
    dry, wet, b2b, n_occ_births = [], [], [], 0
    per_site = [transitions(occ[:, k]) for k in range(len(cols))]
    for b, d in per_site:
        n_occ_births += len(b)
        for bt in b:
            prev = d[d < bt]
            if prev.size:
                dry.append(bt - prev[-1])
        for dt_ in d:
            prev = b[b < dt_]
            if prev.size:
                wet.append(dt_ - prev[-1])
        b2b.extend(np.diff(b).tolist())
    dry, wet, b2b = (np.array(a, dtype=int) for a in (dry, wet, b2b))

    # ---- B. the counter's births, and how many sit on a listed site
    n_birth, on_site, events = counter_births_on_site(full, cols)
    assert n_birth == count_nucleation_events(full)["total_events"], "replay drifted from counter"

    # ---- C. window-limited clock-rule forecaster
    modal = Counter(dry.tolist()).most_common(1)[0][0] if dry.size else None
    tp = fp = fn = 0
    if modal is not None:
        allowed = {modal - 1, modal, modal + 1}
        for k, (b, d) in enumerate(per_site):
            bset = set(b.tolist())
            for t in range(WINDOW - 1, T - 1):          # forecast frame t+1 from frames t-4..t
                last = d[(d <= t) & (d >= t - WINDOW + 1)]
                pred = bool((not occ[t, k]) and last.size and (t + 1 - last[-1]) in allowed)
                truth = (t + 1) in bset
                tp += pred and truth
                fp += pred and not truth
                fn += truth and not pred

    # ---- D. the same questions, scored against the COUNTER's births (real >=16-cell seeds) only.
    # The occupancy series also fires on wall flicker, which inflates A and C; D does not.
    delays = []
    for t, k in events:
        d = per_site[k][1]
        prev = d[d <= t]
        delays.append(t - prev[-1] if prev.size else -1)
    delays = np.array(delays, dtype=int)
    known = delays[delays >= 0]
    tp2 = fp2 = fn2 = 0
    if modal is not None:
        ev = {(t, k) for t, k in events}
        for k, (b, d) in enumerate(per_site):
            for t in range(WINDOW - 1, min(T, VALID_FRAMES + 1) - 1):
                last = d[(d <= t) & (d >= t - WINDOW + 1)]
                pred = bool((not occ[t, k]) and last.size and (t + 1 - last[-1]) in allowed)
                truth = (t + 1, k) in ev
                tp2 += pred and truth
                fp2 += pred and not truth
                fn2 += truth and not pred

    q = lambda a, p: float(np.percentile(a, p)) if a.size else float("nan")
    gaps = np.diff(np.sort(cols))
    return dict(
        study=study, twall=m["twall"], n_sites=len(cols), wait_frames=m["wait"] / m["dt"],
        frames=T, min_site_gap_cells=int(gaps.min()) if gaps.size else -1,
        births_occupancy=n_occ_births,
        counter_births_100_400=n_birth, counter_births_on_site=on_site,
        dry_n=int(dry.size), dry_median=q(dry, 50), dry_p10=q(dry, 10), dry_p90=q(dry, 90),
        dry_mode=modal,
        dry_frac_in_window=float((dry <= WINDOW).mean()) if dry.size else np.nan,
        dry_frac_mode_pm1=float(np.isin(dry, [modal - 1, modal, modal + 1]).mean()) if dry.size else np.nan,
        wet_median=q(wet, 50), wet_p10=q(wet, 10), wet_p90=q(wet, 90),
        b2b_median=q(b2b, 50), b2b_cv=float(b2b.std() / b2b.mean()) if b2b.size > 1 else np.nan,
        clock_precision=tp / (tp + fp) if tp + fp else np.nan,
        clock_recall=tp / (tp + fn) if tp + fn else np.nan,
        cb_no_prior_departure=int((delays < 0).sum()),
        cb_delay_median=q(known, 50), cb_delay_p10=q(known, 10), cb_delay_p90=q(known, 90),
        cb_delay_frac_in_window=float((known <= WINDOW).mean()) if known.size else np.nan,
        cb_clock_precision=tp2 / (tp2 + fp2) if tp2 + fp2 else np.nan,
        cb_clock_recall=tp2 / (tp2 + fn2) if tp2 + fn2 else np.nan,
        cb_delay_hist=dict(sorted(Counter(np.clip(known, 0, 30).tolist()).items())),
        dry_hist=dict(sorted(Counter(np.clip(dry, 0, 30).tolist()).items())),
    )


def main():
    import pandas as pd
    rows = []
    for study, root in STUDIES.items():
        tw = lambda p: float(os.path.basename(p).split("_")[1].split(".")[0])
        paths = sorted(glob.glob(os.path.join(root, "*.hdf5")), key=tw)
        if "--all" not in sys.argv:
            paths = [p for p in paths if tw(p) in SUBSET[study]]
        for h5 in paths:
            r = analyse(study, h5)
            rows.append(r)
            print(f"{study:>6} Twall {r['twall']:>5.0f} sites {r['n_sites']:>2} wait {r['wait_frames']:.0f}f "
                  f"| occ births {r['births_occupancy']:>5} | counter@[100,400) {r['counter_births_100_400']:>4} "
                  f"on-site {r['counter_births_on_site']:>4} | dry med {r['dry_median']:.0f} "
                  f"p10-p90 {r['dry_p10']:.0f}-{r['dry_p90']:.0f} mode {r['dry_mode']} "
                  f"in-window {r['dry_frac_in_window']:.2f} mode±1 {r['dry_frac_mode_pm1']:.2f} "
                  f"| wet med {r['wet_median']:.0f} ({r['wet_p10']:.0f}-{r['wet_p90']:.0f}) "
                  f"| b2b med {r['b2b_median']:.0f} cv {r['b2b_cv']:.2f} "
                  f"| clock P {r['clock_precision']:.2f} R {r['clock_recall']:.2f} "
                  f"| min gap {r['min_site_gap_cells']}c", flush=True)
            print(f"         COUNTER births: delay since site freed med {r['cb_delay_median']:.0f} "
                  f"p10-p90 {r['cb_delay_p10']:.0f}-{r['cb_delay_p90']:.0f} "
                  f"in-window {r['cb_delay_frac_in_window']:.2f} no-prior {r['cb_no_prior_departure']} "
                  f"| clock P {r['cb_clock_precision']:.2f} R {r['cb_clock_recall']:.2f} "
                  f"| hist {r['cb_delay_hist']}", flush=True)
            print(f"         dry-period histogram (frames: count): {r['dry_hist']}", flush=True)
    pd.DataFrame([{k: v for k, v in r.items() if not k.endswith("_hist")} for r in rows]).to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
