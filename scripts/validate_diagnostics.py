#!/usr/bin/env python3
"""Validate the diagnostics on reference data, and measure the reference nucleation curve.

1. Validation: mean |∇φ| on reference frames must lie within EXPECT_GRAD_MEAN; otherwise the run
   reports FAIL and exits non-zero.
2. Reference curve: nucleation rate, void fraction, component count and interface width per
   simulation across the site-count sweep (Twall 75-107, 5-37 sites). Model rollouts are compared
   against this curve (rollout_diagnostics.py).

Usage:
    python scripts/validate_diagnostics.py                                  # pool boiling, writes reference_diagnostics.csv
    python scripts/validate_diagnostics.py --data-dir $BUBBLEML_ROOT/SingleBubble-Saturated-FC72-2D \
        --sims 90 92 94 96 98 100 --name singlebubble_reference_diagnostics.csv
    add --min-area 4 --name ..._minarea4.csv for the 4-cell threshold comparison
"""
import argparse
import glob
import json
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diagnostics import (DX, MIN_AREA_CELLS, count_nucleation_events, component_stats,
                         eikonal_stats, interface_width)
from paths import POOL as DATA

# accepted range of mean |∇φ| on reference frames
EXPECT_GRAD_MEAN = (0.97, 1.02)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=300, help="frames per sim, from start_time")
    ap.add_argument("--start", type=int, default=100, help="skip initial transient")
    ap.add_argument("--out", default="results/diagnostics")
    ap.add_argument("--sims", nargs="*", default=None, help="Twall values; default = spread of 6")
    # SingleBubble (the control) has the same grid spacing as pool boiling (1/32), so the diagnostics
    # apply unchanged. Defaults measure pool boiling.
    ap.add_argument("--data-dir", default=DATA, help="study directory to measure")
    # rollout_diagnostics' self-check compares against the curve written here at the same threshold.
    ap.add_argument("--min-area", type=int, default=MIN_AREA_CELLS,
                    help=f"min component area in cells (default {MIN_AREA_CELLS})")
    ap.add_argument("--name", default="reference_diagnostics.csv",
                    help="output CSV name; use a distinct one per study")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    sims = a.sims or ["75", "82", "90", "96", "102", "107"]
    rows, failures = [], []

    for tw in sims:
        f = os.path.join(a.data_dir, f"Twall_{tw}.hdf5")
        if not os.path.exists(f):
            print(f"  skip Twall_{tw}: not on disk"); continue
        meta = json.load(open(f.replace(".hdf5", ".json")))
        n_sites = len(meta["nuc_sites_x"])

        with h5py.File(f, "r") as h:
            phi = h["dfun"][a.start:a.start + a.frames].astype(np.float32)

        mid = phi[len(phi) // 2].astype(np.float64)
        ek = eikonal_stats(mid, DX)
        ek_face = eikonal_stats(mid, DX, band=4 * DX)
        nuc = count_nucleation_events(phi, DX, min_area_cells=a.min_area)
        comp = component_stats(mid, min_area_cells=a.min_area)
        width = interface_width(mid, DX)

        ok = EXPECT_GRAD_MEAN[0] <= ek["grad_mean"] <= EXPECT_GRAD_MEAN[1]
        if not ok:
            failures.append((tw, ek["grad_mean"]))

        rows.append(dict(twall=int(tw), n_sites=n_sites, min_area=a.min_area,
                         grad_mean=ek["grad_mean"], grad_median=ek["grad_median"],
                         grad_p5=ek["grad_p5"], grad_p95=ek["grad_p95"],
                         frac_off=ek["frac_off_manifold"],
                         grad_mean_nearface=ek_face["grad_mean"],
                         nuc_total=nuc["total_events"], nuc_rate=nuc["rate_per_frame"],
                         n_components=comp["n_components"], void_fraction=comp["void_fraction"],
                         interface_width=width))
        print(f"  Twall_{tw:<4} sites={n_sites:<3} |grad|={ek['grad_mean']:.4f} "
              f"(p5={ek['grad_p5']:.4f} p95={ek['grad_p95']:.4f}) {'OK' if ok else 'FAIL'}  "
              f"nuc/frame={nuc['rate_per_frame']:.3f}  comps={comp['n_components']:<3} "
              f"void={comp['void_fraction']:.4f}  width={width:.2f}", flush=True)

    # ---- validation verdict
    print("\n=== VALIDATION ===")
    if failures:
        print(f"  FAILED on {len(failures)} sims — |grad phi| outside {EXPECT_GRAD_MEAN}:")
        for tw, g in failures:
            print(f"    Twall_{tw}: {g:.4f}")
        print("  The diagnostic does not reproduce the reference; do not use it on model output.")
    else:
        print(f"  PASS — |grad phi| within {EXPECT_GRAD_MEAN} on all {len(rows)} sims.")

    # ---- the reference curve
    if rows:
        print("\n=== REFERENCE CURVE (ground truth) ===")
        print(f"  {'sites':>6} {'nuc/frame':>10} {'components':>11} {'void':>8}")
        for r in sorted(rows, key=lambda r: r["n_sites"]):
            print(f"  {r['n_sites']:>6} {r['nuc_rate']:>10.3f} {r['n_components']:>11} "
                  f"{r['void_fraction']:>8.4f}")
        s = np.array([r["n_sites"] for r in rows], float)
        n = np.array([r["nuc_rate"] for r in rows], float)
        if len(s) > 2 and n.std() > 0:
            print(f"\n  corr(n_sites, nucleation rate) = {np.corrcoef(s, n)[0,1]:+.3f}")

        out = os.path.join(a.out, a.name)
        with open(out, "w", encoding="utf-8") as fh:
            keys = list(rows[0].keys())
            fh.write(",".join(keys) + "\n")
            for r in sorted(rows, key=lambda r: r["n_sites"]):
                fh.write(",".join(f"{r[k]:.6g}" if isinstance(r[k], float) else str(r[k])
                                  for k in keys) + "\n")
        print(f"\nwrote {out}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
