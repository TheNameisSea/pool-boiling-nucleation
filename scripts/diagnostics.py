#!/usr/bin/env python3
"""Physics diagnostics for boiling forecasts: eikonal residual, nucleation-event counting,
bubble-population statistics, interface width and per-step relative L2.

Every diagnostic is validated on reference data by `validate_diagnostics.py` before use on model output.
Fields follow BubbleML ordering: [dfun, temperature, velx, vely], with dfun > 0 = vapour.
Grid spacing dx = dy = 1/32 for both PoolBoiling (512^2, domain 16 x 16) and SingleBubble.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

DX = 1.0 / 32.0
SDF, TEMP, VELX, VELY = 0, 1, 2, 3

# Minimum component area, in cells, for a vapour blob to count as a bubble. Chosen from the reference
# birth-size distributions (Section 3.3 of the paper). Changing it requires regenerating the reference
# curves (validate_diagnostics.py), or the rollout self-check is silently disabled.
MIN_AREA_CELLS = 16


# ----------------------------------------------------------------------------- eikonal
def eikonal_residual(phi: np.ndarray, dx: float = DX) -> np.ndarray:
    """|∇φ| − 1, pointwise. Zero everywhere for a valid signed distance function."""
    gy, gx = np.gradient(phi.astype(np.float64), dx, dx)
    return np.sqrt(gx * gx + gy * gy) - 1.0


def eikonal_stats(phi: np.ndarray, dx: float = DX, band: float | None = None) -> dict:
    """Summary of eikonal violation.

    band: if set, restrict to |φ| < band, i.e. near the interface.
    """
    r = eikonal_residual(phi, dx)
    g = r + 1.0
    if band is not None:
        m = np.abs(phi) < band
        if m.sum() == 0:
            return {"n": 0}
        g, r = g[m], r[m]
    return {
        "n": int(g.size),
        "grad_mean": float(g.mean()),
        "grad_median": float(np.median(g)),
        "grad_p5": float(np.percentile(g, 5)),
        "grad_p95": float(np.percentile(g, 95)),
        "grad_std": float(g.std()),
        "abs_residual_mean": float(np.abs(r).mean()),
        "frac_off_manifold": float((np.abs(r) > 0.1).mean()),  # >10% violation
    }


# ----------------------------------------------------------------------------- nucleation
def vapour_labels(phi: np.ndarray) -> tuple[np.ndarray, int]:
    """Connected components of the vapour phase (φ > 0)."""
    return ndimage.label(phi > 0)


def count_nucleation_events(
    phi_seq: np.ndarray,
    dx: float = DX,
    wall_band: float = 0.5,
    min_area_cells: int = MIN_AREA_CELLS,
) -> dict:
    """Count bubbles appearing at the heated wall over a frame sequence.

    A nucleation event = a vapour component present in frame t that (a) touches the near-wall
    band, and (b) does not overlap ANY vapour component of frame t-1. The overlap test is what
    separates genuine nucleation from a bubble that merely grew or drifted.

    phi_seq: (T, H, W), row 0 = the wall (y = 0).
    wall_band: physical height above the wall to consider, in domain units.
    min_area_cells: ignore blobs below this area (see MIN_AREA_CELLS above).

    Returns per-frame counts and the total.
    """
    T = phi_seq.shape[0]
    rows = max(1, int(round(wall_band / dx)))
    events, areas = [], []

    prev_mask = phi_seq[0] > 0
    for t in range(1, T):
        cur = phi_seq[t] > 0
        lab, n = ndimage.label(cur)
        n_new = 0
        if n:
            # which labels touch the wall band?
            near_wall = set(np.unique(lab[:rows, :])) - {0}
            # which labels overlap any previous vapour?
            overlapping = set(np.unique(lab[prev_mask])) - {0}
            sizes = ndimage.sum(cur, lab, index=np.arange(1, n + 1))
            for l in near_wall - overlapping:
                if sizes[l - 1] >= min_area_cells:
                    n_new += 1
                    areas.append(float(sizes[l - 1]))
        events.append(n_new)
        prev_mask = cur

    return {
        "events_per_frame": events,
        "total_events": int(sum(events)),
        "frames": T - 1,
        "rate_per_frame": float(sum(events) / max(1, T - 1)),
        "mean_birth_area_cells": float(np.mean(areas)) if areas else 0.0,
    }


def reinit_sdf(phi: np.ndarray, dx: float = DX) -> np.ndarray:
    """Reinitialise a level-set field to a true signed distance function (`|∇φ| = 1`).

    Used as the per-step reinitialisation in rollouts (`rollout_diagnostics.py --project`): it restores
    |∇φ| = 1 without moving the interface. The zero level set is preserved to within half a cell;
    everything else is recomputed as a distance.

    Vapour is φ > 0, matching `dfun`. Uses the exact Euclidean distance transform, so the result is
    a genuine distance field rather than an iterative approximation.

    A frame with no interface (all vapour or none) has no level set to reinitialise from, so it
    is returned unchanged rather than filled with ±inf.
    """
    m = phi > 0
    if not m.any() or m.all():
        return phi.astype(np.float32, copy=True)
    d_in = ndimage.distance_transform_edt(m)     # inside vapour: distance to nearest liquid cell
    d_out = ndimage.distance_transform_edt(~m)   # outside: distance to nearest vapour cell
    # −0.5 puts the interface midway between the last vapour and first liquid cell, which is where
    # the original zero crossing sits; without it the level set shifts by half a cell per call and
    # would drift across an autoregressive rollout.
    return np.where(m, dx * (d_in - 0.5), -dx * (d_out - 0.5)).astype(np.float32)


def component_stats(phi: np.ndarray, min_area_cells: int = MIN_AREA_CELLS) -> dict:
    """Bubble-population statistics for a single frame."""
    lab, n = vapour_labels(phi)
    if n == 0:
        return {"n_components": 0, "void_fraction": 0.0, "mean_area_cells": 0.0}
    sizes = ndimage.sum(phi > 0, lab, index=np.arange(1, n + 1))
    sizes = sizes[sizes >= min_area_cells]
    return {
        "n_components": int(sizes.size),
        "void_fraction": float((phi > 0).mean()),
        "mean_area_cells": float(sizes.mean()) if sizes.size else 0.0,
        "max_area_cells": float(sizes.max()) if sizes.size else 0.0,
    }


# ----------------------------------------------------------------------------- sharpness
def interface_width(phi: np.ndarray, dx: float = DX) -> float:
    """Effective interface width, in cells.

    For a true SDF the zero crossing is sharp and this is ~1. Estimated as (cells with |φ| < dx) / (length of the zero contour).
    """
    near = np.abs(phi) < dx
    if near.sum() == 0:
        return 0.0
    gy, gx = np.gradient((phi > 0).astype(np.float64), dx, dx)
    contour = np.sqrt(gx * gx + gy * gy) > 0
    return float(near.sum() / max(1, contour.sum()))


# ----------------------------------------------------------------------------- rollout
def relative_l2_per_step(pred: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    """(T, C) relative L2, matching upstream's LpLoss definition so numbers stay comparable."""
    d = np.linalg.norm((pred - tgt).reshape(*pred.shape[:2], -1), axis=-1)
    b = np.linalg.norm(tgt.reshape(*tgt.shape[:2], -1), axis=-1)
    return d / np.maximum(b, 1e-12)


def summarize_rollout(pred: np.ndarray, tgt: np.ndarray, dx: float = DX) -> dict:
    """Full diagnostic battery over a rollout. pred/tgt: (T, C, H, W)."""
    rel = relative_l2_per_step(pred, tgt)
    out = {
        "rel_l2_mean": rel.mean(axis=0).tolist(),
        "rel_l2_first": rel[0].tolist(),
        "rel_l2_last": rel[-1].tolist(),
    }
    for tag, arr in (("pred", pred), ("ref", tgt)):
        phi = arr[:, SDF]
        mid = phi[len(phi) // 2]
        out[f"{tag}_eikonal"] = eikonal_stats(mid, dx)
        out[f"{tag}_eikonal_nearface"] = eikonal_stats(mid, dx, band=4 * dx)
        out[f"{tag}_nucleation"] = count_nucleation_events(phi, dx)
        out[f"{tag}_components"] = component_stats(mid)
        out[f"{tag}_interface_width"] = interface_width(mid, dx)
    return out
