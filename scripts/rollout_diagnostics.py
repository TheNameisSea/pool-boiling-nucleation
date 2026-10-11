#!/usr/bin/env python3
"""Roll out a trained checkpoint across the site-count sweep and measure the physics diagnostics.

Produces the model-side counterpart of `results/diagnostics/reference_diagnostics.csv`
(rollout nucleation ratio, void ratio, eikonal statistics, interface width, relative L2).
Three things are matched to `validate_diagnostics.py` exactly:

1. Frame alignment. The reference reads raw frames [100, 400). A rollout's first predicted frame is
   `start_time + time_window`, so the dataset is opened at `start_time = align_to - time_window` (95)
   and the predicted frames land on [100, 400).
2. Aggregation. Eikonal / components / interface width are single-frame statistics taken at the
   middle frame; nucleation is counted over the whole sequence.
3. Units. Diagnostics are physical (|∇φ| = 1 presumes an SDF on a dx = 1/32 grid). Predictions are
   de-normalised before measurement (the identity for the `normalize: none` configs used here).

Each rollout also re-measures its own targets (`ref_*` columns) and checks them against the
reference CSV (the self-check); a mismatch is reported as FAILED and the run exits non-zero.

Modes:
    default            300-frame autoregressive rollout from true frames 95-99
    --project          reinitialise the predicted SDF at every step (diagnostics.reinit_sdf)
    --teacher-forced   one step from true inputs at every start frame; writes *_teacherforced.csv
    --save-fields DIR  also save the predicted dfun sequences (rollout: dfun_pred_Twall_<tw>.npy;
                       teacher-forced: first predicted frame, dfun_tf_first_Twall_<tw>.npy)

Usage:
    python scripts/rollout_diagnostics.py --checkpoint <ckpt> --job <run id> --label <name> \
        [--teacher-forced] [--project] [--save-fields $WORK_ROOT/tf_fields_<run id>]
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diagnostics import (DX, MIN_AREA_CELLS, SDF, component_stats,  # noqa: E402
                         count_nucleation_events, eikonal_stats, interface_width,
                         reinit_sdf, relative_l2_per_step)
from inference import build_model  # noqa: E402
from paths import POOL as DATA  # noqa: E402

from bubbleformer.data import BubbleForecast  # noqa: E402

REFERENCE = "results/diagnostics/reference_diagnostics.csv"
# The six sims the reference curve is measured on: 5 -> 37 sites.
DEFAULT_SIMS = ["75", "82", "90", "96", "102", "107"]
FIELDS = ["dfun", "temp", "velx", "vely"]
# Tolerance of the self-check: the rollout's own targets are raw frames and must reproduce the reference.
SELF_CHECK_RTOL = 0.01
# The window `reference_diagnostics.csv` is measured over (validate_diagnostics.py defaults). The
# self-check runs only when the rollout uses the same window.
REFERENCE_WINDOW = (100, 300)  # (align_to, frames)
# The counting threshold the reference CSVs are measured at; the self-check runs only at this threshold.
REFERENCE_MIN_AREA = MIN_AREA_CELLS


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--job", required=True, help="run id of the training run (see configs/runs.csv); used in the output file name")
    p.add_argument("--label", default=None, help="short model label for the CSV (default: from ckpt)")
    p.add_argument("--sims", nargs="*", default=DEFAULT_SIMS, help="Twall values to roll out")
    p.add_argument("--frames", type=int, default=300, help="predicted frames per sim (match reference)")
    p.add_argument("--align-to", type=int, default=100,
                   help="absolute frame the FIRST PREDICTED frame lands on (reference start_time)")
    p.add_argument("--data-dir", default=DATA)
    p.add_argument("--out", default="results/diagnostics")
    p.add_argument("--reference", default=REFERENCE)
    p.add_argument("--device", default="cuda")
    # A threshold other than the default disables the self-check (the reference CSVs use the default).
    p.add_argument("--min-area", type=int, default=MIN_AREA_CELLS,
                   help=f"min component area in cells for a birth to count "
                        f"(default {MIN_AREA_CELLS})")
    p.add_argument("--project", action="store_true",
                   help="reinitialise the predicted SDF at every rollout step (|grad phi| = 1), leaving "
                        "the interface in place")
    p.add_argument("--teacher-forced", action="store_true",
                   help="one-step evaluation: at every start frame feed the TRUE 5-frame window and "
                        "count births in the prediction. Writes *_teacherforced.csv; no self-check.")
    p.add_argument("--save-fields", default=None,
                   help="optional dir for the predicted dfun sequences (~315 MB/sim)")
    return p.parse_args()


def denormalizer(ds, fields):
    """Return f(tensor[T,C,H,W]) -> physical units, using the dataset's own constants.

    `normalize: none` makes this the identity; it is applied anyway so that a checkpoint trained
    with std/minmax normalization still yields physical units.
    """
    diff = torch.tensor([ds.diff_terms[f] for f in fields], dtype=torch.float32)
    div = torch.tensor([ds.div_terms[f] for f in fields], dtype=torch.float32)
    identity = bool(torch.allclose(diff, torch.zeros_like(diff), atol=1e-8)
                    and torch.allclose(div, torch.ones_like(div), atol=1e-6))
    denorm = lambda x: x * div.view(1, -1, 1, 1) + diff.view(1, -1, 1, 1)
    # The inverse is needed by --project: reinitialisation is done in physical units (dx = 1/32),
    # then the state is re-normalised before it is fed back to the model.
    renorm = lambda x: (x - diff.view(1, -1, 1, 1)) / div.view(1, -1, 1, 1)
    return denorm, renorm, identity


def rollout(model, ds, conditioned, n_frames, tw, device, denorm, renorm=None, project=False):
    """Autoregressive rollout. Returns (pred_dfun, ref_dfun, rel_l2) in PHYSICAL units.

    The model is fed its own previous output in NORMALIZED units (what it was trained on);
    de-normalization is applied only to the copies kept for diagnostics.
    """
    n_win = int(np.ceil(n_frames / tw))
    if n_win * tw > len(ds):
        raise SystemExit(f"need {n_win * tw} frames but dataset offers {len(ds)} — "
                         f"lower --frames or --align-to")

    phi_p, phi_r, rels, prev = [], [], [], None
    with torch.no_grad():
        for w in range(n_win):
            itr = w * tw
            sample = ds[itr]
            if conditioned:
                inp, tgt, fluid_params = sample
            else:
                (inp, tgt), fluid_params = sample, None

            # After the first window the model consumes its own prediction (autoregressive rollout).
            inp = (prev if prev is not None else inp).to(device).float().unsqueeze(0)
            if conditioned:
                pred = model(inp, fluid_params.to(device).float().unsqueeze(0))
            else:
                pred = model(inp)

            prev = pred.squeeze(0).detach().cpu()          # normalized: feeds the next window

            if project:
                # Reinitialise the SDF before it is fed back: the interface stays where the model
                # put it; only |grad phi| is repaired.
                phys = denorm(prev.unsqueeze(0)).squeeze(0).clone()
                phys[:, SDF] = torch.from_numpy(
                    np.stack([reinit_sdf(f, DX) for f in phys[:, SDF].numpy()]))
                prev = renorm(phys.unsqueeze(0)).squeeze(0)

            pred_phys = denorm(prev.unsqueeze(0)).squeeze(0)
            tgt_phys = denorm(tgt.detach().cpu().unsqueeze(0)).squeeze(0)

            phi_p.append(pred_phys[:, SDF].numpy().astype(np.float32))
            phi_r.append(tgt_phys[:, SDF].numpy().astype(np.float32))
            rels.append(relative_l2_per_step(pred_phys.numpy(), tgt_phys.numpy()))

    return (np.concatenate(phi_p)[:n_frames],
            np.concatenate(phi_r)[:n_frames],
            np.concatenate(rels)[:n_frames])


def teacher_forced(model, ds, conditioned, n_frames, device, denorm, min_area, site_cols,
                   keep_first=False):
    """One-step nucleation, from TRUE inputs at every start frame (stride 1).

    Sample s feeds true frames [start+s, start+s+tw) and predicts the next tw frames, so over
    s = 0..n_frames-1 the FIRST predicted frame visits every frame of [align_to, align_to+n_frames)
    exactly once. Births are counted against the model's actual input: the counter sees
    [true last input frame, prediction...], never a previous prediction.

      first : births in the first predicted frame (the cleanest one-step measure)
      window: births anywhere in the tw-frame output, per predicted frame
    Every count is repeated on the TRUE targets with the identical recipe, so pred/ref is exact.

    Against a true previous frame, a model that merely displaces a bubble produces a component with
    no overlap, which the counter calls a birth. `*_onsite` therefore also counts only births
    touching a listed site (±3 cells); the plain count is kept for comparability with rollouts.
    Matching predicted to real births is done separately by tf_birth_matching.py.
    """
    from birth_predictability import counter_births_on_site
    if n_frames > len(ds):
        raise SystemExit(f"need {n_frames} start frames but dataset offers {len(ds)}")
    assert min_area == MIN_AREA_CELLS, "on-site replay uses MIN_AREA_CELLS"
    keys = ("first_pred", "first_ref", "win_pred", "win_ref")
    c = {k: 0 for k in keys} | {k + "_on": 0 for k in keys} | {"win_frames": 0}
    rel0, firsts = [], []

    def add(key, seq):
        n, on, _ = counter_births_on_site(seq, site_cols)
        c[key] += n
        c[key + "_on"] += on
    with torch.no_grad():
        for s in range(n_frames):
            sample = ds[s]
            if conditioned:
                inp, tgt, fluid_params = sample
                pred = model(inp.to(device).float().unsqueeze(0),
                             fluid_params.to(device).float().unsqueeze(0))
            else:
                inp, tgt = sample
                pred = model(inp.to(device).float().unsqueeze(0))
            p = denorm(pred.squeeze(0).cpu())[:, SDF].numpy().astype(np.float32)
            t = denorm(tgt.unsqueeze(0)).squeeze(0)[:, SDF].numpy().astype(np.float32)
            last = denorm(inp.unsqueeze(0)).squeeze(0)[-1, SDF].numpy().astype(np.float32)
            add("first_pred", np.stack([last, p[0]]))
            add("first_ref", np.stack([last, t[0]]))
            add("win_pred", np.concatenate([last[None], p]))
            add("win_ref", np.concatenate([last[None], t]))
            c["win_frames"] += len(p)
            rel0.append(float(np.linalg.norm(p[0] - t[0]) / np.linalg.norm(t[0])))
            if keep_first:
                firsts.append(p[0])
    ratio = lambda a, b: a / b if b else float("nan")
    return dict(tf_first_pred=c["first_pred"], tf_first_ref=c["first_ref"],
                tf_first_ratio=ratio(c["first_pred"], c["first_ref"]),
                tf_win_pred_rate=c["win_pred"] / c["win_frames"],
                tf_win_ref_rate=c["win_ref"] / c["win_frames"],
                tf_win_ratio=ratio(c["win_pred"], c["win_ref"]),
                tf_first_pred_onsite=c["first_pred_on"], tf_first_ref_onsite=c["first_ref_on"],
                tf_first_ratio_onsite=ratio(c["first_pred_on"], c["first_ref_on"]),
                tf_win_ratio_onsite=ratio(c["win_pred_on"], c["win_ref_on"]),
                tf_onestep_rel_l2_dfun=float(np.mean(rel0))), (np.stack(firsts) if keep_first else None)


def measure(phi_seq, min_area=4):
    """The diagnostic battery, aggregated exactly as validate_diagnostics.py does it."""
    mid = phi_seq[len(phi_seq) // 2].astype(np.float64)
    ek = eikonal_stats(mid, DX)
    ek_face = eikonal_stats(mid, DX, band=4 * DX)
    nuc = count_nucleation_events(phi_seq, DX, min_area_cells=min_area)
    comp = component_stats(mid, min_area_cells=min_area)
    # The banded statistic is undefined when the mid-frame has no interface (no cell with
    # |φ| < 4·dx); eikonal_stats then returns {"n": 0} and NaN is reported.
    if ek_face.get("n", 0) == 0:
        print("    note: mid-frame has NO interface (|φ| < 4·dx is empty) — grad_mean_nearface=NaN")
    return dict(grad_mean=ek["grad_mean"], grad_median=ek["grad_median"],
                grad_p5=ek["grad_p5"], grad_p95=ek["grad_p95"],
                frac_off=ek["frac_off_manifold"],
                grad_mean_nearface=ek_face.get("grad_mean", float("nan")),
                nuc_total=nuc["total_events"], nuc_rate=nuc["rate_per_frame"],
                n_components=comp["n_components"], void_fraction=comp["void_fraction"],
                interface_width=interface_width(mid, DX))


def load_reference(path):
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        return {int(r["twall"]): {k: float(v) for k, v in r.items()} for r in csv.DictReader(fh)}


def main():
    a = parse_args()
    os.makedirs(a.out, exist_ok=True)

    ckpt = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    model, model_name, hp = build_model(ckpt)
    data_cfg = hp["data_cfg"]
    conditioned = "film" in model_name.lower()
    tw = data_cfg["time_window"]
    label = a.label or model_name
    model = model.to(a.device).eval()

    start_time = a.align_to - tw  # so the first PREDICTED frame lands on --align-to
    print(f"model={model_name} label={label} job={a.job} conditioned={conditioned}")
    print(f"time_window={tw}  dataset start_time={start_time}  "
          f"predicted frames [{a.align_to}, {a.align_to + a.frames})")

    reference = load_reference(a.reference)
    comparable = ((a.align_to, a.frames) == REFERENCE_WINDOW
                  and a.min_area == REFERENCE_MIN_AREA)
    if (a.align_to, a.frames) != REFERENCE_WINDOW:
        print(f"  note: rollout window ({a.align_to}, {a.frames}) != reference window "
              f"{REFERENCE_WINDOW} — self-check SKIPPED and ref_* columns are measured over this "
              f"window, not the reference curve.")
    elif a.min_area != REFERENCE_MIN_AREA:
        # ref_* is re-measured here at --min-area, so pred/ref stays internally consistent; it
        # just cannot be checked against the reference CSV.
        print(f"  note: --min-area {a.min_area} != {REFERENCE_MIN_AREA}, the threshold "
              f"reference_diagnostics.csv was "
              f"measured at — self-check SKIPPED. ref_* columns are re-measured at {a.min_area}, "
              f"so ratios are internally consistent.")
    rows, warnings = [], []

    for tw_val in a.sims:
        path = os.path.join(a.data_dir, f"Twall_{tw_val}.hdf5")
        if not os.path.exists(path):
            print(f"  skip Twall_{tw_val}: not on disk")
            continue
        meta = json.load(open(path.replace(".hdf5", ".json"), encoding="utf-8"))
        n_sites = len(meta["nuc_sites_x"])

        ds = BubbleForecast(
            filenames=[path],
            input_fields=list(data_cfg["input_fields"]),
            output_fields=list(data_cfg["output_fields"]),
            norm=data_cfg["normalize"],
            downsample_factor=data_cfg["downsample_factor"],
            time_window=tw,
            start_time=start_time,
            return_fluid_params=conditioned,
        )
        ds.normalize(*hp["normalization_constants"])
        denorm, renorm, identity = denormalizer(ds, list(data_cfg["input_fields"]))
        if not identity:
            print("  note: non-identity normalization — de-normalizing before diagnostics")

        t0 = time.time()
        if a.teacher_forced:
            cols = np.round((np.atleast_1d(np.array(meta["nuc_sites_x"], float)) - meta["x_min"])
                            / DX - 0.5).astype(int)
            r, firsts = teacher_forced(model, ds, conditioned, a.frames, a.device, denorm, a.min_area,
                                       cols, keep_first=bool(a.save_fields))
            if a.save_fields:
                # frame k = the model's ONE-STEP prediction of absolute frame align_to + k, made from
                # true frames [align_to + k - tw, align_to + k). Aligned with raw frames [100, 400).
                os.makedirs(a.save_fields, exist_ok=True)
                np.save(os.path.join(a.save_fields, f"dfun_tf_first_Twall_{tw_val}.npy"), firsts)
            rows.append(dict(model=label, job=a.job, twall=int(tw_val), n_sites=n_sites,
                             min_area=a.min_area, **r))
            print(f"  Twall_{tw_val:<4} sites={n_sites:<3} TEACHER-FORCED  first-frame births "
                  f"{r['tf_first_pred']} vs ref {r['tf_first_ref']} (ratio {r['tf_first_ratio']:.3f})"
                  f"  |  window rate {r['tf_win_pred_rate']:.3f} vs ref {r['tf_win_ref_rate']:.3f} "
                  f"(ratio {r['tf_win_ratio']:.3f})  |  ON-SITE first {r['tf_first_pred_onsite']} vs "
                  f"{r['tf_first_ref_onsite']} (ratio {r['tf_first_ratio_onsite']:.3f}), window ratio "
                  f"{r['tf_win_ratio_onsite']:.3f}  |  one-step rel L2 dfun "
                  f"{r['tf_onestep_rel_l2_dfun']:.4f}  [{time.time() - t0:.0f}s]", flush=True)
            continue
        phi_p, phi_r, rel = rollout(model, ds, conditioned, a.frames, tw, a.device, denorm,
                                    renorm, a.project)
        pred, ref = measure(phi_p, a.min_area), measure(phi_r, a.min_area)

        # Self-check: the rollout's own targets are raw frames and must reproduce the reference curve.
        if comparable and int(tw_val) in reference:
            r = reference[int(tw_val)]
            for key in ("nuc_rate", "void_fraction", "grad_mean"):
                got, want = ref[key], r[key]
                if abs(got - want) > SELF_CHECK_RTOL * max(abs(want), 1e-9):
                    warnings.append(f"Twall_{tw_val} {key}: rollout targets {got:.4f} "
                                    f"vs reference {want:.4f}")

        row = dict(model=label, job=a.job, twall=int(tw_val), n_sites=n_sites,
                   min_area=a.min_area)
        row.update(pred)
        row.update({f"ref_{k}": v for k, v in ref.items()})
        row["nuc_deficit"] = ref["nuc_rate"] - pred["nuc_rate"]
        # NaN, not a huge number, when the reference rate is ~0.
        row["nuc_ratio"] = (pred["nuc_rate"] / ref["nuc_rate"]
                            if ref["nuc_rate"] > 1e-6 else float("nan"))
        row["void_ratio"] = (pred["void_fraction"] / ref["void_fraction"]
                             if ref["void_fraction"] > 1e-6 else float("nan"))
        for c, f in enumerate(FIELDS[: rel.shape[1]]):
            row[f"rel_l2_{f}"] = float(rel[:, c].mean())
            row[f"rel_l2_{f}_last"] = float(rel[-1, c])
        rows.append(row)

        print(f"  Twall_{tw_val:<4} sites={n_sites:<3} "
              f"|grad|={pred['grad_mean']:.4f} (ref {ref['grad_mean']:.4f})  "
              f"nuc/frame={pred['nuc_rate']:.3f} (ref {ref['nuc_rate']:.3f}, "
              f"deficit {row['nuc_deficit']:+.3f})  void={pred['void_fraction']:.4f} "
              f"(ref {ref['void_fraction']:.4f})  width={pred['interface_width']:.2f}  "
              f"[{time.time() - t0:.0f}s]", flush=True)

        if a.save_fields:
            os.makedirs(a.save_fields, exist_ok=True)
            np.save(os.path.join(a.save_fields, f"dfun_pred_Twall_{tw_val}.npy"), phi_p)

    if not rows:
        raise SystemExit("no sims measured")

    if a.teacher_forced:
        out = os.path.join(a.out, f"model_diagnostics_{a.job}_minarea{a.min_area}_teacherforced.csv")
        keys = list(rows[0].keys())
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(",".join(keys) + "\n")
            for r in sorted(rows, key=lambda r: (r["n_sites"], r["twall"])):
                fh.write(",".join(f"{r[k]:.6g}" if isinstance(r[k], float) else str(r[k])
                                  for k in keys) + "\n")
        print(f"\nwrote {out}")
        return 0

    # ---- self-check verdict
    print("\n=== SELF-CHECK (rollout targets vs reference curve) ===")
    if not comparable:
        if (a.align_to, a.frames) != REFERENCE_WINDOW:
            print(f"  SKIPPED — rollout window ({a.align_to}, {a.frames}) != {REFERENCE_WINDOW}.")
            print("  Re-run at the reference window to compare with the reference curve.")
        else:
            print(f"  SKIPPED — --min-area {a.min_area} != {REFERENCE_MIN_AREA}, the threshold "
                  f"the reference CSV was")
            print("  measured at. ref_* is re-measured here, so pred/ref is internally consistent.")
    elif warnings:
        print(f"  FAILED — {len(warnings)} mismatches beyond {SELF_CHECK_RTOL:.0%}:")
        for w in warnings:
            print(f"    {w}")
        print("  Frame alignment, units or aggregation is wrong. The model numbers above are")
        print("  NOT comparable to reference_diagnostics.csv.")
    else:
        print(f"  PASS — targets reproduce the reference on all {len(rows)} sims.")
        print("  Alignment, units and aggregation are right; model numbers are comparable.")

    # ---- nucleation against site count
    print("\n=== nucleation vs site count ===")
    print(f"  {'sites':>6} {'ref':>8} {'pred':>8} {'deficit':>9} {'ratio':>7} {'void ratio':>11}")
    for r in sorted(rows, key=lambda r: r["n_sites"]):
        print(f"  {r['n_sites']:>6} {r['ref_nuc_rate']:>8.3f} {r['nuc_rate']:>8.3f} "
              f"{r['nuc_deficit']:>+9.3f} {r['nuc_ratio']:>7.3f} {r['void_ratio']:>11.3f}")

    s = np.array([r["n_sites"] for r in rows], float)
    d = np.array([r["nuc_deficit"] for r in rows], float)
    if len(s) > 2 and d.std() > 0:
        corr = np.corrcoef(s, d)[0, 1]
        print(f"\n  corr(n_sites, deficit) = {corr:+.3f}")

    # The deficit grows with the reference rate even for a model with a constant rate; the ratio
    # (pred / ref) does not, so it is the quantity to compare across site counts.
    ratios = np.array([r["nuc_ratio"] for r in rows], float)
    finite = np.isfinite(ratios)
    if finite.sum() > 2 and ratios[finite].std() > 0:
        rc = np.corrcoef(s[finite], ratios[finite])[0, 1]
        print(f"  corr(n_sites, ratio)   = {rc:+.3f}")

    # The file name carries the counting threshold, so runs at different thresholds do not overwrite each other.
    suffix = f"_minarea{a.min_area}" + ("_projected" if a.project else "")
    out = os.path.join(a.out, f"model_diagnostics_{a.job}{suffix}.csv")
    keys = list(rows[0].keys())
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(",".join(keys) + "\n")
        for r in sorted(rows, key=lambda r: r["n_sites"]):
            fh.write(",".join(f"{r[k]:.6g}" if isinstance(r[k], float) else str(r[k])
                              for k in keys) + "\n")
    print(f"\nwrote {out}")
    return 1 if warnings else 0


if __name__ == "__main__":
    sys.exit(main())
