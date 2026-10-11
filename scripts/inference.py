#!/usr/bin/env python3
"""Autoregressive rollout + relative-L2 evaluation for a trained Bubbleformer checkpoint.

Adapted from upstream Bubbleformer's scripts/inference.py, with command-line arguments instead of
hard-coded paths and support for conditioned (FiLM) models, which take `forward(x, fluid_params)`.
The rollout logic and the relative-L2 definition are identical to upstream.
`build_model` (checkpoint -> model) is also used by rollout_diagnostics.py and representation_probe.py.

Usage:
    python scripts/inference.py --checkpoint <ckpt> --test-file <sim.hdf5> --out-dir <dir> [--steps 200] [--frames 8]
"""
import argparse
import os
import sys
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from bubbleformer.models import get_model
from bubbleformer.data import BubbleForecast
from bubbleformer.utils.losses import LpLoss

FIELD_LABELS = ["SDF", "Temp", "VelX", "VelY"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--test-file", required=True)
    p.add_argument("--out-dir", required=True,
                   help="small outputs (CSV, summary plot)")
    p.add_argument("--bulk-dir", default=None,
                   help="where predictions.pt (~350 MB) goes; defaults to a sibling of the checkpoint")
    p.add_argument("--steps", type=int, default=200,
                   help="number of timesteps to roll out (default 200, as upstream)")
    p.add_argument("--frames", type=int, default=8,
                   help="how many field snapshots to render (0 = none)")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def build_model(ckpt):
    """Reconstruct the model from the checkpoint's own hyper_parameters."""
    hp = ckpt["hyper_parameters"]
    name = hp["model_cfg"]["name"]
    params = dict(hp["model_cfg"]["params"])
    params["input_fields"] = len(hp["data_cfg"]["input_fields"])
    params["output_fields"] = len(hp["data_cfg"]["output_fields"])
    params["time_window"] = hp["data_cfg"]["time_window"]
    if any(k.startswith("model.wall_path.") for k in ckpt["state_dict"]):   # model trained with src/wall_path.py
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
        import wall_path
        wall_path.install()
    model = get_model(name, **params)

    # Lightning prefixes every key with "model."
    state = OrderedDict((k[6:], v) for k, v in ckpt["state_dict"].items() if k.startswith("model."))
    model.load_state_dict(state)
    return model, name, hp


def plot_relative_l2(preds, targets, timesteps, out_dir):
    """Per-channel relative L2 over rollout time. Definition identical to upstream."""
    diff_norm = torch.norm(preds - targets, p=2, dim=(2, 3))
    bnorm = torch.norm(targets, p=2, dim=(2, 3))
    rel = diff_norm / bnorm

    plt.figure(figsize=(10, 6))
    for c, label in enumerate(FIELD_LABELS[: rel.shape[1]]):
        plt.plot(timesteps.numpy(), rel[:, c].numpy(), label=label)
    plt.xlabel("Time (timesteps)")
    plt.ylabel("Relative L2 error")
    plt.title("Relative L2 error over rollout")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(out_dir, "relative_l2_error.png"), dpi=130, bbox_inches="tight")
    plt.close()
    return rel


def plot_frames(preds, targets, out_dir, n_frames):
    """A few side-by-side SDF/temperature snapshots."""
    if n_frames <= 0:
        return
    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    idxs = np.linspace(0, preds.shape[0] - 1, n_frames).astype(int)

    sdf_t = targets[:, 0]
    sdf_lo, sdf_hi = sdf_t.mean().item() - 3 * sdf_t.std().item(), sdf_t.mean().item() + 3 * sdf_t.std().item()
    tmp_t = targets[:, 1]
    tmp_lo, tmp_hi = tmp_t.min().item(), tmp_t.max().item()

    for i in idxs:
        f, ax = plt.subplots(2, 2, figsize=(10, 9), layout="constrained")
        for row, (arr, title) in enumerate([(targets, "Label"), (preds, "Pred")]):
            im0 = ax[row][0].imshow(arr[i, 0].numpy(), vmin=sdf_lo, vmax=sdf_hi,
                                    cmap="Blues", origin="lower")
            ax[row][0].set_title(f"SDF {title} t={i}")
            ax[row][0].axis("off")
            f.colorbar(im0, ax=ax[row][0], fraction=0.04, pad=0.04)
            im1 = ax[row][1].imshow(arr[i, 1].numpy(), vmin=tmp_lo, vmax=tmp_hi,
                                    cmap="turbo", origin="lower")
            ax[row][1].set_title(f"Temp {title} t={i}")
            ax[row][1].axis("off")
            f.colorbar(im1, ax=ax[row][1], fraction=0.04, pad=0.04)
        plt.savefig(os.path.join(plot_dir, f"{str(i).zfill(4)}.png"), dpi=110, bbox_inches="tight")
        plt.close()
    print(f"rendered {len(idxs)} frames -> {plot_dir}")


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model, model_name, hp = build_model(ckpt)
    data_cfg = hp["data_cfg"]
    conditioned = "film" in model_name.lower()
    print(f"model={model_name}  conditioned={conditioned}  "
          f"time_window={data_cfg['time_window']}  start_time={data_cfg['start_time']}")

    dataset = BubbleForecast(
        filenames=[args.test_file],
        input_fields=list(data_cfg["input_fields"]),
        output_fields=list(data_cfg["output_fields"]),
        norm=data_cfg["normalize"],
        downsample_factor=data_cfg["downsample_factor"],
        time_window=data_cfg["time_window"],
        start_time=data_cfg["start_time"],
        return_fluid_params=conditioned,
    )
    diff_term, div_term = hp["normalization_constants"]
    dataset.normalize(diff_term, div_term)

    model = model.to(args.device).eval()
    criterion = LpLoss(d=2, p=2, reduce_dims=[0, 1], reductions=["mean", "mean"])

    start_time = dataset.start_time
    tw = dataset.time_window
    preds, targets, timesteps, step_losses = [], [], [], []

    with torch.no_grad():
        for itr in range(0, args.steps, tw):
            sample = dataset[itr]
            if conditioned:
                inp, tgt, fluid_params = sample
            else:
                inp, tgt = sample
                fluid_params = None

            # Autoregressive: after the first window, feed the model its own output.
            if preds:
                inp = preds[-1]
            inp = inp.to(args.device).float().unsqueeze(0)

            if conditioned:
                fp = fluid_params.to(args.device).float().unsqueeze(0)
                pred = model(inp, fp)
            else:
                pred = model(inp)

            pred = pred.squeeze(0).detach().cpu()
            tgt = tgt.detach().cpu()
            loss = criterion(pred, tgt).item()
            step_losses.append(loss)
            preds.append(pred)
            targets.append(tgt)
            timesteps.append(torch.arange(start_time + itr + tw, start_time + itr + 2 * tw))
            print(f"  itr {itr:4d}  window [{start_time+itr+tw}, {start_time+itr+2*tw})  LpLoss {loss:.4f}")

    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)
    timesteps = torch.cat(timesteps, dim=0)

    # Full prediction tensors, ~350 MB for a 200-step rollout.
    bulk_dir = args.bulk_dir or os.path.join(os.path.dirname(args.checkpoint), "eval_outputs")
    os.makedirs(bulk_dir, exist_ok=True)
    bulk_path = os.path.join(bulk_dir, "predictions.pt")
    torch.save({"preds": preds, "targets": targets, "timesteps": timesteps}, bulk_path)
    print(f"bulk tensors -> {bulk_path} ({os.path.getsize(bulk_path)/1e6:.0f} MB, not tracked)")

    rel = plot_relative_l2(preds, targets, timesteps, args.out_dir)
    plot_frames(preds, targets, args.out_dir, args.frames)

    # Per-step relative L2 per field.
    with open(os.path.join(args.out_dir, "rollout_metrics.csv"), "w", encoding="utf-8") as f:
        f.write("timestep," + ",".join(f"rel_l2_{l.lower()}" for l in FIELD_LABELS[: rel.shape[1]]) + "\n")
        for t in range(rel.shape[0]):
            f.write(f"{int(timesteps[t])}," + ",".join(f"{rel[t, c]:.6f}" for c in range(rel.shape[1])) + "\n")

    print(f"\nrollout {preds.shape[0]} steps from {os.path.basename(args.test_file)}")
    print(f"mean LpLoss over windows : {np.mean(step_losses):.4f}")
    for c, label in enumerate(FIELD_LABELS[: rel.shape[1]]):
        print(f"  rel L2 {label:5s}  first={rel[0, c]:.4f}  last={rel[-1, c]:.4f}  mean={rel[:, c].mean():.4f}")
    print(f"\nwrote -> {args.out_dir}")


if __name__ == "__main__":
    main()
