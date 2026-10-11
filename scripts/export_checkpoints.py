#!/usr/bin/env python3
"""Export the 24 trained models for deposit: weights and training configuration only.

For each run in configs/runs.csv, loads the final Lightning checkpoint (epoch 150) and writes
<out>/run<id>.ckpt holding `state_dict`, `hyper_parameters` (with local data paths replaced by
`${data_home}`) and `epoch`/`global_step`, without optimiser and scheduler states. The result loads
with `scripts/inference.build_model` exactly like the original. Also writes <out>/manifest.csv
(run, paper label, architecture, file, size, sha256) and checks that every exported tensor equals
the original.

    python scripts/export_checkpoints.py --ckpt-root $WORK_ROOT/checkpoints --out <dir> [--data-root <local data path>]
"""
import argparse
import csv
import glob
import hashlib
import os

import torch


def scrub(obj, old, new):
    if isinstance(obj, dict):
        return {k: scrub(v, old, new) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(scrub(v, old, new) for v in obj)
    if isinstance(obj, str) and old:
        return obj.replace(old, new)
    return obj


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-root", default=None, help="local data path to replace by ${data_home}")
    ap.add_argument("--runs", default=os.path.join(os.path.dirname(__file__), "..", "configs", "runs.csv"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    runs = list(csv.DictReader(open(a.runs)))
    rows = []
    for r in runs:
        src = glob.glob(os.path.join(a.ckpt_root, f"*_{r['run']}", "lightning_logs", "version_0", "checkpoints",
                                     "epoch=149-*.ckpt"))
        assert len(src) == 1, (r["run"], src)
        ck = torch.load(src[0], map_location="cpu", weights_only=False)
        hp = dict(ck["hyper_parameters"])
        if a.data_root:
            hp = scrub(hp, a.data_root.rstrip("/"), "${data_home}")
        out = {"state_dict": ck["state_dict"], "hyper_parameters": hp,
               "epoch": ck.get("epoch"), "global_step": ck.get("global_step"),
               "run": r["run"], "paper_label": r["paper_label"]}
        dst = os.path.join(a.out, f"run{r['run']}.ckpt")
        torch.save(out, dst)
        back = torch.load(dst, map_location="cpu", weights_only=False)
        assert back["state_dict"].keys() == ck["state_dict"].keys()
        assert all(torch.equal(back["state_dict"][k], ck["state_dict"][k]) for k in ck["state_dict"])
        size = os.path.getsize(dst)
        rows.append(dict(run=r["run"], paper_label=r["paper_label"], architecture=r["architecture"],
                         seed=r["seed"], file=os.path.basename(dst), bytes=size, sha256=sha256(dst)))
        print(f"run {r['run']:>5}: {size / 1e6:8.1f} MB  <- {os.path.getsize(src[0]) / 1e6:8.1f} MB", flush=True)
    with open(os.path.join(a.out, "manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} checkpoints and manifest.csv to {a.out}")


if __name__ == "__main__":
    main()
