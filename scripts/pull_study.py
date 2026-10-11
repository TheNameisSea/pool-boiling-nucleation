#!/usr/bin/env python3
"""Download BubbleML 2.0 data from Hugging Face (hpcforge/BubbleML_2, DOI 10.57967/hf/5594) to $BUBBLEML_ROOT.

Usage:
    python scripts/pull_study.py PoolBoiling-Saturated-FC72-2D --meta-only          # JSON metadata only (MB)
    python scripts/pull_study.py PoolBoiling-Saturated-FC72-2D --sims 75 82 90 96 102 107
    python scripts/pull_study.py PoolBoiling-Saturated-FC72-2D                       # whole study (~336 GB)
    python scripts/pull_study.py SingleBubble-Saturated-FC72-2D                      # control (~39 GB)

Refuses to run if the download would leave less than MIN_FREE_GB free on the target filesystem.
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_ROOT  # noqa: E402

REPO = "hpcforge/BubbleML_2"
DEST = Path(DATA_ROOT)
MIN_FREE_GB = 50


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("study")
    ap.add_argument("--yes", action="store_true", help="skip confirmation")
    ap.add_argument("--sims", nargs="*", help="only these Twall values (e.g. 75 82 90)")
    ap.add_argument("--meta-only", action="store_true", help="only the per-simulation JSON metadata")
    a = ap.parse_args()

    api = HfApi()
    info = api.repo_info(REPO, repo_type="dataset", files_metadata=True)
    pats = [f"{a.study}/Twall_{t}." for t in a.sims] if a.sims else [a.study + "/"]
    files = [s for s in info.siblings if any(s.rfilename.startswith(p) for p in pats)
             and (not a.meta_only or s.rfilename.endswith(".json"))]
    if not files:
        sys.exit(f"no files under '{a.study}' in {REPO}")
    need = sum((s.lfs.size if s.lfs else s.size) or 0 for s in files)

    DEST.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(DEST).free
    after = (free - need) / 1e9
    print(f"study      : {a.study}\nfiles      : {len(files)}\nsize       : {need/1e9:.2f} GB")
    print(f"free now   : {free/1e9:.1f} GB\nfree after : {after:.1f} GB")
    if after < MIN_FREE_GB:
        sys.exit(f"REFUSING: would leave {after:.1f} GB (< {MIN_FREE_GB} GB)")
    if not a.yes and input("proceed? [y/N] ").strip().lower() != "y":
        sys.exit("aborted")

    snapshot_download(REPO, repo_type="dataset", local_dir=DEST,
                      allow_patterns=[s.rfilename for s in files], max_workers=8)
    print(f"done -> {DEST / a.study}")


if __name__ == "__main__":
    main()
