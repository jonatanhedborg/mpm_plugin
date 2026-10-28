"""Bake a cache dir (written by one of the scene_*.py scripts) and report timing.

    python examples/bench.py OUT_DIR [--device cuda:0]
"""

import argparse
import json
import time
from pathlib import Path

from mpm_core.bake import bake

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    t0 = time.perf_counter()
    result = bake(args.out_dir, device=args.device)
    total = time.perf_counter() - t0
    progress = json.loads((args.out_dir / "progress.json").read_text())
    print(f"status: {result.status} {result.message}".strip())
    print(f"particles: {progress['particles']:,}")
    print(f"frames: {result.frames_written}, total {total:.1f} s, "
          f"avg {progress['ms_per_frame']:.0f} ms/frame (incl. first-run kernel compile)")
    for w in progress["warnings"]:
        print(f"warning: {w}")
