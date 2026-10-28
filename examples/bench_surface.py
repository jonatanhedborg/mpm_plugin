"""Surface a baked cache (all emitters, default settings unless surface_settings.json exists) and report timing.

    python examples/bench_surface.py CACHE_DIR [--device cuda:0]
"""

import argparse
import json
import time
from pathlib import Path

from mpm_surface.run import surface

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cache_dir", type=Path)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    t0 = time.perf_counter()
    result = surface(args.cache_dir, device=args.device)
    total = time.perf_counter() - t0
    progress = json.loads((args.cache_dir / "surface" / "progress.json").read_text())
    print(f"status: {result.status} {result.message}".strip())
    print(f"particles: {progress['particles']:,}")
    print(f"frames meshed: {result.frames_written}, unchanged: {result.frames_skipped}, total {total:.1f} s, "
          f"avg {progress['ms_per_frame'] or 0:.0f} ms/frame (incl. first-run kernel compile)")
    for w in progress["warnings"]:
        print(f"warning: {w}")
