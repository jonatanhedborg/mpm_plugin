"""Surface a baked particle cache.

    python -m mpm_surface CACHE_DIR [--frames A-B] [--device cuda:0]

Settings come from CACHE_DIR/surface_settings.json (defaults if missing).
Exit code 0 when done or cancelled, 1 on error.
"""

import argparse
import sys
from pathlib import Path

from mpm_surface.run import surface


def _frame_range(text: str) -> list[int]:
    a, _, b = text.partition("-")
    return list(range(int(a), int(b or a) + 1))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m mpm_surface", description=__doc__.split("\n\n")[0])
    ap.add_argument("cache_dir", type=Path)
    ap.add_argument("--frames", type=_frame_range, default=None, help="e.g. 1-48")
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)
    try:
        result = surface(args.cache_dir, frames=args.frames, device=args.device)
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"{result.status}: {result.frames_written} frames meshed, {result.frames_skipped} unchanged")
    return 1 if result.status == "error" else 0


if __name__ == "__main__":
    sys.exit(main())
