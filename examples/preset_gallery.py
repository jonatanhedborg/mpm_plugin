"""Bake every material preset in the same small test: a block dropped onto the floor, and a stream poured
next to it. For comparing presets, and for checking new ones.

    python examples/preset_gallery.py [OUT_DIR] [--only PRESET ...]

Writes one cache per preset into OUT_DIR (default out/presets). View them side by side with
tools/blender_view_cache.py, setting CACHE_DIR to OUT_DIR.
"""

import argparse
import json
import shutil
import time
from pathlib import Path

from _meshes import box_mesh, still, translate
from test_scenes import box

from mpm_core.bake import bake
from mpm_core.materials import list_presets, load_preset
from mpm_core.scene import Domain, Emitter, SceneDescription, write_scene

FRAMES = 72  # 3 s: 1 s of pouring, 2 s to settle
DOMAIN = Domain((0, 0, 0), (0.8, 0.4, 0.6), 64)


def gallery_scene(preset: str) -> SceneDescription:
    block = Emitter("block", "fill", 0, box_mesh(), still(box((0.1, 0.1, 0.25), (0.3, 0.3, 0.45)), FRAMES))
    pour = Emitter("pour", "inflow", 0, box_mesh(), still(box((0.53, 0.17, 0.5), (0.59, 0.23, 0.56)), FRAMES),
                   velocity=(0, 0, -1), frame_start=1, frame_end=24, substeps=4)
    return SceneDescription(domain=DOMAIN, materials=[load_preset(preset).material], emitters=[block, pour],
                            colliders=[], frame_start=1, frame_end=FRAMES)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", nargs="?", type=Path, default=Path("out/presets"))
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    for preset in args.only or list_presets():
        cache = args.out_dir / preset
        shutil.rmtree(cache, ignore_errors=True)
        write_scene(gallery_scene(preset), cache / "scene")
        t0 = time.perf_counter()
        result = bake(cache)
        progress = json.loads((cache / "progress.json").read_text())
        print(f"{preset:15s} {result.status:6s} {time.perf_counter() - t0:5.1f} s  "
              f"{progress['particles']:7,d} particles  {result.message}", flush=True)
