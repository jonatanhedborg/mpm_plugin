"""THROWAWAY SPIKE: jelly cube dropped on a sharp wedge, unbreakable vs breakable.

    python spike/fracture_spike.py [OUT_DIR] [--res 128]

Bakes into OUT_DIR/<variant> (default out/spike_fracture); view with tools/blender_view_cache.py.
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from _meshes import box_mesh, scale, still, translate  # noqa: E402

from mpm_core.bake import bake  # noqa: E402
from mpm_core.materials import MaterialParams, load_preset  # noqa: E402
from mpm_core.scene import Collider, Domain, Emitter, MeshData, SceneDescription, write_scene  # noqa: E402

FRAMES = 48


def wedge_mesh() -> MeshData:
    """Closed triangular prism: base [-0.5, 0.5] in x at z=0, ridge at x=0, z=1; y in [-0.5, 0.5]."""
    v = np.array([[-0.5, -0.5, 0], [0.5, -0.5, 0], [0, -0.5, 1],
                  [-0.5, 0.5, 0], [0.5, 0.5, 0], [0, 0.5, 1]], np.float32)
    t = [[0, 1, 2], [3, 5, 4], [0, 3, 4], [0, 4, 1], [1, 4, 5], [1, 5, 2], [0, 2, 5], [0, 5, 3]]
    return MeshData(v, np.array(t, np.int32))


def jelly(critical_stretch: float | None) -> MaterialParams:
    base = load_preset("jelly").material.to_dict()
    if critical_stretch is not None:
        base.update(name=f"jelly_break_{critical_stretch}", model="elastic_fracture",
                    critical_stretch=critical_stretch)
    return MaterialParams.from_dict(base)


def build(res: int, critical_stretch: float | None) -> SceneDescription:
    cube = translate(0.41, 0.4, 0.45) @ scale(0.2)  # slightly off the ridge so it doesn't split perfectly evenly
    wedge = translate(0.5, 0.5, 0.0) @ scale(0.35, 0.6, 0.3)
    return SceneDescription(
        domain=Domain((0, 0, 0), (1, 1, 1), res),
        materials=[jelly(critical_stretch)],
        emitters=[Emitter("jelly", "fill", 0, box_mesh(), still(cube, FRAMES))],
        colliders=[Collider("wedge", wedge_mesh(), still(wedge, FRAMES), friction=0.3)],
        frame_start=1, frame_end=FRAMES)


VARIANTS = {"1_unbreakable": None, "2_tough_stretch_0.5": 0.5, "3_brittle_stretch_0.15": 0.15}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path, nargs="?", default=Path("out/spike_fracture"))
    ap.add_argument("--res", type=int, default=128)
    args = ap.parse_args()
    for name, crit in VARIANTS.items():
        out = args.out_dir / name
        shutil.rmtree(out, ignore_errors=True)
        write_scene(build(args.res, crit), out / "scene")
        t0 = time.perf_counter()
        r = bake(out)
        print(f"{name:26s} {r.status} {r.frames_written} frames {time.perf_counter() - t0:5.1f} s {r.message}",
              flush=True)
