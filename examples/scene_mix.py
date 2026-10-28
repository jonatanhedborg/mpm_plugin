"""Spike scene 1: a cream ball, a sand column and a thrown snowball in one domain.

    python examples/scene_mix.py OUT_DIR [--res 128] [--frames 96]
"""

import argparse
from pathlib import Path

from _meshes import box_mesh, icosphere, scale, still, translate

from mpm_core.materials import load_preset
from mpm_core.scene import Domain, Emitter, SceneDescription, write_scene


def build(res: int, frames: int) -> SceneDescription:
    sphere = icosphere(3)
    emitters = [
        Emitter("cream", "fill", 0, sphere, still(translate(0.22, 0.5, 0.3) @ scale(0.11), frames)),
        Emitter("sand", "fill", 1, box_mesh(), still(translate(0.44, 0.44, 0.0) @ scale(0.12, 0.12, 0.33), frames)),
        Emitter("snow", "fill", 2, sphere, still(translate(0.82, 0.5, 0.3) @ scale(0.08), frames),
                velocity=(-2.0, 0.0, 0.5)),
    ]
    return SceneDescription(
        domain=Domain((0, 0, 0), (1, 1, 1), res),
        materials=[load_preset(p).material for p in ("whipped_cream", "sand_dry", "snow_packed")],
        emitters=emitters, colliders=[], frame_start=1, frame_end=frames)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--frames", type=int, default=96)
    args = ap.parse_args()
    write_scene(build(args.res, args.frames), args.out_dir / "scene")
    print(f"wrote {args.out_dir / 'scene'}")
