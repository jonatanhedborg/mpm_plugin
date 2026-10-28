"""Spike scene 2: a ball rolling through a layer of packed snow (animated mesh collider).

    python examples/scene_ball.py OUT_DIR [--res 128] [--frames 96]
"""

import argparse
from pathlib import Path

import numpy as np
from _meshes import box_mesh, icosphere, rot_y, scale, still, translate

from mpm_core.materials import load_preset
from mpm_core.scene import Collider, Domain, Emitter, SceneDescription, write_scene

RADIUS = 0.08
SPEED = 0.2  # m/s along +x
FPS = 24.0


def ball_transforms(frames: int) -> np.ndarray:
    out = []
    for f in range(frames):
        t = f / FPS
        # rolling without slipping: rotation about +y by distance / radius
        out.append(translate(0.1 + SPEED * t, 0.5, RADIUS) @ rot_y(SPEED * t / RADIUS) @ scale(RADIUS))
    return np.stack(out)


def build(res: int, frames: int) -> SceneDescription:
    snow = Emitter("snow", "fill", 0, box_mesh(),
                   still(translate(0.2, 0.25, 0.0) @ scale(0.77, 0.5, 0.07), frames))
    ball = Collider("ball", icosphere(3), ball_transforms(frames), friction=1.0, closed=True)
    return SceneDescription(
        domain=Domain((0, 0, 0), (1, 1, 1), res),
        materials=[load_preset("snow_packed").material],
        emitters=[snow], colliders=[ball], fps=FPS, frame_start=1, frame_end=frames)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--frames", type=int, default=96)
    args = ap.parse_args()
    write_scene(build(args.res, args.frames), args.out_dir / "scene")
    print(f"wrote {args.out_dir / 'scene'}")
