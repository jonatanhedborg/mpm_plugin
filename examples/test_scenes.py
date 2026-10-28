"""Bake the GPU test setups (plus a few variants) as scenes you can scrub in Blender.

    python examples/test_scenes.py [OUT_DIR] [--only NAME ...]

Writes one cache per scene into OUT_DIR (default out/tests). View them all side by side with
tools/blender_view_cache.py, setting CACHE_DIR to OUT_DIR.
"""

import argparse
import shutil
import time
from pathlib import Path

import numpy as np
from _meshes import box_mesh, icosphere, plane_mesh, rot_x, rot_y, scale, still, translate

from mpm_core.bake import bake
from mpm_core.materials import load_preset
from mpm_core.scene import Collider, Domain, Emitter, SceneDescription, write_scene

UNIT = Domain((0, 0, 0), (1, 1, 1), 64)
DX = UNIT.dx


def box(lo, hi) -> np.ndarray:
    return translate(*lo) @ scale(*np.subtract(hi, lo))


def fill(name, preset_index, xf, frames, mesh=None, velocity=(0, 0, 0)) -> Emitter:
    return Emitter(name, "fill", preset_index, mesh or box_mesh(), still(xf, frames), velocity=velocity)


def scene(presets, emitters, frames, colliders=(), domain=UNIT, gravity=(0, 0, -9.81)) -> SceneDescription:
    return SceneDescription(domain=domain, materials=[load_preset(p).material for p in presets],
                            emitters=list(emitters), colliders=list(colliders), gravity=gravity,
                            frame_start=1, frame_end=frames)


def tilted_cube(degrees, friction, frames=25, preset="whipped_cream") -> SceneDescription:
    plane_xf = translate(0.5, 0.5, 0.3) @ rot_y(np.radians(degrees))
    cube = plane_xf @ box((-0.05, -0.05, 0.25 * DX), (0.05, 0.05, 0.1))
    plane = Collider("plane", plane_mesh(), still(plane_xf @ scale(0.8), frames), friction=friction, closed=False)
    return scene([preset], [fill("cube", 0, cube, frames)], frames, [plane])


def moving_box() -> SceneDescription:
    frames = 25

    def pusher(x):
        return translate(x, 0.5, 0.05) @ scale(0.1, 0.5, 0.1) @ translate(-0.5, -0.5, -0.5)

    xfs = np.stack([pusher(0.15 + 0.3 * k / 24) for k in range(frames)])
    sand = fill("sand", 0, box((0.1, 0.3, 0), (0.9, 0.7, 0.06)), frames)
    return scene(["sand_dry"], [sand], frames, [Collider("pusher", box_mesh(), xfs, friction=0.5)])


def spinning_sphere(deg_per_frame) -> SceneDescription:
    frames = 8

    def ball(deg):
        return translate(0.5, 0.5, 0.1) @ rot_x(np.radians(deg)) @ scale(0.12)

    xfs = np.stack([ball(0)] + [ball(deg_per_frame * k) for k in range(frames - 1)])
    sand = fill("sand", 0, box((0.25, 0.25, 0), (0.75, 0.75, 0.15)), frames)
    return scene(["sand_dry"], [sand], frames, [Collider("ball", icosphere(5), xfs, friction=0.0)])


def tilted_gravity_floor(zmin) -> SceneDescription:
    a = np.radians(20)
    d = Domain((0, 0, zmin), (1, 1, zmin + 1), 64, wall_friction=0.5)
    cube = fill("cream", 0, box((0.45, 0.45, zmin), (0.55, 0.55, zmin + 0.1)), 25)
    return scene(["whipped_cream"], [cube], 25, domain=d, gravity=(9.81 * np.sin(a), 0, -9.81 * np.cos(a)))


def inflow() -> SceneDescription:
    frames = 48
    xf = box((0.45, 0.45, 0.6), (0.55, 0.55, 0.7))
    e = Emitter("cream", "inflow", 0, box_mesh(), still(xf, frames), velocity=(0, 0, -1), frame_start=1,
                frame_end=24)
    return scene(["whipped_cream"], [e], frames)


SCENES = {
    # test_warp_basic
    "free_fall": lambda: scene(["jelly"], [fill("jelly", 0, box((0.45, 0.45, 0.6), (0.55, 0.55, 0.7)), 7)], 7),
    "rests_on_floor": lambda: scene(["jelly"], [fill("jelly", 0, box((0.4, 0.4, 0), (0.5, 0.5, 0.1)), 25)], 25),
    "floor_friction_placement_0.0100": lambda: tilted_gravity_floor(0.0100),
    # test_warp_materials
    **{f"slab_at_rest_{p}": (lambda p=p: scene(
        [p], [fill(p, 0, box((0, 0, 0), (0.5, 0.5, 0.07)), 25)], 25, domain=Domain((0, 0, 0), (0.5, 0.5, 0.5), 64)))
       for p in ("jelly", "sand_dry", "snow_packed", "snow_powder", "whipped_cream")},
    "sand_column_collapses": lambda: scene(
        ["sand_dry"], [fill("sand", 0, box((0.44, 0.44, 0), (0.56, 0.56, 0.33)), 49)], 49),
    "cream_holds_shape": lambda: scene(
        ["whipped_cream"], [fill("cream", 0, translate(0.5, 0.5, 0.11) @ scale(0.11), 49, mesh=icosphere(4))], 49),
    "jelly_recovers": lambda: scene(["jelly"], [fill("jelly", 0, box((0.4, 0.4, 0.2), (0.6, 0.6, 0.4)), 73)], 73),
    "mixed_materials": lambda: scene(["sand_dry", "snow_packed"], [
        fill("sand", 0, box((0.3, 0.3, 0), (0.5, 0.5, 0.1)), 25),
        fill("snow", 1, box((0.5, 0.5, 0), (0.7, 0.7, 0.1)), 25)], 25),
    # test_warp_colliders
    "rests_on_open_plane": lambda: scene(
        ["jelly"], [fill("jelly", 0, box((0.4, 0.4, 0.45), (0.6, 0.6, 0.55)), 37)], 37,
        [Collider("plane", plane_mesh(), still(translate(0.5, 0.5, 0.3) @ scale(0.8), 37), closed=False)]),
    "scaled_box_collider": lambda: scene(
        ["jelly"], [fill("jelly", 0, box((0.45, 0.45, 0.4), (0.55, 0.55, 0.5)), 37)], 37,
        [Collider("box", box_mesh(), still(box((0.2, 0.2, -0.2), (0.8, 0.8, 0.2)), 37))]),
    "slope_10deg_friction_0": lambda: tilted_cube(10, 0.0),
    "slope_10deg_friction_1": lambda: tilted_cube(10, 1.0),
    "slope_30deg_friction_1_KNOWN_ISSUE": lambda: tilted_cube(30, 1.0),
    "slope_30deg_friction_1_jelly_KNOWN_ISSUE": lambda: tilted_cube(30, 1.0, preset="jelly"),
    "moving_box_clears_sand": moving_box,
    "spinning_sphere_0deg": lambda: spinning_sphere(0),
    "spinning_sphere_90deg": lambda: spinning_sphere(90),
    # test_bake
    "inflow": inflow,
}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path, nargs="?", default=Path("out/tests"))
    ap.add_argument("--only", nargs="*", help="scene names to bake (default: all)")
    args = ap.parse_args()
    names = args.only or list(SCENES)
    for name in names:
        out = args.out_dir / name
        shutil.rmtree(out, ignore_errors=True)
        write_scene(SCENES[name](), out / "scene")
        t0 = time.perf_counter()
        result = bake(out)
        print(f"{name:42s} {result.status:9s} {result.frames_written:3d} frames  {time.perf_counter() - t0:5.1f} s",
              flush=True)
