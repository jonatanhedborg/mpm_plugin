"""Bake runner: scene description in, cache out.

    python -m mpm_core.bake CACHE_DIR [--backend warp] [--device cuda:0] [--resume]

Reads ``CACHE_DIR/scene``, writes frames, checkpoints and progress (see mpm_core.cache).
Exit code 0 when done or cancelled, 1 on error.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mpm_core.backends import create_solver
from mpm_core.cache import CacheWriter, read_checkpoint, write_checkpoint, write_json
from mpm_core.geometry import sample_mesh_volume, transform_points
from mpm_core.scene import Emitter, SceneDescription, read_scene, scene_hash
from mpm_core.solver import ColliderGeometry, Solver


@dataclass
class BakeResult:
    status: str  # "done" | "cancelled" | "error"
    frames_written: int
    message: str = ""


class SceneChangedError(RuntimeError):
    pass


class _Progress:
    def __init__(self, cache_dir: Path, frame_end: int):
        self.path = cache_dir / "progress.json"
        self.doc = {"status": "running", "frame": None, "frame_end": frame_end, "particles": 0,
                    "ms_per_frame": None, "message": "", "warnings": []}

    def warn(self, message: str) -> None:
        if message not in self.doc["warnings"]:
            self.doc["warnings"].append(message)

    def update(self, **fields) -> None:
        self.doc.update(fields)
        write_json(self.path, self.doc)


def _emit(scene: SceneDescription, solver: Solver, emitter_index: int, frame: int, progress: _Progress) -> None:
    e: Emitter = scene.emitters[emitter_index]
    k = frame - scene.frame_start
    spacing = scene.domain.dx / round(scene.particles_per_cell ** (1 / 3))
    seed = scene.seed + emitter_index if e.kind == "fill" else scene.seed + 1000 * frame + emitter_index
    pts = sample_mesh_volume(transform_points(e.transforms[k], e.mesh.verts), e.mesh.tris, spacing, seed)

    lo, hi = np.asarray(scene.domain.min), np.asarray(scene.domain.max)
    inside = ((pts >= lo) & (pts <= hi)).all(axis=1)
    if not inside.all():
        progress.warn(f"{e.name}: {int((~inside).sum())} particles outside the domain were dropped")
        pts = pts[inside]
    if e.kind == "inflow" and len(pts) and solver.particle_count:
        pts = pts[~solver.is_occupied(pts)]
    if len(pts) == 0:
        return

    vel = np.tile(np.asarray(e.velocity, np.float32), (len(pts), 1))
    if e.use_object_velocity and k > 0:
        local = transform_points(np.linalg.inv(e.transforms[k]), pts)
        vel += (transform_points(e.transforms[k], local) - transform_points(e.transforms[k - 1], local)) * scene.fps
    n = len(pts)
    solver.add_particles(pts, vel, np.full(n, e.material, np.int32), np.full(n, emitter_index, np.int32))


def _active_emitters(scene: SceneDescription, frame: int) -> list[int]:
    out = []
    for i, e in enumerate(scene.emitters):
        if e.kind == "fill" and frame == scene.frame_start:
            out.append(i)
        elif e.kind == "inflow" and e.frame_start <= frame <= e.frame_end:
            out.append(i)
    return out


def bake(cache_dir: Path, *, backend: str = "warp", device: str | None = None, resume: bool = False,
         should_cancel: Callable[[], bool] | None = None) -> BakeResult:
    cache_dir = Path(cache_dir)
    scene_dir = cache_dir / "scene"
    scene = read_scene(scene_dir)
    current_hash = scene_hash(scene_dir)
    progress = _Progress(cache_dir, scene.frame_end)
    cancel_file = cache_dir / "cancel"
    written = 0

    def cancelled() -> bool:
        if cancel_file.exists():
            cancel_file.unlink()
            return True
        return bool(should_cancel and should_cancel())

    try:
        solver = create_solver(backend, domain=scene.domain, materials=scene.materials, gravity=scene.gravity,
                               particles_per_cell=scene.particles_per_cell, device=device)
        solver.set_colliders([ColliderGeometry(c.mesh, c.closed, c.friction) for c in scene.colliders])
        writer = CacheWriter(cache_dir, [e.name for e in scene.emitters], scene.fps, scene.frame_start,
                             scene.frame_end, scene.domain.dx, backend)

        if resume:
            checkpoint = read_checkpoint(cache_dir)
            if checkpoint is None:
                raise FileNotFoundError(f"no checkpoint to resume from in {cache_dir}")
            last_frame, state, saved_hash = checkpoint
            if saved_hash != current_hash:
                raise SceneChangedError("the scene changed since this bake was started; bake from scratch")
            solver.load_state(state)
            first = last_frame + 1
        else:
            shutil.rmtree(cache_dir / "frames", ignore_errors=True)
            shutil.rmtree(cache_dir / "state", ignore_errors=True)
            (cache_dir / "frames").mkdir()
            first = scene.frame_start

        timings = []
        for frame in range(first, scene.frame_end + 1):
            t0 = time.perf_counter()
            if frame > scene.frame_start:
                k = frame - scene.frame_start
                if scene.colliders:
                    solver.set_collider_transforms(np.stack([c.transforms[k - 1] for c in scene.colliders]),
                                                   np.stack([c.transforms[k] for c in scene.colliders]))
                solver.advance(1.0 / scene.fps)
            for i in _active_emitters(scene, frame):
                _emit(scene, solver, i, frame, progress)
            particles = solver.get_particles()
            if not np.isfinite(particles.positions).all():
                message = f"simulation became unstable at frame {frame} (NaN/inf in particle positions)"
                progress.update(status="error", frame=frame, message=message)
                return BakeResult("error", written, message)
            writer.write_frame(frame, particles)
            write_checkpoint(cache_dir, frame, solver.save_state(), current_hash)
            written += 1
            timings.append((time.perf_counter() - t0) * 1000)
            progress.update(frame=frame, particles=len(particles.positions),
                            ms_per_frame=round(float(np.mean(timings)), 1))
            if frame < scene.frame_end and cancelled():
                progress.update(status="cancelled")
                return BakeResult("cancelled", written)
        progress.update(status="done")
        return BakeResult("done", written)
    except Exception as exc:
        progress.update(status="error", message=f"{type(exc).__name__}: {exc}")
        raise


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m mpm_core.bake", description=__doc__.split("\n\n")[0])
    ap.add_argument("cache_dir", type=Path)
    ap.add_argument("--backend", default="warp")
    ap.add_argument("--device", default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)

    last = [0.0]

    def report() -> bool:
        # one line per frame for whoever watches stdout
        doc_path = args.cache_dir / "progress.json"
        if doc_path.exists() and doc_path.stat().st_mtime != last[0]:
            last[0] = doc_path.stat().st_mtime
            print(doc_path.read_text(encoding="utf-8").replace("\n", " "), flush=True)
        return False

    try:
        result = bake(args.cache_dir, backend=args.backend, device=args.device, resume=args.resume,
                      should_cancel=report)
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"{result.status}: {result.frames_written} frames written. {result.message}".strip())
    return 1 if result.status == "error" else 0


if __name__ == "__main__":
    sys.exit(main())
