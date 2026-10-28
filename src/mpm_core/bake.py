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
from fractions import Fraction
from pathlib import Path

import numpy as np

from mpm_core.backends import create_solver
from mpm_core.cache import (CacheWriter, exclusive_lock, read_checkpoint, stdin_closed, write_checkpoint,
                            write_json)
from mpm_core.geometry import interpolate_transform, mesh_is_closed, sample_mesh_volume, transform_points
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
    def __init__(self, cache_dir: Path):
        self.path = cache_dir / "progress.json"
        self.doc = {"status": "starting", "frame": None, "frame_end": None, "particles": 0,
                    "ms_per_frame": None, "message": "", "warnings": []}

    def warn(self, message: str) -> None:
        if message not in self.doc["warnings"]:
            self.doc["warnings"].append(message)

    def update(self, **fields) -> None:
        self.doc.update(fields)
        write_json(self.path, self.doc)


def _emit(scene: SceneDescription, solver: Solver, emitter_index: int, frame: int, s: Fraction,
          progress: _Progress) -> None:
    """Emit at time ``frame - 1 + s`` (just ``frame`` on the first frame), the pose interpolated in between."""
    e: Emitter = scene.emitters[emitter_index]
    k = frame - scene.frame_start
    xf = interpolate_transform(e.transforms[k - 1], e.transforms[k], float(s)) if k > 0 else e.transforms[k]
    spacing = scene.domain.dx / round(scene.particles_per_cell ** (1 / 3))
    seed = scene.seed + emitter_index if e.kind == "fill" else [scene.seed, frame, emitter_index, s.numerator,
                                                                 s.denominator]
    pts = sample_mesh_volume(transform_points(xf, e.mesh.verts), e.mesh.tris, spacing, seed)
    if len(pts) == 0:
        progress.warn(f"{e.name}: no particles, the mesh is smaller than the particle spacing ({spacing:.4g} m)")
        return

    lo, hi = np.asarray(scene.domain.min), np.asarray(scene.domain.max)
    inside = ((pts >= lo) & (pts <= hi)).all(axis=1)
    if not inside.all():
        progress.warn(f"{e.name}: particles outside the domain were dropped")
        pts = pts[inside]
    if e.kind == "inflow" and len(pts) and solver.particle_count:
        pts = pts[~solver.is_occupied(pts)]
    if len(pts) == 0:
        return

    vel = np.tile(np.asarray(e.velocity, np.float32), (len(pts), 1))
    if e.use_object_velocity and len(e.transforms) > 1:
        a, b = (k - 1, k) if k > 0 else (0, 1)  # forward difference on the first frame
        local = transform_points(np.linalg.inv(xf), pts)
        vel += (transform_points(e.transforms[b], local) - transform_points(e.transforms[a], local)) * scene.fps
    n = len(pts)
    solver.add_particles(pts, vel, np.full(n, e.material, np.int32), np.full(n, emitter_index, np.int32))


def _active_emitters(scene: SceneDescription, frame: int) -> list[int]:
    """Emitters that emit at the start frame itself."""
    out = []
    for i, e in enumerate(scene.emitters):
        if e.kind == "fill" and frame == scene.frame_start:
            out.append(i)
        elif e.kind == "inflow" and e.frame_start <= frame <= e.frame_end:
            out.append(i)
    return out


def _emission_times(scene: SceneDescription, frame: int) -> list[tuple[Fraction, list[int]]]:
    """Inflow emissions while advancing from ``frame - 1`` to ``frame``: (fraction of the frame, emitters).

    An inflow with m substeps emits at i/m, i = 1..m, of each frame it is active; on its first frame only
    at 1, the frame itself. In time order, always ending at 1 (possibly with no emitters).
    """
    times: dict[Fraction, list[int]] = {Fraction(1): []}
    for i, e in enumerate(scene.emitters):
        if e.kind != "inflow" or not e.frame_start <= frame <= e.frame_end:
            continue
        steps = range(1, e.substeps + 1) if frame > e.frame_start else [e.substeps]
        for j in steps:
            times.setdefault(Fraction(j, e.substeps), []).append(i)
    return sorted(times.items())


def bake(cache_dir: Path, *, backend: str = "warp", device: str | None = None, resume: bool = False,
         should_cancel: Callable[[], bool] | None = None) -> BakeResult:
    cache_dir = Path(cache_dir)
    try:
        lock = exclusive_lock(cache_dir / "bake.lock")
        lock.__enter__()
    except RuntimeError as exc:
        raise RuntimeError(f"a bake is already running on {cache_dir}") from exc
    try:
        return _bake_locked(cache_dir, backend, device, resume, should_cancel)
    finally:
        lock.__exit__(None, None, None)


def _bake_locked(cache_dir: Path, backend: str, device: str | None, resume: bool,
                 should_cancel: Callable[[], bool] | None) -> BakeResult:
    scene_dir = cache_dir / "scene"
    cancel_file = cache_dir / "cancel"
    cancel_file.unlink(missing_ok=True)  # a leftover request from an earlier bake
    progress = _Progress(cache_dir)
    progress.update()  # replace the previous bake's status right away
    written = 0

    def cancelled() -> bool:
        if cancel_file.exists():
            cancel_file.unlink()
            return True
        return bool(should_cancel and should_cancel())

    try:
        scene = read_scene(scene_dir)
        current_hash = scene_hash(scene_dir)
        progress.doc["frame_end"] = scene.frame_end
        for e in scene.emitters:
            if not mesh_is_closed(e.mesh.tris):
                raise ValueError(f"emitter {e.name!r}: mesh is not closed (every edge must be shared by exactly "
                                 f"two triangles; in Blender, try Merge by Distance)")
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
            shutil.rmtree(cache_dir / "surface", ignore_errors=True)  # meshes of the previous simulation
            (cache_dir / "frames").mkdir()
            first = scene.frame_start

        timings = []
        for frame in range(first, scene.frame_end + 1):
            t0 = time.perf_counter()
            if frame == scene.frame_start:
                for i in _active_emitters(scene, frame):
                    _emit(scene, solver, i, frame, Fraction(1), progress)
            else:
                k = frame - scene.frame_start
                done = Fraction(0)
                for s, emitters in _emission_times(scene, frame):  # advance in pieces, emitting in between
                    if scene.colliders:
                        solver.set_collider_transforms(
                            np.stack([interpolate_transform(c.transforms[k - 1], c.transforms[k], float(done))
                                      for c in scene.colliders]),
                            np.stack([interpolate_transform(c.transforms[k - 1], c.transforms[k], float(s))
                                      for c in scene.colliders]))
                    solver.advance(float(s - done) / scene.fps)
                    for i in emitters:
                        _emit(scene, solver, i, frame, s, progress)
                    done = s
            particles = solver.get_particles()
            if not (np.isfinite(particles.positions).all() and np.isfinite(particles.velocities).all()):
                message = f"simulation became unstable at frame {frame} (NaN/inf in particle positions)"
                progress.update(status="error", frame=frame, message=message)
                return BakeResult("error", written, message)
            writer.write_frame(frame, particles)
            write_checkpoint(cache_dir, frame, solver.save_state(), current_hash)
            written += 1
            timings.append((time.perf_counter() - t0) * 1000)
            progress.update(status="running", frame=frame, particles=len(particles.positions),
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
    ap.add_argument("--watch-stdin", action="store_true",
                    help="stop (as if cancelled) when stdin closes, i.e. when the launching process exits")
    args = ap.parse_args(argv)
    parent_gone = stdin_closed() if args.watch_stdin else None

    last = [0.0]

    def report() -> bool:
        # one line per frame for whoever watches stdout
        doc_path = args.cache_dir / "progress.json"
        if doc_path.exists() and doc_path.stat().st_mtime != last[0]:
            last[0] = doc_path.stat().st_mtime
            print(doc_path.read_text(encoding="utf-8").replace("\n", " "), flush=True)
        return parent_gone is not None and parent_gone.is_set()

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
