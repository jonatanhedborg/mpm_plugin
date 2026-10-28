"""Surface runner: particle cache in, mesh cache out. Re-running only re-meshes what changed.

Progress goes to ``<cache>/surface/progress.json`` (same schema as the bake runner's); creating
``<cache>/surface/cancel`` stops the run after the current frame.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mpm_core.cache import CacheReader, write_json
from mpm_surface.field import surface_particles
from mpm_surface.meshcache import MeshCacheWriter
from mpm_surface.settings import SurfaceSettings, read_settings


@dataclass
class SurfaceResult:
    status: str  # "done" | "cancelled" | "error"
    frames_written: int
    frames_skipped: int
    message: str = ""


def _particle_key(positions: np.ndarray, settings: SurfaceSettings) -> str:
    return hashlib.sha1(np.ascontiguousarray(positions).tobytes()).hexdigest() + ":" + settings.hash()


def surface(cache_dir: Path, *, settings: dict[str, SurfaceSettings] | None = None, frames: list[int] | None = None,
            device: str | None = None, should_cancel: Callable[[], bool] | None = None) -> SurfaceResult:
    cache_dir = Path(cache_dir)
    surf_dir = cache_dir / "surface"
    surf_dir.mkdir(parents=True, exist_ok=True)
    cancel_file = surf_dir / "cancel"
    cancel_file.unlink(missing_ok=True)  # a leftover request from an earlier run
    doc = {"status": "starting", "frame": None, "frame_end": None, "particles": 0, "ms_per_frame": None,
           "message": "", "warnings": []}

    def update(**fields):
        doc.update(fields)
        write_json(surf_dir / "progress.json", doc)

    def cancelled() -> bool:
        if cancel_file.exists():
            cancel_file.unlink()
            return True
        return bool(should_cancel and should_cancel())

    update()
    written = skipped = 0
    try:
        reader = CacheReader(cache_dir)
        ppc = json.loads((cache_dir / "scene" / "scene.json").read_text(encoding="utf-8"))["particles_per_cell"]
        spacing = reader.dx / round(ppc ** (1 / 3))
        settings = read_settings(cache_dir) if settings is None else settings
        for name in settings:
            if name not in reader.emitter_names:
                doc["warnings"].append(f"surface settings for unknown emitter {name} ignored")
        resolved = {name: settings.get(name, SurfaceSettings()) for name in reader.emitter_names}
        writer = MeshCacheWriter(cache_dir)
        for name, s in resolved.items():
            if not s.enabled:
                writer.remove_emitter(name)  # display falls back to points
        active = [name for name, s in resolved.items() if s.enabled]

        todo = reader.available_frames()
        if frames is not None:
            todo = [f for f in todo if f in set(frames)]
        doc["frame_end"] = todo[-1] if todo else None
        timings = []
        for index, frame in enumerate(todo):
            t0 = time.perf_counter()
            data = reader.read_frame(frame)
            meshed = False
            for name in active:
                positions, velocities = data[name]
                key = _particle_key(positions, resolved[name])
                if writer.key(name, frame) == key:
                    continue
                mesh = surface_particles(positions, velocities, spacing, resolved[name], device)
                writer.write(name, frame, mesh, key)
                meshed = True
            if meshed:
                written += 1
                timings.append((time.perf_counter() - t0) * 1000)
            else:
                skipped += 1
            update(status="running", frame=frame,
                   particles=int(sum(len(data[n][0]) for n in active)),
                   ms_per_frame=round(float(np.mean(timings)), 1) if timings else None)
            if index < len(todo) - 1 and cancelled():
                update(status="cancelled")
                return SurfaceResult("cancelled", written, skipped)
        update(status="done")
        return SurfaceResult("done", written, skipped)
    except Exception as exc:
        update(status="error", message=f"{type(exc).__name__}: {exc}")
        raise
