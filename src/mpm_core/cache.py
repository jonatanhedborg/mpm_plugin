"""Bake cache on disk. Backend-independent; the Blender add-on reads it directly.

Layout of a cache directory::

    scene/                      scene description (see mpm_core.scene), written by the add-on
    cache.json                  {"version": 1, "emitters": [names], "fps", "frame_start", "frame_end",
                                 "dx", "backend"}
    frames/frame_NNNN.npz       per emitter i: pos_i (n,3) float32, vel_i (n,3) float32
    state/checkpoint.npz        full solver state of the last completed frame, plus
                                __frame and __scene_hash entries
    progress.json               {"status": "running|done|cancelled|error", "frame", "frame_end",
                                 "particles", "ms_per_frame", "message", "warnings": [...]}
    cancel                      if present, the runner stops after the current frame and deletes it

Every file is written to a ``.tmp`` sibling and then renamed, so readers never see a partial file
and frames can be read while a bake is still running.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from mpm_core.solver import ParticleData

FORMAT_VERSION = 1


def _atomic_write_bytes(path: Path, write) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        write(f)
    os.replace(tmp, path)


def write_json(path: Path, doc: dict) -> None:
    _atomic_write_bytes(Path(path), lambda f: f.write(json.dumps(doc, indent=2).encode("utf-8")))


def frame_path(cache_dir: Path, frame: int) -> Path:
    return Path(cache_dir) / "frames" / f"frame_{frame:04d}.npz"


class CacheWriter:
    def __init__(self, cache_dir: Path, emitter_names: list[str], fps: float, frame_start: int, frame_end: int,
                 dx: float, backend: str):
        self.cache_dir = Path(cache_dir)
        self.emitter_names = list(emitter_names)
        (self.cache_dir / "frames").mkdir(parents=True, exist_ok=True)
        write_json(self.cache_dir / "cache.json", {
            "version": FORMAT_VERSION, "emitters": self.emitter_names, "fps": fps, "frame_start": frame_start,
            "frame_end": frame_end, "dx": dx, "backend": backend})

    def write_frame(self, frame: int, particles: ParticleData) -> None:
        arrays = {}
        for i in range(len(self.emitter_names)):
            mask = particles.emitter_ids == i
            arrays[f"pos_{i}"] = np.ascontiguousarray(particles.positions[mask], np.float32)
            arrays[f"vel_{i}"] = np.ascontiguousarray(particles.velocities[mask], np.float32)
        _atomic_write_bytes(frame_path(self.cache_dir, frame), lambda f: np.savez(f, **arrays))


class CacheReader:
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        doc = json.loads((self.cache_dir / "cache.json").read_text(encoding="utf-8"))
        if doc.get("version") != FORMAT_VERSION:
            raise ValueError(f"unsupported cache format version {doc.get('version')!r}")
        self.emitter_names: list[str] = doc["emitters"]
        self.fps: float = doc["fps"]
        self.frame_start: int = doc["frame_start"]
        self.frame_end: int = doc["frame_end"]
        self.dx: float = doc["dx"]

    def available_frames(self) -> list[int]:
        return sorted(int(p.stem.split("_")[1]) for p in (self.cache_dir / "frames").glob("frame_*.npz"))

    def read_frame(self, frame: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        with np.load(frame_path(self.cache_dir, frame)) as data:
            return {name: (data[f"pos_{i}"], data[f"vel_{i}"]) for i, name in enumerate(self.emitter_names)}


def _checkpoint_path(cache_dir: Path) -> Path:
    return Path(cache_dir) / "state" / "checkpoint.npz"


def write_checkpoint(cache_dir: Path, frame: int, state: dict[str, np.ndarray], scene_hash: str) -> None:
    path = _checkpoint_path(cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {**state, "__frame": np.array(frame), "__scene_hash": np.array(scene_hash)}
    _atomic_write_bytes(path, lambda f: np.savez(f, **arrays))


def read_checkpoint(cache_dir: Path) -> tuple[int, dict[str, np.ndarray], str] | None:
    path = _checkpoint_path(cache_dir)
    if not path.exists():
        return None
    with np.load(path) as data:
        state = {k: data[k] for k in data.files if not k.startswith("__")}
        return int(data["__frame"]), state, str(data["__scene_hash"])
