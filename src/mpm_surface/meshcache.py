"""Mesh cache: ``<cache>/surface/<emitter>/frame_NNNN.npz``.

Each file holds ``verts`` (V,3) float32 world-space positions, ``tris`` (T,3) int32 outward-wound
triangles, ``vel`` (V,3) float32 vertex velocities in m/s, and ``key``: a fingerprint of the particles
and settings it was made from, so unchanged frames can be skipped.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mpm_core.cache import atomic_write_npz


@dataclass
class SurfaceMesh:
    verts: np.ndarray
    tris: np.ndarray
    vel: np.ndarray

    @staticmethod
    def empty() -> SurfaceMesh:
        return SurfaceMesh(np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32), np.zeros((0, 3), np.float32))


def _emitter_dir(cache_dir: Path, emitter: str) -> Path:
    return Path(cache_dir) / "surface" / emitter


def _frame_path(cache_dir: Path, emitter: str, frame: int) -> Path:
    return _emitter_dir(cache_dir, emitter) / f"frame_{frame:04d}.npz"


class MeshCacheWriter:
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)

    def write(self, emitter: str, frame: int, mesh: SurfaceMesh, key: str) -> None:
        _emitter_dir(self.cache_dir, emitter).mkdir(parents=True, exist_ok=True)
        atomic_write_npz(_frame_path(self.cache_dir, emitter, frame), {
            "verts": np.ascontiguousarray(mesh.verts, np.float32),
            "tris": np.ascontiguousarray(mesh.tris, np.int32),
            "vel": np.ascontiguousarray(mesh.vel, np.float32),
            "key": np.array(key),
        })

    def key(self, emitter: str, frame: int) -> str | None:
        path = _frame_path(self.cache_dir, emitter, frame)
        if not path.exists():
            return None
        with np.load(path) as data:
            return str(data["key"])

    def remove_emitter(self, emitter: str) -> None:
        shutil.rmtree(_emitter_dir(self.cache_dir, emitter), ignore_errors=True)


class MeshCacheReader:
    def __init__(self, cache_dir: Path):
        self.root = Path(cache_dir) / "surface"

    def emitters(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(d.name for d in self.root.iterdir() if d.is_dir())

    def available_frames(self, emitter: str) -> list[int]:
        return sorted(int(p.stem.split("_")[1]) for p in (self.root / emitter).glob("frame_*.npz"))

    def read(self, emitter: str, frame: int) -> SurfaceMesh:
        with np.load(self.root / emitter / f"frame_{frame:04d}.npz") as data:
            return SurfaceMesh(data["verts"], data["tris"], data["vel"])
