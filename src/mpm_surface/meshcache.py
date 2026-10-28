"""Mesh cache under ``<cache>/surface/``.

    surface/meshes.json                   {"version": 1, "emitters": {"<emitter name>": "<folder>"}}
    surface/meshes/<folder>/frame_NNNN.npz

Folders are ``<index>_<sanitized name>`` (index = emitter order in cache.json), so emitter names never
act as paths and names differing only in case don't collide. Each npz holds ``verts`` (V,3) float32
world-space positions, ``tris`` (T,3) int32 outward-wound triangles, ``vel`` (V,3) float32 vertex
velocities in m/s, and ``key``: a fingerprint of the particles and settings it was made from.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mpm_core.cache import atomic_write_npz, write_json

FORMAT_VERSION = 1


@dataclass
class SurfaceMesh:
    verts: np.ndarray
    tris: np.ndarray
    vel: np.ndarray

    @staticmethod
    def empty() -> SurfaceMesh:
        return SurfaceMesh(np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32), np.zeros((0, 3), np.float32))


def _folder(index: int, name: str) -> str:
    return f"{index:02d}_{re.sub(r'[^A-Za-z0-9_-]+', '_', name)[:40]}"


class MeshCacheWriter:
    """Owns the layout for the given emitters; folders of emitters no longer in the cache are removed."""

    def __init__(self, cache_dir: Path, emitter_names: list[str]):
        self.root = Path(cache_dir) / "surface"
        self.folders = {name: _folder(i, name) for i, name in enumerate(emitter_names)}
        meshes = self.root / "meshes"
        meshes.mkdir(parents=True, exist_ok=True)
        for d in meshes.iterdir():
            if d.is_dir() and d.name not in self.folders.values():
                shutil.rmtree(d, ignore_errors=True)
        write_json(self.root / "meshes.json", {"version": FORMAT_VERSION, "emitters": self.folders})

    def _path(self, emitter: str, frame: int) -> Path:
        return self.root / "meshes" / self.folders[emitter] / f"frame_{frame:04d}.npz"

    def write(self, emitter: str, frame: int, mesh: SurfaceMesh, key: str) -> None:
        path = self._path(emitter, frame)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_npz(path, {
            "verts": np.ascontiguousarray(mesh.verts, np.float32),
            "tris": np.ascontiguousarray(mesh.tris, np.int32),
            "vel": np.ascontiguousarray(mesh.vel, np.float32),
            "key": np.array(key),
        })

    def key(self, emitter: str, frame: int) -> str | None:
        path = self._path(emitter, frame)
        if not path.exists():
            return None
        with np.load(path) as data:
            return str(data["key"])

    def remove_frame(self, emitter: str, frame: int) -> None:
        self._path(emitter, frame).unlink(missing_ok=True)

    def remove_emitter(self, emitter: str) -> None:
        shutil.rmtree(self.root / "meshes" / self.folders[emitter], ignore_errors=True)


class MeshCacheReader:
    def __init__(self, cache_dir: Path):
        self.root = Path(cache_dir) / "surface"
        manifest = self.root / "meshes.json"
        self.folders: dict[str, str] = (json.loads(manifest.read_text(encoding="utf-8"))["emitters"]
                                        if manifest.exists() else {})

    def emitters(self) -> list[str]:
        """Emitters that have meshes, in cache order."""
        return [name for name, folder in self.folders.items() if (self.root / "meshes" / folder).is_dir()]

    def available_frames(self, emitter: str) -> list[int]:
        if emitter not in self.folders:
            return []
        folder = self.root / "meshes" / self.folders[emitter]
        return sorted(int(p.stem.split("_")[1]) for p in folder.glob("frame_*.npz"))

    def read(self, emitter: str, frame: int) -> SurfaceMesh:
        with np.load(self.root / "meshes" / self.folders[emitter] / f"frame_{frame:04d}.npz") as data:
            return SurfaceMesh(data["verts"], data["tris"], data["vel"])
