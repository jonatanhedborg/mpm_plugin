"""Scene description: the contract between the Blender add-on and the bake runner.

On disk a scene is a directory with ``scene.json`` (format version 1) plus one
``.npy`` file per array:

    emitter_{i}_verts.npy  (V,3) float32   mesh vertices, object local space
    emitter_{i}_tris.npy   (T,3) int32     triangle indices
    emitter_{i}_xf.npy     (F,4,4) float32 world matrix per frame, frame_start..frame_end
    collider_{i}_*.npy     same for colliders

``transforms[k]`` is the object's world matrix at frame ``frame_start + k``.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from mpm_core.materials import MaterialParams

FORMAT_VERSION = 1
WALL_SIDES = ("x-", "x+", "y-", "y+", "z-", "z+")
EMITTER_KINDS = ("fill", "inflow")
MAX_SUBSTEPS = 16


def _all_walls() -> dict[str, bool]:
    return {side: True for side in WALL_SIDES}


@dataclass
class Domain:
    min: tuple[float, float, float]
    max: tuple[float, float, float]
    resolution: int
    walls: dict[str, bool] = field(default_factory=_all_walls)
    wall_friction: float = 0.5

    def __post_init__(self):
        self.min = tuple(float(x) for x in self.min)
        self.max = tuple(float(x) for x in self.max)
        if any(b <= a for a, b in zip(self.min, self.max)):
            raise ValueError("domain max must be greater than min on every axis")
        if self.resolution < 4:
            raise ValueError("domain resolution must be at least 4")
        self.walls = {**_all_walls(), **self.walls}
        unknown = set(self.walls) - set(WALL_SIDES)
        if unknown:
            raise ValueError(f"unknown wall sides {sorted(unknown)}")

    @property
    def size(self) -> tuple[float, float, float]:
        return tuple(b - a for a, b in zip(self.min, self.max))

    @property
    def dx(self) -> float:
        return max(self.size) / self.resolution

    @property
    def grid_shape(self) -> tuple[int, int, int]:
        # small tolerance so exact multiples of dx don't round up
        return tuple(max(1, math.ceil(s / self.dx - 1e-6)) for s in self.size)


@dataclass
class MeshData:
    verts: np.ndarray
    tris: np.ndarray

    def __post_init__(self):
        self.verts = np.asarray(self.verts, np.float32).reshape(-1, 3)
        self.tris = np.asarray(self.tris, np.int32).reshape(-1, 3)


@dataclass
class Emitter:
    name: str
    kind: str
    material: int
    mesh: MeshData
    transforms: np.ndarray
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    use_object_velocity: bool = False
    frame_start: int = 1
    frame_end: int = 1
    substeps: int = 1  # inflow: emissions per frame, at evenly spaced times with the pose interpolated

    def __post_init__(self):
        if self.kind not in EMITTER_KINDS:
            raise ValueError(f"emitter {self.name!r}: kind must be one of {EMITTER_KINDS}")
        if not 1 <= self.substeps <= MAX_SUBSTEPS:
            raise ValueError(f"emitter {self.name!r}: substeps must be in 1..{MAX_SUBSTEPS}")
        self.transforms = np.asarray(self.transforms, np.float32).reshape(-1, 4, 4)
        self.velocity = tuple(float(x) for x in self.velocity)


@dataclass
class Collider:
    name: str
    mesh: MeshData
    transforms: np.ndarray
    friction: float = 0.5
    closed: bool = True

    def __post_init__(self):
        self.transforms = np.asarray(self.transforms, np.float32).reshape(-1, 4, 4)


@dataclass
class SceneDescription:
    domain: Domain
    materials: list[MaterialParams]
    emitters: list[Emitter]
    colliders: list[Collider]
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    fps: float = 24.0
    frame_start: int = 1
    frame_end: int = 250
    seed: int = 0
    particles_per_cell: int = 8

    def __post_init__(self):
        self.gravity = tuple(float(x) for x in self.gravity)
        if self.frame_end < self.frame_start:
            raise ValueError("frame_end must be >= frame_start")
        root = round(self.particles_per_cell ** (1 / 3))
        if root < 1 or root**3 != self.particles_per_cell:
            raise ValueError("particles_per_cell must be a perfect cube (1, 8, 27, ...)")
        n = self.frame_count
        for obj in [*self.emitters, *self.colliders]:
            if len(obj.transforms) != n:
                raise ValueError(f"{obj.name!r} has {len(obj.transforms)} transforms, expected {n} "
                                 f"(frames {self.frame_start}..{self.frame_end})")
        for e in self.emitters:
            if not 0 <= e.material < len(self.materials):
                raise ValueError(f"emitter {e.name!r} references missing material {e.material}")

    @property
    def frame_count(self) -> int:
        return self.frame_end - self.frame_start + 1


def write_scene(scene: SceneDescription, directory: Path) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for old in directory.glob("*.npy"):
        old.unlink()

    def save_mesh_obj(prefix: str, mesh: MeshData, transforms: np.ndarray) -> None:
        np.save(directory / f"{prefix}_verts.npy", mesh.verts)
        np.save(directory / f"{prefix}_tris.npy", mesh.tris)
        np.save(directory / f"{prefix}_xf.npy", transforms)

    emitters = []
    for i, e in enumerate(scene.emitters):
        save_mesh_obj(f"emitter_{i}", e.mesh, e.transforms)
        emitters.append({"name": e.name, "kind": e.kind, "material": e.material, "velocity": list(e.velocity),
                         "use_object_velocity": e.use_object_velocity,
                         "frame_start": e.frame_start, "frame_end": e.frame_end, "substeps": e.substeps})
    colliders = []
    for i, c in enumerate(scene.colliders):
        save_mesh_obj(f"collider_{i}", c.mesh, c.transforms)
        colliders.append({"name": c.name, "friction": c.friction, "closed": c.closed})

    d = scene.domain
    doc = {
        "version": FORMAT_VERSION,
        "domain": {"min": list(d.min), "max": list(d.max), "resolution": d.resolution,
                   "walls": d.walls, "wall_friction": d.wall_friction},
        "materials": [m.to_dict() for m in scene.materials],
        "emitters": emitters,
        "colliders": colliders,
        "gravity": list(scene.gravity),
        "fps": scene.fps,
        "frame_start": scene.frame_start,
        "frame_end": scene.frame_end,
        "seed": scene.seed,
        "particles_per_cell": scene.particles_per_cell,
    }
    (directory / "scene.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")


def read_scene(directory: Path) -> SceneDescription:
    directory = Path(directory)
    doc = json.loads((directory / "scene.json").read_text(encoding="utf-8"))
    if doc.get("version") != FORMAT_VERSION:
        raise ValueError(f"unsupported scene format version {doc.get('version')!r}")

    def load_mesh(prefix: str) -> tuple[MeshData, np.ndarray]:
        mesh = MeshData(np.load(directory / f"{prefix}_verts.npy"), np.load(directory / f"{prefix}_tris.npy"))
        return mesh, np.load(directory / f"{prefix}_xf.npy")

    emitters = []
    for i, e in enumerate(doc["emitters"]):
        mesh, xf = load_mesh(f"emitter_{i}")
        emitters.append(Emitter(mesh=mesh, transforms=xf, **e))
    colliders = []
    for i, c in enumerate(doc["colliders"]):
        mesh, xf = load_mesh(f"collider_{i}")
        colliders.append(Collider(mesh=mesh, transforms=xf, **c))

    dom = doc["domain"]
    return SceneDescription(
        domain=Domain(dom["min"], dom["max"], dom["resolution"], dom["walls"], dom["wall_friction"]),
        materials=[MaterialParams.from_dict(m) for m in doc["materials"]],
        emitters=emitters,
        colliders=colliders,
        gravity=doc["gravity"],
        fps=doc["fps"],
        frame_start=doc["frame_start"],
        frame_end=doc["frame_end"],
        seed=doc["seed"],
        particles_per_cell=doc["particles_per_cell"],
    )


def scene_hash(directory: Path) -> str:
    """sha256 over scene.json and every .npy file, in sorted name order."""
    directory = Path(directory)
    h = hashlib.sha256()
    for path in [directory / "scene.json", *sorted(directory.glob("*.npy"))]:
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()
