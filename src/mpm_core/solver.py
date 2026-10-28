"""Backend-independent solver interface. numpy in, numpy out; no backend types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from mpm_core.materials import MaterialParams
from mpm_core.scene import Domain, MeshData


@dataclass
class ParticleData:
    positions: np.ndarray  # (N,3) float32
    velocities: np.ndarray  # (N,3) float32
    material_ids: np.ndarray  # (N,) int32, index into the solver's material table
    emitter_ids: np.ndarray  # (N,) int32, carried through untouched for the cache


@dataclass
class AdvanceStats:
    substeps: int
    dt: float
    removed: int


@dataclass
class ColliderGeometry:
    mesh: MeshData  # object local space
    closed: bool
    friction: float


class Solver(ABC):
    """An MPM solver over one domain.

    All materials share one grid, so materials in contact stick together.
    Particle volume is ``dx**3 / particles_per_cell``.
    """

    def __init__(self, domain: Domain, materials: Sequence[MaterialParams], gravity: tuple[float, float, float],
                 particles_per_cell: int = 8, device: str | None = None):
        self.domain = domain
        self.materials = list(materials)
        self.gravity = tuple(float(g) for g in gravity)
        self.particles_per_cell = particles_per_cell
        self.device = device

    @abstractmethod
    def add_particles(self, positions: np.ndarray, velocities: np.ndarray, material_ids: np.ndarray,
                      emitter_ids: np.ndarray) -> None: ...

    @abstractmethod
    def set_colliders(self, colliders: Sequence[ColliderGeometry]) -> None:
        """Replace all colliders."""

    @abstractmethod
    def set_collider_transforms(self, start: np.ndarray, end: np.ndarray) -> None:
        """(C,4,4) world matrices at the start and end of the next advance()."""

    @abstractmethod
    def advance(self, frame_dt: float) -> AdvanceStats: ...

    @property
    @abstractmethod
    def particle_count(self) -> int: ...

    @abstractmethod
    def get_particles(self) -> ParticleData: ...

    @abstractmethod
    def is_occupied(self, points: np.ndarray) -> np.ndarray:
        """(M,) bool: material exists at each point (about half a cell's reach)."""

    @abstractmethod
    def save_state(self) -> dict[str, np.ndarray]: ...

    @abstractmethod
    def load_state(self, state: dict[str, np.ndarray]) -> None: ...
