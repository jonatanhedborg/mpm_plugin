"""Shared test helpers: meshes, matrices, small scenes."""

from __future__ import annotations

import numpy as np

from mpm_core.materials import load_preset
from mpm_core.scene import Collider, Domain, Emitter, MeshData, SceneDescription

WALL_SIDES = ("x-", "x+", "y-", "y+", "z-", "z+")
ALL_WALLS = {side: True for side in WALL_SIDES}


def translate(x, y, z) -> np.ndarray:
    m = np.eye(4, dtype=np.float32)
    m[:3, 3] = (x, y, z)
    return m


def scale(x, y=None, z=None) -> np.ndarray:
    y = x if y is None else y
    z = x if z is None else z
    return np.diag([x, y, z, 1.0]).astype(np.float32)


def unit_cube_mesh(centered: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Closed cube, outward winding: [0,1]^3, or [-0.5,0.5]^3 if centered."""
    v = np.array([[x, y, z] for z in (0, 1) for y in (0, 1) for x in (0, 1)], np.float32)
    if centered:
        v -= 0.5
    t = np.array([
        [0, 2, 1], [1, 2, 3],  # z-
        [4, 5, 6], [5, 7, 6],  # z+
        [0, 1, 4], [1, 5, 4],  # y-
        [2, 6, 3], [3, 6, 7],  # y+
        [0, 4, 2], [2, 4, 6],  # x-
        [1, 3, 5], [3, 7, 5],  # x+
    ], np.int32)
    return v, t


def plane_mesh(size: float) -> tuple[np.ndarray, np.ndarray]:
    """Open square in the xy plane centered at the origin, normal +z."""
    h = size / 2
    v = np.array([[-h, -h, 0], [h, -h, 0], [h, h, 0], [-h, h, 0]], np.float32)
    return v, np.array([[0, 1, 2], [0, 2, 3]], np.int32)


def icosphere(radius: float, subdivisions: int) -> tuple[np.ndarray, np.ndarray]:
    p = (1 + 5 ** 0.5) / 2
    v = [[-1, p, 0], [1, p, 0], [-1, -p, 0], [1, -p, 0], [0, -1, p], [0, 1, p],
         [0, -1, -p], [0, 1, -p], [p, 0, -1], [p, 0, 1], [-p, 0, -1], [-p, 0, 1]]
    f = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2],
         [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5],
         [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]]
    verts = [np.array(x, float) / np.linalg.norm(x) for x in v]
    for _ in range(subdivisions):
        cache, nf = {}, []

        def mid(a, b):
            key = (min(a, b), max(a, b))
            if key not in cache:
                m = verts[a] + verts[b]
                verts.append(m / np.linalg.norm(m))
                cache[key] = len(verts) - 1
            return cache[key]

        for a, b, c in f:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nf += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        f = nf
    return (np.array(verts) * radius).astype(np.float32), np.array(f, np.int32)


def box_particles(lo, hi, dx: float, seed: int = 0) -> np.ndarray:
    """Particles filling the box [lo, hi] at 8 per cell (spacing dx/2)."""
    from mpm_core.geometry import sample_mesh_volume, transform_points

    v, t = unit_cube_mesh()
    size = np.subtract(hi, lo)
    m = translate(*lo) @ scale(*size)
    return sample_mesh_volume(transform_points(m, v), t, dx / 2, seed)


def sphere_particles(center, radius: float, dx: float, seed: int = 0) -> np.ndarray:
    from mpm_core.geometry import sample_mesh_volume

    v, t = icosphere(radius, 4)
    return sample_mesh_volume(v + np.asarray(center, np.float32), t, dx / 2, seed)


def make_solver(device, domain, materials, gravity=(0.0, 0.0, -9.81)):
    from mpm_core.backends import create_solver

    return create_solver("warp", domain=domain, materials=list(materials), gravity=gravity, device=device)


def add(solver, positions, material_id: int = 0, emitter_id: int = 0) -> None:
    n = len(positions)
    solver.add_particles(positions, np.zeros_like(positions), np.full(n, material_id, np.int32),
                         np.full(n, emitter_id, np.int32))


def solver_with_box(device, domain, material, lo, hi):
    s = make_solver(device, domain, [material])
    add(s, box_particles(lo, hi, domain.dx))
    return s


def solver_with_cube(device, domain, lo, hi):
    """Jelly box (the plan calls it a cube even when it isn't one)."""
    return solver_with_box(device, domain, load_preset("jelly").material, lo, hi)


def solver_with_sphere(device, domain, material, center, radius):
    s = make_solver(device, domain, [material])
    add(s, sphere_particles(center, radius, domain.dx))
    return s


def solver_with_presets(device, domain, boxes):
    """boxes: [(preset_id, lo, hi)]; material and emitter id = list index."""
    s = make_solver(device, domain, [load_preset(p).material for p, _, _ in boxes])
    for i, (_, lo, hi) in enumerate(boxes):
        add(s, box_particles(lo, hi, domain.dx, seed=i), material_id=i, emitter_id=i)
    return s


def run(solver, frames: int, fps: float = 24.0) -> None:
    for _ in range(frames):
        solver.advance(1.0 / fps)


def small_scene(frame_end: int = 10, gravity=(0.0, 0.0, -9.81), particles_per_cell: int = 8,
                emitter_frames: int | None = None, emitter_offset=(0.0, 0.0, 0.0), kind: str = "fill",
                velocity=(0.0, 0.0, 0.0), emit_frames: tuple[int, int] | None = None,
                resolution: int = 32, preset: str = "jelly") -> SceneDescription:
    """Unit-cube domain, one cube emitter named "emitter_0", one open plane collider at z=0.05."""
    n = frame_end if emitter_frames is None else emitter_frames
    xf = translate(0.4 + emitter_offset[0], 0.4 + emitter_offset[1], 0.1 + emitter_offset[2]) @ scale(0.2)
    start, end = emit_frames or (1, 1)
    emitter = Emitter(name="emitter_0", kind=kind, material=0, mesh=MeshData(*unit_cube_mesh()),
                      transforms=np.repeat(xf[None], n, 0), velocity=tuple(velocity),
                      frame_start=start, frame_end=end)
    collider = Collider(name="floor", mesh=MeshData(*plane_mesh(0.8)),
                        transforms=np.repeat(translate(0.5, 0.5, 0.05)[None], frame_end, 0),
                        friction=0.5, closed=False)
    return SceneDescription(
        domain=Domain((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), resolution),
        materials=[load_preset(preset).material],
        emitters=[emitter],
        colliders=[collider],
        gravity=tuple(gravity),
        frame_start=1,
        frame_end=frame_end,
        particles_per_cell=particles_per_cell,
    )
