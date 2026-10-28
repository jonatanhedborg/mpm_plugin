"""Mesh and matrix helpers for the example scenes."""

import numpy as np

from mpm_core.scene import MeshData


def translate(x, y, z) -> np.ndarray:
    m = np.eye(4, dtype=np.float32)
    m[:3, 3] = (x, y, z)
    return m


def scale(x, y=None, z=None) -> np.ndarray:
    return np.diag([x, x if y is None else y, x if z is None else z, 1.0]).astype(np.float32)


def rot_y(radians: float) -> np.ndarray:
    c, s = np.cos(radians), np.sin(radians)
    m = np.eye(4, dtype=np.float32)
    m[0, 0], m[0, 2], m[2, 0], m[2, 2] = c, s, -s, c
    return m


def rot_x(radians: float) -> np.ndarray:
    c, s = np.cos(radians), np.sin(radians)
    m = np.eye(4, dtype=np.float32)
    m[1, 1], m[1, 2], m[2, 1], m[2, 2] = c, -s, s, c
    return m


def plane_mesh() -> MeshData:
    """Open unit square in the xy plane centered at the origin, normal +z."""
    v = np.array([[-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, 0.5, 0], [-0.5, 0.5, 0]], np.float32)
    return MeshData(v, np.array([[0, 1, 2], [0, 2, 3]], np.int32))


def box_mesh() -> MeshData:
    """Unit cube [0,1]^3, outward winding."""
    v = np.array([[x, y, z] for z in (0, 1) for y in (0, 1) for x in (0, 1)], np.float32)
    t = [[0, 2, 1], [1, 2, 3], [4, 5, 6], [5, 7, 6], [0, 1, 4], [1, 5, 4],
         [2, 6, 3], [3, 6, 7], [0, 4, 2], [2, 4, 6], [1, 3, 5], [3, 7, 5]]
    return MeshData(v, np.array(t, np.int32))


def icosphere(subdivisions: int = 3) -> MeshData:
    """Unit-radius icosphere centered at the origin."""
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
    return MeshData(np.array(verts, np.float32), np.array(f, np.int32))


def still(matrix: np.ndarray, frames: int) -> np.ndarray:
    return np.repeat(matrix[None], frames, 0)
