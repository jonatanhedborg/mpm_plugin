"""Mesh helpers: closedness check, transforms and their interpolation, and volume sampling."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# Rays are nudged off the lattice column by this fraction of the spacing so they
# don't pass exactly through mesh edges or vertices (which would break parity).
_RAY_NUDGE = np.array([1.234567e-4, 2.345678e-4])


def mesh_is_closed(tris: np.ndarray) -> bool:
    """True if every undirected edge is shared by exactly two triangles."""
    tris = np.asarray(tris)
    if len(tris) == 0:
        return False
    edges = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    edges.sort(axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return bool((counts == 2).all())


def transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, np.float64)
    return (np.asarray(points, np.float64) @ m[:3, :3].T + m[:3, 3]).astype(np.float32)


def decompose_transform(m: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """World matrix -> (translation, rotation quaternion xyzw, symmetric stretch) with M = [R S | t]."""
    A = np.asarray(m[:3, :3], np.float64)
    U, sig, Vt = np.linalg.svd(A)
    if np.linalg.det(U @ Vt) < 0:  # mirrored: keep R a proper rotation, put the flip into S
        U[:, -1] = -U[:, -1]
        sig[-1] = -sig[-1]
    R = U @ Vt
    S = Vt.T @ np.diag(sig) @ Vt
    return np.asarray(m[:3, 3], np.float64), _quat_from_matrix(R), S


def _quat_from_matrix(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion (x, y, z, w)."""
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.copysign(np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def _matrix_from_quat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _slerp(q0: np.ndarray, q1: np.ndarray, s: float) -> np.ndarray:
    if np.dot(q0, q1) < 0:
        q1 = -q1  # the short way round
    d = min(float(np.dot(q0, q1)), 1.0)
    angle = np.arccos(d)
    if angle < 1e-6:
        q = q0 + s * (q1 - q0)
    else:
        q = (np.sin((1 - s) * angle) * q0 + np.sin(s * angle) * q1) / np.sin(angle)
    return q / np.linalg.norm(q)


def interpolate_transform(a: np.ndarray, b: np.ndarray, s: float) -> np.ndarray:
    """The pose a fraction ``s`` of the way from world matrix ``a`` to ``b``.

    Rotation is slerped, translation and stretch lerped, the same motion the Warp backend gives colliders
    within a frame, so a spinning object keeps its size mid-frame.
    """
    if s == 0:
        return np.asarray(a, np.float32)
    if s == 1:
        return np.asarray(b, np.float32)
    t0, q0, S0 = decompose_transform(a)
    t1, q1, S1 = decompose_transform(b)
    m = np.eye(4)
    m[:3, :3] = _matrix_from_quat(_slerp(q0, q1, s)) @ (S0 + s * (S1 - S0))
    m[:3, 3] = t0 + s * (t1 - t0)
    return m.astype(np.float32)


def sample_mesh_volume(verts: np.ndarray, tris: np.ndarray, spacing: float,
                       seed: int | Sequence[int]) -> np.ndarray:
    """Points inside a closed mesh on a cell-centered lattice, jittered by ±0.25 spacing.

    The lattice is ``(i + 0.5) * spacing`` on every axis, aligned to the world origin.
    Inside-ness uses z-ray parity per (x, y) lattice column.
    """
    if not mesh_is_closed(tris):
        raise ValueError("emitter mesh is not closed (every edge must be shared by exactly two triangles)")
    v = np.asarray(verts, np.float64)
    tri = v[np.asarray(tris)]  # (T, 3, 3)
    lo, hi = v.min(0), v.max(0)
    first = np.ceil(lo / spacing - 0.5).astype(np.int64)
    last = np.floor(hi / spacing - 0.5).astype(np.int64)
    if (last < first).any():
        return np.zeros((0, 3), np.float32)
    xs, ys, zs = [(np.arange(first[a], last[a] + 1) + 0.5) * spacing for a in range(3)]
    nudge = _RAY_NUDGE * spacing

    hit_cols, hit_z = [], []
    for a, b, c in tri:
        # column index range covered by the triangle's xy bounding box
        tmin = np.minimum(np.minimum(a, b), c)[:2]
        tmax = np.maximum(np.maximum(a, b), c)[:2]
        i0 = max(int(np.ceil((tmin[0] - nudge[0]) / spacing - 0.5)) - first[0], 0)
        i1 = min(int(np.floor((tmax[0] - nudge[0]) / spacing - 0.5)) - first[0], len(xs) - 1)
        j0 = max(int(np.ceil((tmin[1] - nudge[1]) / spacing - 0.5)) - first[1], 0)
        j1 = min(int(np.floor((tmax[1] - nudge[1]) / spacing - 0.5)) - first[1], len(ys) - 1)
        if i1 < i0 or j1 < j0:
            continue
        area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        if area == 0.0:
            continue  # vertical triangle: rays never cross it
        ii, jj = np.meshgrid(np.arange(i0, i1 + 1), np.arange(j0, j1 + 1), indexing="ij")
        px = xs[ii] + nudge[0]
        py = ys[jj] + nudge[1]
        w0 = ((b[0] - px) * (c[1] - py) - (b[1] - py) * (c[0] - px)) / area
        w1 = ((c[0] - px) * (a[1] - py) - (c[1] - py) * (a[0] - px)) / area
        w2 = 1.0 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue
        hit_cols.append((ii[inside] * len(ys) + jj[inside]).ravel())
        hit_z.append((w0 * a[2] + w1 * b[2] + w2 * c[2])[inside].ravel())

    if not hit_cols:
        return np.zeros((0, 3), np.float32)
    cols = np.concatenate(hit_cols)
    z = np.concatenate(hit_z)
    order = np.lexsort((z, cols))
    cols, z = cols[order], z[order]
    starts = np.flatnonzero(np.r_[True, cols[1:] != cols[:-1]])
    ends = np.r_[starts[1:], len(cols)]

    out = []
    for s, e in zip(starts, ends):
        crossings = np.searchsorted(z[s:e], zs)  # hits below each lattice z
        zin = zs[crossings % 2 == 1]
        if len(zin):
            col = cols[s]
            x, y = xs[col // len(ys)], ys[col % len(ys)]
            out.append(np.column_stack([np.full(len(zin), x), np.full(len(zin), y), zin]))
    if not out:
        return np.zeros((0, 3), np.float32)
    pts = np.concatenate(out)
    pts += np.random.default_rng(seed).uniform(-0.25, 0.25, pts.shape) * spacing
    return pts.astype(np.float32)
