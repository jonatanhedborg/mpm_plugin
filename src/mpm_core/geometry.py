"""Mesh helpers: closedness check, transforms, and volume sampling."""

from __future__ import annotations

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


def sample_mesh_volume(verts: np.ndarray, tris: np.ndarray, spacing: float, seed: int) -> np.ndarray:
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
