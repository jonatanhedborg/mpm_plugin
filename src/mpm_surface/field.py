"""GPU surfacing of one particle set: density splat -> marching cubes -> vertex velocities.

Density is an SPH-style sum of a cubic B-spline kernel with support radius
``h = 2 * spacing * smoothing``, normalized so a uniform particle lattice at ``spacing`` reads 1.
The surface is the ``settings.iso`` level set. Velocity is the kernel-weighted particle average,
sampled trilinearly at the vertices.
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

from mpm_surface.meshcache import SurfaceMesh
from mpm_surface.settings import SurfaceSettings

# Peak GPU bytes per field cell: density + velocity grids and Warp's marching cubes working set
# (measured ~52 on an RTX 5080 with Warp 1.18).
BYTES_PER_CELL = 56


@wp.func
def cubic_kernel(q: float) -> float:
    """Unnormalized cubic B-spline, support q in [0, 1)."""
    if q < 0.5:
        return 1.0 - 6.0 * q * q + 6.0 * q * q * q
    if q < 1.0:
        t = 1.0 - q
        return 2.0 * t * t * t
    return 0.0


@wp.kernel
def splat(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), origin: wp.vec3, inv_cell: float,
          cell: float, h: float, density: wp.array3d(dtype=float), momentum: wp.array3d(dtype=wp.vec3)):
    p = wp.tid()
    g = (x[p] - origin) * inv_cell
    r = h * inv_cell
    lo_i = wp.max(int(wp.floor(g[0] - r)), 0)
    lo_j = wp.max(int(wp.floor(g[1] - r)), 0)
    lo_k = wp.max(int(wp.floor(g[2] - r)), 0)
    hi_i = wp.min(int(wp.ceil(g[0] + r)), density.shape[0] - 1)
    hi_j = wp.min(int(wp.ceil(g[1] + r)), density.shape[1] - 1)
    hi_k = wp.min(int(wp.ceil(g[2] + r)), density.shape[2] - 1)
    for i in range(lo_i, hi_i + 1):
        for j in range(lo_j, hi_j + 1):
            for k in range(lo_k, hi_k + 1):
                d = wp.length(wp.vec3(float(i), float(j), float(k)) - g) * cell
                w = cubic_kernel(d / h)
                if w > 0.0:
                    wp.atomic_add(density, i, j, k, w)
                    wp.atomic_add(momentum, i, j, k, w * v[p])


@wp.kernel
def normalize(density: wp.array3d(dtype=float), momentum: wp.array3d(dtype=wp.vec3), inv_rest: float):
    i, j, k = wp.tid()
    d = density[i, j, k]
    if d > 0.0:
        momentum[i, j, k] = momentum[i, j, k] / d  # now velocity
    density[i, j, k] = d * inv_rest


@wp.kernel
def sample_velocity(verts: wp.array(dtype=wp.vec3), velocity: wp.array3d(dtype=wp.vec3),
                    out: wp.array(dtype=wp.vec3)):
    """Trilinear sample; verts are in grid index units."""
    t = wp.tid()
    g = verts[t]
    i = wp.clamp(int(wp.floor(g[0])), 0, velocity.shape[0] - 2)
    j = wp.clamp(int(wp.floor(g[1])), 0, velocity.shape[1] - 2)
    k = wp.clamp(int(wp.floor(g[2])), 0, velocity.shape[2] - 2)
    fx = g[0] - float(i)
    fy = g[1] - float(j)
    fz = g[2] - float(k)
    acc = wp.vec3()
    for a in range(2):
        wa = wp.where(a == 0, 1.0 - fx, fx)
        for b in range(2):
            wb = wp.where(b == 0, 1.0 - fy, fy)
            for c in range(2):
                wc = wp.where(c == 0, 1.0 - fz, fz)
                acc = acc + (wa * wb * wc) * velocity[i + a, j + b, k + c]
    out[t] = acc


def _rest_density(spacing: float, h: float) -> float:
    """Mean kernel sum of a uniform particle lattice, averaged over a few sample offsets."""
    n = math.ceil(h / spacing) + 1
    lattice = np.stack(np.meshgrid(*[np.arange(-n, n + 1)] * 3, indexing="ij"), -1).reshape(-1, 3) * spacing
    sums = []
    for offset in np.array([[x, y, z] for x in (0, 0.5) for y in (0, 0.5) for z in (0, 0.5)]) * spacing:
        q = np.linalg.norm(lattice - offset, axis=1) / h
        w = np.where(q < 0.5, 1 - 6 * q**2 + 6 * q**3, np.where(q < 1, 2 * (1 - q) ** 3, 0.0))
        sums.append(w.sum())
    return float(np.mean(sums))


def surface_particles(positions: np.ndarray, velocities: np.ndarray, spacing: float, settings: SurfaceSettings,
                      device: str | None = None, max_bytes: int | None = None) -> SurfaceMesh:
    """max_bytes defaults to 70% of the device's currently free memory."""
    positions = np.asarray(positions, np.float32).reshape(-1, 3)
    if len(positions) == 0:
        return SurfaceMesh.empty()
    wp.config.quiet = True
    wp.init()
    device = device or str(wp.get_preferred_device())

    cell = spacing / settings.detail
    h = 2.0 * spacing * settings.smoothing
    pad = h + 2.0 * cell  # so the field drops below iso before the grid edge and the mesh closes
    origin = positions.min(0).astype(np.float64) - pad
    dims = tuple(int(math.ceil(e / cell)) + 1 for e in positions.max(0) + pad - origin)
    cells = dims[0] * dims[1] * dims[2]
    needed = cells * BYTES_PER_CELL
    if max_bytes is None:
        max_bytes = int(0.7 * (getattr(wp.get_device(device), "free_memory", 0) or 8 * 1024**3))
    if needed > max_bytes:
        raise ValueError(f"surfacing needs a {dims[0]}x{dims[1]}x{dims[2]} grid ({cells:,} cells, about "
                         f"{needed / 1e9:.1f} GB of GPU memory), over the budget of {max_bytes / 1e9:.1f} GB; "
                         f"lower the detail or split the emitter")

    density = wp.zeros(dims, dtype=float, device=device)
    momentum = wp.zeros(dims, dtype=wp.vec3, device=device)
    x = wp.array(positions, dtype=wp.vec3, device=device)
    v = wp.array(np.asarray(velocities, np.float32).reshape(-1, 3), dtype=wp.vec3, device=device)
    wp.launch(splat, len(positions), inputs=[x, v, wp.vec3(*origin), 1.0 / cell, cell, h, density, momentum],
              device=device)
    wp.launch(normalize, dims, inputs=[density, momentum, 1.0 / _rest_density(spacing, h)], device=device)

    mc = wp.MarchingCubes(*dims, device=device)
    mc.surface(density, settings.iso)
    if mc.verts is None or len(mc.verts) == 0:
        return SurfaceMesh.empty()
    vel = wp.zeros(len(mc.verts), dtype=wp.vec3, device=device)
    wp.launch(sample_velocity, len(mc.verts), inputs=[mc.verts, momentum, vel], device=device)

    verts = (origin + mc.verts.numpy().astype(np.float64) * cell).astype(np.float32)
    tris = mc.indices.numpy().reshape(-1, 3)[:, [0, 2, 1]].astype(np.int32)  # MC winds inward for "field > iso"
    return SurfaceMesh(verts, np.ascontiguousarray(tris), vel.numpy())
