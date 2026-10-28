import numpy as np
import pytest
from helpers import box_particles, sphere_particles

from mpm_core.geometry import mesh_is_closed
from mpm_surface.field import surface_particles
from mpm_surface.settings import SurfaceSettings

SPACING = 0.005  # helpers sample at dx / 2
DX = 0.01


def signed_volume(verts, tris):
    a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6)


def components(tris):
    parent = list(range(int(tris.max()) + 1))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b, c in tris:
        for u, v in ((a, b), (b, c)):
            parent[find(u)] = find(v)
    return len({find(i) for i in np.unique(tris)})


def mesh(p, vel=None, settings=SurfaceSettings(), device="cuda:0"):
    return surface_particles(p, np.zeros_like(p) if vel is None else vel, SPACING, settings, device)


@pytest.mark.gpu
def test_sphere_closed_outward_right_size(device):
    m = mesh(sphere_particles((0.5, 0.5, 0.5), 0.1, DX), device=device)
    assert mesh_is_closed(m.tris)
    vol = signed_volume(m.verts, m.tris)
    assert vol > 0 and vol == pytest.approx(4 / 3 * np.pi * 0.1**3, rel=0.10)
    assert np.allclose(m.verts.mean(0), 0.5, atol=0.005)


@pytest.mark.gpu
def test_two_blobs_two_components(device):
    p = np.concatenate([sphere_particles((0.3, 0.5, 0.5), 0.05, DX), sphere_particles((0.7, 0.5, 0.5), 0.05, DX)])
    assert components(mesh(p, device=device).tris) == 2


@pytest.mark.gpu
def test_resting_on_floor_closed(device):  # Review Focus: material at the domain floor
    m = mesh(box_particles((0.4, 0.4, 0.0), (0.6, 0.6, 0.1), DX), device=device)
    assert mesh_is_closed(m.tris) and m.verts[:, 2].min() < 0.0  # closes just below z=0, no hole


@pytest.mark.gpu
def test_empty(device):  # Review Focus
    m = mesh(np.zeros((0, 3), np.float32), device=device)
    assert m.verts.shape == (0, 3) and m.tris.shape == (0, 3) and m.vel.shape == (0, 3)


@pytest.mark.gpu
def test_vertex_velocity(device):
    p = sphere_particles((0.5, 0.5, 0.5), 0.05, DX)
    m = mesh(p, vel=np.tile([1.0, 0, 0], (len(p), 1)).astype(np.float32), device=device)
    assert len(m.vel) == len(m.verts) and np.allclose(m.vel, [1.0, 0, 0], atol=1e-3)


@pytest.mark.gpu
def test_detail_increases_triangles(device):
    p = sphere_particles((0.5, 0.5, 0.5), 0.05, DX)
    assert len(mesh(p, settings=SurfaceSettings(detail=2), device=device).tris) > 2 * len(mesh(p, device=device).tris)


@pytest.mark.gpu
def test_too_many_cells_raises(device):  # Review Focus
    p = np.array([[0, 0, 0], [4, 4, 0.1]], np.float32)
    with pytest.raises(ValueError, match="GPU memory"):
        surface_particles(p, np.zeros_like(p), 0.001, SurfaceSettings(detail=2), device)


@pytest.mark.gpu
def test_memory_budget_counts_bytes(device):  # surfacer review finding 3
    p = sphere_particles((0.5, 0.5, 0.5), 0.05, DX)
    with pytest.raises(ValueError, match="GPU memory"):
        surface_particles(p, np.zeros_like(p), SPACING, SurfaceSettings(), device, max_bytes=1_000_000)
