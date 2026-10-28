import numpy as np
import pytest
from helpers import (add, box_particles, make_solver, plane_mesh, run, scale, solver_with_box, solver_with_cube,
                     translate, unit_cube_mesh)

from mpm_core.geometry import transform_points
from mpm_core.materials import load_preset
from mpm_core.scene import Domain, MeshData
from mpm_core.solver import ColliderGeometry

DOMAIN = Domain((0, 0, 0), (1, 1, 1), 64)
DX = DOMAIN.dx


def rot_y(degrees: float) -> np.ndarray:
    a = np.radians(degrees)
    m = np.eye(4, dtype=np.float32)
    m[0, 0], m[0, 2], m[2, 0], m[2, 2] = np.cos(a), np.sin(a), -np.sin(a), np.cos(a)
    return m


def jelly_cube(device, lo, hi):
    return solver_with_cube(device, DOMAIN, lo, hi)


def sand_layer(device, lo, hi):
    return solver_with_box(device, DOMAIN, load_preset("sand_dry").material, lo, hi)


def cube_on_plane_tilted(device, degrees, friction, preset):
    plane_xf = translate(0.5, 0.5, 0.3) @ rot_y(degrees)
    s = make_solver(device, DOMAIN, [load_preset(preset).material])
    local = box_particles((-0.05, -0.05, 0.25 * DX), (0.05, 0.05, 0.1), DX)
    add(s, transform_points(plane_xf, local))
    s.set_colliders([ColliderGeometry(MeshData(*plane_mesh(0.8)), closed=False, friction=friction)])
    s.set_collider_transforms(plane_xf[None], plane_xf[None])
    return s


def points_in_box(pos, box_xf):
    """Points more than 0.25 dx inside the box (on-surface particles are resting, not inside)."""
    local = transform_points(np.linalg.inv(box_xf), pos)
    half = 0.5 - 0.25 * DX / np.linalg.norm(box_xf[:3, :3], axis=0)
    return (np.abs(local) < half).all(axis=1)


@pytest.mark.gpu
def test_rests_on_open_plane(device):
    s = jelly_cube(device, lo=(0.4, 0.4, 0.45), hi=(0.6, 0.6, 0.55))
    s.set_colliders([ColliderGeometry(MeshData(*plane_mesh(0.8)), closed=False, friction=0.5)])
    xf = translate(0.5, 0.5, 0.3)
    s.set_collider_transforms(xf[None], xf[None])
    run(s, 36)
    zmin = s.get_particles().positions[:, 2].min()
    assert 0.3 - 0.25 * DX <= zmin <= 0.3 + 1.0 * DX


@pytest.mark.gpu
def test_scaled_box_collider(device):  # Review Focus: scaled colliders
    s = jelly_cube(device, lo=(0.45, 0.45, 0.4), hi=(0.55, 0.55, 0.5))
    s.set_colliders([ColliderGeometry(MeshData(*unit_cube_mesh(centered=True)), closed=True, friction=0.5)])
    xf = translate(0.5, 0.5, 0.0) @ scale(0.6, 0.6, 0.4)  # top face at z = 0.2
    s.set_collider_transforms(xf[None], xf[None])
    run(s, 36)
    assert s.get_particles().positions[:, 2].min() == pytest.approx(0.2, abs=1.0 * DX)


@pytest.mark.gpu
@pytest.mark.parametrize("friction,slides", [(0.0, True), (1.0, False)])
def test_tilted_plane_friction(device, friction, slides):
    # 10°, cream: on steeper slanted colliders friction is weaker than nominal (known limitation), and
    # undamped elastic jelly keeps vibrating and creeps even at high friction
    s = cube_on_plane_tilted(device, degrees=10, friction=friction, preset="whipped_cream")
    start = s.get_particles().positions.mean(0)
    run(s, 24)
    moved = np.linalg.norm(s.get_particles().positions.mean(0) - start)
    assert (moved > 0.05) if slides else (moved < 0.01)


@pytest.mark.gpu
def test_moving_box_clears_sand(device):
    s = sand_layer(device, lo=(0.1, 0.3, 0), hi=(0.9, 0.7, 0.06))
    s.set_colliders([ColliderGeometry(MeshData(*unit_cube_mesh(centered=True)), closed=True, friction=0.5)])

    def box(x):
        return translate(x, 0.5, 0.05) @ scale(0.1, 0.5, 0.1)

    for f in range(24):  # 0.3 m/s along x
        s.set_collider_transforms(box(0.15 + 0.3 * f / 24)[None], box(0.15 + 0.3 * (f + 1) / 24)[None])
        s.advance(1 / 24)
    inside = points_in_box(s.get_particles().positions, box(0.45))
    assert inside.sum() <= 0.001 * s.particle_count
