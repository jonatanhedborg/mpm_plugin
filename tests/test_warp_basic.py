import numpy as np
import pytest
from helpers import ALL_WALLS, box_particles, make_solver, solver_with_cube

from mpm_core.backends import create_solver
from mpm_core.materials import load_preset
from mpm_core.scene import Domain

JELLY = load_preset("jelly").material


@pytest.mark.gpu
def test_free_fall(device):
    d = Domain((0, 0, 0), (1, 1, 1), 64)
    s = create_solver("warp", domain=d, materials=[JELLY], gravity=(0, 0, -9.81), device=device)
    p = box_particles((0.45, 0.45, 0.6), (0.55, 0.55, 0.7), d.dx)
    s.add_particles(p, np.zeros_like(p), np.zeros(len(p), np.int32), np.zeros(len(p), np.int32))
    z0 = p[:, 2].mean()
    for _ in range(6):
        s.advance(1 / 24)
    assert z0 - s.get_particles().positions[:, 2].mean() == pytest.approx(0.5 * 9.81 * 0.25**2, rel=0.02)


@pytest.mark.gpu
def test_rests_on_floor(device):  # contact accuracy + no sag (10 cm: jelly's own elastic squash stays < 0.5 dx)
    d = Domain((0, 0, 0), (1, 1, 1), 64)
    s = solver_with_cube(device, d, lo=(0.4, 0.4, 0.0), hi=(0.5, 0.5, 0.1))
    start = s.get_particles().positions.copy()
    for _ in range(24):
        s.advance(1 / 24)
    pos = s.get_particles().positions
    assert pos[:, 2].min() >= -0.25 * d.dx
    assert np.abs(pos - start).max() < 0.5 * d.dx


@pytest.mark.gpu
def test_non_cubic_domain(device):  # Review Focus: flat, wide domain
    d = Domain((-0.5, -0.2, 0.0), (0.5, 0.05, 0.5), 64)
    assert d.grid_shape == (64, 16, 32)
    s = solver_with_cube(device, d, lo=(-0.05, -0.1, 0.3), hi=(0.05, 0.0, 0.4))
    for _ in range(24):
        s.advance(1 / 24)
    pos = s.get_particles().positions
    assert pos[:, 2].min() >= -0.25 * d.dx and np.isfinite(pos).all()


@pytest.mark.gpu
def test_open_side_removes(device):
    d = Domain((0, 0, 0), (1, 1, 1), 32, walls={**ALL_WALLS, "z-": False})
    s = solver_with_cube(device, d, lo=(0.4, 0.4, 0.1), hi=(0.6, 0.6, 0.2))
    removed = sum(s.advance(1 / 24).removed for _ in range(24))
    assert s.particle_count == 0 and removed > 0


@pytest.mark.gpu
def test_state_roundtrip(device):
    d = Domain((0, 0, 0), (1, 1, 1), 32)
    a = solver_with_cube(device, d, lo=(0.4, 0.4, 0.3), hi=(0.6, 0.6, 0.5))
    a.advance(1 / 24)
    b = make_solver(device, d, [JELLY])
    b.load_state(a.save_state())
    a.advance(1 / 24)
    b.advance(1 / 24)
    assert np.allclose(a.get_particles().positions, b.get_particles().positions, atol=1e-5)


@pytest.mark.gpu
def test_is_occupied(device):
    d = Domain((0, 0, 0), (1, 1, 1), 32)
    s = solver_with_cube(device, d, lo=(0.4, 0.4, 0.0), hi=(0.6, 0.6, 0.2))
    s.advance(1 / 24)
    assert s.is_occupied(np.array([[0.5, 0.5, 0.1], [0.9, 0.9, 0.9]], np.float32)).tolist() == [True, False]


def test_unknown_backend():
    with pytest.raises(KeyError, match="warp"):
        create_solver("nope", domain=None, materials=[], gravity=(0, 0, 0))
