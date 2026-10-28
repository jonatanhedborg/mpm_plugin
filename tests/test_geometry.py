import time

import numpy as np
import pytest
from helpers import icosphere, unit_cube_mesh

from mpm_core.geometry import mesh_is_closed, sample_mesh_volume, transform_points


def test_closed():
    v, t = unit_cube_mesh()
    assert mesh_is_closed(t) and not mesh_is_closed(t[:-1])


def test_cube_count():
    v, t = unit_cube_mesh()
    pts = sample_mesh_volume(v * 0.2, t, 0.01, seed=0)
    assert abs(len(pts) - 8000) <= 8000 * 0.02
    assert pts.min() >= -0.0025 and pts.max() <= 0.2025


def test_sphere_volume():
    v, t = icosphere(0.1, 4)
    t0 = time.perf_counter()
    pts = sample_mesh_volume(v, t, 0.005, seed=0)
    assert time.perf_counter() - t0 < 2.0
    assert len(pts) == pytest.approx(4 / 3 * np.pi * 0.1**3 / 0.005**3, rel=0.03)


def test_deterministic():
    v, t = icosphere(0.1, 3)
    assert np.array_equal(sample_mesh_volume(v, t, 0.01, 7), sample_mesh_volume(v, t, 0.01, 7))


def test_open_mesh_raises():
    v, t = unit_cube_mesh()
    with pytest.raises(ValueError, match="not closed"):
        sample_mesh_volume(v, t[:-2], 0.01, 0)


def test_scaled_mesh_volume():  # Review Focus: scaled emitters
    v, t = unit_cube_mesh()
    m = np.diag([0.2, 0.1, 0.4, 1.0]).astype(np.float32)
    pts = sample_mesh_volume(transform_points(m, v), t, 0.01, 0)
    assert len(pts) == pytest.approx(0.2 * 0.1 * 0.4 / 0.01**3, rel=0.03)
