import numpy as np
import pytest
from helpers import small_scene

from mpm_core.scene import Domain, read_scene, scene_hash, write_scene


def test_domain_grid():
    d = Domain(min=(0, 0, 0), max=(1.0, 0.25, 0.5), resolution=64)
    assert d.dx == pytest.approx(1 / 64) and d.grid_shape == (64, 16, 32)


def test_roundtrip(tmp_path):
    scene = small_scene()
    write_scene(scene, tmp_path)
    back = read_scene(tmp_path)
    assert back.domain == scene.domain and back.materials == scene.materials
    assert np.array_equal(back.emitters[0].transforms, scene.emitters[0].transforms)
    assert back.colliders[0].closed is False


def test_hash_changes(tmp_path):
    write_scene(small_scene(), tmp_path)
    h = scene_hash(tmp_path)
    write_scene(small_scene(gravity=(0, 0, -5)), tmp_path)
    assert scene_hash(tmp_path) != h


def test_wrong_transform_count():
    with pytest.raises(ValueError):
        small_scene(emitter_frames=5)  # frames 1..10 need 10 transforms


def test_bad_particles_per_cell():
    with pytest.raises(ValueError):
        small_scene(particles_per_cell=6)
