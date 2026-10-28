import numpy as np
import pytest
from helpers import solver_with_box, solver_with_presets, solver_with_sphere

from mpm_core.materials import list_presets, load_preset
from mpm_core.scene import Domain


@pytest.mark.gpu
@pytest.mark.parametrize("preset", list_presets())
def test_slab_at_rest(device, preset):  # spec: no sag at frame 1
    d = Domain((0, 0, 0), (0.5, 0.5, 0.5), 64)
    s = solver_with_box(device, d, load_preset(preset).material, lo=(0, 0, 0), hi=(0.5, 0.5, 0.07))  # wall to wall
    start = s.get_particles().positions.copy()
    for _ in range(24):
        s.advance(1 / 24)
    assert np.abs(s.get_particles().positions - start).max() < 0.5 * d.dx


@pytest.mark.gpu
def test_sand_column_collapses(device):
    d = Domain((0, 0, 0), (1, 1, 1), 64)
    s = solver_with_box(device, d, load_preset("sand_dry").material, lo=(0.44, 0.44, 0), hi=(0.56, 0.56, 0.33))
    for _ in range(48):
        s.advance(1 / 24)
    pos = s.get_particles().positions
    assert pos[:, 2].max() < 0.6 * 0.33
    assert 0.12 < pos[:, 0].max() - 0.5 < 0.45


@pytest.mark.gpu
def test_cream_holds_shape(device):
    d = Domain((0, 0, 0), (1, 1, 1), 64)
    s = solver_with_sphere(device, d, load_preset("whipped_cream").material, center=(0.5, 0.5, 0.11), radius=0.11)
    for _ in range(48):
        s.advance(1 / 24)
    z = s.get_particles().positions[:, 2]
    assert z.max() - z.min() > 0.5 * 0.22


@pytest.mark.gpu
def test_jelly_recovers(device):  # elastic: drops, bounces, comes back to ~original height
    d = Domain((0, 0, 0), (1, 1, 1), 64)
    s = solver_with_box(device, d, load_preset("jelly").material, lo=(0.4, 0.4, 0.2), hi=(0.6, 0.6, 0.4))
    for _ in range(72):
        s.advance(1 / 24)
    z = s.get_particles().positions[:, 2]
    assert z.max() - z.min() == pytest.approx(0.2, rel=0.15)


@pytest.mark.gpu
def test_mixed_materials_one_domain(device):
    d = Domain((0, 0, 0), (1, 1, 1), 64)
    s = solver_with_presets(device, d, [("sand_dry", (0.3, 0.3, 0), (0.5, 0.5, 0.1)),
                                         ("snow_packed", (0.5, 0.5, 0), (0.7, 0.7, 0.1))])
    for _ in range(24):
        s.advance(1 / 24)
    assert np.isfinite(s.get_particles().positions).all()
