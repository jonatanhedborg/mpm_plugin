import math

import bpy
import numpy as np
import pytest
from conftest import add_cube, add_plane, add_sphere

from mpm_core.scene import read_scene


@pytest.fixture
def basic(clean_scene, tmp_path):
    """1 m domain (z 0..1), cream sphere emitter, plane collider moving +x over frames 1..5."""
    scene = clean_scene
    scene.frame_end = 5
    domain = add_cube("Domain", location=(0, 0, 0.5))
    domain.mpm.role = "DOMAIN"
    domain.mpm.domain.cache_dir = str(tmp_path / "cache")
    emitter = add_sphere("Emitter", location=(0, 0, 0.5), radius=0.1)
    emitter.mpm.role = "EMITTER"
    emitter.mpm.emitter.preset = "whipped_cream"
    emitter.mpm.emitter.firmness = 2.0
    plane = add_plane("Collider", location=(0, 0, 0.2), size=0.6)
    plane.mpm.role = "COLLIDER"
    for frame, x in ((1, 0.0), (5, 0.4)):
        plane.location.x = x
        plane.keyframe_insert("location", frame=frame)
    scene.frame_set(1)
    return scene, domain, emitter, plane


def export(scene, domain, quality="FINAL"):
    from mpm_blender.export import export_scene

    return export_scene(scene, domain, quality)


def test_domain_and_quality(basic):
    scene, domain, *_ = basic
    s, names = export(scene, domain, "FINAL")
    assert s.domain.resolution == 128 and export(scene, domain, "PREVIEW")[0].domain.resolution == 64
    assert np.allclose(s.domain.min, (-0.5, -0.5, 0.0), atol=1e-5)
    assert np.allclose(s.domain.max, (0.5, 0.5, 1.0), atol=1e-5)
    assert names == ["Emitter"] and (s.frame_start, s.frame_end) == (1, 5)
    assert s.fps == pytest.approx(scene.render.fps / scene.render.fps_base)


def test_emitter_transforms_and_local_mesh(basic):
    scene, domain, *_ = basic
    s, _ = export(scene, domain)
    e = s.emitters[0]
    assert len(e.transforms) == 5
    assert np.abs(np.linalg.norm(e.mesh.verts, axis=1) - 0.1).max() < 0.01  # local space, around the origin
    assert np.allclose(e.transforms[0][:3, 3], (0, 0, 0.5), atol=1e-6)
    xs = [m[0, 3] for m in s.colliders[0].transforms]
    assert xs[0] == pytest.approx(0.0, abs=1e-6) and xs[-1] == pytest.approx(0.4, abs=1e-6)
    assert all(b > a for a, b in zip(xs, xs[1:]))
    assert scene.frame_current == 1  # restored


def test_material_from_tweaks(basic):
    scene, domain, *_ = basic
    s, _ = export(scene, domain)
    assert s.materials[0].yield_stress == pytest.approx(5000) and s.materials[0].name == "Emitter"
    assert s.emitters[0].material == 0


def test_collider_open_and_closed(basic):
    scene, domain, *_ = basic
    assert export(scene, domain)[0].colliders[0].closed is False
    box = add_cube("Box", location=(0.3, 0.3, 0.1), size=0.1)
    box.mpm.role = "COLLIDER"
    s, _ = export(scene, domain)
    assert [c.name for c in s.colliders] == ["Box", "Collider"]
    assert s.colliders[0].closed is True


def test_walls_and_inflow(basic):
    scene, domain, emitter, _ = basic
    domain.mpm.domain.wall_z_pos = False
    e = emitter.mpm.emitter
    e.kind = "INFLOW"
    e.velocity = (0, 0, -1)
    e.inflow_start, e.inflow_end = 2, 4
    e.inflow_substeps = 3
    s, _ = export(scene, domain)
    assert s.domain.walls["z+"] is False and s.domain.walls["z-"] is True
    em = s.emitters[0]
    assert em.kind == "inflow" and em.velocity == (0, 0, -1) and (em.frame_start, em.frame_end) == (2, 4)
    assert em.substeps == 3


def test_gravity_off(basic):
    scene, domain, *_ = basic
    scene.use_gravity = False
    assert export(scene, domain)[0].gravity == (0, 0, 0)
    scene.use_gravity = True


def test_rotated_domain_refused(basic):  # Review Focus
    from mpm_blender.export import ExportError

    scene, domain, *_ = basic
    domain.rotation_euler.z = math.radians(10)
    with pytest.raises(ExportError, match="Domain"):
        export(scene, domain)


def test_unsaved_relative_cache_refused(basic):  # Review Focus
    from mpm_blender.export import ExportError, cache_root

    _, domain, *_ = basic
    domain.mpm.domain.cache_dir = ""
    assert bpy.data.filepath == ""
    with pytest.raises(ExportError, match="save the .blend"):
        cache_root(domain)


def test_write_export_roundtrip(basic):
    from mpm_blender.export import write_export

    scene, domain, *_ = basic
    path = write_export(scene, domain, "PREVIEW")
    assert path.name == "preview"
    back = read_scene(path / "scene")
    expected, _ = export(scene, domain, "PREVIEW")
    assert back.domain == expected.domain and back.materials == expected.materials
    assert len(back.colliders[0].transforms) == 5


def test_no_emitters_refused(basic):
    from mpm_blender.export import ExportError

    scene, domain, emitter, _ = basic
    emitter.mpm.role = "NONE"
    with pytest.raises(ExportError, match="no emitters"):
        export(scene, domain)
