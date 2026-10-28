import bpy
import pytest
from conftest import add_cube, cache_of, small_scene

from mpm_surface.meshcache import MeshCacheReader
from mpm_surface.settings import read_settings


@pytest.fixture
def two_emitters(clean_scene, tmp_path):
    """Jelly (AUTO -> surface) and sand (AUTO -> points) in one small domain."""
    domain, jelly = small_scene(clean_scene, tmp_path)
    jelly.name = "Jelly"
    sand = add_cube("Sand", location=(0.12, 0.12, 0.1), size=0.08)
    sand.mpm.role = "EMITTER"
    sand.mpm.emitter.preset_id = "sand_dry"
    return clean_scene, domain, jelly, sand


def out(name):
    return bpy.data.objects[f"MPM {name}"]


def test_auto_surface_after_bake(two_emitters):
    scene, domain, *_ = two_emitters
    domain.mpm.domain.auto_surface = True
    bpy.ops.mpm.bake(wait=True, domain="Domain")
    assert MeshCacheReader(cache_of(domain)).emitters() == ["Jelly"]
    scene.frame_set(2)
    assert len(out("Jelly").data.polygons) > 0 and "MPM Points" not in out("Jelly").modifiers
    assert "MPM Points" in out("Sand").modifiers


def test_surface_settings_written(two_emitters):
    _, domain, jelly, _ = two_emitters
    bpy.ops.mpm.bake(wait=True, domain="Domain")
    jelly.mpm.emitter.surface_detail = 2.0
    assert bpy.ops.mpm.bake_surface(wait=True, domain="Domain") == {"FINISHED"}
    settings = read_settings(cache_of(domain))
    assert settings["Jelly"].detail == 2.0 and settings["Sand"].enabled is False


def test_display_points_override(two_emitters):
    _, domain, jelly, _ = two_emitters
    jelly.mpm.emitter.display = "POINTS"
    bpy.ops.mpm.bake(wait=True, domain="Domain")
    bpy.ops.mpm.bake_surface(wait=True, domain="Domain")
    assert MeshCacheReader(cache_of(domain)).emitters() == []
    assert "MPM Points" in out("Jelly").modifiers


def test_bake_surface_without_bake_errors(two_emitters):
    with pytest.raises(RuntimeError, match="bake first"):
        bpy.ops.mpm.bake_surface(wait=True, domain="Domain")


def test_free_surface(two_emitters):
    scene, domain, *_ = two_emitters
    bpy.ops.mpm.bake(wait=True, domain="Domain")
    bpy.ops.mpm.bake_surface(wait=True, domain="Domain")
    assert bpy.ops.mpm.free_surface(domain="Domain") == {"FINISHED"}
    assert not (cache_of(domain) / "surface").exists()
    assert "MPM Points" in out("Jelly").modifiers
