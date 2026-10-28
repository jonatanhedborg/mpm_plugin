import bpy
import pytest
from conftest import cache_of, small_scene

from mpm_core.cache import CacheReader, frame_path


@pytest.fixture
def baked(clean_scene, tmp_path):
    domain, emitter = small_scene(clean_scene, tmp_path)
    bpy.ops.mpm.bake(wait=True, domain="Domain")
    return clean_scene, domain, emitter


def output(name="Emitter"):
    return bpy.data.objects.get(f"MPM {name}")


def test_output_object_per_emitter(baked):
    scene, domain, _ = baked
    coll = bpy.data.collections.get("MPM Domain")
    assert coll is not None and [o.name for o in coll.objects] == ["MPM Emitter"]
    scene.frame_set(3)
    pos = CacheReader(cache_of(domain)).read_frame(3)["Emitter"][0]
    assert len(output().data.vertices) == len(pos)


def test_material_from_preset(baked):
    mat = output().data.materials[0]
    assert mat.name == "MPM Jelly"
    color = mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value
    assert tuple(round(c, 3) for c in color[:3]) == (0.8, 0.15, 0.2)


def test_existing_material_kept(clean_scene, tmp_path):
    mine = bpy.data.materials.new("MPM Jelly")
    mine.diffuse_color = (0, 1, 0, 1)
    small_scene(clean_scene, tmp_path)
    bpy.ops.mpm.bake(wait=True, domain="Domain")
    assert output().data.materials[0] == mine and tuple(mine.diffuse_color) == (0, 1, 0, 1)


def test_scrub_unbaked_frame(baked):  # Review Focus
    scene, domain, _ = baked
    scene.frame_set(3)
    count = len(output().data.vertices)
    frame_path(cache_of(domain), 4).unlink()
    scene.frame_set(4)  # must not raise
    assert len(output().data.vertices) == count


def test_renamed_emitter(baked):  # Review Focus
    scene, _, emitter = baked
    emitter.name = "Renamed"
    scene.frame_set(2)  # stale output object, must not raise
    bpy.ops.mpm.bake(wait=True, domain="Domain")
    assert output("Emitter") is None and output("Renamed") is not None


def test_reload_file(baked, tmp_path):  # Review Focus
    path = str(tmp_path / "scene.blend")
    bpy.ops.wm.save_as_mainfile(filepath=path)
    bpy.ops.wm.open_mainfile(filepath=path)
    from mpm_blender import display

    assert display.load_frame in bpy.app.handlers.frame_change_post
    try:
        bpy.context.scene.frame_set(2)
        assert len(output().data.vertices) > 0
    finally:
        bpy.ops.wm.read_homefile(use_empty=True)  # back to an unsaved session for the other tests


def test_free_bake_removes_outputs(baked):
    bpy.ops.mpm.free_bake(domain="Domain")
    assert output() is None
