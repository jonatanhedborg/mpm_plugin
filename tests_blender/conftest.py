"""Runs inside Blender (see tools/blender_tests.py)."""

import bpy
import pytest


@pytest.fixture(scope="session", autouse=True)
def addon(tmp_path_factory):
    import mpm_blender
    from mpm_blender import presets

    presets.USER_DIR_OVERRIDE = tmp_path_factory.mktemp("user_presets")  # never the real user folder
    mpm_blender.register()
    yield mpm_blender
    mpm_blender.unregister()


@pytest.fixture
def clean_scene():
    """An empty scene with frames 1..10."""
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob)
    for coll in list(bpy.data.collections):
        bpy.data.collections.remove(coll)
    for block in (bpy.data.meshes, bpy.data.materials, bpy.data.node_groups, bpy.data.curves):
        for item in list(block):
            block.remove(item)
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, 10
    scene.render.use_lock_interface = False
    scene.frame_set(1)
    return scene


def _add(op, name, **kwargs):
    op(**kwargs)
    ob = bpy.context.object
    ob.name = name
    ob.data.name = name
    return ob


def add_cube(name, location=(0, 0, 0), size=1.0):
    return _add(bpy.ops.mesh.primitive_cube_add, name, size=size, location=location)


def add_sphere(name, location=(0, 0, 0), radius=0.1):
    return _add(bpy.ops.mesh.primitive_uv_sphere_add, name, radius=radius, location=location, segments=24,
                ring_count=12)


def add_plane(name, location=(0, 0, 0), size=1.0):
    return _add(bpy.ops.mesh.primitive_plane_add, name, size=size, location=location)


def small_scene(scene, tmp_path, frames=4, resolution=16):
    """0.5 m domain (z 0..0.5), jelly cube emitter, FINAL quality, absolute cache folder."""
    scene.frame_end = frames
    domain = add_cube("Domain", location=(0, 0, 0.25), size=0.5)
    domain.mpm.role = "DOMAIN"
    domain.mpm.domain.resolution = resolution
    domain.mpm.domain.quality = "FINAL"
    domain.mpm.domain.cache_dir = str(tmp_path / "cache")
    domain.mpm.domain.auto_surface = False
    emitter = add_cube("Emitter", location=(0, 0, 0.15), size=0.1)
    emitter.mpm.role = "EMITTER"
    emitter.mpm.emitter.preset_id = "jelly"
    return domain, emitter


def cache_of(domain):
    from mpm_blender.export import cache_path

    return cache_path(domain, domain.mpm.domain.quality)
