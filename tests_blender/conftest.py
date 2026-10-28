"""Runs inside Blender (see tools/blender_tests.py)."""

import bpy
import pytest


@pytest.fixture(scope="session", autouse=True)
def addon():
    import mpm_blender

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
