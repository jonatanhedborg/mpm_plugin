"""DEV TOOL: view an mpm_core cache in Blender as point clouds. Superseded by the add-on.

Set CACHE_DIR below (or the MPM_CACHE environment variable), then run from Blender's
Text Editor. Reads the cache files directly, so the mpm_core package isn't needed in Blender.
Creates one point object per emitter in an "MPM" collection; re-running replaces them.
"""

import json
import os

import bpy
import numpy as np

CACHE_DIR = r""  # e.g. r"D:\projects\mpm_solver\out\mix"

COLORS = [(0.95, 0.93, 0.88), (0.76, 0.6, 0.38), (0.85, 0.9, 1.0), (0.8, 0.15, 0.2), (0.4, 0.7, 0.4)]


def _cache_dir() -> str:
    path = os.environ.get("MPM_CACHE") or CACHE_DIR
    if not path or not os.path.exists(os.path.join(path, "cache.json")):
        raise RuntimeError(f"set CACHE_DIR (or MPM_CACHE) to a baked cache directory, got {path!r}")
    return path


CACHE = _cache_dir()
META = json.load(open(os.path.join(CACHE, "cache.json")))
NAMES = META["emitters"]
RADIUS = META["dx"] * 0.3


def frame_file(frame: int) -> str:
    return os.path.join(CACHE, "frames", f"frame_{frame:04d}.npz")


def points_node_group(name, material):
    ng = bpy.data.node_groups.new(name, "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    gin, gout = ng.nodes.new("NodeGroupInput"), ng.nodes.new("NodeGroupOutput")
    to_pts = ng.nodes.new("GeometryNodeMeshToPoints")
    to_pts.inputs["Radius"].default_value = RADIUS
    set_mat = ng.nodes.new("GeometryNodeSetMaterial")
    set_mat.inputs["Material"].default_value = material
    ng.links.new(gin.outputs[0], to_pts.inputs["Mesh"])
    ng.links.new(to_pts.outputs["Points"], set_mat.inputs["Geometry"])
    ng.links.new(set_mat.outputs["Geometry"], gout.inputs[0])
    return ng


def rebuild_objects():
    coll = bpy.data.collections.get("MPM") or bpy.data.collections.new("MPM")
    if coll.name not in bpy.context.scene.collection.children:
        bpy.context.scene.collection.children.link(coll)
    for ob in list(coll.objects):
        data = ob.data
        bpy.data.objects.remove(ob)
        if data is not None and data.users == 0:
            bpy.data.meshes.remove(data)
    for i, name in enumerate(NAMES):
        mat = bpy.data.materials.get(f"MPM_{name}") or bpy.data.materials.new(f"MPM_{name}")
        mat.use_nodes = True
        mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (*COLORS[i % len(COLORS)], 1)
        ob = bpy.data.objects.new(f"MPM_{name}", bpy.data.meshes.new(f"MPM_{name}"))
        coll.objects.link(ob)
        ob.modifiers.new("points", "NODES").node_group = points_node_group(f"MPM_points_{name}", mat)


def load_frame(scene, *_):
    frame = min(max(scene.frame_current, META["frame_start"]), META["frame_end"])
    if not os.path.exists(frame_file(frame)):
        return  # not baked (yet)
    with np.load(frame_file(frame)) as data:
        for i, name in enumerate(NAMES):
            pos = data[f"pos_{i}"]
            me = bpy.data.objects[f"MPM_{name}"].data
            if len(me.vertices) != len(pos):
                me.clear_geometry()
                me.vertices.add(len(pos))
            me.vertices.foreach_set("co", pos.ravel())
            me.update()


rebuild_objects()
scene = bpy.context.scene
scene.frame_start, scene.frame_end = META["frame_start"], META["frame_end"]
scene.render.fps = int(META["fps"])
bpy.app.handlers.frame_change_post[:] = [h for h in bpy.app.handlers.frame_change_post
                                         if h.__name__ != "load_frame"]
bpy.app.handlers.frame_change_post.append(load_frame)
load_frame(scene)
print(f"MPM cache: {', '.join(NAMES)} from {CACHE}")
