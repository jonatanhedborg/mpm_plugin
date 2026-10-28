"""THROWAWAY SPIKE: play back the MPM cache in Blender as point clouds.

Open in Blender's Text Editor and Run Script, or:
    blender --python spike/blender_load.py
Loads the most recently baked spike/cache_* dir; set MPM_CACHE to override.
Re-running replaces the particle objects, so it can switch between caches.
"""

import os

import bpy
import numpy as np


def _script_dirs():
    space = getattr(bpy.context, "space_data", None)
    text = getattr(space, "text", None)
    if text is not None and text.filepath:  # run from the Text Editor
        yield os.path.dirname(bpy.path.abspath(text.filepath))
    if "__file__" in globals():
        yield os.path.dirname(os.path.abspath(__file__))


def _find_cache():
    if os.environ.get("MPM_CACHE"):
        return os.environ["MPM_CACHE"]
    tried = []
    for d in _script_dirs():
        caches = [os.path.join(d, c) for c in os.listdir(d) if c.startswith("cache")]
        caches = [c for c in caches if os.path.exists(os.path.join(c, "meta.txt"))]
        if caches:
            return max(caches, key=lambda c: os.path.getmtime(os.path.join(c, "meta.txt")))
        tried.append(d)
    raise RuntimeError(f"No MPM cache_* dir found next to the script, looked in: {tried}")


CACHE = _find_cache()

meta = dict(line.split("=", 1) for line in open(os.path.join(CACHE, "meta.txt")).read().split())
RES = int(meta["res"])
MAT_NAMES = meta["materials"].split(",")
DX = 1.0 / RES
# sim domain is the unit cube; center it on the origin with the floor at z=0
# (particles settle ~2-3 cells up: the grid boundary layer is 3 cells thick and fuzzy)
OFFSET = np.array([-0.5, -0.5, -2 * DX], np.float32)
RADIUS = DX * 0.3

SHADERS = {  # base color, roughness, subsurface weight
    "cream": ((0.95, 0.93, 0.88), 0.45, 0.6),
    "sand": ((0.76, 0.6, 0.38), 0.9, 0.0),
    "snow": ((0.9, 0.94, 1.0), 0.35, 0.4),
}

mat_ids = np.load(os.path.join(CACHE, "material.npy"))
frame_files = sorted(f for f in os.listdir(CACHE) if f.startswith("frame_"))
indices = {name: np.nonzero(mat_ids == i)[0] for i, name in enumerate(MAT_NAMES)}
indices = {name: idx for name, idx in indices.items() if len(idx)}
ball_path = os.path.join(CACHE, "ball.npy")
ball_track = np.load(ball_path) if os.path.exists(ball_path) else None


def make_material(name):
    m = bpy.data.materials.get(f"MPM_{name}") or bpy.data.materials.new(f"MPM_{name}")
    m.use_nodes = True
    bsdf = m.node_tree.nodes["Principled BSDF"]
    color, rough, sss = SHADERS[name]
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Subsurface Weight"].default_value = sss
    return m


def points_node_group(material):
    ng = bpy.data.node_groups.new(f"MPM_points_{material.name}", "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    n = ng.nodes
    gin, gout = n.new("NodeGroupInput"), n.new("NodeGroupOutput")
    to_pts = n.new("GeometryNodeMeshToPoints")
    to_pts.inputs["Radius"].default_value = RADIUS
    set_mat = n.new("GeometryNodeSetMaterial")
    set_mat.inputs["Material"].default_value = material
    ng.links.new(gin.outputs[0], to_pts.inputs["Mesh"])
    ng.links.new(to_pts.outputs["Points"], set_mat.inputs["Geometry"])
    ng.links.new(set_mat.outputs["Geometry"], gout.inputs[0])
    return ng


def checker_material():
    m = bpy.data.materials.get("MPM_ball") or bpy.data.materials.new("MPM_ball")
    m.use_nodes = True
    nt = m.node_tree
    checker = nt.nodes.get("Checker") or nt.nodes.new("ShaderNodeTexChecker")
    checker.inputs["Scale"].default_value = 8.0
    checker.inputs["Color1"].default_value = (0.8, 0.1, 0.05, 1.0)
    checker.inputs["Color2"].default_value = (0.9, 0.85, 0.2, 1.0)
    nt.links.new(checker.outputs["Color"], nt.nodes["Principled BSDF"].inputs["Base Color"])
    return m


def rebuild_objects():
    coll = bpy.data.collections.get("MPM") or bpy.data.collections.new("MPM")
    if coll.name not in bpy.context.scene.collection.children:
        bpy.context.scene.collection.children.link(coll)
    for ob in list(coll.objects):
        data = ob.data
        bpy.data.objects.remove(ob)
        if data is not None and data.users == 0:
            bpy.data.meshes.remove(data)
    for name, idx in indices.items():
        me = bpy.data.meshes.new(f"MPM_{name}")
        me.vertices.add(len(idx))
        ob = bpy.data.objects.new(f"MPM_{name}", me)
        coll.objects.link(ob)
        mod = ob.modifiers.new("points", "NODES")
        mod.node_group = points_node_group(make_material(name))
    if ball_track is not None:
        bpy.ops.mesh.primitive_uv_sphere_add(radius=float(ball_track[0, 3]), segments=48, ring_count=24)
        ball = bpy.context.object
        ball.name = "MPM_ball"
        bpy.ops.object.shade_smooth()
        for c in ball.users_collection:
            c.objects.unlink(ball)
        coll.objects.link(ball)
        ball.data.materials.append(checker_material())
        for f, (x, y, z, _r, angle) in enumerate(ball_track, start=1):
            ball.location = np.array([x, y, z], np.float32) + OFFSET
            ball.rotation_euler = (0.0, angle, 0.0)
            ball.keyframe_insert("location", frame=f)
            ball.keyframe_insert("rotation_euler", frame=f)


def load_frame(scene, *_):
    f = min(max(scene.frame_current, 1), len(frame_files))
    pos = np.load(os.path.join(CACHE, frame_files[f - 1])) + OFFSET
    for name, idx in indices.items():
        ob = bpy.data.objects.get(f"MPM_{name}")
        if ob is None:
            continue
        ob.data.vertices.foreach_set("co", pos[idx].ravel())
        ob.data.update()


def setup_stage():
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, len(frame_files)
    scene.render.fps = int(float(meta["fps"]))
    if "MPM_floor" not in bpy.data.objects:
        bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0, 0, 0))
        bpy.context.object.name = "MPM_floor"
    if "MPM_cam" not in bpy.data.objects:
        cam = bpy.data.objects.new("MPM_cam", bpy.data.cameras.new("MPM_cam"))
        scene.collection.objects.link(cam)
        cam.location = (0.0, -1.25, 0.55)
        cam.rotation_euler = (1.22, 0.0, 0.0)
        scene.camera = cam
    if "MPM_sun" not in bpy.data.objects:
        sun = bpy.data.objects.new("MPM_sun", bpy.data.lights.new("MPM_sun", "SUN"))
        sun.data.energy = 3.0
        sun.rotation_euler = (0.7, 0.2, 0.6)
        scene.collection.objects.link(sun)


rebuild_objects()
setup_stage()
bpy.app.handlers.frame_change_post[:] = [h for h in bpy.app.handlers.frame_change_post if h.__name__ != "load_frame"]
bpy.app.handlers.frame_change_post.append(load_frame)
load_frame(bpy.context.scene)
print(f"MPM spike: {len(mat_ids):,} particles, {len(frame_files)} frames from {CACHE}")
