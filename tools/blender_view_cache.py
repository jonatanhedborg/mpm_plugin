"""DEV TOOL: view mpm_core caches in Blender. Superseded by the add-on.

Set CACHE_DIR below (or the MPM_CACHE environment variable) to either one baked cache, or a
folder of caches (e.g. out/tests): each cache is then laid out side by side in a grid with its
name, domain box and animated colliders, all driven by the one timeline. Emitters that have been
surfaced (python -m mpm_surface CACHE) show as meshes with a "velocity" attribute; others as points.

Run from Blender's Text Editor. Reads the cache files directly, so the mpm_core package isn't
needed in Blender. Re-running replaces everything in the "MPM" collection.
"""

import json
import os

import bpy
import numpy as np
from mathutils import Matrix

CACHE_DIR = r""  # e.g. r"D:\projects\mpm_solver\out\mix"; empty = <repo>/out/tests next to this script
COLUMNS = 5
GAP = 0.4  # metres between domains

# by material (preset id); anything else cycles through FALLBACK_COLORS
PRESET_COLORS = {"whipped_cream": (0.97, 0.93, 0.8), "sand_dry": (0.76, 0.6, 0.38), "snow_packed": (0.6, 0.75, 1.0),
                 "snow_powder": (0.8, 0.88, 1.0), "jelly": (0.8, 0.15, 0.2)}
FALLBACK_COLORS = [(0.4, 0.7, 0.4), (0.7, 0.4, 0.7), (0.9, 0.6, 0.2)]


def _script_dir() -> str | None:
    space = getattr(bpy.context, "space_data", None)
    text = getattr(space, "text", None)
    if text is not None and text.filepath:  # run from the Text Editor
        return os.path.dirname(bpy.path.abspath(text.filepath))
    return os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else None


def _root() -> str:
    path = os.environ.get("MPM_CACHE") or CACHE_DIR
    if not path and _script_dir():
        path = os.path.join(os.path.dirname(_script_dir()), "out", "tests")
    if not path or not os.path.isdir(path):
        raise RuntimeError(f"no cache folder at {path!r}. Set the line at the top of this script, e.g. "
                           f'CACHE_DIR = r"D:\\projects\\mpm_solver\\out\\mix" (keep the r before the quotes)')
    return path


def _find_caches(root: str) -> list[str]:
    if os.path.exists(os.path.join(root, "cache.json")):
        return [root]
    subdirs = sorted(os.path.join(root, d) for d in os.listdir(root))
    caches = [d for d in subdirs if os.path.exists(os.path.join(d, "cache.json"))]
    if not caches:
        raise RuntimeError(f"no baked caches (cache.json) in {root}")
    return caches


class CacheView:
    """One cache: an empty at a grid offset, parenting points, colliders, domain box and label."""

    def __init__(self, path: str, index: int, coll, offset):
        self.path = path
        self.name = os.path.basename(os.path.normpath(path))
        self.meta = json.load(open(os.path.join(path, "cache.json")))
        scene_doc = json.load(open(os.path.join(path, "scene", "scene.json")))
        dom = scene_doc["domain"]
        self.dmin, self.dmax = np.array(dom["min"]), np.array(dom["max"])
        self.coll = coll

        self.root = bpy.data.objects.new(f"MPM {self.name}", None)
        self.root.location = offset - self.dmin * np.array([1, 1, 0])
        coll.objects.link(self.root)
        self._domain_box()
        self._label()
        materials = [scene_doc["materials"][e["material"]]["name"] for e in scene_doc["emitters"]]
        self.outputs = [self._output(name, mat) for name, mat in zip(self.meta["emitters"], materials)]
        for i, c in enumerate(scene_doc["colliders"]):
            self._collider(i, c["name"], scene_doc["frame_start"])

    def _child(self, ob):
        self.coll.objects.link(ob)
        ob.parent = self.root
        return ob

    def _domain_box(self):
        bpy.ops.mesh.primitive_cube_add(size=1.0)
        box = bpy.context.object
        for c in box.users_collection:
            c.objects.unlink(box)
        box.name = f"{self.name} domain"
        self._child(box)
        box.location = (self.dmin + self.dmax) / 2
        box.scale = self.dmax - self.dmin
        box.display_type = "WIRE"
        box.hide_render = True

    def _label(self):
        curve = bpy.data.curves.new(f"{self.name} label", "FONT")
        curve.body = self.name.replace("_", " ")
        curve.size = 0.06
        label = self._child(bpy.data.objects.new(f"{self.name} label", curve))
        label.location = (self.dmin[0], self.dmin[1] - 0.12, self.dmin[2])

    def _output(self, name, material_name):
        """(object, surface dir or None): a mesh object if the emitter was surfaced, else points."""
        mat = bpy.data.materials.get(f"MPM_{material_name}") or bpy.data.materials.new(f"MPM_{material_name}")
        mat.use_nodes = True
        color = PRESET_COLORS.get(material_name, FALLBACK_COLORS[len(material_name) % len(FALLBACK_COLORS)])
        mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (*color, 1)
        mat.diffuse_color = (*color, 1)  # viewport solid mode
        me = bpy.data.meshes.new(f"{self.name} {name}")
        me.materials.append(mat)
        ob = self._child(bpy.data.objects.new(f"{self.name} {name}", me))
        manifest = os.path.join(self.path, "surface", "meshes.json")
        folder = json.load(open(manifest))["emitters"].get(name) if os.path.exists(manifest) else None
        surface_dir = os.path.join(self.path, "surface", "meshes", folder) if folder else ""
        if folder and os.path.isdir(surface_dir):
            return ob, surface_dir
        ob.modifiers.new("points", "NODES").node_group = _points_node_group(mat, self.meta["dx"] * 0.3)
        return ob, None

    def _collider(self, i, name, frame_start):
        scene_dir = os.path.join(self.path, "scene")
        verts = np.load(os.path.join(scene_dir, f"collider_{i}_verts.npy"))
        tris = np.load(os.path.join(scene_dir, f"collider_{i}_tris.npy"))
        xfs = np.load(os.path.join(scene_dir, f"collider_{i}_xf.npy"))
        me = bpy.data.meshes.new(f"{self.name} {name}")
        me.from_pydata(verts.tolist(), [], tris.tolist())
        ob = self._child(bpy.data.objects.new(f"{self.name} {name}", me))
        ob.display_type = "SOLID"
        ob.color = (0.3, 0.3, 0.35, 1)
        for k, m in enumerate(xfs):
            ob.matrix_basis = Matrix(m.tolist())
            for path in ("location", "rotation_euler", "scale"):
                ob.keyframe_insert(path, frame=frame_start + k)

    def load(self, frame: int):
        frame = min(max(frame, self.meta["frame_start"]), self.meta["frame_end"])
        path = os.path.join(self.path, "frames", f"frame_{frame:04d}.npz")
        if not os.path.exists(path):
            return  # not baked (yet)
        with np.load(path) as data:
            for i, (ob, surface_dir) in enumerate(self.outputs):
                if surface_dir is None:
                    _set_points(ob.data, data[f"pos_{i}"])
                    continue
                mesh_path = os.path.join(surface_dir, f"frame_{frame:04d}.npz")
                if os.path.exists(mesh_path):
                    with np.load(mesh_path) as m:
                        _set_mesh(ob.data, m["verts"], m["tris"], m["vel"])
                else:
                    ob.data.clear_geometry()  # not surfaced (yet)


def _set_points(me, pos):
    if len(me.vertices) != len(pos):
        me.clear_geometry()
        me.vertices.add(len(pos))
    me.vertices.foreach_set("co", pos.ravel())
    me.update()


def _set_mesh(me, verts, tris, vel):
    me.clear_geometry()
    me.vertices.add(len(verts))
    me.vertices.foreach_set("co", verts.ravel())
    me.loops.add(tris.size)
    me.loops.foreach_set("vertex_index", tris.ravel())
    me.polygons.add(len(tris))
    me.polygons.foreach_set("loop_start", np.arange(0, tris.size, 3, dtype=np.int32))
    me.polygons.foreach_set("use_smooth", np.ones(len(tris), bool))
    attr = me.attributes.get("velocity") or me.attributes.new("velocity", "FLOAT_VECTOR", "POINT")
    attr.data.foreach_set("vector", vel.ravel())
    me.update()


def _points_node_group(material, radius):
    ng = bpy.data.node_groups.new(f"MPM points {material.name}", "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    gin, gout = ng.nodes.new("NodeGroupInput"), ng.nodes.new("NodeGroupOutput")
    to_pts = ng.nodes.new("GeometryNodeMeshToPoints")
    to_pts.inputs["Radius"].default_value = radius
    set_mat = ng.nodes.new("GeometryNodeSetMaterial")
    set_mat.inputs["Material"].default_value = material
    ng.links.new(gin.outputs[0], to_pts.inputs["Mesh"])
    ng.links.new(to_pts.outputs["Points"], set_mat.inputs["Geometry"])
    ng.links.new(set_mat.outputs["Geometry"], gout.inputs[0])
    return ng


def build() -> list[CacheView]:
    coll = bpy.data.collections.get("MPM") or bpy.data.collections.new("MPM")
    if coll.name not in bpy.context.scene.collection.children:
        bpy.context.scene.collection.children.link(coll)
    for ob in list(coll.objects):
        bpy.data.objects.remove(ob)
    views, x, y, row_depth = [], 0.0, 0.0, 0.0
    for i, path in enumerate(_find_caches(_root())):
        if i and i % COLUMNS == 0:
            x, y, row_depth = 0.0, y - row_depth - GAP, 0.0
        v = CacheView(path, i, coll, np.array([x, y, 0.0]))
        size = v.dmax - v.dmin
        x += size[0] + GAP
        row_depth = max(row_depth, size[1] + 0.15)
        views.append(v)
    return views


VIEWS = build()


def load_frame(scene, *_):
    for v in VIEWS:
        v.load(scene.frame_current)


scene = bpy.context.scene
scene.frame_start = min(v.meta["frame_start"] for v in VIEWS)
scene.frame_end = max(v.meta["frame_end"] for v in VIEWS)
scene.render.fps = int(VIEWS[0].meta["fps"])
bpy.app.handlers.frame_change_post[:] = [h for h in bpy.app.handlers.frame_change_post
                                         if h.__name__ != "load_frame"]
bpy.app.handlers.frame_change_post.append(load_frame)
load_frame(scene)
print(f"MPM: {len(VIEWS)} cache(s) from {_root()}")
