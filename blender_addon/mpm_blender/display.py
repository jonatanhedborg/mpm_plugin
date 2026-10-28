"""Playback: one output object per emitter, loaded from the cache on frame change.

Output objects live in a collection ``MPM <domain>`` and are named ``MPM <emitter>``. They are
points (Geometry Nodes Mesh to Points) or, once surfaced, meshes from the mesh cache with a
``velocity`` attribute for motion blur. Nothing here ever raises into Blender: playback errors are
printed and the last loaded geometry stays.
"""

from __future__ import annotations

import json
import time
import traceback
import zipfile
from contextlib import contextmanager
from pathlib import Path

import bpy
import numpy as np
from bpy.app.handlers import persistent


from . import jobs, operators, presets, props
from .export import ExportError, cache_path

POINTS_MODIFIER = "MPM Points"
RUNNING_STATUSES = ("starting", "running", "surfacing")

_LOADED: dict[int, tuple[str, float]] = {}  # output object session_uid -> (file, mtime) it shows
_SUSPENDED = [0]


@contextmanager
def suspended():
    """No playback loads inside (scene.frame_set while exporting would reload the old cache every frame)."""
    _SUSPENDED[0] += 1
    try:
        yield
    finally:
        _SUSPENDED[0] -= 1


def _retry(read, attempts: int = 5):
    """Readers retry: on Windows a file being replaced is briefly locked."""
    for i in range(attempts):
        try:
            return read()
        except (PermissionError, zipfile.BadZipFile, EOFError):
            if i == attempts - 1:
                raise
            time.sleep(0.02)


def _domains(scene):
    return [ob for ob in scene.objects if ob.type == "MESH" and ob.mpm.role == "DOMAIN"]


def _cache_of(domain_obj) -> Path | None:
    try:
        path = cache_path(domain_obj, domain_obj.mpm.domain.quality)
    except ExportError:
        return None
    return path if (path / "cache.json").exists() else None


def output_collection(domain_obj):
    """The domain's output collection, found by the domain's persistent id (renames don't lose it)."""
    uid = domain_obj.mpm.domain.uid
    if not uid:
        return None
    return next((c for c in bpy.data.collections if c.get("mpm_domain_uid") == uid), None)


def _collection(scene, domain_obj, create: bool):
    coll = output_collection(domain_obj)
    if coll is None and create:
        coll = bpy.data.collections.new(f"MPM {domain_obj.name}")
        coll["mpm_domain_uid"] = operators.domain_uid(domain_obj)
    if coll is not None and create:
        if coll.name != f"MPM {domain_obj.name}":
            coll.name = f"MPM {domain_obj.name}"  # follow a renamed domain
        if coll.name not in scene.collection.children:
            scene.collection.children.link(coll)
        # load_frame rewrites output meshes from the render thread while rendering; without the lock the
        # viewport draws them at the same time and Blender crashes (see "Note on Altering Data" in the
        # bpy.app.handlers docs)
        scene.render.use_lock_interface = True
    return coll


def _material(emitter):
    """The shared render material of the emitter's preset, created from the preset's look on first use.

    ``emitter`` is the emitter's settings, or None when the emitter object is gone.
    """
    preset = presets.get(emitter.preset_id) if emitter is not None else None
    label = preset.label if preset is not None else (emitter.preset_label if emitter is not None else "") or "Custom"
    name = f"MPM {label}"
    mat = bpy.data.materials.get(name)
    if mat is not None:
        return mat  # never overwrite: the user may have edited it
    look = preset.look if preset is not None else presets.DEFAULT_LOOK
    mat = bpy.data.materials.new(name)
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*look.base_color, 1.0)
    bsdf.inputs["Roughness"].default_value = look.roughness
    bsdf.inputs["Subsurface Weight"].default_value = look.subsurface
    mat.diffuse_color = (*look.base_color, 1.0)
    return mat


def _points_group(material, radius: float):
    ng = bpy.data.node_groups.new("MPM Points", "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    gin, gout = ng.nodes.new("NodeGroupInput"), ng.nodes.new("NodeGroupOutput")
    to_points = ng.nodes.new("GeometryNodeMeshToPoints")
    to_points.inputs["Radius"].default_value = radius
    set_mat = ng.nodes.new("GeometryNodeSetMaterial")
    set_mat.inputs["Material"].default_value = material
    ng.links.new(gin.outputs[0], to_points.inputs["Mesh"])
    ng.links.new(to_points.outputs["Points"], set_mat.inputs["Geometry"])
    ng.links.new(set_mat.outputs["Geometry"], gout.inputs[0])
    return ng


def _mesh_folders(cache: Path) -> dict[str, str]:
    manifest = cache / "surface" / "meshes.json"
    if not manifest.exists():
        return {}
    folders = _retry(lambda: json.loads(manifest.read_text(encoding="utf-8")))["emitters"]
    return {name: f for name, f in folders.items() if (cache / "surface" / "meshes" / f).is_dir()}


def remove_outputs(scene, domain_obj) -> None:
    coll = _collection(scene, domain_obj, create=False)
    if coll is None:
        return
    for ob in list(coll.objects):
        mesh = ob.data
        bpy.data.objects.remove(ob)
        if mesh is not None and mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    bpy.data.collections.remove(coll)


def refresh_outputs(scene, domain_obj, cache: Path | None = None) -> None:
    """Match the output objects to the cache: create, update, delete; then load the current frame."""
    cache = cache or _cache_of(domain_obj)
    if cache is None:
        remove_outputs(scene, domain_obj)
        return
    meta = _retry(lambda: json.loads((cache / "cache.json").read_text(encoding="utf-8")))
    names = meta["emitters"]
    coll = _collection(scene, domain_obj, create=True)
    existing = {ob.get("mpm_emitter"): ob for ob in coll.objects}
    for name, ob in existing.items():
        if name not in names:
            bpy.data.objects.remove(ob)
    folders = _mesh_folders(cache)
    for index, name in enumerate(names):
        ob = existing.get(name)
        if ob is None:
            ob = bpy.data.objects.new(f"MPM {name}", bpy.data.meshes.new(f"MPM {name}"))
            coll.objects.link(ob)
            ob["mpm_emitter"] = name
        ob["mpm_index"] = index
        emitter_obj = bpy.data.objects.get(name)
        if not ob.data.materials:
            ob.data.materials.append(_material(emitter_obj.mpm.emitter if emitter_obj is not None else None))
        mod = ob.modifiers.get(POINTS_MODIFIER)
        if name in folders:
            ob["mpm_mesh_folder"] = folders[name]
            if mod is not None:
                ob.modifiers.remove(mod)
        else:
            if "mpm_mesh_folder" in ob:
                del ob["mpm_mesh_folder"]
            if mod is None:
                mod = ob.modifiers.new(POINTS_MODIFIER, "NODES")
                mod.node_group = _points_group(ob.data.materials[0], meta["dx"] * 0.3)
    load_domain_frame(scene, domain_obj, scene.frame_current, cache, force=True)


def _read_npz(path: Path) -> dict:
    return _retry(lambda: dict(np.load(path)))


def _set_points(me, pos) -> None:
    if len(me.polygons) or len(me.vertices) != len(pos):
        me.clear_geometry()
        me.vertices.add(len(pos))
    me.vertices.foreach_set("co", np.ascontiguousarray(pos, np.float32).ravel())
    me.update()


def _set_mesh(me, verts, tris, vel) -> None:
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


def load_domain_frame(scene, domain_obj, frame: int, cache: Path | None = None, force: bool = False) -> None:
    """Show ``frame``; objects already showing that exact file (same mtime) are left alone unless forced."""
    if _SUSPENDED[0]:
        return
    cache = cache or _cache_of(domain_obj)
    coll = output_collection(domain_obj)
    if cache is None or coll is None:
        return
    meta = _retry(lambda: json.loads((cache / "cache.json").read_text(encoding="utf-8")))
    frame = min(max(frame, meta["frame_start"]), meta["frame_end"])
    particle_file = cache / "frames" / f"frame_{frame:04d}.npz"
    if not particle_file.exists():
        return  # not baked yet: keep what's shown
    particles = None
    for ob in coll.objects:
        index = ob.get("mpm_index")
        if index is None:
            continue
        folder = ob.get("mpm_mesh_folder")
        source = cache / "surface" / "meshes" / folder / f"frame_{frame:04d}.npz" if folder else particle_file
        if not source.exists():
            continue
        stamp = (str(source), source.stat().st_mtime)
        if not force and _LOADED.get(ob.session_uid) == stamp:
            continue
        if folder:
            m = _read_npz(source)
            _set_mesh(ob.data, m["verts"], m["tris"], m["vel"])
        else:
            particles = particles if particles is not None else _read_npz(particle_file)
            if f"pos_{index}" not in particles:
                continue
            _set_points(ob.data, particles[f"pos_{index}"])
        _LOADED[ob.session_uid] = stamp


@persistent
def load_frame(scene, depsgraph=None) -> None:
    for domain_obj in _domains(scene):
        try:
            load_domain_frame(scene, domain_obj, scene.frame_current)
        except Exception:  # noqa: BLE001 - playback must never break Blender
            traceback.print_exc()


@persistent
def on_load_post(_dummy=None) -> None:
    _ensure_frame_handler()
    _LOADED.clear()
    props.migrate_all()
    for ob in bpy.data.objects:  # a saved "running" status means the bake was cut off by closing Blender
        if ob.type == "MESH" and ob.mpm.role == "DOMAIN" and ob.mpm.domain.status in RUNNING_STATUSES:
            ob.mpm.domain.status = "interrupted"
            ob.mpm.domain.status_message = "Interrupted when Blender closed: Resume to continue"
    _restore_outputs()


@persistent
def on_undo_post(*_args) -> None:
    """Undo removes output objects created after the undo step; rebuild them from the cache."""
    _restore_outputs()


def _restore_outputs() -> None:
    for scene in bpy.data.scenes:
        for domain_obj in _domains(scene):
            try:
                if _cache_of(domain_obj) is not None:
                    refresh_outputs(scene, domain_obj)
            except Exception:  # noqa: BLE001
                traceback.print_exc()


@persistent
def on_load_pre(_dummy=None) -> None:
    jobs.cancel_all()


def _ensure_frame_handler() -> None:
    if load_frame not in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.append(load_frame)


def _after_bake(scene, domain_obj, cache) -> None:
    refresh_outputs(scene, domain_obj, Path(cache))


def _on_progress(scene, domain_obj, cache) -> None:
    if output_collection(domain_obj) is None:
        refresh_outputs(scene, domain_obj, Path(cache))
    else:
        load_domain_frame(scene, domain_obj, scene.frame_current, Path(cache))


_HANDLERS = [(bpy.app.handlers.load_post, on_load_post), (bpy.app.handlers.load_pre, on_load_pre),
             (bpy.app.handlers.undo_post, on_undo_post), (bpy.app.handlers.redo_post, on_undo_post)]
_HOOKS = [(operators.AFTER_BAKE, _after_bake), (operators.AFTER_SURFACE, _after_bake),
          (operators.AFTER_FREE, remove_outputs),
          (operators.ON_PROGRESS, _on_progress)]


def register():
    _ensure_frame_handler()
    for handlers, fn in _HANDLERS:
        if fn not in handlers:
            handlers.append(fn)
    for hooks, fn in _HOOKS:
        hooks.append(fn)


def unregister():
    for handlers, fn in [(bpy.app.handlers.frame_change_post, load_frame), *_HANDLERS]:
        if fn in handlers:
            handlers.remove(fn)
    for hooks, fn in _HOOKS:
        if fn in hooks:
            hooks.remove(fn)
    jobs.cancel_all()
