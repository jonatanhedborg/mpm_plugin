"""Blender scene -> mpm_core SceneDescription."""

from __future__ import annotations

import re
from pathlib import Path

import bpy
import numpy as np

from mpm_core.geometry import mesh_is_closed
from mpm_core.scene import Collider, Domain, Emitter, MeshData, SceneDescription, write_scene

from .props import emitter_material


class ExportError(Exception):
    """Shown to the user as is."""


def cache_root(domain_obj) -> Path:
    raw = domain_obj.mpm.domain.cache_dir or f"//mpm_cache/{re.sub(r'[^A-Za-z0-9_.-]+', '_', domain_obj.name)}"
    if raw.startswith("//") and not bpy.data.filepath:
        raise ExportError("save the .blend file first, or set an absolute cache folder on the domain")
    return Path(bpy.path.abspath(raw))


def cache_path(domain_obj, quality: str) -> Path:
    return cache_root(domain_obj) / quality.lower()


def _domains(scene):
    return [ob for ob in scene.objects if ob.type == "MESH" and ob.mpm.role == "DOMAIN"]


def _belongs(settings, domain_obj, only_domain: bool) -> bool:
    return settings.domain == domain_obj or (settings.domain is None and only_domain)


def members(scene, domain_obj) -> tuple[list, list]:
    """(emitters, colliders) of this domain, sorted by name; the order defines emitter indices."""
    only = len(_domains(scene)) == 1
    emitters = [ob for ob in scene.objects if ob.type == "MESH" and ob.mpm.role == "EMITTER"
                and _belongs(ob.mpm.emitter, domain_obj, only)]
    colliders = [ob for ob in scene.objects if ob.type == "MESH" and ob.mpm.role == "COLLIDER"
                 and _belongs(ob.mpm.collider, domain_obj, only)]
    return sorted(emitters, key=lambda o: o.name), sorted(colliders, key=lambda o: o.name)


def _domain_box(domain_obj) -> tuple[np.ndarray, np.ndarray]:
    _, rotation, _ = domain_obj.matrix_world.decompose()
    if rotation.angle > 1e-4:
        raise ExportError(f"domain {domain_obj.name!r} is rotated; MPM domains must be axis-aligned "
                          f"(apply or clear its rotation)")
    corners = np.array([domain_obj.matrix_world @ v.co for v in domain_obj.data.vertices]
                       if len(domain_obj.data.vertices) else [domain_obj.matrix_world.translation])
    return corners.min(0), corners.max(0)


def _local_mesh(ob, depsgraph) -> MeshData:
    """Evaluated (modifiers applied), triangulated mesh in object local space."""
    ob_eval = ob.evaluated_get(depsgraph)
    mesh = ob_eval.to_mesh()
    try:
        mesh.calc_loop_triangles()
        verts = np.empty(len(mesh.vertices) * 3, np.float32)
        mesh.vertices.foreach_get("co", verts)
        tris = np.empty(len(mesh.loop_triangles) * 3, np.int32)
        mesh.loop_triangles.foreach_get("vertices", tris)
        return MeshData(verts.reshape(-1, 3), tris.reshape(-1, 3))
    finally:
        ob_eval.to_mesh_clear()


def _sample_transforms(scene, objects) -> list[np.ndarray]:
    """World matrices per frame (frame_start..frame_end) for each object; restores the current frame."""
    current = scene.frame_current
    frames = range(scene.frame_start, scene.frame_end + 1)
    out = [np.empty((len(frames), 4, 4), np.float32) for _ in objects]
    try:
        for k, frame in enumerate(frames):
            scene.frame_set(frame)
            for i, ob in enumerate(objects):
                out[i][k] = np.array(ob.matrix_world, np.float32)
    finally:
        scene.frame_set(current)
    return out


def export_scene(scene, domain_obj, quality: str) -> tuple[SceneDescription, list[str]]:
    d = domain_obj.mpm.domain
    bpy.context.view_layer.update()  # matrix_world is stale after property edits until the depsgraph updates
    emitters, colliders = members(scene, domain_obj)
    if not emitters:
        raise ExportError(f"domain {domain_obj.name!r} has no emitters")
    lo, hi = _domain_box(domain_obj)
    resolution = d.resolution if quality == "FINAL" else max(8, round(d.resolution / 2))
    walls = {f"{axis}{sign}": getattr(d, f"wall_{axis}_{name}")
             for axis in "xyz" for sign, name in (("-", "neg"), ("+", "pos"))}

    current = scene.frame_current
    scene.frame_set(scene.frame_start)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    emitter_meshes = [_local_mesh(ob, depsgraph) for ob in emitters]
    collider_meshes = [_local_mesh(ob, depsgraph) for ob in colliders]
    scene.frame_set(current)
    transforms = _sample_transforms(scene, emitters + colliders)

    scene_emitters = []
    for i, (ob, mesh) in enumerate(zip(emitters, emitter_meshes)):
        e = ob.mpm.emitter
        inflow = e.kind == "INFLOW"
        scene_emitters.append(Emitter(
            name=ob.name, kind="inflow" if inflow else "fill", material=i, mesh=mesh, transforms=transforms[i],
            velocity=tuple(e.velocity), use_object_velocity=e.use_object_velocity,
            frame_start=e.inflow_start if inflow else scene.frame_start,
            frame_end=e.inflow_end if inflow else scene.frame_start))
    scene_colliders = [
        Collider(name=ob.name, mesh=mesh, transforms=transforms[len(emitters) + i],
                 friction=ob.mpm.collider.friction, closed=mesh_is_closed(mesh.tris))
        for i, (ob, mesh) in enumerate(zip(colliders, collider_meshes))]

    description = SceneDescription(
        domain=Domain(tuple(lo), tuple(hi), resolution, walls, d.wall_friction),
        materials=[emitter_material(ob) for ob in emitters],
        emitters=scene_emitters,
        colliders=scene_colliders,
        gravity=tuple(scene.gravity) if scene.use_gravity else (0.0, 0.0, 0.0),
        fps=scene.render.fps / scene.render.fps_base,
        frame_start=scene.frame_start,
        frame_end=scene.frame_end,
        particles_per_cell=int(d.particles_per_cell),
    )
    return description, [ob.name for ob in emitters]


def write_export(scene, domain_obj, quality: str) -> Path:
    """Export and write the scene description to <cache>/<quality>/scene; returns the cache path."""
    description, _ = export_scene(scene, domain_obj, quality)
    path = cache_path(domain_obj, quality)
    write_scene(description, path / "scene")
    return path
