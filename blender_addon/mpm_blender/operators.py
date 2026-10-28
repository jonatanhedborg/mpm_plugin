"""Bake / cancel / free operators. Bakes run as subprocess jobs; the UI follows them with a modal timer."""

from __future__ import annotations

import json
import shutil
import tempfile
import uuid
from pathlib import Path

import bpy

from mpm_core.materials import Look, user_preset_id
from mpm_core.scene import scene_hash, write_scene

from mpm_surface.settings import SurfaceSettings, write_settings  # no warp import

from . import jobs, presets
from .export import ExportError, cache_path, export_scene, write_export
from .props import apply_preset, effective_display, emitter_material

# Called as hook(scene, domain_obj, cache_path) after a job finishes successfully. Display and
# surfacing register here, so this module doesn't import them.
AFTER_BAKE: list = []
AFTER_SURFACE: list = []
ON_PROGRESS: list = []  # hook(scene, domain_obj, cache_path) while a bake runs, so new frames show up


def find_domain(context, name: str):
    ob = bpy.data.objects.get(name) if name else context.object
    return ob if ob is not None and ob.type == "MESH" and ob.mpm.role == "DOMAIN" else None


def domain_uid(domain_obj) -> str:
    """A persistent id for the domain (survives renames and file reloads); assigned on first use."""
    d = domain_obj.mpm.domain
    if not d.uid:
        d.uid = uuid.uuid4().hex
    return d.uid


def find_domain_by_uid(uid: str):
    return next((ob for ob in bpy.data.objects if ob.type == "MESH" and ob.mpm.role == "DOMAIN"
                 and ob.mpm.domain.uid == uid), None)


def is_running(domain_obj) -> bool:
    """A job is actually alive for this domain (the saved status string can be stale)."""
    job = jobs.JOBS.get(domain_obj.mpm.domain.uid)
    return job is not None and job.running


def set_status(domain_obj, status: str, message: str = "", progress: float | None = None) -> None:
    d = domain_obj.mpm.domain
    d.status, d.status_message = status, message
    if progress is not None:
        d.progress = progress


def _redraw(context) -> None:
    screen = getattr(context, "screen", None)
    for area in screen.areas if screen else []:
        if area.type in ("PROPERTIES", "VIEW_3D"):
            area.tag_redraw()


class _JobOperator:
    """Shared start / follow / finish logic for bake and surface jobs."""

    wait: bpy.props.BoolProperty(default=False, options={"SKIP_SAVE"},
                                 description="Block until done (for scripts)")
    domain: bpy.props.StringProperty(default="", options={"SKIP_SAVE"},
                                     description="Domain object name; empty = active object")
    running_status = "running"
    hooks: list = []

    def start_job(self, context, domain_obj, job):
        self._uid = domain_uid(domain_obj)
        jobs.JOBS[self._uid] = job
        job.start()
        set_status(domain_obj, self.running_status, "Starting…", 0.0)
        if self.wait:
            job.wait()
            return self.finish(context, domain_obj, job)
        self._timer = context.window_manager.event_timer_add(0.25, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        domain_obj = find_domain_by_uid(self._uid)  # by id: the domain may be renamed meanwhile
        job = jobs.JOBS.get(self._uid)
        if job is None:
            context.window_manager.event_timer_remove(self._timer)
            return {"CANCELLED"}
        if domain_obj is None:  # domain deleted: stop following, let the job finish on its own
            job.cancel()
            context.window_manager.event_timer_remove(self._timer)
            return {"CANCELLED"}
        progress = job.poll()
        if progress and progress.get("frame"):
            start = context.scene.frame_start
            done = progress["frame"] - start + 1
            total = (progress.get("frame_end") or progress["frame"]) - start + 1
            set_status(domain_obj, self.running_status, f"Frame {progress['frame']} / {progress.get('frame_end')}",
                       max(0.0, min(1.0, done / max(total, 1))))
            self.on_progress(context, domain_obj, job)
        _redraw(context)
        if job.running:
            return {"PASS_THROUGH"}
        context.window_manager.event_timer_remove(self._timer)
        job.wait()
        return self.finish(context, domain_obj, job)

    def on_progress(self, context, domain_obj, job):
        pass

    def finish(self, context, domain_obj, job):
        progress = job.poll() or {}
        status = progress.get("status")
        if job.returncode == 0 and status == "done":
            set_status(domain_obj, "done", self.done_message(progress), 1.0)
            for hook in self.hooks:
                hook(context.scene, domain_obj, job.cache_dir)
            return {"FINISHED"}
        if status == "cancelled":
            set_status(domain_obj, "cancelled", f"Cancelled at frame {progress.get('frame')}")
            return {"CANCELLED"}
        message = progress.get("message") or job.log_tail(3) or f"{job.kind} failed (exit code {job.returncode})"
        set_status(domain_obj, "error", message.splitlines()[-1][:300])
        self.report({"ERROR"}, f"MPM {job.kind} failed: {message}\n(log: {job.log_path})")
        return {"CANCELLED"}

    def done_message(self, progress) -> str:
        return "Done"

    def fail(self, domain_obj, message: str):
        if domain_obj is not None:
            set_status(domain_obj, "error", message)
        self.report({"ERROR"}, message)
        return {"CANCELLED"}

    def busy(self, domain_obj) -> bool:
        return is_running(domain_obj)


class MPM_OT_bake(_JobOperator, bpy.types.Operator):
    bl_idname = "mpm.bake"
    bl_label = "Bake"
    bl_description = "Simulate this domain and write the cache"

    resume: bpy.props.BoolProperty(default=False, options={"SKIP_SAVE"},
                                   description="Continue the last bake from where it stopped")
    hooks = AFTER_BAKE

    def execute(self, context):
        domain_obj = find_domain(context, self.domain)
        if domain_obj is None:
            return self.fail(None, "select an MPM domain")
        if self.busy(domain_obj):
            return self.fail(domain_obj, "a bake or surface run is already running for this domain")
        quality = domain_obj.mpm.domain.quality
        d = domain_obj.mpm.domain
        try:
            if self.resume:
                path = cache_path(domain_obj, quality)
                if not (path / "state").exists():
                    raise ExportError("nothing to resume: bake first")
                if _scene_changed(context.scene, domain_obj, quality, path):
                    raise ExportError("the scene changed since this bake started: Bake from scratch "
                                      "(Resume would mix old and new settings)")
            else:
                if not d.cache_dir:  # pin the default folder so renaming the domain doesn't lose the cache
                    path = cache_path(domain_obj, quality)  # raises for an unsaved file
                    d.cache_dir = bpy.path.relpath(str(path.parent)) if bpy.data.filepath else str(path.parent)
                path = write_export(context.scene, domain_obj, quality)
        except ExportError as exc:
            return self.fail(domain_obj, str(exc))
        return self.start_job(context, domain_obj, jobs.bake_job(path, self.resume))

    def on_progress(self, context, domain_obj, job):
        for hook in ON_PROGRESS:
            hook(context.scene, domain_obj, job.cache_dir)

    def finish(self, context, domain_obj, job):
        result = super().finish(context, domain_obj, job)
        if result == {"FINISHED"} and domain_obj.mpm.domain.auto_surface and _surfaced_emitters(job.cache_dir):
            bpy.ops.mpm.bake_surface(wait=self.wait, domain=domain_obj.name)
        return result

    def done_message(self, progress) -> str:
        return f"Baked {progress.get('frame')} frames, {progress.get('particles', 0):,} particles"


def _scene_changed(scene, domain_obj, quality: str, cache: Path) -> bool:
    """Would a fresh export differ from the scene the cached bake was made from?"""
    with tempfile.TemporaryDirectory() as tmp:
        description, _ = export_scene(scene, domain_obj, quality)
        write_scene(description, Path(tmp))
        return scene_hash(Path(tmp)) != scene_hash(cache / "scene")


def _surfaced_emitters(cache) -> list[str]:
    """Emitters in the cache whose object is displayed as a surface."""
    names = json.loads((cache / "cache.json").read_text(encoding="utf-8"))["emitters"]
    return [n for n in names if (ob := bpy.data.objects.get(n)) is not None and effective_display(ob) == "MESH"]


def write_surface_settings(cache) -> None:
    """surface_settings.json from each emitter object's Display & Surface settings."""
    names = json.loads((cache / "cache.json").read_text(encoding="utf-8"))["emitters"]
    meshed = set(_surfaced_emitters(cache))
    settings = {}
    for name in names:
        ob = bpy.data.objects.get(name)
        if ob is None:
            settings[name] = SurfaceSettings(enabled=False)
            continue
        e = ob.mpm.emitter
        settings[name] = SurfaceSettings(enabled=name in meshed, detail=e.surface_detail,
                                         smoothing=e.surface_smoothing, iso=e.surface_iso)
    write_settings(cache, settings)


class MPM_OT_bake_surface(_JobOperator, bpy.types.Operator):
    bl_idname = "mpm.bake_surface"
    bl_label = "Bake Surface"
    bl_description = "Turn the baked particles of emitters displayed as surfaces into meshes"

    running_status = "surfacing"
    hooks = AFTER_SURFACE

    def execute(self, context):
        domain_obj = find_domain(context, self.domain)
        if domain_obj is None:
            return self.fail(None, "select an MPM domain")
        if self.busy(domain_obj):
            return self.fail(domain_obj, "a bake or surface run is already running for this domain")
        try:
            cache = cache_path(domain_obj, domain_obj.mpm.domain.quality)
        except ExportError as exc:
            return self.fail(domain_obj, str(exc))
        if not (cache / "cache.json").exists():
            return self.fail(domain_obj, "nothing to surface: bake first")
        write_surface_settings(cache)
        return self.start_job(context, domain_obj, jobs.surface_job(cache))

    def done_message(self, progress) -> str:
        return f"Surfaced up to frame {progress.get('frame')}"


class MPM_OT_free_surface(bpy.types.Operator):
    bl_idname = "mpm.free_surface"
    bl_label = "Free Surface"
    bl_description = "Delete the surface meshes; emitters show as points again"

    domain: bpy.props.StringProperty(default="", options={"SKIP_SAVE"})

    def execute(self, context):
        domain_obj = find_domain(context, self.domain)
        if domain_obj is None:
            return {"CANCELLED"}
        if is_running(domain_obj):
            self.report({"ERROR"}, "cancel the running job first")
            return {"CANCELLED"}
        try:
            cache = cache_path(domain_obj, domain_obj.mpm.domain.quality)
        except ExportError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        shutil.rmtree(cache / "surface", ignore_errors=True)
        for hook in AFTER_SURFACE:
            hook(context.scene, domain_obj, cache)
        return {"FINISHED"}


class MPM_OT_cancel(bpy.types.Operator):
    bl_idname = "mpm.cancel"
    bl_label = "Cancel"
    bl_description = "Stop the running bake or surface run after the current frame"

    domain: bpy.props.StringProperty(default="", options={"SKIP_SAVE"})

    def execute(self, context):
        domain_obj = find_domain(context, self.domain)
        if domain_obj is None:
            return {"CANCELLED"}
        job = jobs.JOBS.get(domain_obj.mpm.domain.uid)
        if job is None or not job.running:
            # no job in this session (e.g. a bake left running by an earlier session): ask through the files
            try:
                cache = cache_path(domain_obj, domain_obj.mpm.domain.quality)
            except ExportError:
                return {"CANCELLED"}
            if not cache.exists():
                return {"CANCELLED"}
            (cache / "cancel").touch()
            if (cache / "surface").exists():
                (cache / "surface" / "cancel").touch()
            set_status(domain_obj, "cancelled", "Cancel requested")
            return {"FINISHED"}
        job.cancel()
        set_status(domain_obj, domain_obj.mpm.domain.status, "Cancelling…")
        return {"FINISHED"}


class MPM_OT_free_bake(bpy.types.Operator):
    bl_idname = "mpm.free_bake"
    bl_label = "Free Bake"
    bl_description = "Delete this domain's cache for the current quality"

    domain: bpy.props.StringProperty(default="", options={"SKIP_SAVE"})

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        domain_obj = find_domain(context, self.domain)
        if domain_obj is None:
            return {"CANCELLED"}
        if is_running(domain_obj):
            self.report({"ERROR"}, "cancel the running bake first")
            return {"CANCELLED"}
        try:
            path = cache_path(domain_obj, domain_obj.mpm.domain.quality)
        except ExportError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        shutil.rmtree(path, ignore_errors=True)
        for hook in AFTER_FREE:
            hook(context.scene, domain_obj)
        set_status(domain_obj, "", "", 0.0)
        return {"FINISHED"}


AFTER_FREE: list = []  # hook(scene, domain_obj) after a cache was deleted
ON_DISABLE: list = []  # hook(scene, obj, old_role) after MPM was turned off for an object


class MPM_OT_toggle(bpy.types.Operator):
    bl_idname = "mpm.toggle"
    bl_label = "MPM"
    bl_description = "Add or remove MPM simulation (cream, snow, sand and more) for this object"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.object is not None and context.object.type == "MESH"

    def execute(self, context):
        ob = context.object
        settings = ob.mpm
        if not settings.enabled:
            settings.enabled = True
            return {"FINISHED"}
        if settings.role == "DOMAIN" and is_running(ob):
            self.report({"ERROR"}, "Cancel the running bake first")
            return {"CANCELLED"}
        old_role = settings.role
        settings.role = "NONE"  # gives a domain back its display and render visibility
        settings.enabled = False
        for hook in ON_DISABLE:
            hook(context.scene, ob, old_role)
        return {"FINISHED"}


# ---------------------------------------------------------------------------- material presets

def _active_emitter(context):
    ob = context.object
    return ob if ob is not None and ob.type == "MESH" and ob.mpm.role == "EMITTER" else None


def _current_look(ob) -> Look:
    """The look to save with a preset: the output object's render material if baked, else the preset's."""
    e = ob.mpm.emitter
    preset = presets.get(e.preset_id)
    look = preset.look if preset is not None else presets.DEFAULT_LOOK
    display = "mesh" if effective_display(ob) == "MESH" else "points"
    out = next((o for o in bpy.data.objects if o.get("mpm_emitter") == ob.name), None)
    mat = out.data.materials[0] if out is not None and out.data.materials else None
    bsdf = mat.node_tree.nodes.get("Principled BSDF") if mat is not None and mat.node_tree else None
    if bsdf is None:
        return Look(display, look.base_color, look.roughness, look.subsurface)
    return Look(display, tuple(bsdf.inputs["Base Color"].default_value[:3]), bsdf.inputs["Roughness"].default_value,
                bsdf.inputs["Subsurface Weight"].default_value)


class MPM_OT_apply_preset(bpy.types.Operator):
    bl_idname = "mpm.apply_preset"
    bl_label = "Apply Material Preset"
    bl_options = {"REGISTER", "UNDO"}

    preset_id: bpy.props.StringProperty(options={"SKIP_SAVE"})

    @classmethod
    def poll(cls, context):
        return _active_emitter(context) is not None

    @classmethod
    def description(cls, context, properties):
        preset = presets.get(properties.preset_id)
        return (preset.description or preset.label) if preset is not None else ""

    def execute(self, context):
        preset = presets.get(self.preset_id)
        if preset is None:
            self.report({"ERROR"}, f"Unknown preset {self.preset_id!r}")
            return {"CANCELLED"}
        e = _active_emitter(context).mpm.emitter
        e.preset_id = preset.id
        apply_preset(e, preset)  # also when the id didn't change: re-applying resets edited values
        return {"FINISHED"}


class MPM_OT_save_preset(bpy.types.Operator):
    bl_idname = "mpm.save_preset"
    bl_label = "Save Material Preset"
    bl_description = "Save this emitter's material and look as a preset of your own, in every file on this computer"
    bl_options = {"REGISTER", "UNDO"}

    name: bpy.props.StringProperty(name="Name", default="", options={"SKIP_SAVE"})

    @classmethod
    def poll(cls, context):
        return _active_emitter(context) is not None

    def invoke(self, context, event):
        preset = presets.get(_active_emitter(context).mpm.emitter.preset_id)
        if not self.name and preset is not None and preset.is_user:
            self.name = preset.label  # updating your own preset is the common case
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        ob = _active_emitter(context)
        if not self.name.strip():
            self.report({"ERROR"}, "Give the preset a name")
            return {"CANCELLED"}
        try:
            existed = presets.get(user_preset_id(self.name)) is not None
            preset = presets.save(self.name, emitter_material(ob), _current_look(ob))
        except (ValueError, OSError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        ob.mpm.emitter.preset_id = preset.id
        self.report({"INFO"}, f"{'Updated' if existed else 'Saved'} preset {preset.label!r}")
        return {"FINISHED"}


class MPM_OT_delete_preset(bpy.types.Operator):
    bl_idname = "mpm.delete_preset"
    bl_label = "Delete Material Preset"
    bl_description = "Delete this preset of your own. Emitters using it keep their values"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        ob = _active_emitter(context)
        preset = presets.get(ob.mpm.emitter.preset_id) if ob is not None else None
        if preset is None or not preset.is_user:
            cls.poll_message_set("Only your own presets can be deleted")
            return False
        return True

    def invoke(self, context, event):
        label = presets.get(_active_emitter(context).mpm.emitter.preset_id).label
        return context.window_manager.invoke_confirm(self, event, title="Delete Preset",
                                                     message=f"Delete the preset {label!r}?", confirm_text="Delete")

    def execute(self, context):
        e = _active_emitter(context).mpm.emitter
        label = presets.get(e.preset_id).label
        try:
            presets.delete(e.preset_id)
        except (KeyError, OSError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        e.preset_id = ""
        e.preset_label = ""
        self.report({"INFO"}, f"Deleted preset {label!r}")
        return {"FINISHED"}


CLASSES = [MPM_OT_bake, MPM_OT_bake_surface, MPM_OT_cancel, MPM_OT_free_bake, MPM_OT_free_surface, MPM_OT_toggle,
           MPM_OT_apply_preset, MPM_OT_save_preset, MPM_OT_delete_preset]


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
