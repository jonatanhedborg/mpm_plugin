"""Bake / cancel / free operators. Bakes run as subprocess jobs; the UI follows them with a modal timer."""

from __future__ import annotations

import shutil

import bpy

from . import jobs
from .export import ExportError, cache_path, write_export

# Called as hook(scene, domain_obj, cache_path) after a job finishes successfully. Display and
# surfacing register here, so this module doesn't import them.
AFTER_BAKE: list = []
AFTER_SURFACE: list = []


def find_domain(context, name: str):
    ob = bpy.data.objects.get(name) if name else context.object
    return ob if ob is not None and ob.type == "MESH" and ob.mpm.role == "DOMAIN" else None


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
        jobs.JOBS[domain_obj.name] = job
        job.start()
        self._domain_name = domain_obj.name
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
        domain_obj = bpy.data.objects.get(self._domain_name)
        job = jobs.JOBS.get(self._domain_name)
        if domain_obj is None or job is None:
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
        job = jobs.JOBS.get(domain_obj.name)
        return job is not None and job.running


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
        try:
            if self.resume:
                path = cache_path(domain_obj, quality)
                if not (path / "state").exists():
                    raise ExportError("nothing to resume: bake first")
            else:
                path = write_export(context.scene, domain_obj, quality)
        except ExportError as exc:
            return self.fail(domain_obj, str(exc))
        return self.start_job(context, domain_obj, jobs.bake_job(path, self.resume))

    def done_message(self, progress) -> str:
        return f"Baked {progress.get('frame')} frames, {progress.get('particles', 0):,} particles"


class MPM_OT_cancel(bpy.types.Operator):
    bl_idname = "mpm.cancel"
    bl_label = "Cancel"
    bl_description = "Stop the running bake or surface run after the current frame"

    domain: bpy.props.StringProperty(default="", options={"SKIP_SAVE"})

    def execute(self, context):
        domain_obj = find_domain(context, self.domain)
        job = jobs.JOBS.get(domain_obj.name) if domain_obj else None
        if job is None or not job.running:
            return {"CANCELLED"}
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
        job = jobs.JOBS.get(domain_obj.name)
        if job is not None and job.running:
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

CLASSES = [MPM_OT_bake, MPM_OT_cancel, MPM_OT_free_bake]


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
