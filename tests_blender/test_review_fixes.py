"""Regression tests for the add-on review findings (C1, I1-I7)."""

from types import SimpleNamespace

import bpy
import pytest
from conftest import cache_of, small_scene

from mpm_core.cache import CacheReader


@pytest.fixture
def baked(clean_scene, tmp_path):
    domain, emitter = small_scene(clean_scene, tmp_path)
    bpy.ops.mpm.bake(wait=True, domain="Domain")
    return clean_scene, domain, emitter


class RecordingLayout:
    def __init__(self):
        self.operators = []
        self.labels = []

    def operator(self, idname, **kwargs):
        self.operators.append(idname)
        return self

    def label(self, text="", **kwargs):
        self.labels.append(text)
        return self

    def __getattr__(self, name):
        return lambda *args, **kwargs: self


def test_stale_running_status_reset_on_load(baked):  # C1
    from mpm_blender import display, operators

    _, domain, _ = baked
    domain.mpm.domain.status = "running"  # as saved in a .blend mid-bake
    assert not operators.is_running(domain)
    display.on_load_post()
    assert domain.mpm.domain.status == "interrupted"


def test_panel_offers_bake_when_no_job_runs(baked):  # C1
    from mpm_blender import ui

    _, domain, _ = baked
    domain.mpm.domain.status = "running"
    layout = RecordingLayout()
    ui.MPM_PT_domain_bake.draw(SimpleNamespace(layout=layout), SimpleNamespace(object=domain, scene=bpy.context.scene))
    assert "mpm.bake" in layout.operators


def test_panel_warns_without_lock_interface(baked):
    from mpm_blender import ui

    scene, domain, _ = baked
    scene.render.use_lock_interface = False
    layout = RecordingLayout()
    ui.MPM_PT_domain_bake.draw(SimpleNamespace(layout=layout), SimpleNamespace(object=domain, scene=scene))
    assert any("Lock Interface" in text for text in layout.labels)


def test_cancel_without_job_writes_cancel_file(baked):  # I1: orphan from an earlier session
    _, domain, _ = baked
    assert bpy.ops.mpm.cancel(domain="Domain") == {"FINISHED"}
    assert (cache_of(domain) / "cancel").exists()


def test_rename_domain_after_bake(clean_scene, tmp_path):  # I2
    from mpm_blender import display

    domain, _ = small_scene(clean_scene, tmp_path)
    bpy.ops.wm.save_as_mainfile(filepath=str(tmp_path / "scene.blend"))
    try:
        domain = bpy.data.objects["Domain"]
        domain.mpm.domain.cache_dir = ""  # default, relative to the .blend
        bpy.ops.mpm.bake(wait=True, domain="Domain")
        cache = cache_of(domain)
        domain.name = "Renamed"
        assert cache_of(domain) == cache  # the cache folder doesn't move with the name
        scene = bpy.context.scene
        scene.frame_set(3)
        coll = display.output_collection(domain)  # found by id although the name no longer matches
        assert coll is not None
        out = coll.objects["MPM Emitter"]
        assert len(out.data.vertices) == len(CacheReader(cache).read_frame(3)["Emitter"][0])
        assert bpy.ops.mpm.bake(wait=True, domain="Renamed") == {"FINISHED"}
        assert display.output_collection(domain).name == "MPM Renamed"  # follows on the next bake
    finally:
        bpy.ops.wm.read_homefile(use_empty=True)


def test_same_frame_not_reloaded(baked, monkeypatch):  # I3
    from mpm_blender import display

    scene, domain, _ = baked
    scene.frame_set(3)
    reads = []
    real = display._read_npz
    monkeypatch.setattr(display, "_read_npz", lambda path: reads.append(path) or real(path))
    display.load_domain_frame(scene, domain, 3)
    assert reads == []
    scene.frame_set(2)
    assert len(reads) >= 1


def test_export_does_not_load_cache(baked, monkeypatch):  # I4
    from mpm_blender import display
    from mpm_blender.export import export_scene

    scene, domain, _ = baked
    reads = []
    monkeypatch.setattr(display, "_read_npz", lambda path: reads.append(path))
    export_scene(scene, domain, "FINAL")
    assert reads == []


def test_quality_switch_refreshes_outputs(baked):  # I5
    from mpm_blender import display

    _, domain, _ = baked
    domain.mpm.domain.quality = "PREVIEW"  # nothing baked at preview quality
    assert display.output_collection(domain) is None
    domain.mpm.domain.quality = "FINAL"
    assert display.output_collection(domain) is not None


def test_undo_restores_outputs(baked):  # I6
    from mpm_blender import display

    scene, domain, _ = baked
    coll = display.output_collection(domain)
    for ob in list(coll.objects):
        bpy.data.objects.remove(ob)
    bpy.data.collections.remove(coll)  # what undo does to objects created after the undo step
    display.on_undo_post(scene)
    assert display.output_collection(domain) is not None


def test_resume_refuses_changed_scene(baked):  # I7
    _, domain, emitter = baked
    emitter.mpm.emitter.youngs_modulus *= 2
    with pytest.raises(RuntimeError, match="changed"):
        bpy.ops.mpm.bake(wait=True, resume=True, domain="Domain")
    emitter.mpm.emitter.youngs_modulus /= 2
    assert bpy.ops.mpm.bake(wait=True, resume=True, domain="Domain") == {"FINISHED"}
