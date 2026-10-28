import os
import sys
import time

import bpy
import pytest
from conftest import add_cube

from mpm_core.cache import CacheReader


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
    emitter.mpm.emitter.preset = "jelly"
    return domain, emitter


def cache_of(domain):
    from mpm_blender.export import cache_path

    return cache_path(domain, domain.mpm.domain.quality)


def test_bake_wait_writes_cache(clean_scene, tmp_path):
    domain, _ = small_scene(clean_scene, tmp_path)
    assert bpy.ops.mpm.bake(wait=True, domain="Domain") == {"FINISHED"}
    assert CacheReader(cache_of(domain)).available_frames() == [1, 2, 3, 4]
    assert domain.mpm.domain.status == "done"


def test_bake_error_is_reported(clean_scene, tmp_path):
    domain, emitter = small_scene(clean_scene, tmp_path)
    emitter.mpm.role = "NONE"
    with pytest.raises(RuntimeError, match="no emitters"):  # error reports raise when called from a script
        bpy.ops.mpm.bake(wait=True, domain="Domain")
    assert domain.mpm.domain.status == "error" and "no emitters" in domain.mpm.domain.status_message


def test_resume(clean_scene, tmp_path):
    from mpm_blender import jobs
    from mpm_blender.export import write_export

    domain, _ = small_scene(clean_scene, tmp_path, frames=60, resolution=32)
    path = write_export(clean_scene, domain, "FINAL")
    job = jobs.bake_job(path, resume=False)
    job.start()
    deadline = time.time() + 120
    while time.time() < deadline:
        progress = job.poll()
        if progress and (progress.get("frame") or 0) >= 2:
            break
        time.sleep(0.05)
    job.cancel()
    job.wait()
    written = CacheReader(path).available_frames()
    assert 2 <= len(written) < 60
    assert bpy.ops.mpm.bake(wait=True, resume=True, domain="Domain") == {"FINISHED"}
    assert CacheReader(path).available_frames() == list(range(1, 61))


def test_free_bake(clean_scene, tmp_path):
    domain, _ = small_scene(clean_scene, tmp_path)
    bpy.ops.mpm.bake(wait=True, domain="Domain")
    assert cache_of(domain).exists()
    assert bpy.ops.mpm.free_bake(domain="Domain") == {"FINISHED"}
    assert not cache_of(domain).exists()


def test_job_env_has_blender_sys_path(tmp_path):
    from mpm_blender import jobs

    job = jobs.bake_job(tmp_path, resume=False)
    entries = job.env["PYTHONPATH"].split(os.pathsep)
    assert all(p in entries for p in sys.path if p and os.path.exists(p))
    assert job.argv[:3] == [sys.executable, "-m", "mpm_core.bake"]


def test_bake_needs_a_domain(clean_scene):
    with pytest.raises(RuntimeError, match="select an MPM domain"):
        bpy.ops.mpm.bake(wait=True, domain="Nope")
