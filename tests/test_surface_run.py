import json
import subprocess
import sys

import numpy as np
import pytest
from helpers import small_scene, sphere_particles

from mpm_core.cache import CacheWriter, frame_path
from mpm_core.geometry import mesh_is_closed
from mpm_core.scene import write_scene
from mpm_core.solver import ParticleData
from mpm_surface.meshcache import MeshCacheReader
from mpm_surface.run import surface
from mpm_surface.settings import SurfaceSettings

DX = 1 / 32


def write_particle_frame(writer, frame, empty=False):
    p = sphere_particles((0.4 + 0.01 * (frame - 1), 0.5, 0.5), 0.1, DX)
    if empty:
        p = p[:0]
    n = len(p)
    writer.write_frame(frame, ParticleData(p, np.tile([0.24, 0, 0], (n, 1)).astype(np.float32),
                                           np.zeros(n, np.int32), np.zeros(n, np.int32)))


@pytest.fixture
def particle_cache(tmp_path):
    """4-frame particle cache for emitter_0: a sphere moving +x 1 cm per frame. No solver involved."""
    write_scene(small_scene(frame_end=4), tmp_path / "scene")
    writer = CacheWriter(tmp_path, ["emitter_0"], 24, 1, 4, DX, "test")
    for f in range(1, 5):
        write_particle_frame(writer, f)
    return tmp_path


def progress(cache):
    return json.loads((cache / "surface" / "progress.json").read_text())


@pytest.mark.gpu
def test_meshes_all_frames(particle_cache):
    res = surface(particle_cache)
    assert res.status == "done" and res.frames_written == 4
    r = MeshCacheReader(particle_cache)
    assert r.available_frames("emitter_0") == [1, 2, 3, 4]
    assert all(mesh_is_closed(r.read("emitter_0", f).tris) for f in range(1, 5))
    assert progress(particle_cache)["status"] == "done"


@pytest.mark.gpu
def test_second_run_skips(particle_cache):
    surface(particle_cache)
    res = surface(particle_cache)
    assert (res.frames_written, res.frames_skipped) == (0, 4)


@pytest.mark.gpu
def test_changed_settings_remesh(particle_cache):
    surface(particle_cache)
    assert surface(particle_cache, settings={"emitter_0": SurfaceSettings(detail=2)}).frames_written == 4


@pytest.mark.gpu
def test_disabled_emitter_removed(particle_cache):
    surface(particle_cache)
    surface(particle_cache, settings={"emitter_0": SurfaceSettings(enabled=False)})
    assert MeshCacheReader(particle_cache).emitters() == []


@pytest.mark.gpu
def test_unknown_emitter_warns(particle_cache):  # Review Focus
    res = surface(particle_cache, settings={"nope": SurfaceSettings()})
    assert res.frames_written == 4
    assert "surface settings for unknown emitter nope ignored" in progress(particle_cache)["warnings"]


@pytest.mark.gpu
def test_partial_cache_then_rest(particle_cache):  # Review Focus: surfacing while the bake still runs
    saved = {f: frame_path(particle_cache, f).read_bytes() for f in (3, 4)}
    for f in saved:
        frame_path(particle_cache, f).unlink()
    assert surface(particle_cache).frames_written == 2
    for f, data in saved.items():
        frame_path(particle_cache, f).write_bytes(data)
    res = surface(particle_cache)
    assert (res.frames_written, res.frames_skipped) == (2, 2)


@pytest.mark.gpu
def test_empty_frame(particle_cache):  # Review Focus
    write_particle_frame(CacheWriter(particle_cache, ["emitter_0"], 24, 1, 4, DX, "test"), 2, empty=True)
    assert surface(particle_cache).status == "done"
    assert MeshCacheReader(particle_cache).read("emitter_0", 2).tris.shape == (0, 3)


@pytest.mark.gpu
def test_cancel(particle_cache):
    res = surface(particle_cache, should_cancel=lambda: True)
    assert res.status == "cancelled" and res.frames_written == 1
    assert progress(particle_cache)["status"] == "cancelled"


@pytest.mark.gpu
def test_cli(particle_cache):
    out = subprocess.run([sys.executable, "-m", "mpm_surface", str(particle_cache)], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert MeshCacheReader(particle_cache).available_frames("emitter_0") == [1, 2, 3, 4]
