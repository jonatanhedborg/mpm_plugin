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


@pytest.mark.gpu
def test_meshes_without_particle_frames_are_pruned(particle_cache):  # surfacer review finding 1
    surface(particle_cache)
    frame_path(particle_cache, 4).unlink()  # e.g. the frame range got shorter
    surface(particle_cache)
    assert MeshCacheReader(particle_cache).available_frames("emitter_0") == [1, 2, 3]


@pytest.mark.gpu
def test_awkward_emitter_names(tmp_path):  # surfacer review finding 2: names are not paths
    names = ["cancel", "Cream", "cream", "a/b", "progress.json"]
    write_scene(small_scene(frame_end=1), tmp_path / "scene")
    p, ids = [], []
    for i in range(len(names)):
        s = sphere_particles((0.15 + 0.17 * i, 0.5, 0.5), 0.06, DX)
        p.append(s)
        ids.append(np.full(len(s), i, np.int32))
    p = np.concatenate(p)
    CacheWriter(tmp_path, names, 24, 1, 1, DX, "test").write_frame(
        1, ParticleData(p, np.zeros_like(p), np.zeros(len(p), np.int32), np.concatenate(ids)))
    assert surface(tmp_path).status == "done"
    r = MeshCacheReader(tmp_path)
    assert r.emitters() == names
    for i, name in enumerate(names):
        assert r.read(name, 1).verts[:, 0].mean() == pytest.approx(0.15 + 0.17 * i, abs=0.01)


@pytest.mark.gpu
def test_second_concurrent_run_refused(particle_cache):  # surfacer review finding 5
    from mpm_core.cache import exclusive_lock

    (particle_cache / "surface").mkdir(exist_ok=True)
    with exclusive_lock(particle_cache / "surface" / "lock"):
        with pytest.raises(RuntimeError, match="already running"):
            surface(particle_cache)


@pytest.mark.gpu
def test_particle_frame_vanishing_mid_run(particle_cache):  # surfacer review finding 5: bake restarted
    calls = iter(range(100))

    def bake_restarts():
        if next(calls) == 0:
            frame_path(particle_cache, 2).unlink()
        return False

    res = surface(particle_cache, should_cancel=bake_restarts)
    assert res.status == "done" and res.frames_written == 3
    assert any("frame 2" in w for w in progress(particle_cache)["warnings"])
