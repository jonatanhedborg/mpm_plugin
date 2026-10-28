import json
import subprocess
import sys

import numpy as np
import pytest
from helpers import scale, small_scene, translate

from mpm_core.backends import register_backend
from mpm_core.bake import SceneChangedError, bake
from mpm_core.cache import CacheReader
from mpm_core.scene import write_scene
from mpm_core.solver import AdvanceStats, ParticleData, Solver


class FakeSolver(Solver):
    """Ballistic particles, no grid. Optionally produces NaN after a number of advances."""

    def __init__(self, nan_at_advance=None, **kwargs):
        super().__init__(**kwargs)
        self.nan_at_advance = nan_at_advance
        self.advances = 0
        self.state = {"x": np.zeros((0, 3), np.float32), "v": np.zeros((0, 3), np.float32),
                      "mat": np.zeros(0, np.int32), "emitter": np.zeros(0, np.int32)}

    def add_particles(self, positions, velocities, material_ids, emitter_ids):
        for k, a in zip(("x", "v", "mat", "emitter"), (positions, velocities, material_ids, emitter_ids)):
            self.state[k] = np.concatenate([self.state[k], np.asarray(a, self.state[k].dtype)])

    def set_colliders(self, colliders):
        pass

    def set_collider_transforms(self, start, end):
        pass

    def advance(self, frame_dt):
        self.advances += 1
        self.state["v"] = self.state["v"] + np.float32(frame_dt) * np.asarray(self.gravity, np.float32)
        self.state["x"] = self.state["x"] + np.float32(frame_dt) * self.state["v"]
        if self.advances == self.nan_at_advance:
            self.state["x"][:] = np.nan
        return AdvanceStats(1, frame_dt, 0)

    @property
    def particle_count(self):
        return len(self.state["x"])

    def get_particles(self):
        s = self.state
        return ParticleData(s["x"].copy(), s["v"].copy(), s["mat"].copy(), s["emitter"].copy())

    def is_occupied(self, points):
        return np.zeros(len(points), bool)

    def save_state(self):
        return {k: v.copy() for k, v in self.state.items()}

    def load_state(self, state):
        self.state = {k: np.asarray(v) for k, v in state.items()}


register_backend("fake", lambda **kw: FakeSolver(**kw))
register_backend("fake_nan4", lambda **kw: FakeSolver(nan_at_advance=3, **kw))  # frame 4 = 3rd advance


def make_cache_dir(path, scene):
    write_scene(scene, path / "scene")


def progress(path):
    return json.loads((path / "progress.json").read_text())


def test_bake_writes_all_frames(tmp_path):
    make_cache_dir(tmp_path, small_scene(frame_end=6))
    res = bake(tmp_path, backend="fake")
    assert res.status == "done" and CacheReader(tmp_path).available_frames() == [1, 2, 3, 4, 5, 6]
    assert progress(tmp_path)["status"] == "done"


def test_nan_stops(tmp_path):
    make_cache_dir(tmp_path, small_scene(frame_end=6))
    res = bake(tmp_path, backend="fake_nan4")
    assert res.status == "error" and "frame 4" in res.message
    assert CacheReader(tmp_path).available_frames() == [1, 2, 3]
    assert progress(tmp_path)["status"] == "error"


def test_cancel_and_resume(tmp_path):
    make_cache_dir(tmp_path, small_scene(frame_end=8))
    calls = iter(range(100))
    res = bake(tmp_path, backend="fake", should_cancel=lambda: next(calls) >= 2)
    assert res.status == "cancelled"
    n = len(CacheReader(tmp_path).available_frames())
    assert bake(tmp_path, backend="fake", resume=True).status == "done"
    assert CacheReader(tmp_path).available_frames() == list(range(1, 9)) and n < 8


def test_resume_continues_simulation(tmp_path):
    make_cache_dir(tmp_path / "a", small_scene(frame_end=6))
    make_cache_dir(tmp_path / "b", small_scene(frame_end=6))
    bake(tmp_path / "a", backend="fake")
    calls = iter(range(100))
    bake(tmp_path / "b", backend="fake", should_cancel=lambda: next(calls) >= 2)
    bake(tmp_path / "b", backend="fake", resume=True)
    pa = CacheReader(tmp_path / "a").read_frame(6)["emitter_0"][0]
    pb = CacheReader(tmp_path / "b").read_frame(6)["emitter_0"][0]
    assert np.allclose(pa, pb)


def test_cancel_file(tmp_path):
    make_cache_dir(tmp_path, small_scene(frame_end=6))
    calls = iter(range(100))

    def add_on_requests_cancel():  # the add-on creates the file while the bake runs
        if next(calls) == 1:
            (tmp_path / "cancel").touch()
        return False

    res = bake(tmp_path, backend="fake", should_cancel=add_on_requests_cancel)
    assert res.status == "cancelled" and res.frames_written == 3  # seen after the next frame
    assert not (tmp_path / "cancel").exists()
    assert progress(tmp_path)["status"] == "cancelled"


def test_stale_cancel_file_ignored(tmp_path):  # review finding 3
    make_cache_dir(tmp_path, small_scene(frame_end=4))
    (tmp_path / "cancel").touch()  # left over from an earlier bake
    assert bake(tmp_path, backend="fake").status == "done"


def test_progress_reset_at_start(tmp_path):  # review finding 4
    make_cache_dir(tmp_path, small_scene(frame_end=3))
    bake(tmp_path, backend="fake")
    seen = []

    def factory(**kw):
        seen.append(progress(tmp_path)["status"])  # what the add-on sees while the solver starts up
        return FakeSolver(**kw)

    register_backend("fake_probe", factory)
    bake(tmp_path, backend="fake_probe")
    assert seen == ["starting"]


def test_unreadable_scene_reports_error(tmp_path):  # review finding 4
    make_cache_dir(tmp_path, small_scene(frame_end=3))
    (tmp_path / "scene" / "scene.json").write_text("{ not json")
    with pytest.raises(ValueError):
        bake(tmp_path, backend="fake")
    assert progress(tmp_path)["status"] == "error"


def test_open_inflow_mesh_fails_before_baking(tmp_path):  # review finding 5
    scene = small_scene(kind="inflow", emit_frames=(3, 4), frame_end=5)
    scene.emitters[0].mesh.tris = scene.emitters[0].mesh.tris[:-2]
    make_cache_dir(tmp_path, scene)
    with pytest.raises(ValueError, match="emitter_0.*not closed"):
        bake(tmp_path, backend="fake")
    assert not list((tmp_path / "frames").glob("*.npz"))


def test_emitter_smaller_than_spacing_warns(tmp_path):  # review finding 6
    scene = small_scene(frame_end=2)
    scene.emitters[0].transforms[:] = translate(0.5, 0.5, 0.5) @ scale(0.002)
    make_cache_dir(tmp_path, scene)
    bake(tmp_path, backend="fake")
    assert any("emitter_0" in w and "no particles" in w for w in progress(tmp_path)["warnings"])


def test_fill_object_velocity(tmp_path):  # review finding 14
    scene = small_scene(frame_end=3)
    e = scene.emitters[0]
    e.use_object_velocity = True
    e.transforms = np.stack([translate(0.1 * f, 0, 0) @ e.transforms[0] for f in range(3)])  # 2.4 m/s in x
    make_cache_dir(tmp_path, scene)
    bake(tmp_path, backend="fake")
    vel = CacheReader(tmp_path).read_frame(1)["emitter_0"][1]
    assert np.allclose(vel[:, 0], 2.4, atol=1e-3)


def test_resume_refuses_changed_scene(tmp_path):  # Review Focus
    make_cache_dir(tmp_path, small_scene(frame_end=8))
    bake(tmp_path, backend="fake", should_cancel=lambda: True)
    write_scene(small_scene(frame_end=8, gravity=(0, 0, -5)), tmp_path / "scene")
    with pytest.raises(SceneChangedError):
        bake(tmp_path, backend="fake", resume=True)


def test_emitter_outside_domain_warns(tmp_path):  # Review Focus
    make_cache_dir(tmp_path, small_scene(emitter_offset=(0.5, 0, 0)))  # cube x 0.9..1.1, half outside x+
    bake(tmp_path, backend="fake")
    warnings = progress(tmp_path)["warnings"]
    assert any("outside the domain were dropped" in w for w in warnings)
    pos = CacheReader(tmp_path).read_frame(1)["emitter_0"][0]
    assert len(pos) > 0 and (pos[:, 0] <= 1.0).all()


def test_empty_scene(tmp_path):
    scene = small_scene(frame_end=3)
    scene.emitters = []
    make_cache_dir(tmp_path, scene)
    assert bake(tmp_path, backend="fake").status == "done"
    r = CacheReader(tmp_path)
    assert r.available_frames() == [1, 2, 3] and r.read_frame(2) == {}


@pytest.mark.gpu
def test_inflow_emits_while_active(tmp_path):
    make_cache_dir(tmp_path, small_scene(kind="inflow", velocity=(0, 0, -1), emit_frames=(1, 10), frame_end=20))
    assert bake(tmp_path).status == "done"
    r = CacheReader(tmp_path)
    counts = [len(r.read_frame(f)["emitter_0"][0]) for f in range(1, 21)]
    assert counts[9] > counts[1] and counts[10:] == [counts[10]] * 10


@pytest.mark.gpu
def test_cli(tmp_path):
    make_cache_dir(tmp_path, small_scene(frame_end=3))
    out = subprocess.run([sys.executable, "-m", "mpm_core.bake", str(tmp_path)], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert CacheReader(tmp_path).available_frames() == [1, 2, 3]


def test_fresh_bake_clears_old_meshes(tmp_path):  # surfacer review finding 1
    make_cache_dir(tmp_path, small_scene(frame_end=3))
    (tmp_path / "surface" / "meshes").mkdir(parents=True)
    (tmp_path / "surface" / "meshes" / "stale.npz").write_bytes(b"old")
    bake(tmp_path, backend="fake")
    assert not (tmp_path / "surface").exists()


def test_resume_keeps_meshes(tmp_path):
    make_cache_dir(tmp_path, small_scene(frame_end=4))
    bake(tmp_path, backend="fake", should_cancel=lambda: True)
    (tmp_path / "surface").mkdir()
    (tmp_path / "surface" / "keep").write_bytes(b"x")
    bake(tmp_path, backend="fake", resume=True)
    assert (tmp_path / "surface" / "keep").exists()
