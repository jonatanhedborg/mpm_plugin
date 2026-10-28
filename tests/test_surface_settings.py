import numpy as np
import pytest

from mpm_surface.meshcache import MeshCacheReader, MeshCacheWriter, SurfaceMesh
from mpm_surface.settings import SurfaceSettings, read_settings, write_settings


def test_defaults_and_roundtrip():
    s = SurfaceSettings()
    assert (s.enabled, s.detail, s.smoothing, s.iso) == (True, 1.0, 2.0, 0.5)
    assert SurfaceSettings.from_dict(s.to_dict()) == s


def test_hash_changes():
    assert SurfaceSettings(detail=2).hash() != SurfaceSettings().hash()
    assert SurfaceSettings().hash() == SurfaceSettings().hash()


@pytest.mark.parametrize("kw", [{"detail": 0}, {"smoothing": -1}, {"iso": 1.0}])
def test_invalid(kw):
    with pytest.raises(ValueError):
        SurfaceSettings(**kw)


def test_settings_file_roundtrip(tmp_path):
    assert read_settings(tmp_path) == {}
    settings = {"cream": SurfaceSettings(detail=2), "sand": SurfaceSettings(enabled=False)}
    write_settings(tmp_path, settings)
    assert read_settings(tmp_path) == settings


def _triangle():
    return SurfaceMesh(np.eye(3, dtype=np.float32), np.array([[0, 1, 2]], np.int32), np.zeros((3, 3), np.float32))


def test_mesh_cache_roundtrip(tmp_path):
    MeshCacheWriter(tmp_path).write("cream", 3, _triangle(), "k")
    r = MeshCacheReader(tmp_path)
    assert r.emitters() == ["cream"] and r.available_frames("cream") == [3]
    m = r.read("cream", 3)
    assert np.array_equal(m.verts, _triangle().verts) and np.array_equal(m.tris, [[0, 1, 2]])
    assert MeshCacheWriter(tmp_path).key("cream", 3) == "k"
    assert MeshCacheWriter(tmp_path).key("cream", 4) is None


def test_remove_emitter(tmp_path):
    w = MeshCacheWriter(tmp_path)
    w.write("cream", 1, _triangle(), "k")
    w.remove_emitter("cream")
    assert MeshCacheReader(tmp_path).emitters() == []


def test_empty_mesh(tmp_path):
    e = SurfaceMesh.empty()
    assert e.verts.shape == (0, 3) and e.tris.shape == (0, 3) and e.vel.shape == (0, 3)
    MeshCacheWriter(tmp_path).write("cream", 1, e, "k")
    assert MeshCacheReader(tmp_path).read("cream", 1).tris.shape == (0, 3)
