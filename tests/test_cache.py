import json
import sys
import threading

import numpy as np
import pytest

from mpm_core.cache import CacheReader, CacheWriter, read_checkpoint, write_checkpoint, write_json
from mpm_core.solver import ParticleData


def _particles():
    return ParticleData(np.random.rand(5, 3).astype(np.float32), np.zeros((5, 3), np.float32),
                        np.zeros(5, np.int32), np.array([0, 1, 1, 0, 1], np.int32))


def test_write_read_frame(tmp_path):
    w = CacheWriter(tmp_path, ["a", "b"], 24, 1, 10, 0.01, "warp")
    w.write_frame(3, _particles())
    r = CacheReader(tmp_path)
    assert r.available_frames() == [3] and r.emitter_names == ["a", "b"]
    f = r.read_frame(3)
    assert f["a"][0].shape == (2, 3) and f["b"][0].shape == (3, 3)


def test_no_tmp_left(tmp_path):
    CacheWriter(tmp_path, ["a", "b"], 24, 1, 10, 0.01, "warp").write_frame(1, _particles())
    assert list((tmp_path / "frames").glob("*.tmp")) == []


def test_checkpoint_roundtrip(tmp_path):
    state = {"x": np.arange(6, dtype=np.float32).reshape(2, 3), "mat": np.array([0, 1], np.int32)}
    write_checkpoint(tmp_path, 5, state, "abc")
    frame, back, h = read_checkpoint(tmp_path)
    assert frame == 5 and h == "abc"
    assert set(back) == set(state) and all(np.array_equal(back[k], state[k]) for k in state)


def test_no_checkpoint(tmp_path):
    assert read_checkpoint(tmp_path) is None


@pytest.mark.skipif(sys.platform != "win32", reason="Windows file locking")
def test_atomic_write_survives_reader_holding_file(tmp_path):
    # the add-on polls progress.json; on Windows an open handle blocks os.replace
    path = tmp_path / "progress.json"
    write_json(path, {"status": "running"})
    reader = open(path, "rb")
    threading.Timer(0.05, reader.close).start()
    write_json(path, {"status": "done"})
    assert json.loads(path.read_text())["status"] == "done"


def test_concurrent_writes_to_one_path(tmp_path):  # surfacer review finding 5: unique tmp names
    path = tmp_path / "progress.json"
    errors = []

    def writer(n):
        try:
            for i in range(30):
                write_json(path, {"writer": n, "i": i, "pad": "x" * 20000})
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert json.loads(path.read_text())["i"] == 29
    assert list(tmp_path.glob("*.tmp")) == []
