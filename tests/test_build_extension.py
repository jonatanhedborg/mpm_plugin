import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from build_extension import wheel_version  # noqa: E402


def _tree(root: Path, text: str) -> Path:
    (root / "pkg").mkdir(parents=True, exist_ok=True)
    (root / "pkg" / "a.py").write_text(text)
    (root / "pkg" / "b.toml").write_text("x = 1")
    return root


def test_version_is_stable_for_same_sources(tmp_path):
    a = wheel_version("0.1.0", [_tree(tmp_path / "one", "print(1)") / "pkg"])
    b = wheel_version("0.1.0", [_tree(tmp_path / "two", "print(1)") / "pkg"])
    assert a == b and a.startswith("0.1.0.dev")


def test_version_changes_with_sources(tmp_path):
    a = wheel_version("0.1.0", [_tree(tmp_path / "one", "print(1)") / "pkg"])
    b = wheel_version("0.1.0", [_tree(tmp_path / "two", "print(2)") / "pkg"])
    assert a != b


def test_pycache_ignored(tmp_path):
    root = _tree(tmp_path / "one", "print(1)") / "pkg"
    a = wheel_version("0.1.0", [root])
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "a.cpython-312.pyc").write_bytes(b"junk")
    assert wheel_version("0.1.0", [root]) == a
