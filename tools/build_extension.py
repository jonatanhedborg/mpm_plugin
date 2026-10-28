"""Build the Blender extension zips (Windows x64, Linux x64) with all wheels bundled.

    python tools/build_extension.py [--blender PATH]

1. builds the mpm_core wheel (contains mpm_core and mpm_surface) from this repo
2. downloads the Warp wheels for both platforms
3. writes the wheel list into blender_manifest.toml
4. runs `blender --command extension build --split-platforms` into dist/
"""

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blender_tests import find_blender  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ADDON = ROOT / "blender_addon" / "mpm_blender"
WHEELS = ADDON / "wheels"
DIST = ROOT / "dist"
WARP = "warp-lang==1.18.0"
PLATFORMS = ["win_amd64", "manylinux_2_28_x86_64"]


def wheel_version(base: str, package_dirs: list[Path]) -> str:
    """``<base>.dev<N>`` with N from a hash of the package sources.

    Blender skips installing a wheel whose version is already installed, so an unchanged version
    would keep stale code after an extension update. Same sources give the same version.
    """
    h = hashlib.sha1()
    for root in package_dirs:
        for path in sorted(Path(root).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                h.update(path.relative_to(root).as_posix().encode())
                h.update(path.read_bytes())
    return f"{base}.dev{int(h.hexdigest()[:8], 16)}"


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    # don't leak this interpreter's venv into tools that bring their own Python (uvx pip, Blender)
    env = {k: v for k, v in os.environ.items()
           if k not in ("__PYVENV_LAUNCHER__", "VIRTUAL_ENV", "PYTHONHOME", "PYTHONPATH")}
    subprocess.run(cmd, check=True, cwd=ROOT, env=env)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blender", default=None)
    args = ap.parse_args()

    shutil.rmtree(WHEELS, ignore_errors=True)
    WHEELS.mkdir(parents=True)
    pyproject = ROOT / "pyproject.toml"
    original = pyproject.read_text(encoding="utf-8")
    base = re.search(r'(?m)^version = "([^"]+)"', original).group(1)
    version = wheel_version(base, [ROOT / "src" / "mpm_core", ROOT / "src" / "mpm_surface"])
    try:  # build the wheel under the source-hash version, then put pyproject.toml back
        pyproject.write_text(re.sub(r'(?m)^version = "[^"]+"', f'version = "{version}"', original, count=1),
                             encoding="utf-8")
        run(["uv", "build", "--wheel", "--out-dir", str(WHEELS)])
    finally:
        pyproject.write_text(original, encoding="utf-8")
    for platform in PLATFORMS:
        run(["uvx", "pip", "download", WARP, "--no-deps", "--only-binary=:all:", "--platform", platform,
             "-d", str(WHEELS), "-q"])

    wheels = sorted(p.name for p in WHEELS.glob("*.whl"))
    manifest = ADDON / "blender_manifest.toml"
    listing = "wheels = [\n" + "".join(f'  "./wheels/{w}",\n' for w in wheels) + "]"
    text, n = re.subn(r"(?ms)^wheels = \[.*?\]", listing, manifest.read_text(encoding="utf-8"))
    if n != 1:
        raise SystemExit("blender_manifest.toml: no 'wheels = [...]' entry to fill in")
    manifest.write_text(text, encoding="utf-8")

    shutil.rmtree(DIST, ignore_errors=True)
    DIST.mkdir()
    run([args.blender or find_blender(), "--factory-startup", "--command", "extension", "build",
         "--split-platforms", "--source-dir", str(ADDON), "--output-dir", str(DIST)])
    for zip_path in sorted(DIST.glob("*.zip")):
        print(f"built {zip_path} ({zip_path.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
