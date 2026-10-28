"""Install what the in-Blender tests need into out/blender_site (pure Python or ABI-independent only).

    python tools/setup_blender_test_env.py

Blender's own numpy is used; Warp's wheel is py3-none so it works with Blender's Python.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "out" / "blender_site"
PACKAGES = ["warp-lang==1.18.0", "pytest", "iniconfig", "pluggy", "packaging", "colorama", "pygments"]

if __name__ == "__main__":
    TARGET.mkdir(parents=True, exist_ok=True)
    cmd = ["uv", "pip", "install", "--python", sys.executable, "--target", str(TARGET), "--no-deps", *PACKAGES]
    sys.exit(subprocess.call(cmd))
