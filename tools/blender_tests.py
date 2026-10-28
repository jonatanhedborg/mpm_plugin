"""Run tests_blender/ with pytest inside headless Blender.

    python tools/blender_tests.py [pytest args...]

Blender is found via the BLENDER environment variable, else the default install location.
Run tools/setup_blender_test_env.py once first.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def find_blender() -> str:
    if os.environ.get("BLENDER"):
        return os.environ["BLENDER"]
    if sys.platform == "win32":
        path = Path("C:/Program Files/Blender Foundation/Blender 5.2/blender.exe")
        if path.exists():
            return str(path)
    found = shutil.which("blender")
    if not found:
        raise SystemExit("Blender not found: set the BLENDER environment variable")
    return found


def main() -> int:
    site = ROOT / "out" / "blender_site"
    if not (site / "pytest").exists():
        raise SystemExit("run tools/setup_blender_test_env.py first")
    paths = [str(ROOT / "src"), str(ROOT / "blender_addon"), str(site)]
    expr = (
        "import sys; "
        f"sys.path[:0] = {paths!r}; "
        "import pytest; "
        f"code = pytest.main([{str(ROOT / 'tests_blender')!r}, '-p', 'no:cacheprovider', *{sys.argv[1:]!r}]); "
        "sys.stdout.flush(); "
        "import os; os._exit(int(code))"
    )
    return subprocess.call([find_blender(), "-b", "--factory-startup", "--python-exit-code", "1",
                            "--python-expr", expr], cwd=ROOT)


if __name__ == "__main__":
    sys.exit(main())
