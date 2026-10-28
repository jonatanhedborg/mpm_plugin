"""Bake and surface runs as subprocesses of Blender's own Python, watched through their progress files.

The child gets Blender's sys.path as PYTHONPATH, so it finds mpm_core, mpm_surface and Warp exactly
where the extension's wheels put them. A GPU fault kills the child, never Blender.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


class Job:
    def __init__(self, kind: str, cache_dir: Path, argv: list[str], log_path: Path, progress_path: Path,
                 cancel_path: Path):
        self.kind = kind
        self.cache_dir = Path(cache_dir)
        self.argv = argv
        self.log_path = log_path
        self.progress_path = progress_path
        self.cancel_path = cancel_path
        self.env = dict(os.environ,
                        PYTHONPATH=os.pathsep.join(p for p in sys.path if p and os.path.exists(p)),
                        PYTHONNOUSERSITE="1")
        self.proc: subprocess.Popen | None = None
        self._log = None

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.progress_path.unlink(missing_ok=True)  # never show the previous run's status
        self._log = open(self.log_path, "wb")
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        # stdin stays an open pipe we never write to: when Blender exits the OS closes it and the child
        # (started with --watch-stdin) stops, so no orphan bake keeps writing the cache
        self.proc = subprocess.Popen(self.argv, env=self.env, stdout=self._log, stderr=subprocess.STDOUT,
                                     stdin=subprocess.PIPE, creationflags=flags)

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    @property
    def returncode(self) -> int | None:
        return None if self.proc is None else self.proc.poll()

    def poll(self) -> dict | None:
        """Latest progress document, or None if not written yet."""
        for _ in range(5):
            try:
                return json.loads(self.progress_path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                return None
            except (PermissionError, json.JSONDecodeError):  # being replaced right now (Windows)
                time.sleep(0.02)
        return None

    def cancel(self) -> None:
        self.cancel_path.parent.mkdir(parents=True, exist_ok=True)
        self.cancel_path.touch()

    def wait(self) -> int:
        code = self.proc.wait() if self.proc else 0
        if self.proc and self.proc.stdin and not self.proc.stdin.closed:
            self.proc.stdin.close()
        if self._log:
            self._log.close()
            self._log = None
        return code

    def terminate(self, timeout: float = 2.0) -> None:
        if not self.running:
            return
        self.cancel()
        try:
            self.proc.wait(timeout)
        except subprocess.TimeoutExpired:
            self.proc.terminate()
        self.wait()

    def log_tail(self, lines: int = 12) -> str:
        try:
            text = self.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        return "\n".join(text.strip().splitlines()[-lines:])


def bake_job(cache: Path, resume: bool) -> Job:
    cache = Path(cache)
    argv = [sys.executable, "-m", "mpm_core.bake", str(cache), "--watch-stdin"] + (["--resume"] if resume else [])
    return Job("bake", cache, argv, cache / "bake.log", cache / "progress.json", cache / "cancel")


def surface_job(cache: Path) -> Job:
    cache = Path(cache)
    argv = [sys.executable, "-m", "mpm_surface", str(cache), "--watch-stdin"]
    surf = cache / "surface"
    return Job("surface", cache, argv, surf / "surface.log", surf / "progress.json", surf / "cancel")


JOBS: dict[str, Job] = {}  # domain uid (MPMDomainSettings.uid) -> its running (or last) job


def cancel_all() -> None:
    for job in JOBS.values():
        job.terminate()
    JOBS.clear()
