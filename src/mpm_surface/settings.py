"""Per-emitter surfacing settings and the settings file the add-on writes.

``<cache>/surface_settings.json``::

    {"version": 1, "emitters": {"<emitter name>": {"enabled", "detail", "smoothing", "iso"}}}

Emitters missing from the file use the defaults.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from mpm_core.cache import write_json

FORMAT_VERSION = 1


@dataclass(frozen=True)
class SurfaceSettings:
    enabled: bool = True
    detail: float = 1.0  # field cells per particle spacing
    smoothing: float = 1.0  # scales the kernel radius (2 particle spacings)
    iso: float = 0.5  # surface at this fraction of the rest density

    def __post_init__(self):
        if not self.detail > 0:
            raise ValueError("detail must be > 0")
        if not self.smoothing > 0:
            raise ValueError("smoothing must be > 0")
        if not 0 < self.iso < 1:
            raise ValueError("iso must be in (0, 1)")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> SurfaceSettings:
        return cls(**d)

    def hash(self) -> str:
        return hashlib.sha1(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()


def _path(cache_dir: Path) -> Path:
    return Path(cache_dir) / "surface_settings.json"


def read_settings(cache_dir: Path) -> dict[str, SurfaceSettings]:
    path = _path(cache_dir)
    if not path.exists():
        return {}
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("version") != FORMAT_VERSION:
        raise ValueError(f"unsupported surface settings version {doc.get('version')!r}")
    return {name: SurfaceSettings.from_dict(d) for name, d in doc["emitters"].items()}


def write_settings(cache_dir: Path, settings: dict[str, SurfaceSettings]) -> None:
    write_json(_path(cache_dir), {"version": FORMAT_VERSION,
                                  "emitters": {name: s.to_dict() for name, s in settings.items()}})
