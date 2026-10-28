"""Material parameters and presets.

A preset bundles physics (MaterialParams), a render look (Look), a label and an optional description.
Presets are TOML files with a top-level ``label`` (and ``description``) and ``[material]`` / ``[look]``
tables. Built-in presets live in ``mpm_core/presets/<id>.toml``; user presets in a folder the caller
passes as ``user_dir`` (the Blender add-on uses its user folder), with ids ``user:<file stem>``.
"""

from __future__ import annotations

import json
import os
import re
import tomllib
from dataclasses import asdict, dataclass, fields
from importlib import resources
from pathlib import Path

MODELS = ("elastic", "von_mises", "drucker_prager", "snow")
DISPLAYS = ("mesh", "points")
USER_PREFIX = "user:"


@dataclass(frozen=True)
class MaterialParams:
    name: str
    model: str
    youngs_modulus: float
    poisson_ratio: float
    density: float
    yield_stress: float = 0.0
    friction_angle_deg: float = 0.0
    hardening: float = 0.0
    critical_compression: float = 0.0
    critical_stretch: float = 0.0
    max_hardening: float = 5.0

    def __post_init__(self):
        if self.model not in MODELS:
            raise ValueError(f"unknown material model {self.model!r}, expected one of {MODELS}")
        if not self.youngs_modulus > 0:
            raise ValueError("youngs_modulus must be > 0")
        if not 0 <= self.poisson_ratio < 0.5:
            raise ValueError("poisson_ratio must be in [0, 0.5)")
        if not self.density > 0:
            raise ValueError("density must be > 0")

    def lame(self) -> tuple[float, float]:
        """Lamé parameters (mu, lam)."""
        e, nu = self.youngs_modulus, self.poisson_ratio
        return e / (2 * (1 + nu)), e * nu / ((1 + nu) * (1 - 2 * nu))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> MaterialParams:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass(frozen=True)
class Look:
    display: str
    base_color: tuple[float, float, float]
    roughness: float
    subsurface: float

    def __post_init__(self):
        if self.display not in DISPLAYS:
            raise ValueError(f"unknown display {self.display!r}, expected one of {DISPLAYS}")


@dataclass(frozen=True)
class Preset:
    id: str
    label: str
    material: MaterialParams
    look: Look
    description: str = ""

    @property
    def is_user(self) -> bool:
        return self.id.startswith(USER_PREFIX)


def _builtin_files():
    return {p.name.removesuffix(".toml"): p for p in resources.files("mpm_core.presets").iterdir()
            if p.name.endswith(".toml")}


def _user_files(user_dir: Path | None) -> dict[str, Path]:
    if user_dir is None or not Path(user_dir).is_dir():
        return {}
    return {USER_PREFIX + p.stem: p for p in Path(user_dir).glob("*.toml")}


def list_presets(user_dir: Path | None = None) -> list[str]:
    """Built-in preset ids, sorted, then the user presets found in ``user_dir``, sorted by label."""
    user = sorted(_user_files(user_dir), key=lambda i: load_preset(i, user_dir).label.lower())
    return sorted(_builtin_files()) + user


def load_preset(preset_id: str, user_dir: Path | None = None) -> Preset:
    files = {**_builtin_files(), **_user_files(user_dir)}
    if preset_id not in files:
        raise KeyError(f"unknown preset {preset_id!r}, available: {sorted(files)}")
    data = tomllib.loads(files[preset_id].read_text(encoding="utf-8"))
    look = data["look"]
    return Preset(
        id=preset_id,
        label=data["label"],
        material=MaterialParams(name=preset_id, **data["material"]),
        look=Look(look["display"], tuple(look["base_color"]), look["roughness"], look["subsurface"]),
        description=data.get("description", ""),
    )


def user_preset_id(label: str) -> str:
    """The id a user preset with this label gets (labels differing only in case or punctuation share one)."""
    stem = re.sub(r"[^a-z0-9]+", "_", label.strip().lower()).strip("_")
    if not stem:
        raise ValueError("a preset name needs at least one letter or digit")
    return USER_PREFIX + stem


def save_user_preset(user_dir: Path, label: str, material: MaterialParams, look: Look,
                     description: str = "") -> Preset:
    """Write a user preset, replacing one with the same id. Built-in labels are refused."""
    label = label.strip()
    if label.lower() in {load_preset(i).label.lower() for i in _builtin_files()}:
        raise ValueError(f"{label!r} is the name of a built-in preset")
    preset_id = user_preset_id(label)
    values = {k: v for k, v in material.to_dict().items() if k != "name"}
    lines = [f"label = {json.dumps(label)}"]
    if description:
        lines.append(f"description = {json.dumps(description)}")
    lines += ["", "[material]", *(f"{k} = {json.dumps(v)}" for k, v in values.items()),
              "", "[look]", f"display = {json.dumps(look.display)}",
              f"base_color = {json.dumps([float(c) for c in look.base_color])}",
              f"roughness = {json.dumps(float(look.roughness))}", f"subsurface = {json.dumps(float(look.subsurface))}",
              ""]
    user_dir = Path(user_dir)
    user_dir.mkdir(parents=True, exist_ok=True)
    path = user_dir / f"{preset_id.removeprefix(USER_PREFIX)}.toml"
    tmp = path.with_suffix(".toml.tmp")
    tmp.write_text("\n".join(lines), encoding="utf-8")
    os.replace(tmp, path)
    return load_preset(preset_id, user_dir)


def delete_user_preset(user_dir: Path, preset_id: str) -> None:
    if not preset_id.startswith(USER_PREFIX):
        raise ValueError(f"{preset_id!r} is a built-in preset")
    path = _user_files(user_dir).get(preset_id)
    if path is None:
        raise KeyError(f"unknown user preset {preset_id!r}")
    path.unlink()
