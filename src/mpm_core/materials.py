"""Material parameters and presets.

A preset bundles physics (MaterialParams), a render look (Look) and a label.
Presets live as TOML files in ``mpm_core/presets/<id>.toml`` with a top-level
``label`` and ``[material]`` / ``[look]`` tables.
"""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, fields
from importlib import resources

MODELS = ("elastic", "von_mises", "drucker_prager", "snow")
DISPLAYS = ("mesh", "points")


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


def _preset_files():
    return {p.name.removesuffix(".toml"): p for p in resources.files("mpm_core.presets").iterdir()
            if p.name.endswith(".toml")}


def list_presets() -> list[str]:
    return sorted(_preset_files())


def load_preset(preset_id: str) -> Preset:
    files = _preset_files()
    if preset_id not in files:
        raise KeyError(f"unknown preset {preset_id!r}, available: {sorted(files)}")
    data = tomllib.loads(files[preset_id].read_text(encoding="utf-8"))
    look = data["look"]
    return Preset(
        id=preset_id,
        label=data["label"],
        material=MaterialParams(name=preset_id, **data["material"]),
        look=Look(look["display"], tuple(look["base_color"]), look["roughness"], look["subsurface"]),
    )
