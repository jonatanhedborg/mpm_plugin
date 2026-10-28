"""Friendly preset tweaks, mapped to physical material parameters.

Every preset gets ``stiffness`` (multiplies Young's modulus) and ``density`` (kg/m³). Each model adds
one tweak of its own: ``firmness`` (von Mises: multiplies the yield stress), ``friction_angle``
(Drucker-Prager: degrees) or ``clumping`` (snow: hardening).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from mpm_core.materials import MaterialParams, load_preset


@dataclass(frozen=True)
class TweakSpec:
    name: str
    label: str
    default: float
    min: float
    max: float
    unit: str
    description: str


def tweaks_for(preset_id: str) -> list[TweakSpec]:
    m = load_preset(preset_id).material
    specs = [
        TweakSpec("stiffness", "Stiffness", 1.0, 0.1, 10.0, "×", "How hard the material pushes back when deformed"),
        TweakSpec("density", "Density", m.density, 10.0, 5000.0, "kg/m³", "Mass per volume"),
    ]
    if m.model == "von_mises":
        specs.append(TweakSpec("firmness", "Firmness", 1.0, 0.05, 20.0, "×",
                               "How much force it takes before the material starts to flow"))
    elif m.model == "drucker_prager":
        specs.append(TweakSpec("friction_angle", "Friction Angle", m.friction_angle_deg, 0.0, 60.0, "°",
                               "Steepest slope a pile can hold"))
    elif m.model == "snow":
        specs.append(TweakSpec("clumping", "Clumping", m.hardening, 0.0, 30.0, "",
                               "How much the material stiffens when packed together"))
    return specs


def apply_tweaks(preset_id: str, values: dict[str, float], overrides: dict[str, float] | None = None,
                 name: str | None = None) -> MaterialParams:
    m = load_preset(preset_id).material
    known = {s.name for s in tweaks_for(preset_id)}
    unknown = set(values) - known
    if unknown:
        raise KeyError(f"tweaks {sorted(unknown)} don't apply to preset {preset_id!r} (known: {sorted(known)})")
    changes: dict[str, float | str] = {}
    if "stiffness" in values:
        changes["youngs_modulus"] = m.youngs_modulus * values["stiffness"]
    if "density" in values:
        changes["density"] = values["density"]
    if "firmness" in values:
        changes["yield_stress"] = m.yield_stress * values["firmness"]
    if "friction_angle" in values:
        changes["friction_angle_deg"] = values["friction_angle"]
    if "clumping" in values:
        changes["hardening"] = values["clumping"]
    changes.update(overrides or {})
    if name is not None:
        changes["name"] = name
    return replace(m, **changes)
