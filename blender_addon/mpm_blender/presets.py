"""Material presets as the add-on sees them: mpm_core's built-in presets plus the user's own.

User presets are TOML files in the extension's user folder, so they are per machine and survive add-on
updates. They are read once and cached; saving or deleting one refreshes the cache.
"""

from __future__ import annotations

from pathlib import Path

import bpy

from mpm_core.materials import (Look, MaterialParams, Preset, delete_user_preset, list_presets, load_preset,
                                save_user_preset)

DEFAULT_LOOK = Look("mesh", (0.8, 0.8, 0.8), 0.5, 0.0)  # for a preset that isn't on this machine
USER_DIR_OVERRIDE: Path | None = None  # tests point this at a temporary folder
_cache: dict[str, Preset] | None = None


def user_dir() -> Path:
    if USER_DIR_OVERRIDE is not None:
        return USER_DIR_OVERRIDE
    if __package__.startswith("bl_ext."):
        return Path(bpy.utils.extension_path_user(__package__, path="presets"))
    # loaded as a plain add-on (development): keep presets next to Blender's config
    return Path(bpy.utils.user_resource("CONFIG", path="mpm_presets"))


def all_presets() -> dict[str, Preset]:
    """Built-in presets in id order, then user presets by label."""
    global _cache
    if _cache is None:
        folder = user_dir()
        _cache = {}
        for preset_id in list_presets(folder):
            try:
                _cache[preset_id] = load_preset(preset_id, folder)
            except Exception as exc:  # noqa: BLE001 - one broken user file must not hide the others
                print(f"MPM: skipping preset {preset_id!r}: {exc}")
    return _cache


def get(preset_id: str) -> Preset | None:
    return all_presets().get(preset_id)


def save(label: str, material: MaterialParams, look: Look) -> Preset:
    global _cache
    preset = save_user_preset(user_dir(), label, material, look)
    _cache = None
    return preset


def delete(preset_id: str) -> None:
    global _cache
    delete_user_preset(user_dir(), preset_id)
    _cache = None
