import pytest

from mpm_core.materials import list_presets, load_preset
from mpm_core.tweaks import apply_tweaks, tweaks_for


def test_tweaks_for_cream():
    specs = tweaks_for("whipped_cream")
    assert [s.name for s in specs] == ["stiffness", "density", "firmness"]
    assert {s.name: s.default for s in specs}["density"] == 400


def test_tweaks_for_sand():
    assert {s.name: s.default for s in tweaks_for("sand_dry")}["friction_angle"] == 30


def test_tweaks_for_snow_and_jelly():
    assert [s.name for s in tweaks_for("snow_packed")] == ["stiffness", "density", "clumping"]
    assert [s.name for s in tweaks_for("jelly")] == ["stiffness", "density"]


def test_stiffness_and_firmness_scale():
    m = apply_tweaks("whipped_cream", {"stiffness": 2, "firmness": 0.5})
    assert m.youngs_modulus == pytest.approx(2e5) and m.yield_stress == pytest.approx(1250)


def test_absolute_tweaks():
    assert apply_tweaks("sand_dry", {"friction_angle": 40}).friction_angle_deg == 40
    assert apply_tweaks("snow_packed", {"clumping": 5}).hardening == 5
    assert apply_tweaks("jelly", {"density": 1200}).density == 1200


def test_overrides_win():
    m = apply_tweaks("whipped_cream", {"stiffness": 2}, overrides={"youngs_modulus": 7e4})
    assert m.youngs_modulus == 7e4


def test_unknown_tweak_raises():
    with pytest.raises(KeyError):
        apply_tweaks("jelly", {"firmness": 2})  # firmness doesn't apply to jelly


def test_name():
    assert apply_tweaks("jelly", {}, name="Blob").name == "Blob"


@pytest.mark.parametrize("preset", list_presets())
def test_defaults_reproduce_preset(preset):
    assert apply_tweaks(preset, {}) == load_preset(preset).material
