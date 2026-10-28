from types import SimpleNamespace

import bpy
import pytest
from conftest import add_cube

from mpm_core.materials import list_presets


def test_role_defaults(clean_scene):
    d = add_cube("Domain").mpm.domain
    assert d.resolution == 128 and d.quality == "PREVIEW" and d.particles_per_cell == "8"
    assert all(getattr(d, f"wall_{a}_{s}") for a in "xyz" for s in ("neg", "pos"))
    assert d.auto_surface is True


def test_preset_enum_lists_presets(clean_scene):
    e = add_cube("Emitter").mpm.emitter
    items = e.bl_rna.properties["preset"].enum_items
    assert [i.identifier for i in items] == list_presets()
    assert "Whipped cream" in [i.name for i in items]


def test_preset_change_resets_tweaks(clean_scene):
    e = add_cube("Emitter").mpm.emitter
    e.preset = "sand_dry"
    e.friction_angle = 40
    e.preset = "snow_packed"
    assert e.clumping == 10
    e.preset = "sand_dry"
    assert e.friction_angle == 30


def test_emitter_material_uses_tweaks(clean_scene):
    from mpm_blender.props import emitter_material

    ob = add_cube("Emitter")
    e = ob.mpm.emitter
    e.preset = "whipped_cream"
    e.firmness = 0.5
    m = emitter_material(ob)
    assert m.yield_stress == pytest.approx(1250) and m.name == "Emitter"


def test_emitter_material_advanced_overrides(clean_scene):
    from mpm_blender.props import emitter_material

    ob = add_cube("Emitter")
    e = ob.mpm.emitter
    e.preset = "whipped_cream"
    e.stiffness = 2
    e.use_advanced = True
    e.youngs_modulus = 7e4
    assert emitter_material(ob).youngs_modulus == pytest.approx(7e4)


def _panels():
    from mpm_blender import ui

    return ui.PANELS


def test_panels_poll_by_role(clean_scene):
    from mpm_blender import ui

    ob = add_cube("Thing")
    ob.mpm.role = "EMITTER"
    ctx = SimpleNamespace(object=ob)
    assert ui.MPM_PT_emitter_material.poll(ctx)
    assert not ui.MPM_PT_domain_settings.poll(ctx)
    ob.mpm.emitter.kind = "FILL"
    assert not ui.MPM_PT_emitter_inflow.poll(ctx)
    ob.mpm.emitter.kind = "INFLOW"
    assert ui.MPM_PT_emitter_inflow.poll(ctx)


class FakeLayout:
    """Stands in for UILayout; prop() checks the property really exists on the data."""

    def __init__(self):
        self.props = []

    def prop(self, data, name, **kwargs):
        assert name in data.bl_rna.properties, f"{data.bl_rna.identifier} has no property {name!r}"
        self.props.append(name)
        return self

    def __getattr__(self, name):
        def method(*args, **kwargs):
            return self

        return method


@pytest.mark.parametrize("role", ["NONE", "DOMAIN", "EMITTER", "COLLIDER"])
@pytest.mark.parametrize("preset", list_presets())
def test_panels_draw(clean_scene, role, preset):
    ob = add_cube("Thing")
    ob.mpm.role = role
    ob.mpm.emitter.preset = preset
    ob.mpm.emitter.kind = "INFLOW"
    ctx = SimpleNamespace(object=ob, scene=clean_scene)
    for panel_cls in _panels():
        if panel_cls.poll(ctx):
            panel = SimpleNamespace(layout=FakeLayout())
            panel_cls.draw(panel, ctx)
