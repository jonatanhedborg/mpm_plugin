from dataclasses import asdict, replace
from types import SimpleNamespace

import bpy
import pytest
from conftest import add_cube, small_scene

from mpm_core.materials import list_presets, load_preset


def assert_same_material(a, b):
    """Same model and the same values for what that model uses (Blender stores them as float32)."""
    from mpm_blender.props import FROM_CORE_MODEL, _TO_CORE_FIELD, used_fields

    keys = ["name", "model", *(_TO_CORE_FIELD.get(f, f) for f in used_fields(FROM_CORE_MODEL[b.model]))]
    expected = asdict(b)
    assert {k: asdict(a)[k] for k in keys} == {k: pytest.approx(expected[k], rel=1e-6) if isinstance(expected[k], float)
                                              else expected[k] for k in keys}


@pytest.fixture
def baked_jelly(clean_scene, tmp_path):
    domain, emitter = small_scene(clean_scene, tmp_path)
    bpy.ops.mpm.bake(wait=True, domain="Domain")
    return clean_scene, domain, emitter


def test_role_defaults(clean_scene):
    d = add_cube("Domain").mpm.domain
    assert d.resolution == 128 and d.quality == "PREVIEW" and d.particles_per_cell == "8"
    assert all(getattr(d, f"wall_{a}_{s}") for a in "xyz" for s in ("neg", "pos"))
    assert d.auto_surface is True


def test_domain_role_hides_from_render(clean_scene):
    ob = add_cube("Domain")
    ob.mpm.role = "DOMAIN"
    assert ob.hide_render and ob.display_type == "WIRE"
    ob.mpm.role = "COLLIDER"
    assert not ob.hide_render and ob.display_type == "TEXTURED"


def test_new_emitter_is_whipped_cream(clean_scene):
    from mpm_blender.props import emitter_material

    ob = add_cube("Emitter")
    assert_same_material(emitter_material(ob), replace(load_preset("whipped_cream").material, name="Emitter"))


@pytest.mark.parametrize("preset_id", list_presets())
def test_choosing_preset_copies_values(clean_scene, preset_id):
    from mpm_blender.props import emitter_material, is_modified

    ob = add_cube("Emitter")
    ob.mpm.emitter.preset_id = preset_id
    assert_same_material(emitter_material(ob), replace(load_preset(preset_id).material, name="Emitter"))
    assert not is_modified(ob.mpm.emitter)


def test_edit_marks_modified_and_apply_resets(clean_scene):
    from mpm_blender import ui
    from mpm_blender.props import is_modified

    ob = add_cube("Emitter")
    bpy.context.view_layer.objects.active = ob
    ob.mpm.role = "EMITTER"
    e = ob.mpm.emitter
    e.preset_id = "sand_dry"
    e.friction_angle = 20
    assert is_modified(e) and ui.preset_text(e) == "Sand, dry (edited)"
    e.yield_stress = 1  # not used by granular material: not an edit
    e.friction_angle = 47
    assert not is_modified(e)
    e.friction_angle = 20
    assert bpy.ops.mpm.apply_preset(preset_id="sand_dry") == {"FINISHED"}  # same preset again resets
    assert e.friction_angle == 47 and ui.preset_text(e) == "Sand, dry"


def test_switching_model_after_preset_starts_from_defaults(clean_scene):
    from mpm_blender.props import emitter_material

    ob = add_cube("Emitter")
    ob.mpm.emitter.preset_id = "whipped_cream"  # whose preset has no snow values
    ob.mpm.emitter.model = "SNOW"
    m = emitter_material(ob)
    assert m.model == "snow" and m.hardening == 10 and m.critical_stretch == pytest.approx(0.0045)


def test_save_and_delete_user_preset(clean_scene):
    from mpm_blender import presets, ui

    ob = add_cube("Emitter")
    bpy.context.view_layer.objects.active = ob
    ob.mpm.role = "EMITTER"
    e = ob.mpm.emitter
    e.preset_id = "clay"
    e.yield_stress = 9000
    assert bpy.ops.mpm.save_preset(name="Soft clay") == {"FINISHED"}
    assert e.preset_id == "user:soft_clay" and ui.preset_text(e) == "Soft clay"
    saved = presets.get("user:soft_clay")
    assert saved.material.yield_stress == 9000 and saved.look == load_preset("clay").look
    assert (presets.user_dir() / "soft_clay.toml").exists()

    other = add_cube("Other")
    other.mpm.emitter.preset_id = "user:soft_clay"
    assert other.mpm.emitter.yield_stress == 9000 and other.mpm.emitter.model == "PLASTIC"

    bpy.context.view_layer.objects.active = ob
    with pytest.raises(RuntimeError, match="built-in"):
        bpy.ops.mpm.save_preset(name="Clay")
    assert bpy.ops.mpm.delete_preset() == {"FINISHED"}
    assert presets.get("user:soft_clay") is None and ui.preset_text(e) == "Custom"
    assert e.yield_stress == 9000  # values stay
    e.preset_id = "jelly"
    assert not bpy.ops.mpm.delete_preset.poll()  # built-ins can't be deleted


def test_saved_preset_takes_look_from_output_material(baked_jelly):
    from mpm_blender import presets

    _, _, emitter = baked_jelly
    bpy.context.view_layer.objects.active = emitter
    mat = bpy.data.objects["MPM Emitter"].data.materials[0]
    mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.1, 0.2, 0.3, 1)
    assert bpy.ops.mpm.save_preset(name="Blue jelly") == {"FINISHED"}
    look = presets.get("user:blue_jelly").look
    assert look.base_color == pytest.approx((0.1, 0.2, 0.3)) and look.display == "mesh"
    presets.delete("user:blue_jelly")


def test_unknown_preset_keeps_label(clean_scene):  # a file from a machine with presets this one lacks
    from mpm_blender import ui

    e = add_cube("Emitter").mpm.emitter
    e.preset_id = "jelly"
    e["preset_id"], e["preset_label"] = "user:theirs", "Their jelly"
    assert ui.preset_text(e) == "Their jelly (not on this computer)" and e.model == "ELASTIC"


@pytest.mark.parametrize("old_index, old_values, expected", [
    (None, {}, None),  # untouched emitter: nothing stored, nothing to convert
    (None, {"firmness": 2.0}, {"model": "PLASTIC", "yield_stress": 5000, "preset_id": "whipped_cream"}),
    (1, {"friction_angle": 30.0, "stiffness": 2.0, "density": 2200.0},
     {"model": "GRANULAR", "friction_angle": 30, "youngs_modulus": 7e5, "preset_id": "sand_dry"}),
    (2, {"clumping": 4.0, "use_advanced": True, "critical_stretch": 0.01, "youngs_modulus": 9e4},
     {"model": "SNOW", "hardening": 4, "critical_stretch": 0.01, "youngs_modulus": 9e4, "critical_compression": 0.025}),
    (0, {"use_advanced": False, "youngs_modulus": 1.0}, {"model": "ELASTIC", "youngs_modulus": 5e4}),
])
def test_migrate_old_settings(clean_scene, old_index, old_values, expected):
    from mpm_blender.props import migrate_emitter

    e = add_cube("Emitter").mpm.emitter
    if old_index is not None:
        e["preset"] = old_index
    for key, value in old_values.items():
        e[key] = value
    assert migrate_emitter(e) == (expected is not None)
    for key, value in (expected or {}).items():
        assert getattr(e, key) == (pytest.approx(value) if isinstance(value, float | int) else value), key
    assert not {"preset", "stiffness", "firmness", "clumping", "use_advanced"} & set(e.keys())


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
    ob.mpm.emitter.preset_id = preset
    ob.mpm.emitter.kind = "INFLOW"
    ctx = SimpleNamespace(object=ob, scene=clean_scene)
    for panel_cls in _panels():
        if panel_cls.poll(ctx):
            panel = SimpleNamespace(layout=FakeLayout())
            panel_cls.draw(panel, ctx)
