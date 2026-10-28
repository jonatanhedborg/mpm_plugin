"""Per-object MPM settings: one ``Object.mpm`` group with a role and per-role settings."""

import bpy

from mpm_core.materials import MaterialParams, load_preset
from mpm_core.scene import MAX_SUBSTEPS
from mpm_surface.settings import SurfaceSettings  # no warp import

from . import presets

ROLES = [
    ("NONE", "None", "Not part of an MPM simulation"),
    ("DOMAIN", "Domain", "The box the simulation runs in"),
    ("EMITTER", "Emitter", "Fills the domain with material, once or continuously"),
    ("COLLIDER", "Collider", "Material bumps into this object"),
]
MODELS = [  # (UI id, mpm_core model, label, description)
    ("ELASTIC", "elastic", "Elastic", "Always springs back to its shape: jelly, rubber"),
    ("PLASTIC", "von_mises", "Plastic", "Holds its shape until pushed hard enough, then flows: cream, clay, mud"),
    ("GRANULAR", "drucker_prager", "Granular", "Grains that pour and pile up: sand, sugar, gravel"),
    ("SNOW", "snow", "Snow", "Packs harder when compressed and breaks apart when stretched"),
]
TO_CORE_MODEL = {ui: core for ui, core, *_ in MODELS}
FROM_CORE_MODEL = {core: ui for ui, core, *_ in MODELS}
# material fields each model uses beyond stiffness, poisson ratio and density (UI property names)
MODEL_FIELDS = {
    "ELASTIC": (),
    "PLASTIC": ("yield_stress",),
    "GRANULAR": ("friction_angle",),
    "SNOW": ("hardening", "max_hardening", "critical_compression", "critical_stretch"),
}
_TO_CORE_FIELD = {"friction_angle": "friction_angle_deg"}  # UI property -> MaterialParams field, where different
MATERIAL_FIELDS = ("youngs_modulus", "poisson_ratio", "density", "yield_stress", "friction_angle", "hardening",
                   "max_hardening", "critical_compression", "critical_stretch")
_SURFACE = SurfaceSettings()


def _is_domain(self, obj):
    return obj.mpm.role == "DOMAIN"


def used_fields(model: str) -> tuple[str, ...]:
    """The material properties a model (UI id) uses."""
    return ("youngs_modulus", "poisson_ratio", "density", *MODEL_FIELDS[model])


def apply_preset(settings, preset) -> None:
    """Copy a preset's model and the values that model uses into emitter settings.

    The other values go back to their defaults, so switching the model afterwards starts from sensible ones
    rather than the preset's unused zeros.
    """
    m = preset.material
    settings.model = FROM_CORE_MODEL[m.model]
    used = used_fields(settings.model)
    for field in MATERIAL_FIELDS:
        if field in used:
            setattr(settings, field, getattr(m, _TO_CORE_FIELD.get(field, field)))
        else:
            settings.property_unset(field)
    settings.preset_label = preset.label


def _preset_changed(self, context):
    """Choosing a preset copies its values; an unknown id (deleted, or another machine's) changes nothing."""
    preset = presets.get(self.preset_id)
    if preset is not None:
        apply_preset(self, preset)


def _role_changed(self, context):
    """A domain is only a box: shown as wireframe, never rendered. Leaving the role restores both."""
    ob = self.id_data
    is_domain = self.role == "DOMAIN"
    ob.hide_render = is_domain
    ob.display_type = "WIRE" if is_domain else "TEXTURED"


def _quality_changed(self, context):
    from . import display  # lazy: display imports this module

    display.refresh_outputs(context.scene, self.id_data)


class MPMDomainSettings(bpy.types.PropertyGroup):
    uid: bpy.props.StringProperty(default="", description="Persistent id, assigned on first bake")
    resolution: bpy.props.IntProperty(name="Resolution", default=128, min=16, max=1024,
                                      description="Grid cells along the domain's longest side (final quality)")
    quality: bpy.props.EnumProperty(name="Quality", default="PREVIEW", update=_quality_changed, items=[
        ("PREVIEW", "Preview", "Half resolution, for quick iteration"),
        ("FINAL", "Final", "Full resolution")])
    particles_per_cell: bpy.props.EnumProperty(name="Particles per Cell", default="8", items=[
        ("1", "1", "Fastest, coarse"), ("8", "8", "Standard"), ("27", "27", "Dense, slow")])
    cache_dir: bpy.props.StringProperty(name="Cache Folder", subtype="DIR_PATH", default="",
                                        options={"PATH_SUPPORTS_BLEND_RELATIVE"},
                                        description="Where bakes are stored; empty = //mpm_cache/<domain name>")
    wall_x_neg: bpy.props.BoolProperty(name="-X", default=True)
    wall_x_pos: bpy.props.BoolProperty(name="+X", default=True)
    wall_y_neg: bpy.props.BoolProperty(name="-Y", default=True)
    wall_y_pos: bpy.props.BoolProperty(name="+Y", default=True)
    wall_z_neg: bpy.props.BoolProperty(name="Floor", default=True)
    wall_z_pos: bpy.props.BoolProperty(name="Ceiling", default=True)
    wall_friction: bpy.props.FloatProperty(name="Wall Friction", default=0.5, min=0.0, max=2.0)
    auto_surface: bpy.props.BoolProperty(name="Surface After Bake", default=True,
                                         description="Mesh emitters displayed as surfaces when a bake finishes")
    # runtime state, written by the bake operators
    progress: bpy.props.FloatProperty(default=0.0, min=0.0, max=1.0, options={"SKIP_SAVE"})
    status: bpy.props.StringProperty(default="", options={"SKIP_SAVE"})
    status_message: bpy.props.StringProperty(default="", options={"SKIP_SAVE"})


class MPMEmitterSettings(bpy.types.PropertyGroup):
    domain: bpy.props.PointerProperty(name="Domain", type=bpy.types.Object, poll=_is_domain,
                                      description="Empty = the scene's only domain")
    kind: bpy.props.EnumProperty(name="Type", default="FILL", items=[
        ("FILL", "Fill", "The object's volume becomes material at the start frame"),
        ("INFLOW", "Inflow", "Material keeps flowing in from the object's volume")])
    # defaults are the whipped cream preset's, so emitters saved without material values get whipped cream
    preset_id: bpy.props.StringProperty(name="Preset", default="whipped_cream", update=_preset_changed,
                                        description="The preset these values came from; empty = custom")
    preset_label: bpy.props.StringProperty(default="Whipped cream", description="Label of the preset, kept so "
                                           "the name shows even where the preset doesn't exist")
    model: bpy.props.EnumProperty(name="Model", default="PLASTIC", items=[(ui, label, desc) for ui, _, label, desc
                                                                           in MODELS])
    youngs_modulus: bpy.props.FloatProperty(
        name="Stiffness", default=1.0e5, min=100.0, soft_max=1.0e7, step=1000, precision=0,
        description="Young's modulus in Pa: how hard the material pushes back when deformed. Stiffer materials "
                    "bake slower")
    poisson_ratio: bpy.props.FloatProperty(
        name="Poisson Ratio", default=0.3, min=0.0, max=0.49,
        description="How much it bulges sideways when squashed: 0 not at all, 0.49 keeps its volume")
    density: bpy.props.FloatProperty(name="Density", default=400.0, min=10.0, max=20000.0, precision=0,
                                     description="Mass per volume in kg/m³")
    yield_stress: bpy.props.FloatProperty(
        name="Yield Stress", default=2500.0, min=0.0, soft_max=1.0e5, step=1000, precision=0,
        description="Stress in Pa at which it starts to flow: higher holds taller shapes")
    friction_angle: bpy.props.FloatProperty(
        name="Friction Angle", default=47.0, min=0.0, max=70.0, precision=1,
        description="Internal friction of the grains, in degrees. Higher holds steeper piles; piles settle below "
                    "this angle (47° settles at about 30°, 55° at about 40°)")
    hardening: bpy.props.FloatProperty(name="Hardening", default=10.0, min=0.0, max=50.0,
                                       description="How much stiffer it gets as it is packed together")
    max_hardening: bpy.props.FloatProperty(
        name="Max Hardening", default=5.0, min=1.0, max=50.0,
        description="Upper limit on how many times stiffer packing can make it. Higher bakes slower")
    critical_compression: bpy.props.FloatProperty(
        name="Compression Limit", default=0.025, min=0.0, max=1.0, precision=4, step=0.1,
        description="How far it can be squashed before it packs for good")
    critical_stretch: bpy.props.FloatProperty(
        name="Stretch Limit", default=0.0045, min=0.0, max=1.0, precision=4, step=0.1,
        description="How far it can be stretched before it breaks apart")
    velocity: bpy.props.FloatVectorProperty(name="Velocity", size=3, subtype="VELOCITY", unit="VELOCITY")
    use_object_velocity: bpy.props.BoolProperty(name="Add Object Velocity", default=False)
    inflow_start: bpy.props.IntProperty(name="Start", default=1)
    inflow_end: bpy.props.IntProperty(name="End", default=250)
    inflow_substeps: bpy.props.IntProperty(
        name="Substeps", default=1, min=1, max=MAX_SUBSTEPS,
        description="Emissions per frame. More give a smoother stream from moving emitters and fast flows")
    display: bpy.props.EnumProperty(name="Display", default="AUTO", items=[
        ("AUTO", "Auto", "As the material preset suggests"),
        ("POINTS", "Points", "Particles as small spheres"),
        ("MESH", "Surface", "A smooth surface mesh (needs Bake Surface)")])
    surface_detail: bpy.props.FloatProperty(name="Detail", default=_SURFACE.detail, min=0.25, max=4.0)
    surface_smoothing: bpy.props.FloatProperty(name="Smoothing", default=_SURFACE.smoothing, min=0.5,
                                               max=4.0)
    surface_iso: bpy.props.FloatProperty(name="Threshold", default=_SURFACE.iso, min=0.05, max=0.95)


class MPMColliderSettings(bpy.types.PropertyGroup):
    domain: bpy.props.PointerProperty(name="Domain", type=bpy.types.Object, poll=_is_domain,
                                      description="Empty = the scene's only domain")
    friction: bpy.props.FloatProperty(name="Friction", default=0.5, min=0.0, max=2.0)


class MPMObjectSettings(bpy.types.PropertyGroup):
    role: bpy.props.EnumProperty(name="Role", items=ROLES, default="NONE", update=_role_changed)
    domain: bpy.props.PointerProperty(type=MPMDomainSettings)
    emitter: bpy.props.PointerProperty(type=MPMEmitterSettings)
    collider: bpy.props.PointerProperty(type=MPMColliderSettings)


def emitter_material(ob) -> MaterialParams:
    """The emitter's MaterialParams, named after the object."""
    e = ob.mpm.emitter
    values = {_TO_CORE_FIELD.get(f, f): getattr(e, f) for f in MATERIAL_FIELDS}
    return MaterialParams(name=ob.name, model=TO_CORE_MODEL[e.model], **values)


def effective_display(ob) -> str:
    """POINTS or MESH: the emitter's choice, else its preset's, else what suits the model."""
    e = ob.mpm.emitter
    if e.display != "AUTO":
        return e.display
    preset = presets.get(e.preset_id)
    if preset is not None:
        return "MESH" if preset.look.display == "mesh" else "POINTS"
    return "MESH" if e.model in ("ELASTIC", "PLASTIC") else "POINTS"


def is_modified(e) -> bool:
    """The emitter's values differ from its preset's (False when the preset is unknown)."""
    preset = presets.get(e.preset_id)
    if preset is None:
        return False
    m = preset.material
    if TO_CORE_MODEL[e.model] != m.model:
        return True
    return any(abs(getattr(e, f) - getattr(m, _TO_CORE_FIELD.get(f, f))) > 1e-6 * max(1.0, abs(getattr(e, f)))
               for f in used_fields(e.model))


# Earlier versions stored a preset enum plus relative tweaks (stiffness and firmness multipliers, clumping =
# hardening) and optional overrides. The enum's stored value is an index into this list; the defaults are
# those versions' property defaults.
_OLD_PRESET_IDS = ["jelly", "sand_dry", "snow_packed", "snow_powder", "whipped_cream"]
_OLD_KEYS = ("preset", "stiffness", "firmness", "clumping", "use_advanced")
_OLD_DEFAULTS = {"stiffness": 1.0, "firmness": 1.0, "density": 400.0, "friction_angle": 30.0, "clumping": 10.0}


def migrate_emitter(e) -> bool:
    """Turn old emitter settings into model + values with the same physics. True if anything changed."""
    stored = set(e.keys())
    if not stored & set(_OLD_KEYS):
        return False
    index = e.get("preset")
    valid = isinstance(index, int) and 0 <= index < len(_OLD_PRESET_IDS)
    preset_id = _OLD_PRESET_IDS[index] if valid else "whipped_cream"
    base = load_preset(preset_id).material  # these presets' physics are unchanged, apart from sand's angle

    def old(key):
        return e.get(key, _OLD_DEFAULTS[key])

    values = {"youngs_modulus": base.youngs_modulus * old("stiffness"),
              "poisson_ratio": base.poisson_ratio,
              "density": old("density"),
              "yield_stress": base.yield_stress * old("firmness"),
              "friction_angle": old("friction_angle"),
              "hardening": old("clumping"),
              "critical_compression": base.critical_compression,
              "critical_stretch": base.critical_stretch}
    if e.get("use_advanced", False):
        for field in ("youngs_modulus", "poisson_ratio", "yield_stress", "critical_compression", "critical_stretch"):
            if field in stored:
                values[field] = e[field]
    for key in _OLD_KEYS:
        if key in stored:
            del e[key]
    e.preset_id = preset_id
    for field, value in values.items():
        setattr(e, field, value)
    return True


def migrate_all() -> None:
    for ob in bpy.data.objects:
        if ob.type == "MESH" and migrate_emitter(ob.mpm.emitter):
            print(f"MPM: converted {ob.name!r} to the new material settings")


CLASSES = [MPMDomainSettings, MPMEmitterSettings, MPMColliderSettings, MPMObjectSettings]


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Object.mpm = bpy.props.PointerProperty(type=MPMObjectSettings)


def unregister():
    del bpy.types.Object.mpm
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
