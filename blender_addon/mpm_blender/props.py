"""Per-object MPM settings: one ``Object.mpm`` group with a role and per-role settings."""

import bpy

from mpm_core.materials import list_presets, load_preset
from mpm_core.tweaks import apply_tweaks, tweaks_for
from mpm_surface.settings import SurfaceSettings  # no warp import

ROLES = [
    ("NONE", "None", "Not part of an MPM simulation"),
    ("DOMAIN", "Domain", "The box the simulation runs in"),
    ("EMITTER", "Emitter", "Fills the domain with material, once or continuously"),
    ("COLLIDER", "Collider", "Material bumps into this object"),
]
PRESET_ITEMS = [(p, load_preset(p).label, "") for p in list_presets()]
_SURFACE = SurfaceSettings()
ADVANCED_FIELDS = ("youngs_modulus", "poisson_ratio", "yield_stress", "critical_compression", "critical_stretch")


def _is_domain(self, obj):
    return obj.mpm.role == "DOMAIN"


def _reset_to_preset(self, context):
    """Changing the preset resets its tweaks and advanced values to the preset's own."""
    m = load_preset(self.preset).material
    self.stiffness = 1.0
    self.firmness = 1.0
    self.density = m.density
    self.friction_angle = m.friction_angle_deg or 30.0
    self.clumping = m.hardening or 10.0
    for field in ADVANCED_FIELDS:
        setattr(self, field, getattr(m, field))


class MPMDomainSettings(bpy.types.PropertyGroup):
    resolution: bpy.props.IntProperty(name="Resolution", default=128, min=16, max=1024,
                                      description="Grid cells along the domain's longest side (final quality)")
    quality: bpy.props.EnumProperty(name="Quality", default="PREVIEW", items=[
        ("PREVIEW", "Preview", "Half resolution, for quick iteration"),
        ("FINAL", "Final", "Full resolution")])
    particles_per_cell: bpy.props.EnumProperty(name="Particles per Cell", default="8", items=[
        ("1", "1", "Fastest, coarse"), ("8", "8", "Standard"), ("27", "27", "Dense, slow")])
    cache_dir: bpy.props.StringProperty(name="Cache Folder", subtype="DIR_PATH", default="",
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
    preset: bpy.props.EnumProperty(name="Material", items=PRESET_ITEMS, default="whipped_cream",
                                   update=_reset_to_preset)
    stiffness: bpy.props.FloatProperty(name="Stiffness", default=1.0, min=0.1, max=10.0)
    density: bpy.props.FloatProperty(name="Density", default=400.0, min=10.0, max=5000.0, subtype="NONE")
    firmness: bpy.props.FloatProperty(name="Firmness", default=1.0, min=0.05, max=20.0)
    friction_angle: bpy.props.FloatProperty(name="Friction Angle", default=30.0, min=0.0, max=60.0)
    clumping: bpy.props.FloatProperty(name="Clumping", default=10.0, min=0.0, max=30.0)
    use_advanced: bpy.props.BoolProperty(name="Override Physical Values", default=False)
    youngs_modulus: bpy.props.FloatProperty(name="Young's Modulus", default=1.0e5, min=1.0, unit="NONE")
    poisson_ratio: bpy.props.FloatProperty(name="Poisson Ratio", default=0.3, min=0.0, max=0.49)
    yield_stress: bpy.props.FloatProperty(name="Yield Stress", default=2500.0, min=0.0)
    critical_compression: bpy.props.FloatProperty(name="Critical Compression", default=0.0, min=0.0, max=1.0,
                                                  precision=4)
    critical_stretch: bpy.props.FloatProperty(name="Critical Stretch", default=0.0, min=0.0, max=1.0, precision=4)
    velocity: bpy.props.FloatVectorProperty(name="Velocity", size=3, subtype="VELOCITY", unit="VELOCITY")
    use_object_velocity: bpy.props.BoolProperty(name="Add Object Velocity", default=False)
    inflow_start: bpy.props.IntProperty(name="Start", default=1)
    inflow_end: bpy.props.IntProperty(name="End", default=250)
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
    role: bpy.props.EnumProperty(name="Role", items=ROLES, default="NONE")
    domain: bpy.props.PointerProperty(type=MPMDomainSettings)
    emitter: bpy.props.PointerProperty(type=MPMEmitterSettings)
    collider: bpy.props.PointerProperty(type=MPMColliderSettings)


def emitter_material(ob):
    """The emitter's MaterialParams: preset + its tweaks (+ advanced overrides), named after the object."""
    e = ob.mpm.emitter
    values = {spec.name: getattr(e, spec.name) for spec in tweaks_for(e.preset)}
    overrides = {field: getattr(e, field) for field in ADVANCED_FIELDS} if e.use_advanced else None
    return apply_tweaks(e.preset, values, overrides, name=ob.name)


def effective_display(ob) -> str:
    """POINTS or MESH."""
    e = ob.mpm.emitter
    if e.display != "AUTO":
        return e.display
    return "MESH" if load_preset(e.preset).look.display == "mesh" else "POINTS"


CLASSES = [MPMDomainSettings, MPMEmitterSettings, MPMColliderSettings, MPMObjectSettings]


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Object.mpm = bpy.props.PointerProperty(type=MPMObjectSettings)


def unregister():
    del bpy.types.Object.mpm
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
