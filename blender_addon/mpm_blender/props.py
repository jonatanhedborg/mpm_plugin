"""Per-object MPM settings."""

import bpy

ROLES = [
    ("NONE", "None", "Not part of an MPM simulation"),
    ("DOMAIN", "Domain", "The box the simulation runs in"),
    ("EMITTER", "Emitter", "Fills the domain with material, once or continuously"),
    ("COLLIDER", "Collider", "Material bumps into this object"),
]


class MPMObjectSettings(bpy.types.PropertyGroup):
    role: bpy.props.EnumProperty(name="Role", items=ROLES, default="NONE")


CLASSES = [MPMObjectSettings]


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Object.mpm = bpy.props.PointerProperty(type=MPMObjectSettings)


def unregister():
    del bpy.types.Object.mpm
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
