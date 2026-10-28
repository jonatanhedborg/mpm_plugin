"""Physics tab panels."""

import bpy

from . import presets
from .operators import is_running
from .props import MODELS, is_modified, used_fields


class _MPMPanel:
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "physics"


class MPM_PT_main(_MPMPanel, bpy.types.Panel):
    bl_label = "MPM"

    @classmethod
    def poll(cls, context):
        return context.object is not None and context.object.type == "MESH"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.prop(context.object.mpm, "role")


class _RolePanel(_MPMPanel):
    bl_parent_id = "MPM_PT_main"
    role = ""

    @classmethod
    def poll(cls, context):
        ob = context.object
        return ob is not None and ob.type == "MESH" and ob.mpm.role == cls.role


# ---------------------------------------------------------------------------- domain

class MPM_PT_domain_settings(_RolePanel, bpy.types.Panel):
    bl_label = "Domain"
    role = "DOMAIN"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        d = context.object.mpm.domain
        layout.prop(d, "resolution")
        layout.prop(d, "quality")
        layout.prop(d, "particles_per_cell")


class MPM_PT_domain_walls(_RolePanel, bpy.types.Panel):
    bl_label = "Walls"
    role = "DOMAIN"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        d = context.object.mpm.domain
        grid = layout.grid_flow(columns=3, align=True)
        for side in ("x_neg", "y_neg", "z_neg", "x_pos", "y_pos", "z_pos"):
            grid.prop(d, f"wall_{side}", toggle=True)
        layout.use_property_split = True
        layout.prop(d, "wall_friction")


class MPM_PT_domain_bake(_RolePanel, bpy.types.Panel):
    bl_label = "Cache & Bake"
    role = "DOMAIN"

    def draw(self, context):
        layout = self.layout
        d = context.object.mpm.domain
        col = layout.column()
        col.use_property_split = True
        col.prop(d, "cache_dir")
        col.prop(d, "auto_surface")
        running = is_running(context.object)  # not the saved status: it can be stale after a reload
        if running:
            layout.progress(factor=d.progress, type="BAR", text=d.status_message or d.status.title())
            layout.operator("mpm.cancel", icon="CANCEL")
            return
        row = layout.row(align=True)
        row.operator("mpm.bake", icon="PHYSICS").resume = False
        row.operator("mpm.bake", text="Resume", icon="PLAY").resume = True
        row.operator("mpm.free_bake", text="", icon="TRASH")
        row = layout.row(align=True)
        row.operator("mpm.bake_surface", icon="MESH_DATA")
        row.operator("mpm.free_surface", text="", icon="TRASH")
        if d.status_message:
            layout.label(text=d.status_message, icon="ERROR" if d.status == "error" else "INFO")
        if not context.scene.render.use_lock_interface:
            box = layout.box()
            box.label(text="Rendering can crash without Lock Interface", icon="ERROR")
            box.prop(context.scene.render, "use_lock_interface")


# ---------------------------------------------------------------------------- emitter

class MPM_PT_emitter(_RolePanel, bpy.types.Panel):
    bl_label = "Emitter"
    role = "EMITTER"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        e = context.object.mpm.emitter
        layout.prop(e, "domain")
        layout.prop(e, "kind")


def preset_text(e) -> str:
    """What the preset menu button shows."""
    preset = presets.get(e.preset_id)
    if preset is None:
        return f"{e.preset_label} (not on this computer)" if e.preset_id and e.preset_label else "Custom"
    return f"{preset.label} (edited)" if is_modified(e) else preset.label


class MPM_MT_material_presets(bpy.types.Menu):
    bl_label = "Material Presets"

    def draw(self, context):
        row = self.layout.row()
        everything = presets.all_presets().values()
        for _, core_model, label, _ in MODELS:
            col = row.column()
            col.label(text=label)
            for preset in sorted((p for p in everything if not p.is_user and p.material.model == core_model),
                                 key=lambda p: p.label):
                col.operator("mpm.apply_preset", text=preset.label).preset_id = preset.id
        mine = [p for p in everything if p.is_user]
        if mine:
            col = row.column()
            col.label(text="Your Presets")
            for preset in mine:
                col.operator("mpm.apply_preset", text=preset.label).preset_id = preset.id


class MPM_PT_emitter_material(_RolePanel, bpy.types.Panel):
    bl_label = "Material"
    role = "EMITTER"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        e = context.object.mpm.emitter
        split = layout.split(factor=0.4)
        split.alignment = "RIGHT"
        split.label(text="Preset")
        row = split.row(align=True)
        row.menu("MPM_MT_material_presets", text=preset_text(e))
        row.operator("mpm.save_preset", text="", icon="ADD")
        row.operator("mpm.delete_preset", text="", icon="REMOVE")
        layout.prop(e, "model")
        col = layout.column()
        for field in used_fields(e.model):
            col.prop(e, field)


class MPM_PT_emitter_inflow(_RolePanel, bpy.types.Panel):
    bl_label = "Inflow"
    role = "EMITTER"

    @classmethod
    def poll(cls, context):
        return super().poll(context) and context.object.mpm.emitter.kind == "INFLOW"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        e = context.object.mpm.emitter
        layout.prop(e, "velocity")
        layout.prop(e, "use_object_velocity")
        col = layout.column(align=True)
        col.prop(e, "inflow_start", text="Frame Start")
        col.prop(e, "inflow_end", text="End")
        layout.prop(e, "inflow_substeps")


class MPM_PT_emitter_display(_RolePanel, bpy.types.Panel):
    bl_label = "Display & Surface"
    role = "EMITTER"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        e = context.object.mpm.emitter
        layout.prop(e, "display")
        col = layout.column()
        col.prop(e, "surface_detail")
        col.prop(e, "surface_smoothing")
        col.prop(e, "surface_iso")


# ---------------------------------------------------------------------------- collider

class MPM_PT_collider_settings(_RolePanel, bpy.types.Panel):
    bl_label = "Collider"
    role = "COLLIDER"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        c = context.object.mpm.collider
        layout.prop(c, "domain")
        layout.prop(c, "friction")


PANELS = [MPM_PT_main, MPM_PT_domain_settings, MPM_PT_domain_walls, MPM_PT_domain_bake, MPM_PT_emitter,
          MPM_PT_emitter_material, MPM_PT_emitter_inflow, MPM_PT_emitter_display, MPM_PT_collider_settings]
CLASSES = [MPM_MT_material_presets, *PANELS]


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
