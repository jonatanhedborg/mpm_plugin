"""Physics tab panels."""

import bpy

from mpm_core.tweaks import tweaks_for


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
        running = d.status in ("starting", "running", "surfacing")
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


# ---------------------------------------------------------------------------- emitter

class MPM_PT_emitter_material(_RolePanel, bpy.types.Panel):
    bl_label = "Emitter"
    role = "EMITTER"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        e = context.object.mpm.emitter
        layout.prop(e, "domain")
        layout.prop(e, "kind")
        layout.prop(e, "preset")
        col = layout.column()
        col.active = not e.use_advanced
        for spec in tweaks_for(e.preset):
            col.prop(e, spec.name, text=spec.label)


class MPM_PT_emitter_advanced(_RolePanel, bpy.types.Panel):
    bl_label = "Advanced"
    bl_parent_id = "MPM_PT_emitter_material"
    bl_options = {"DEFAULT_CLOSED"}
    role = "EMITTER"

    def draw_header(self, context):
        self.layout.prop(context.object.mpm.emitter, "use_advanced", text="")

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        e = context.object.mpm.emitter
        layout.active = e.use_advanced
        for field in ("youngs_modulus", "poisson_ratio", "yield_stress", "critical_compression",
                      "critical_stretch"):
            layout.prop(e, field)


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


PANELS = [MPM_PT_main, MPM_PT_domain_settings, MPM_PT_domain_walls, MPM_PT_domain_bake, MPM_PT_emitter_material,
          MPM_PT_emitter_advanced, MPM_PT_emitter_inflow, MPM_PT_emitter_display, MPM_PT_collider_settings]


def register():
    for cls in PANELS:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(PANELS):
        bpy.utils.unregister_class(cls)
