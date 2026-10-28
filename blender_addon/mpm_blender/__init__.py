"""MPM Solver for Blender: GPU Material Point Method simulation of cream, snow, sand, jelly and more."""

import bpy

from . import display, operators, props, ui

MODULES = [props, operators, display, ui]


def register():
    for module in MODULES:
        module.register()
    # bpy.data isn't available while registering; convert an already open file's old settings right after
    bpy.app.timers.register(props.migrate_all, first_interval=0.0)


def unregister():
    if bpy.app.timers.is_registered(props.migrate_all):
        bpy.app.timers.unregister(props.migrate_all)
    for module in reversed(MODULES):
        module.unregister()
