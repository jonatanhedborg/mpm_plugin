"""MPM Solver for Blender: GPU Material Point Method simulation of cream, snow, sand and jelly."""

from . import operators, props, ui

MODULES = [props, operators, ui]


def register():
    for module in MODULES:
        module.register()


def unregister():
    for module in reversed(MODULES):
        module.unregister()
