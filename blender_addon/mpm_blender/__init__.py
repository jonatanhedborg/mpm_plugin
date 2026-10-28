"""MPM Solver for Blender: GPU Material Point Method simulation of cream, snow, sand and jelly."""

from . import props

MODULES = [props]


def register():
    for module in MODULES:
        module.register()


def unregister():
    for module in reversed(MODULES):
        module.unregister()
