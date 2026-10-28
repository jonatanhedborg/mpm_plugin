import bpy


def test_object_has_mpm_settings(clean_scene):
    from conftest import add_cube

    ob = add_cube("Thing")
    assert hasattr(bpy.types.Object, "mpm")
    assert ob.mpm.role == "NONE"
