"""THROWAWAY SPIKE: render a few frames headless for a quick look.

    blender -b --factory-startup --python spike/render_check.py -- <out_dir> [frames...]
"""

import os
import sys

import bpy

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
out_dir = argv[0] if argv else os.path.join(os.path.dirname(__file__), "renders")
frames = [int(f) for f in argv[1:]] or [1, 12, 24, 48, 96]

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "blender_load.py")).read(),
     {"__file__": os.path.join(os.path.dirname(os.path.abspath(__file__)), "blender_load.py"), "__name__": "spike"})

scene = bpy.context.scene
scene.render.engine = "BLENDER_EEVEE"
scene.render.resolution_x, scene.render.resolution_y = 960, 540
os.makedirs(out_dir, exist_ok=True)
for f in frames:
    scene.frame_set(f)
    scene.render.filepath = os.path.join(out_dir, f"frame_{f:04d}.png")
    bpy.ops.render.render(write_still=True)
