# MPM solver

GPU Material Point Method simulation for Blender: whipped cream, snow, sand, jelly.
Design: `docs/superpowers/specs/2026-10-28-mpm-blender-solver-design.md`.
Picking this up on another machine: `docs/CONTINUING.md`.

## Blender add-on

Install: build the extension (see Building below), then in Blender 5.2 go to Edit → Preferences →
Get Extensions → (dropdown) Install from Disk, and pick `dist/mpm_blender-0.1.0-windows_x64.zip`
(or `-linux_x64.zip`). Needs an NVIDIA GPU.

Workflow, all in the Properties editor's Physics tab, MPM panel:

1. Add a box, set Role to **Domain** (it becomes a wireframe and is hidden from renders). Set resolution,
   quality (Preview = half resolution), walls.
2. Add meshes, set Role to **Emitter**: Fill or Inflow, pick a material preset, tweak it. Display &
   Surface picks points or a surface mesh (Auto follows the preset). Inflow **Substeps** emit several
   times per frame, for smooth streams from moving emitters and fast flows (4 is a good start).
3. Optionally add meshes with Role **Collider** (animated objects work).
4. Save the .blend (the cache goes to `//mpm_cache/<domain>` by default), then **Bake** on the domain.
   Frames appear while it bakes; Cancel / Resume / Free next to it. With "Surface After Bake" on,
   surface meshes are made right after; **Bake Surface** re-makes them after changing surface settings.

Output objects appear in a collection `MPM <domain>`, with materials `MPM <preset>` that you can edit
freely (re-bakes keep them). Surface meshes carry a `velocity` attribute for motion blur.

Add-on tests run inside headless Blender:

```
.venv/Scripts/python tools/setup_blender_test_env.py    # once
.venv/Scripts/python tools/blender_tests.py             # pytest args pass through
```

## Building

```
.venv/Scripts/python tools/build_extension.py          # -> dist/mpm_blender-0.1.0-{windows,linux}_x64.zip
```

## Core libraries

The add-on runs on the Blender-independent core, `mpm_core`:

- `mpm_core.materials`: material parameters and presets (`src/mpm_core/presets/*.toml`)
- `mpm_core.scene`: the scene description the Blender add-on writes
- `mpm_core.solver`: the backend interface; `mpm_core.backends.warp` implements it with NVIDIA Warp
- `mpm_core.cache`: the cache format (layout documented in the module docstring)
- `mpm_core.bake`: the bake runner, `python -m mpm_core.bake CACHE_DIR [--resume]`

It also contains the surfacer, `mpm_surface`, which turns a baked particle cache into meshes.

`spike/` is the original throwaway prototype.

## Setup

Needs an NVIDIA GPU (CUDA) and [uv](https://docs.astral.sh/uv/).

```
uv venv --python 3.12 .venv
uv pip install --python .venv -e ".[dev]"
```

## Tests

```
.venv/Scripts/python -m pytest        # Windows
.venv/bin/python -m pytest            # Linux
```

GPU tests are marked `gpu` and skipped when no CUDA device is found.

## Examples

```
.venv/Scripts/python examples/scene_mix.py out/mix     # cream ball, sand column, thrown snowball
.venv/Scripts/python examples/scene_ball.py out/ball   # ball rolling through packed snow
.venv/Scripts/python examples/bench.py out/mix
```

To look at a cache in Blender, open `tools/blender_view_cache.py` in the Text Editor, set
`CACHE_DIR`, and run it. `CACHE_DIR` can also be a folder of caches, which are then laid out side
by side, each with its name, domain box and animated colliders.

`examples/test_scenes.py` bakes the GPU test setups (plus a few known-issue variants) into
`out/tests/`, so you can see what the tests check: point the viewer at `out/tests`.

## Surfacing

```
.venv/Scripts/python -m mpm_surface out/mix            # mesh every emitter, every baked frame
.venv/Scripts/python -m mpm_surface out/mix --frames 1-48
```

Meshes go to `<cache>/surface/meshes/<folder>/frame_NNNN.npz` (verts, outward triangles, per-vertex
velocity), with `surface/meshes.json` mapping emitter names to folders. Only one surfacer run per
cache at a time; a fresh bake deletes `surface/`, so stop the surfacer before re-baking. Settings per emitter (`enabled`, `detail`, `smoothing`, `iso`) are read from
`<cache>/surface_settings.json` (see `mpm_surface/settings.py`); emitters not listed use the
defaults. Re-running only re-meshes frames whose particles or settings changed. The viewer shows
surfaced emitters as meshes, the rest as points.

## Benchmarks

96 frames at 24 fps, 128³ grid, including cache and checkpoint writes.

| Scene | Particles | RTX 5080 (Windows) | RTX 4090 (Linux) |
|---|---|---|---|
| mix | 204k | 389 ms/frame | 280 ms/frame |
| ball | 454k | 1088 ms/frame | 673 ms/frame |

Surfacing the mix scene (default settings): 30 ms/frame on the RTX 5080, 24 ms/frame on the RTX 4090.

The spike (fixed timestep, analytic ball) ran mix at ~430 ms/frame and ball at ~720 ms/frame. The ball
scene is slower here because it does general mesh distance queries against the collider.

## Known limitations

- On slanted mesh colliders, friction is weaker than the setting implies: at 30° with friction 1.0,
  snow and cream still slide (10° holds). Flat colliders and domain walls are fine.
- Jelly (purely elastic, undamped) keeps vibrating and creeps on slopes regardless of friction.
