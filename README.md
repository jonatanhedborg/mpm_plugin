# MPM solver

GPU Material Point Method simulation for Blender: whipped cream, snow, sand, jelly.
Design: `docs/superpowers/specs/2026-10-28-mpm-blender-solver-design.md`.

This repository currently contains the Blender-independent core, `mpm_core`:

- `mpm_core.materials`: material parameters and presets (`src/mpm_core/presets/*.toml`)
- `mpm_core.scene`: the scene description the Blender add-on writes
- `mpm_core.solver`: the backend interface; `mpm_core.backends.warp` implements it with NVIDIA Warp
- `mpm_core.cache`: the cache format (layout documented in the module docstring)
- `mpm_core.bake`: the bake runner, `python -m mpm_core.bake CACHE_DIR [--resume]`

The Blender add-on is the next step. `spike/` is the original throwaway prototype.

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
`CACHE_DIR`, and run it.

## Benchmarks

96 frames at 24 fps, 128³ grid, including cache and checkpoint writes.

| Scene | Particles | RTX 5080 (Windows) | RTX 4090 (Linux) |
|---|---|---|---|
| mix | 204k | 389 ms/frame | not run yet |
| ball | 454k | 1088 ms/frame | not run yet |

The spike (fixed timestep, analytic ball) ran mix at ~430 ms/frame and ball at ~720 ms/frame. The ball
scene is slower here because it does general mesh distance queries against the collider.

## Known limitations

- On slanted mesh colliders, friction is weaker than the setting implies: at 30° with friction 1.0,
  snow and cream still slide (10° holds). Flat colliders and domain walls are fine.
- Jelly (purely elastic, undamped) keeps vibrating and creeps on slopes regardless of friction.
