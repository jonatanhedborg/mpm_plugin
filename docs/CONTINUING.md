# Continuing on another machine

Notes for picking this project up on a fresh checkout. Last updated 2026-10-28.

## What's in git

Everything committed: design and plans (`docs/superpowers/`), the solver core (`src/mpm_core`), the
surfacer (`src/mpm_surface`), the Blender add-on (`blender_addon/mpm_blender`), tests (`tests/`,
`tests_blender/`), tools, examples, and the original spike (`spike/`).

Branches:

- `master`: all the real work. Working directly on master is fine in this repo.
- `spike/jelly-fracture`: throwaway spike of breakable jelly (brittle damage model on a wedge). Kept for
  reference only; nothing from it is in master.

## What isn't in git (recreate or copy)

| Thing | How to get it back |
|---|---|
| Python environment `.venv` | see Setup below |
| In-Blender test packages `out/blender_site` | `tools/setup_blender_test_env.py` |
| Extension zips `dist/`, wheels in `blender_addon/mpm_blender/wheels/` | `tools/build_extension.py` |
| Bakes and renders in `out/` (mix, ball, tests, mix256, spike_fracture) | regenerate (below) or copy `out/` across |
| Claude Code's saved preference "work on master in this repo" | lives in the old machine's Claude settings; say it again |

## Setup

Needs an NVIDIA GPU (CUDA driver), [uv](https://docs.astral.sh/uv/), and Blender 5.2.

Windows:

```
uv venv --python 3.12 .venv
uv pip install --python .venv -e ".[dev]"
.venv/Scripts/python -m pytest                          # core tests
.venv/Scripts/python tools/setup_blender_test_env.py    # once
.venv/Scripts/python tools/blender_tests.py             # add-on tests inside headless Blender
```

Linux: the same with `.venv/bin/python`. `tools/blender_tests.py` and `tools/build_extension.py`
find Blender through the `BLENDER` environment variable or `blender` on `PATH`.

Build the extension and install it in Blender (Edit > Preferences > Get Extensions > dropdown >
Install from Disk):

```
.venv/Scripts/python tools/build_extension.py    # -> dist/mpm_blender-0.1.0-{windows,linux}_x64.zip
```

Regenerate the example and test bakes:

```
.venv/Scripts/python examples/scene_mix.py out/mix && .venv/Scripts/python examples/bench.py out/mix
.venv/Scripts/python examples/scene_ball.py out/ball && .venv/Scripts/python examples/bench.py out/ball
.venv/Scripts/python examples/test_scenes.py                 # all GPU test setups -> out/tests
.venv/Scripts/python -m mpm_surface out/mix                  # surface meshes
```

View caches in Blender with `tools/blender_view_cache.py` (set `CACHE_DIR` at the top, or leave it
empty for `out/tests`).

## Open items

Needs a decision or a person:

- **Manual check of the add-on in Blender, on both machines** (Windows RTX 5080, Linux RTX 4090): set
  up a mixed scene by hand (cream, sand, snow, a collider), bake a preview while scrubbing, cancel and
  resume, bake final, bake surface, render a frame in Cycles. Panel layout and feel are only checked
  here.
- **Linux run on the 4090**: `pytest`, `tools/blender_tests.py`, `examples/bench.py out/mix`; add the
  numbers to the README benchmark table.
- **Windows path length** (undecided): Warp's wheel has deeply nested files. Normal profile paths fit
  (about 200 of 260 characters), very long Windows profile paths would fail to install the extension.
  Options: accept, strip Warp's nested license folder in the build, or document long-path support.

Known limitations (see README):

- Friction on steep slanted colliders is weaker than its setting (30° with friction 1.0 slides; flat
  colliders and domain walls are fine). Needs a better contact model.
- Jelly is undamped: it keeps wobbling and creeps on slopes. Needs damping in the preset/model.

Ideas discussed, not started:

- Breakable jelly: a proper damage model, damping, and cracked material that settles instead of
  turning to dust (see the `spike/jelly-fracture` branch for the first try).
- Surface quality: mesh smoothing to remove the particle ripple pattern (cheap), anisotropic surfacing
  for flatter, crisper surfaces (bigger).
- Performance: collider queries make the rolling-ball scene slower than the spike; a native backend
  later behind the same `Solver` interface.

## Deferred minor issues from the code reviews

Solver core and bake runner:

- The collider `closed` flag is trusted from the scene file rather than checked.
- Inflow "particles outside the domain were dropped" warning repeats every frame (count in the message).
- A fresh bake with a locked frame file (Windows) gives a confusing error.
- No substep cap, and no up-front GPU memory estimate for the simulation grid.
- Validation gaps: fps ≤ 0, negative friction, inflow end before start, mismatched array lengths in
  `add_particles`, empty collider meshes.
- Inflow re-uploads all particles every frame; checkpoints are uncompressed every frame (~54 MB/frame at
  500k particles).
- A fill emitter overlapping a collider at frame 1 may cause a spike (unverified).

Surfacer:

- Particle spacing is read from `scene/scene.json` rather than `cache.json`, and isn't in the skip key.
- No surfacer version in the skip key, so algorithm changes don't re-mesh existing frames.
- Settings file quirks: `detail=2` vs `2.0` re-meshes needlessly; an unknown key stops the run.
- One failing frame (e.g. over the memory budget) stops the whole run instead of being skipped.
- `SurfaceResult("error")` is never returned (errors re-raise); the CLI's error branch is dead.
- `wp.MarchingCubes` is deprecated in Warp 1.18 (prints a warning).
- The splat loops over a cube for a spherical kernel; marching cubes is re-created per call.
- A mesh write can fail if Blender holds the file open for more than a second.
- `--frames 48-1` silently does nothing.

Blender add-on:

- The progress text shows only "Frame X / Y" (no time per frame or particle count).
- Playback reads velocity arrays it doesn't need (twice the disk reads).
- Point size is fixed at 0.3 × cell size and not updated after a re-bake at another resolution.
- "MPM Points" node groups pile up; removed outputs leave orphan meshes and node groups.
- The current frame's output geometry is saved in the .blend and every undo step.
- Renaming an emitter recreates its output object, so a material you assigned to it is lost.
- Cancel can't force-stop a hung GPU process.
- Opening another file stops running bakes and can wait up to 2 s per bake.
- Switching scenes mid-bake confuses the progress bar.
- An inflow entirely outside the frame range emits nothing, without a warning.
- With two domains and an emitter without a domain set, the error has no hint about the cause.
- A mirrored (negative scale) domain is refused as "rotated".
- A relative cache path without `//` resolves against the working directory.
- The bake process gets Blender's full Python search path, including user script folders (low risk).
