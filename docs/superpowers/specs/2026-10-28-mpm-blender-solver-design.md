# GPU MPM Solver for Blender: Design

Date: 2026-10-28
Status: draft for review

## Intent

A Blender add-on that simulates "stuff" materials (whipped cream, snow, sand, and similar) with the Material Point Method on the GPU. It should feel like Blender's built-in fluid sim: set up a domain, give objects roles, pick material presets, bake, scrub. The author has not used MPM before, so the tool has to produce good-looking results from presets without the user understanding the physics.

It is a personal tool first, running on the author's own machines. The solver backend should be swappable later (e.g. for a native CUDA/Vulkan implementation, or to support other GPU vendors), so the first backend (NVIDIA Warp) sits behind an interface.

A throwaway spike (`spike/`) already proved the core idea. Its findings are summarized at the end and inform several decisions below. The spike code is not the starting point for the real implementation.

## Goals (v1)

- Simulate whipped cream, snow and sand (plus the variants listed under Presets) in one domain, with the materials interacting.
- Blender-native workflow in the Physics tab: domain, emitters (fill and inflow), colliders (static and animated).
- Material presets that bundle physics settings, a render shader, and a display mode (surface mesh or particles), with a few friendly parameters for tweaking.
- Bake on the GPU without blocking or risking Blender: progress, cancel, resume.
- Preview (fast, coarse) and final (full detail) quality from the same setup.
- Results visible in viewport and renderable in Cycles and EEVEE.
- Runs on Windows x64 and Linux x64 with NVIDIA GPUs (author's RTX 5080 on Windows, RTX 4090 on Linux).

## Decided by the author

- **Preview is a rough guide.** Preview bakes use lower resolution (and whatever else the builder finds makes them fast). They should show roughly the same motion and timing as the final bake, but are not expected to predict it faithfully; MPM behaviour changes with resolution, and that is accepted. No cross-resolution tuning or preview-vs-final regression tests in v1.
- **Materials stick where they touch (v1).** All materials share one grid, so different materials in contact move together; no sliding or separation between materials. The solver interface must not rule out adding inter-material frictional contact later (e.g. per-material grids or a contact pass) without changing its signatures.

## Non-goals (v1)

- AMD, Intel or Apple GPUs; macOS. (The backend interface must not preclude them.)
- Two-way coupling, where the material pushes rigid bodies back.
- Deforming colliders (armatures, shape keys, cloth). v1 colliders move rigidly via object animation.
- Sub-frame sampling for fast-moving inflows. Wanted later.
- Sparse grids. v1 uses a dense grid; the interface must not preclude sparse grids later.
- Water, or replacing FLIP-style liquid sims.
- Publishing on the Blender Extensions platform, render-farm/distributed baking.

## Anti-goals (would make it a failure even if it technically works)

- Blender freezes during a bake, or a GPU/solver error crashes Blender.
- Getting a decent result requires understanding physics parameters (Young's modulus, Poisson's ratio, yield stress).
- Particles visibly resting below or inside colliders (spike issue, see findings).
- Cache format or add-on code that depends on Warp, making the backend hard to swap.

## Constraints

- Blender 5.2 LTS (installed on the author's Windows machine; also 4.5 and 5.1 present). Packaged as a Blender extension (`blender_manifest.toml`) with bundled wheels. *My call: 5.2 LTS as the minimum; not tested on older versions.*
- NVIDIA Warp as the v1 backend. Spike used Warp 1.18.0 with CUDA 13.x; sm_120 (RTX 5080) works.
- Units are Blender scene units (metres). Gravity comes from the scene's gravity setting.
- Must work on both Windows and Linux from the same extension.

## User workflow

1. **Domain.** Add a box, enable "MPM Domain" in the Physics tab. The box bounds the simulation. Settings: resolution (cells along the longest axis), preview/final quality toggle, cache directory, frame range.
2. **Emitters.** Add a mesh, enable "MPM Emitter", choose:
   - *Fill*: the mesh volume becomes material at the start frame.
   - *Inflow*: material is emitted from the mesh volume continuously, with an initial velocity (direction and speed, optionally the object's own motion). Has a rate/active-frames control.
   Each emitter picks a material (a preset, possibly tweaked) and belongs to one domain. **The material lives on the emitter**: preset choice and tweaks are emitter properties, and each emitter is its own entry in the solver's material table and gets its own output object. Two emitters using the same preset are still two materials and two output objects, so they can have different tweaks and shaders. *My call (matches the author's description of configuring material per object).*
3. **Colliders.** Enable "MPM Collider" on any mesh. Static or animated by its object transform. Friction setting (sticky to slippery). Closed meshes are solid; open meshes (e.g. a plane) act as one-sided surfaces. *My call: the domain's own floor and walls are optional built-in colliders (toggle per side), so a simple scene needs no extra objects.*
4. **Bake.** Press Bake in the domain panel. A progress bar shows frame, time per frame and particle count. Cancel stops cleanly. Bake can resume from the last completed frame. Already-baked frames can be scrubbed while the bake continues.
5. **View and render.** Each emitter gets its own output object in a collection next to the domain. Scrubbing the timeline loads cached frames. Each output object has its preset's render material, which the user can replace. Surface meshing settings (smoothness, detail) are adjustable after baking without re-simulating.

## Presets

A preset bundles three things:
1. **Physics:** constitutive model and parameters.
2. **Look:** a render material (Principled BSDF based), working in Cycles and EEVEE.
3. **Display:** surface mesh or particles (spheres), plus surfacing defaults.

v1 preset list (*my call; tune in testing*):

| Preset | Model | Display |
|---|---|---|
| Whipped cream | Hencky elasticity + von Mises plasticity | Surface mesh |
| Snow, packed | Fixed-corotated + Stomakhin hardening | Surface mesh |
| Snow, powder | Same, softer, lower density, more brittle | Particles (or mesh, user switchable) |
| Sand, dry | Hencky + Drucker-Prager | Particles |
| Jelly | Pure elastic (Neo-Hookean or fixed-corotated) | Surface mesh |

User-facing tweaks are friendly names mapped to physical parameters, e.g. *Firmness* (yield stress), *Stiffness*, *Friction angle*, *Density*, *Clumpiness/Hardening*. An "Advanced" subpanel exposes the raw physical values.

Presets are defined in data files (not hard-coded) so new ones can be added without code changes. *My call.*

Presets need to work across scene scales; the spike showed that cream with a fixed yield stress holds shape at one size and collapses at another (see findings). The builder decides how to handle this (e.g. document a reference scale, scale-aware defaults, or a "scale" hint on the domain).

## Architecture

Three layers with a hard boundary between each:

```
Blender add-on (bpy)      UI, object roles, presets, scene export, cache loading, display
        │  scene description on disk (JSON + .npy)
        ▼
Bake runner (subprocess)  reads scene description, drives the solver, writes the cache
        │  Python solver interface, numpy arrays only
        ▼
Backend (Warp in v1)      GPU kernels: P2G, grid update, colliders, G2P, plasticity
```

**Add-on.** Pure bpy. Never imports Warp. Exports the scene description, launches the runner, polls progress, loads cache frames.

**Scene description.** Written to the cache directory at bake start. Contains: domain bounds and resolution, material table (resolved physical parameters), emitters (sampled initial particles for fills; mesh data and emission parameters for inflows), colliders (mesh in local space plus per-frame world transforms for the bake range), gravity, frame range, fps. The format is the contract between add-on and runner, and the runner is runnable from the command line on a scene description alone (for debugging, benchmarking and tests). Particle sampling of emitter volumes happens in the add-on or the runner (builder's call), but the result must be deterministic for a given seed.

**Bake runner.** Separate process (`subprocess`, using Blender's bundled Python interpreter and the extension's bundled wheels), so a GPU fault cannot take Blender down and Blender stays responsive. Reports progress via a small status file or stdout lines the add-on polls with a timer. Supports cancel (signal or flag file) and resume (full solver state checkpointed at least at the last completed frame).

**Solver interface.** A Python abstract interface, numpy in and out, no Warp types in signatures. At minimum: initialize from domain and material table; add particles (positions, velocities, material ids); set collider state for a time (or a callback/time-sampled transforms); advance by a frame duration (backend chooses substeps); read particle fields (positions; optionally velocities and other attributes); save and load full state for resume. A native backend later implements the same interface (as a Python extension module).

**Warp backend.** MLS-MPM with APIC transfers and quadratic B-splines on a dense grid; per-particle material id indexing a material table on the GPU (as in the spike). Adaptive timestep from a CFL condition plus a stiffness-based limit, recomputed per substep or per frame. Colliders as signed distance fields: Warp's mesh queries (`wp.Mesh`, signed point queries) against the collider mesh in its local space, with collider velocity from the transform change between substep times. CUDA graph capture where it helps.

**Cache.** Per frame, per material: particle positions (float32), plus attributes useful for shading (velocity at least; *my call*). Format is backend-independent and documented. Readable while the bake is still writing later frames. Preview and final bakes go to separate cache subdirectories so a preview never overwrites a final bake. *My call.*

**Display.** One output object per material, driven by a frame-change handler that loads the cached frame. Points are displayed via Geometry Nodes (mesh-to-points or a PointCloud, builder's call). Surface meshing for mesh-display materials runs in Geometry Nodes (Points to Volume, Volume to Mesh, smoothing) so it stays adjustable without re-simulating. *My call: start with Geometry Nodes surfacing; consider an external mesher like splashsurf only if quality or speed is not good enough.* For final renders, the user may want baked mesh output; builder decides whether v1 includes a "bake surface" step.

## Requirements carried over from the spike

- **Contact accuracy:** particles resting on a collider (including the domain floor) must sit on the visible collider surface, within a fraction of a grid cell. The spike's grid-only boundary let particles settle 0–1 cell inside the boundary.
- **Settling at start:** materials should not visibly sag or shift at frame 1 under their own weight. The spike's snow slab moved at least 1 cm (of a 7 cm layer) at the start. The builder decides how (e.g. stiffer initial state, gravity ramp, pre-settle frames).
- **Stability:** a bake that produces NaN/inf stops with a clear message naming the frame, rather than writing garbage.

## Performance targets

On the RTX 5080, a scene of around 200–500k particles at 128³ should bake at least as fast as the spike (0.43 s/frame at 211k particles, 0.72 s/frame at 456k, with a fixed 5e-5 s timestep). With adaptive timestepping, expect it to be faster for most scenes. Preview quality should give near-interactive iteration (seconds per second of animation for simple scenes). These are goals, not hard requirements.

## Testing

- Solver interface and Warp backend are testable from plain Python (pytest) without Blender, including small regression scenes (e.g. sand column collapse runout within an expected range, a ball at rest on the floor not penetrating it, mass conservation).
- Add-on tested headless (`blender -b --python ...`) on a small scene: set up, bake a few frames, load cache, check object counts and positions.
- Both test levels must run on Windows and Linux.

## Left to the builder

- Package and module layout, naming, the add-on's name.
- Exact scene-description and cache file formats (within the requirements above).
- How progress/cancel communicate between processes.
- Whether particle sampling happens in the add-on or the runner.
- Timestep strategy details, CUDA graph use.
- How presets handle scene scale.
- Point display mechanism, whether v1 includes a baked-mesh export.
- Starting-sag mitigation.

## Spike findings (for reference)

From `spike/` (Warp 1.18, RTX 5080, Blender 5.2):

- Mixed scene (cream ball, sand column, thrown snowball), 211k particles, 128³ dense grid, 834 substeps per frame: about 0.43 s/frame; about 40 s for a 4 s clip. First run spends ~6 s compiling kernels (cached afterwards).
- Rolling ball through a 7 cm snow layer, 456k particles: about 0.72 s/frame. Kinematic ball collider with frictional contact worked: the ball packed a trench (snow in the track compacted to about half height).
- Warp was pleasant: a full 3-material solver in ~250 lines of Python, correct on the first run.
- Cream at yield stress 400 Pa collapsed a 22 cm ball into a puddle; 2500 Pa held a sagging dome. Snow thrown at 4 m/s shattered into powder. Packed snow at the spike's parameters holds unrealistically sharp trench walls. Parameters are scale- and taste-sensitive: presets need tuning in real scenes.
- Particles-as-beads do not read as cream; surface meshing is required for cream-like materials.
- Cream and snow shaders looked too similar.
- Untested: the 4090 on Linux, inflow, meshing, animated mesh colliders (only an analytic sphere was tested).
