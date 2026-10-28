# MPM Surfacer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A separate surfacing stage, `mpm_surface`, that turns a baked particle cache into a per-emitter mesh cache on the GPU, runnable from Blender (subprocess) or the command line, re-runnable without re-simulating.

**Architecture:** `mpm_surface` reads the particle cache (via `mpm_core.cache.CacheReader` and the cache's `scene/scene.json`), splats each emitter's particles into a density field on the GPU, extracts the iso-surface with `wp.MarchingCubes`, samples a velocity field onto the vertices, and writes `<cache>/surface/<emitter>/frame_NNNN.npz`. A runner handles settings, skipping unchanged frames, progress, and cancel, mirroring `mpm_core.bake`. It never imports the solver or its backends.

**Tech Stack:** Python ≥ 3.11, numpy, warp-lang 1.18 (`wp.MarchingCubes`), pytest. Same package distribution as `mpm_core` (src layout).

**Spec:** `docs/superpowers/specs/2026-10-28-mpm-blender-solver-design.md` (sections Surfacer and Display)

## Global Constraints

- `mpm_surface` imports only `mpm_core.cache` (and stdlib/numpy/warp); never `mpm_core.solver`, `mpm_core.backends`, `mpm_core.bake`, or `bpy`.
- Re-running with unchanged particles and settings rewrites nothing; changed settings re-mesh without re-simulating.
- All files written atomically (tmp + replace, with the Windows retry from `mpm_core.cache`).
- Meshes are closed (watertight) where the material is fully inside the domain, wound outward, in world space (metres).
- Each mesh vertex carries a velocity (m/s) for motion blur.
- Windows x64 and Linux x64, NVIDIA GPU. Tests pass on both.
- Work on master; commit after every task. Commit messages end with `Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>`.

## Review Focus

- **Material touching the domain walls** (cream resting on the floor): the mesh must not have a hole where it meets the floor; the field grid needs padding beyond the particles so the surface closes. Test in Task 2.
- **Frames with zero particles for an emitter** (inflow before it starts, everything drained out of an open side): writes an empty mesh, no crash. Tests in Task 2 and Task 3.
- **Reading a cache while the bake is still running:** only frames that exist get meshed; a later run picks up the rest. Test in Task 3.
- **Very large particle extents** (sand spread over a 4 m floor at fine detail): the field grid is bounded by the particle bbox, and a clear error replaces a GPU out-of-memory when it would exceed a cell budget. Test in Task 2.
- **Settings for an emitter name that doesn't exist** in the cache: warn and ignore, rather than fail silently or crash. Test in Task 3.

## Design decisions (builder's calls, recorded)

- **Field:** normalized SPH-style density. Kernel: cubic B-spline with support radius `h = 2 · spacing · smoothing`, where `spacing = dx / cbrt(particles_per_cell)` from the cache. Normalize so a uniform lattice at `spacing` gives density 1: divide by the lattice sum of the kernel, computed once on the host. The iso level is `settings.iso` (default 0.5), which puts the surface about half a spacing out from the outer particles.
- **Field grid:** cell size `spacing / detail`. Bounds are the emitter's particle bbox padded by `h + 2` cells. Abort with `ValueError` above `max_cells = 256·10⁶`, a cell budget named in the message.
- **Velocity:** a kernel-weighted average `Σ W v / Σ W` on the same grid, sampled trilinearly at the vertices.
- **Winding:** `wp.MarchingCubes` with field > iso as inside produces inward-facing triangles (verified with Warp 1.18). Swap two indices per triangle.
- **Mesh cache layout:** `<cache>/surface/<emitter>/frame_NNNN.npz` with `verts (V,3) f32`, `tris (T,3) i32`, `vel (V,3) f32`, and `key` (str): sha1 of that emitter's particle positions for the frame, plus the settings hash. A run skips a frame whose existing `key` matches.
- **Settings file:** `<cache>/surface_settings.json` = `{"version": 1, "emitters": {name: {enabled, detail, smoothing, iso}}}`, written by the add-on. Emitters missing from it get defaults. With no file, all emitters use defaults. A disabled emitter's mesh directory is removed, so display falls back to points.
- **Progress and cancel:** `<cache>/surface/progress.json` (same schema as the bake) and `<cache>/surface/cancel` (cleared at start, honoured after each frame).

## File Structure

```
src/mpm_core/cache.py                 Modify: expose atomic_write_npz(path, arrays) (reuse existing helper)
src/mpm_surface/__init__.py
src/mpm_surface/settings.py           SurfaceSettings, read/write surface_settings.json
src/mpm_surface/field.py              surface_particles(): GPU splat + marching cubes + vertex velocity (warp)
src/mpm_surface/meshcache.py          MeshCacheWriter/Reader for <cache>/surface/
src/mpm_surface/run.py                surface() runner: settings, skip-unchanged, progress, cancel
src/mpm_surface/__main__.py           CLI: python -m mpm_surface CACHE [--frames A-B] [--device D]
tests/test_surface_settings.py, tests/test_surface_field.py, tests/test_surface_run.py
tools/blender_view_cache.py           Modify: show meshes from the mesh cache when present
examples/bench_surface.py             timing on a baked cache
```

---

### Task 1: Settings and mesh cache format

**Files:** create `src/mpm_surface/{__init__,settings,meshcache}.py` and `tests/test_surface_settings.py`; modify `src/mpm_core/cache.py`.

**Interfaces:**
- Produces, in `mpm_core.cache`: `atomic_write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None`. `CacheWriter` and `write_checkpoint` use it.
- Produces, in `settings.py`:
  - `@dataclass(frozen=True) SurfaceSettings(enabled: bool = True, detail: float = 1.0, smoothing: float = 1.0, iso: float = 0.5)`. `__post_init__` raises `ValueError` unless `detail > 0`, `smoothing > 0` and `0 < iso < 1`.
  - `SurfaceSettings` methods: `to_dict()`, `from_dict()` (classmethod), and `hash() -> str` (sha1 of sorted JSON).
  - `read_settings(cache_dir) -> dict[str, SurfaceSettings]`: `{}` if the file is missing.
  - `write_settings(cache_dir, settings: dict[str, SurfaceSettings]) -> None`.
- Produces, in `meshcache.py`:
  - `@dataclass SurfaceMesh(verts: np.ndarray, tris: np.ndarray, vel: np.ndarray)`, with `empty()` (staticmethod).
  - `MeshCacheWriter(cache_dir)`:
    - `write(emitter: str, frame: int, mesh: SurfaceMesh, key: str)`
    - `key(emitter, frame) -> str | None`
    - `remove_emitter(emitter)`
  - `MeshCacheReader(cache_dir)`:
    - `emitters() -> list[str]`
    - `available_frames(emitter) -> list[int]`
    - `read(emitter, frame) -> SurfaceMesh`

- [ ] **Step 1: Write failing tests in `tests/test_surface_settings.py`.**
  - `test_defaults_and_roundtrip`: `from_dict(to_dict())` equals the original.
  - `test_hash_changes`: `detail=2` gives a different hash.
  - `test_invalid`, parametrized over `detail=0`, `smoothing=-1`, `iso=1.0`: raises `ValueError`.
  - `test_settings_file_roundtrip`: write then read gives the same dict; a missing file gives `{}`.
  - `test_mesh_cache_roundtrip`: write a 1-triangle mesh with key "k", read it back; `key()` is "k"; `available_frames` is `[3]`; `emitters()` is `["cream"]`.
  - `test_remove_emitter`: the directory is gone and `emitters()` is `[]`.
  - `test_empty_mesh`: `SurfaceMesh.empty()` has shapes (0,3), (0,3), (0,3) and round-trips.
- [ ] **Step 2: Run** `.venv/Scripts/python -m pytest tests/test_surface_settings.py -q`. Expected: FAIL (import).
- [ ] **Step 3: Implement** `atomic_write_npz` in `mpm_core/cache.py` and switch its two internal npz writes to it, then write `settings.py` and `meshcache.py`.
- [ ] **Step 4: Run** `tests/test_surface_settings.py tests/test_cache.py`. Expected: PASS.
- [ ] **Step 5: Commit** with message `feat: surface settings and mesh cache format`.

---

### Task 2: GPU surfacing of one particle set

**Files:** create `src/mpm_surface/field.py` and `tests/test_surface_field.py`.

**Interfaces:**
- Consumes `SurfaceSettings` and `SurfaceMesh` (Task 1).
- Produces `surface_particles(positions: np.ndarray, velocities: np.ndarray, spacing: float, settings: SurfaceSettings, device: str | None = None, max_cells: int = 256_000_000) -> SurfaceMesh`, following the Design decisions above. Zero particles give `SurfaceMesh.empty()`.

- [ ] **Step 1: Write failing tests in `tests/test_surface_field.py`** (all `gpu`). Use `sphere_particles` and `box_particles` from `tests/helpers.py` with `dx = 0.01` (spacing 0.005). Helpers in the test file: `signed_volume(verts, tris)` and `components(tris)` (union-find over shared vertices).

```python
def test_sphere_closed_outward_right_size(device):
    p = sphere_particles((0.5, 0.5, 0.5), 0.1, 0.01)
    m = surface_particles(p, np.zeros_like(p), 0.005, SurfaceSettings(), device)
    assert mesh_is_closed(m.tris)
    vol = signed_volume(m.verts, m.tris)
    assert vol > 0 and vol == pytest.approx(4 / 3 * np.pi * 0.1**3, rel=0.10)
    assert np.allclose(m.verts.mean(0), 0.5, atol=0.005)

def test_two_blobs_two_components(device): ...   # spheres r=0.05 at x=0.3 and x=0.7 -> components == 2

def test_resting_on_floor_closed(device):        # Review Focus: material at the domain floor
    p = box_particles((0.4, 0.4, 0.0), (0.6, 0.6, 0.1), 0.01)
    m = surface_particles(p, np.zeros_like(p), 0.005, SurfaceSettings(), device)
    assert mesh_is_closed(m.tris) and m.verts[:, 2].min() < 0.0  # closes just below z=0, no hole

def test_empty(device):                           # Review Focus
    m = surface_particles(np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32), 0.005, SurfaceSettings(), device)
    assert m.verts.shape == (0, 3) and m.tris.shape == (0, 3)

def test_vertex_velocity(device):
    p = sphere_particles((0.5, 0.5, 0.5), 0.05, 0.01)
    m = surface_particles(p, np.tile([1.0, 0, 0], (len(p), 1)).astype(np.float32), 0.005, SurfaceSettings(), device)
    assert np.allclose(m.vel, [1.0, 0, 0], atol=1e-3)

def test_detail_increases_triangles(device): ...  # detail=2 gives > 2x the triangles of detail=1

def test_too_many_cells_raises(device):           # Review Focus
    p = np.array([[0, 0, 0], [4, 4, 0.1]], np.float32)
    with pytest.raises(ValueError, match="cell budget"):
        surface_particles(p, np.zeros_like(p), 0.001, SurfaceSettings(detail=2), device)
```

- [ ] **Step 2: Run** `.venv/Scripts/python -m pytest tests/test_surface_field.py -q`. Expected: FAIL (import).
- [ ] **Step 3: Implement `field.py`.**
  - Splat kernel: each particle loops over the nodes within `h`, then `atomic_add`s the weight and the weighted velocity.
  - Run `wp.MarchingCubes`, convert vertices from index space to world (`origin + v · cell`), and flip the winding.
  - A vertex-velocity kernel samples the velocity grid trilinearly at each vertex.
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Commit** with message `feat: GPU particle surfacing with marching cubes`.

---

### Task 3: Surface runner and CLI

**Files:** create `src/mpm_surface/run.py`, `src/mpm_surface/__main__.py` and `tests/test_surface_run.py`.

**Interfaces:**
- Consumes everything from Tasks 1 and 2, plus `mpm_core.cache.CacheReader` and `write_json`.
- Produces:
  - `@dataclass SurfaceResult(status: str, frames_written: int, frames_skipped: int, message: str = "")`
  - `surface(cache_dir: Path, *, settings: dict[str, SurfaceSettings] | None = None, frames: list[int] | None = None, device: str | None = None, should_cancel: Callable[[], bool] | None = None) -> SurfaceResult`
  - Behaviour:
    - `settings=None` reads `surface_settings.json`.
    - Frames default to the particle frames that exist now.
    - Spacing comes from `cache.json` `dx` and `scene/scene.json` `particles_per_cell`.
    - Progress and cancel work as in Design decisions.
    - Settings for an unknown emitter add the warning `"surface settings for unknown emitter <name> ignored"`.
  - CLI: `python -m mpm_surface CACHE [--frames A-B] [--device D]`. Exit code 0 on done or cancelled, 1 on error.

- [ ] **Step 1: Write failing tests in `tests/test_surface_run.py`** (`gpu`). A fixture `particle_cache(tmp_path)` writes a scene (`small_scene` from `helpers`, `write_scene`) and uses `CacheWriter` to write 4 frames for emitter `emitter_0`: a sphere of particles moving +x 1 cm per frame. No solver involved.
  - `test_meshes_all_frames`: status "done", `frames_written == 4`, `MeshCacheReader.available_frames("emitter_0") == [1, 2, 3, 4]`, and each mesh is closed.
  - `test_second_run_skips`: a second `surface()` gives `frames_written == 0` and `frames_skipped == 4`.
  - `test_changed_settings_remesh`: `settings={"emitter_0": SurfaceSettings(detail=2)}` gives `frames_written == 4`.
  - `test_disabled_emitter_removed`: after a run, `enabled=False` makes `MeshCacheReader.emitters() == []`.
  - `test_unknown_emitter_warns` (Review Focus): settings for "nope" add the warning, and meshing still happens.
  - `test_partial_cache_then_rest` (Review Focus): delete particle frames 3–4 and run (2 written), restore them and run again (2 written, 2 skipped).
  - `test_empty_frame`: a frame where `emitter_0` has 0 particles gives an empty mesh and status "done".
  - `test_cancel`: `should_cancel=lambda: True` gives status "cancelled" with `frames_written == 1`, and progress says cancelled.
  - `test_cli`: the subprocess exits 0 and meshes exist.
- [ ] **Step 2: Run.** Expected: FAIL (import).
- [ ] **Step 3: Implement `run.py` and `__main__.py`.**
- [ ] **Step 4: Run the full suite.** Expected: PASS.
- [ ] **Step 5: Commit** with message `feat: surface runner with skip-unchanged, progress, cancel and CLI`.

---

### Task 4: Viewer support, benchmark, visual check

**Files:** modify `tools/blender_view_cache.py` and `README.md`; create `examples/bench_surface.py`.

**Interfaces:**
- The viewer shows each emitter as a mesh object when `<cache>/surface/<emitter>/` exists, and as points otherwise. Meshes are loaded per frame from the npz: `from_pydata` replacement or `foreach_set` on a resized mesh, smooth shading, and a `velocity` FLOAT_VECTOR point attribute. It reads the npz directly, with no `mpm_surface` import.
- `examples/bench_surface.py CACHE`: runs `surface()` and prints frames, total time and ms/frame.

- [ ] **Step 1: Implement the viewer change and `bench_surface.py`.**
- [ ] **Step 2: Run** `examples/bench_surface.py out/mix` (baked earlier). Expected: done. Record ms/frame for 204k particles in the README; the target is under the sim's 389 ms/frame.
- [ ] **Step 3: Visual check.** Render frames 1, 48 and 96 of `out/mix` headless with the viewer. Check that the cream shows a smooth closed dome with no hole at the floor, the sand reads as a heap surface and the snow as a lump. Also surface and render `out/tests/inflow` and `out/spike_fracture`'s tough variant, which shows two separate pieces.
- [ ] **Step 4: README:** add a surfacing section (CLI, settings file, mesh cache layout pointer, benchmark).
- [ ] **Step 5: Commit** with message `feat: view surfaced meshes in Blender, surfacing benchmark`.
