# MPM Blender Add-on Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Blender 5.2 extension that gives the MPM solver a Blender-native workflow: domain, emitter and collider roles in the Physics tab, material presets with friendly tweaks, Bake / Bake Surface with progress and cancel, and timeline playback of points or meshes with preset render materials.

**Architecture:**
- The add-on (`blender_addon/mpm_blender/`) is pure bpy and never imports Warp.
- It exports a scene description with `mpm_core.scene.write_scene`.
- It runs `python -m mpm_core.bake` and `python -m mpm_surface` as subprocesses, using Blender's own interpreter (`sys.executable`) with `PYTHONPATH` set from Blender's `sys.path`.
- It polls their `progress.json` from a modal timer.
- It loads cache frames in a persistent `frame_change_post` handler.
- `mpm_core` and `mpm_surface` ship as a wheel (`mpm_core-*.whl`, built from this repo) next to the Warp wheels in the extension's `wheels/`.
- Pure logic that needs no bpy, such as the preset tweak mapping, lives in `mpm_core` and is tested with normal pytest.
- bpy code is tested headless with pytest running inside Blender.

**Tech Stack:** Blender 5.2 (Python 3.13, numpy 2.3), bpy, the extension manifest with wheels, `mpm_core`/`mpm_surface`, and warp-lang 1.18 wheels (`py3-none-win_amd64`, `py3-none-manylinux_2_28_x86_64`). Tests run with pytest in the dev venv, plus pytest inside Blender via a test site-packages folder.

**Spec:** `docs/superpowers/specs/2026-10-28-mpm-blender-solver-design.md` (User workflow, Presets, Architecture, Display, anti-goals)

## Global Constraints

- Blender 5.2 LTS minimum; Windows x64 and Linux x64; NVIDIA GPU.
- The add-on never imports `warp`, `mpm_core.backends`, `mpm_core.bake` or `mpm_surface.field`. Simulation and surfacing run only in subprocesses, so a GPU fault can't take Blender down and Blender stays responsive.
- Blender never freezes during a bake or surface run: progress comes from polling a file in a modal timer (≤ 4 Hz).
- Each emitter gets its own output object and its own material entry in the scene (spec: material lives on the emitter).
- Presets are chosen by label; friendly tweaks map to physical values. Raw physical values sit behind an "Advanced" subpanel.
- Preview and final bakes live in separate cache subfolders (`<cache>/preview`, `<cache>/final`). A preview never overwrites a final.
- Preview quality = half the domain resolution, rounded and at least 8. Final = the domain resolution.
- Units: Blender scene units (metres); gravity from `scene.gravity`, or zero when `scene.use_gravity` is off.
- Cache readers retry a few times on `PermissionError` (Windows; see the `mpm_core.cache` docstring).
- Work on master; commit after every task. Commit messages end with `Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>`.

## Review Focus

- **Unsaved .blend with a relative cache path (`//...`)**: Bake reports "save the file first (or set an absolute cache path)" instead of writing to the current working directory. Test in Task 4.
- **Rotated domain object**: the solver's domain is axis-aligned, so bake refuses with a clear message naming the domain. A scaled, unrotated cube is fine. Test in Task 4.
- **Emitter or collider renamed or deleted after a bake**: playback must not crash. Stale output objects show nothing until the next bake, and output objects of removed emitters are deleted on the next bake. Test in Task 6.
- **Blender reopened while a bake is running, or after it finished**: the frame handler re-registers on file load (`@persistent`) and playback works from the cache on disk. A bake started in a previous session isn't re-attached (no progress bar) but can be resumed. Test in Task 6.
- **Scrubbing to frames not baked yet** (bake running): output objects keep the last loaded frame or show nothing, never raise. Test in Task 6.

## Design decisions (builder's calls, recorded)

- **Roles:** each object has one `mpm` property group (`MPMObjectSettings`) with `role` ∈ {NONE, DOMAIN, EMITTER, COLLIDER}, plus nested settings per role. One "MPM" panel in the Physics tab shows a role dropdown, then that role's subpanels.
- **Emitters belong to a domain** via `emitter.domain` (an Object pointer, polled to DOMAIN objects). If it's left empty and the scene has exactly one domain, that domain is used. Colliders do the same.
- **Tweaks** (module `mpm_core.tweaks`, pure):

  | Tweak | Applies to | Effect |
  |---|---|---|
  | `stiffness` | all models | multiplies E |
  | `firmness` | von_mises | multiplies yield_stress |
  | `friction_angle` | drucker_prager | degrees, absolute |
  | `clumping` | snow | hardening, absolute |
  | `density` | all models | kg/m³, absolute |

  Defaults come from the preset. "Advanced" overrides set raw values directly and win over tweaks when enabled.
- **Cache folder:** `domain.cache_dir`, default `//mpm_cache/<domain name>`, plus `/preview` or `/final`.
- **Export:**
  - Emitter and collider meshes are the evaluated mesh at the scene's start frame, in object local space, triangulated.
  - Transforms are sampled by `scene.frame_set` over the bake range; the original frame is restored afterwards.
  - Collider `closed` is computed with `mesh_is_closed`.
- **Output objects:** named `MPM <emitter name>` and parented under a collection `MPM <domain name>`. Each carries a custom property `mpm_emitter` holding the emitter object's name *at bake time*, plus its index in the cache.
- **Display:** an emitter's `display` is AUTO, POINTS or MESH. AUTO follows the preset's `look.display`.
  - Points use a Geometry Nodes Mesh-to-Points group with radius 0.3·dx.
  - Meshes come from the mesh cache, with a `velocity` point attribute.
  - Materials are created from the preset look on first bake, named `MPM <preset label>`, and never overwritten once they exist.
- **Surfacing:**
  - `domain.auto_surface` (default on) runs the surfacer after each successful bake, for emitters whose effective display is MESH.
  - "Bake Surface" runs it on demand.
  - The emitter's Surface subpanel holds `detail`, `smoothing` and `iso`, written to `surface_settings.json`. Emitters displayed as points get `enabled: false`.
- **Subprocess environment:** `[sys.executable, "-m", <module>, <cache>, ...]` with `PYTHONPATH = os.pathsep.join(sys.path)`. On Windows, use `CREATE_NO_WINDOW`. stdout/stderr go to `<cache>/bake.log` or `<cache>/surface/surface.log`, so error details survive.
- **Jobs:** one job (bake or surface) per domain at a time, held in a module-level dict keyed by domain name.
  - Cancel writes the cancel file.
  - Unregister or `load_pre` cancels running jobs and terminates them after 2 s.
- **Scripting/tests:** `bpy.ops.mpm.bake(wait=True)` and `bpy.ops.mpm.bake_surface(wait=True)` block until done (no modal). The UI uses `wait=False`.

## File Structure

```
src/mpm_core/tweaks.py                     TweakSpec, tweaks_for(model), apply_tweaks(preset, values, overrides)
tests/test_tweaks.py
blender_addon/mpm_blender/blender_manifest.toml
blender_addon/mpm_blender/__init__.py      register/unregister, persistent load handlers
blender_addon/mpm_blender/props.py         PropertyGroups (domain, emitter, collider, object) + preset enum
blender_addon/mpm_blender/ui.py            Physics tab panels
blender_addon/mpm_blender/export.py        bpy -> SceneDescription, cache dir resolution, validation errors
blender_addon/mpm_blender/jobs.py          subprocess job: start, poll progress, cancel, terminate
blender_addon/mpm_blender/operators.py     bake / resume / cancel / free / bake_surface / free_surface
blender_addon/mpm_blender/display.py       output objects, materials, frame handler, cache readers with retry
blender_addon/mpm_blender/wheels/          build output (git-ignored)
tools/setup_blender_test_env.py            pip --target out/blender_site: warp-lang (no deps), pytest
tools/blender_tests.py                     run tests_blender/ with pytest inside Blender
tools/build_extension.py                   build mpm_core wheel, fetch warp wheels, write manifest, build zips
tests_blender/conftest.py                  sys.path setup, register add-on, fresh scene fixture, helpers
tests_blender/test_*.py
```

---

### Task 1: Preset tweaks (pure Python)

**Files:** create `src/mpm_core/tweaks.py` and `tests/test_tweaks.py`.

**Interfaces:**
- `@dataclass(frozen=True) TweakSpec(name: str, label: str, default: float, min: float, max: float, unit: str, description: str)`
- `tweaks_for(preset_id: str) -> list[TweakSpec]`: the stiffness and density tweaks plus the model-specific one, with defaults taken from the preset.
  - stiffness: default 1.0, range [0.1, 10].
  - density: default = preset density, range [10, 5000].
  - firmness: default 1.0, range [0.05, 20].
  - friction_angle: default = preset value, range [0, 60].
  - clumping: default = preset hardening, range [0, 30].
- `apply_tweaks(preset_id: str, values: dict[str, float], overrides: dict[str, float] | None = None, name: str | None = None) -> MaterialParams`
  - Unknown tweak names raise `KeyError`.
  - `overrides` are raw `MaterialParams` field values applied last.
  - `name` defaults to the preset id.

- [ ] **Step 1: Write failing tests:**
  - `test_tweaks_for_cream`: the names are `["stiffness", "density", "firmness"]` and the density default is 400.
  - `test_tweaks_for_sand` includes `friction_angle` with default 30.
  - `test_stiffness_and_firmness_scale`: `apply_tweaks("whipped_cream", {"stiffness": 2, "firmness": 0.5})` gives E = 2e5 and yield = 1250.
  - `test_absolute_tweaks`: sand `friction_angle` 40 and snow `clumping` 5 come through as raw values.
  - `test_overrides_win`: `overrides={"youngs_modulus": 7e4}` beats stiffness.
  - `test_unknown_tweak_raises`.
  - `test_defaults_reproduce_preset`: for every preset, `apply_tweaks(p, {})` equals `load_preset(p).material`.
- [ ] **Step 2: Run** `.venv/Scripts/python -m pytest tests/test_tweaks.py -q`. Expected: FAIL (import).
- [ ] **Step 3: Implement** `tweaks.py`.
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Commit** with message `feat: friendly preset tweaks mapped to material parameters`.

---

### Task 2: Extension skeleton and in-Blender test harness

**Files:**
- Create: `blender_addon/mpm_blender/{blender_manifest.toml,__init__.py}`, `tools/setup_blender_test_env.py`, `tools/blender_tests.py`, `tests_blender/conftest.py`, `tests_blender/test_register.py`.
- Modify: `.gitignore` (add `blender_addon/mpm_blender/wheels/` and `dist/`).

**Interfaces:**
- `tools/setup_blender_test_env.py`: runs `pip install --target out/blender_site --no-deps warp-lang==1.18.0 pytest iniconfig pluggy packaging` with the venv's pip, so the packages are pure Python or ABI-independent and safe in Blender's Python 3.13.
- `tools/blender_tests.py [pytest args...]`: finds Blender via the `BLENDER` env var or the default install paths (Windows `C:/Program Files/Blender Foundation/Blender 5.2/blender.exe`, Linux `blender` on PATH). It runs `blender -b --factory-startup --python-expr` with code that:
  - prepends `src/`, `blender_addon/` and `out/blender_site` to `sys.path`,
  - runs `pytest.main(["tests_blender", *args])`,
  - and exits with pytest's code.
- `tests_blender/conftest.py`:
  - a session fixture that imports and registers `mpm_blender`;
  - a function fixture `clean_scene` that deletes all objects, collections, materials, meshes and node groups, and sets the frame range to 1..`n`;
  - helpers `add_cube(name, location, size)`, `add_sphere(...)`, `add_plane(...)`.
- Manifest: `schema_version = "1.0.0"`, `id = "mpm_blender"`, `name = "MPM Solver"`, `version = "0.1.0"`, `blender_version_min = "5.2.0"`, `platforms = ["windows-x64", "linux-x64"]`, `permissions = {files = "Read and write simulation caches"}`, `type = "add-on"`. The wheels list is filled in by Task 8.

- [ ] **Step 1: Write the tooling and a failing test.** `tests_blender/test_register.py::test_object_has_mpm_settings` asserts `bpy.types.Object.mpm` exists and a new object's `mpm.role == "NONE"`.
- [ ] **Step 2: Run** `.venv/Scripts/python tools/setup_blender_test_env.py`, then `.venv/Scripts/python tools/blender_tests.py -q`. Expected: FAIL (`mpm_blender` has no `register`, or `Object.mpm` is missing).
- [ ] **Step 3: Implement** `__init__.py` with a minimal `props.py` (`MPMObjectSettings` with `role`), registering `bpy.types.Object.mpm`.
- [ ] **Step 4: Run** `tools/blender_tests.py -q`. Expected: PASS.
- [ ] **Step 5: Commit** with message `feat: add-on skeleton and in-Blender pytest harness`.

---

### Task 3: Properties and Physics tab UI

**Files:** modify `blender_addon/mpm_blender/{props.py,__init__.py}`; create `blender_addon/mpm_blender/ui.py` and `tests_blender/test_props_ui.py`.

**Interfaces:**
- `MPMDomainSettings`:
  - core: `resolution: Int (16..1024, default 128)`, `quality: Enum {PREVIEW, FINAL}` (default PREVIEW), `cache_dir: StringProperty(subtype DIR_PATH, default "")` (empty means `//mpm_cache/<domain name>`)
  - walls: `wall_x_neg..wall_z_pos: Bool` (default True), `wall_friction: Float (0..2, 0.5)`
  - bake: `particles_per_cell: Enum {1, 8, 27}` (default 8), `auto_surface: Bool (True)`
  - runtime state (not user-edited): `progress: Float`, `status: String`, `status_message: String`
- `MPMEmitterSettings`:
  - setup: `domain: PointerProperty(Object, poll=role==DOMAIN)`, `kind: Enum {FILL, INFLOW}`, `preset: Enum` (items from `list_presets()`, labelled by preset label)
  - tweaks: `stiffness`, `density`, `firmness`, `friction_angle`, `clumping` as Floats (min/max from `tweaks_for`). Changing the preset resets them to that preset's defaults via an update callback.
  - advanced: `use_advanced: Bool`, plus advanced raw fields `youngs_modulus`, `poisson_ratio`, `yield_stress`, `critical_compression`, `critical_stretch`
  - inflow: `velocity: FloatVector(3)`, `use_object_velocity: Bool`, `inflow_start`, `inflow_end: Int`
  - display and surface: `display: Enum {AUTO, POINTS, MESH}`, `surface_detail`, `surface_smoothing`, `surface_iso: Float` (defaults from `SurfaceSettings()`)
- `MPMColliderSettings`: `domain` (as for the emitter) and `friction: Float (0..2, 0.5)`.
- `MPMObjectSettings`: `role` plus `domain`, `emitter` and `collider` PointerProperty groups.
- Panels in Physics (`bl_space_type="PROPERTIES"`, `bl_region_type="WINDOW"`, `bl_context="physics"`):
  - `MPM_PT_main` shows the role dropdown.
  - Subpanels poll on the role:
    - domain: Settings, Walls, Cache & Bake (buttons, progress bar via `layout.progress`, status message)
    - emitter: Material (preset and that preset's tweaks only), Advanced, Inflow (only when INFLOW), Display & Surface
    - collider: Settings
- Helper `emitter_material(settings) -> MaterialParams` in `props.py`, built on `apply_tweaks`.

- [ ] **Step 1: Write failing tests** in `tests_blender/test_props_ui.py`:
  - `test_role_defaults`: a new cube's domain resolution is 128, quality PREVIEW, all walls on.
  - `test_preset_enum_lists_presets`: the enum identifiers equal `list_presets()`, and labels include "Whipped cream".
  - `test_preset_change_resets_tweaks`: set sand, then `friction_angle` 40, then switch to snow and back to sand; `friction_angle` is 30 again.
  - `test_emitter_material_uses_tweaks`: cream with firmness 0.5 gives `emitter_material(...).yield_stress == 1250`.
  - `test_panels_poll_by_role`: with the active object's role EMITTER, `MPM_PT_emitter_material.poll(ctx)` is True and the domain panel's poll is False. Use a context override via `bpy.context.temp_override`.
  - `test_panels_draw`: draw every panel once with each role, catching exceptions; no exceptions.
- [ ] **Step 2: Run** `tools/blender_tests.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** `props.py` and `ui.py`.
- [ ] **Step 4: Run** the same. Expected: PASS.
- [ ] **Step 5: Commit** with message `feat: MPM roles, preset tweaks and Physics tab panels`.

---

### Task 4: Scene export

**Files:** create `blender_addon/mpm_blender/export.py` and `tests_blender/test_export.py`.

**Interfaces:**
- `class ExportError(Exception)`: a message shown to the user verbatim.
- `cache_root(domain_obj) -> Path`: absolute; raises `ExportError("save the .blend file first, or set an absolute cache folder")` for a relative path in an unsaved file.
- `cache_path(domain_obj, quality: str) -> Path`: `cache_root / quality.lower()`.
- `members(scene, domain_obj) -> tuple[list[Object], list[Object]]`: emitters and colliders that belong to this domain (explicit pointer, or implicit when it's the only domain). Order is the sorted object name, which defines emitter index and material index.
- `export_scene(scene, domain_obj, quality: str) -> tuple[SceneDescription, list[str]]`: returns the scene and the emitter names in index order.
  - Domain: world AABB of the domain object; rotated → `ExportError` naming the object. Resolution is halved for PREVIEW (≥ 8).
  - Frame range, fps (`scene.render.fps / fps_base`), gravity and walls come from the scene and domain.
  - Emitter material: `emitter_material()`, named after the emitter object.
  - No emitters → `ExportError("domain <name> has no emitters")`.
  - Meshes are evaluated, triangulated via bmesh, in local space; transforms are sampled per frame.
- `write_export(scene, domain_obj, quality) -> Path`: `export_scene`, then `write_scene` to `cache_path/scene`; returns the cache path.

- [ ] **Step 1: Write failing tests** in `tests_blender/test_export.py`. Build the scene in code: a 1 m cube domain at z 0..1, a 0.2 m sphere emitter (cream, firmness 2), and a plane collider moving +x over frames 1..5.
  - `test_domain_and_quality`: `export_scene(...,"FINAL").domain.resolution == 128` and PREVIEW gives 64. The domain min/max matches the cube bounds within 1e-5.
  - `test_emitter_transforms_and_local_mesh`: 5 transforms. Emitter verts stay in local space (radius ~0.1 around the origin) while `transforms[0][:3,3]` is the object location. The plane collider's x translation increases across frames.
  - `test_material_from_tweaks`: `materials[0].yield_stress == 5000` and the name equals the emitter object name.
  - `test_collider_open_plane`: `colliders[0].closed is False`; a cube collider gives True.
  - `test_gravity_off`: `scene.use_gravity=False` gives gravity (0,0,0).
  - `test_rotated_domain_refused` (Review Focus): a domain rotated 10° about z raises `ExportError` with the domain name in the message.
  - `test_unsaved_relative_cache_refused` (Review Focus): an unsaved file with the default cache dir raises `ExportError` matching "save the .blend".
  - `test_write_export_roundtrip`: with an absolute temp cache dir, `read_scene(path/"scene")` equals `export_scene(...)[0]` (domain, materials and transform counts).
  - `test_no_emitters_refused`.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement** `export.py`.
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** with message `feat: export Blender scene to MPM scene description`.

---

### Task 5: Jobs and bake operators

**Files:** create `blender_addon/mpm_blender/{jobs.py,operators.py}` and `tests_blender/test_bake.py`; modify `__init__.py`.

**Interfaces:**
- `jobs.py`:
  - `class Job` with:
    - `__init__(kind: str, cache_dir: Path, argv: list[str], log_path: Path, progress_path: Path, cancel_path: Path)`
    - `start()`
    - `poll() -> dict | None`: reads progress.json, retrying on PermissionError and JSONDecodeError.
    - `running` (property)
    - `returncode`
    - `cancel()`: writes the cancel file.
    - `terminate(timeout=2.0)`
  - Module functions: `bake_job(cache: Path, resume: bool) -> Job` and `surface_job(cache: Path) -> Job`.
  - `JOBS: dict[str, Job]`, keyed by domain name.
  - `cancel_all()`
- `operators.py`, with the domain taken from the context object, or from `domain=<name>` for scripting:
  - `MPM_OT_bake(wait: Bool, resume: Bool)`: export, start the job, then either a modal timer (0.25 s) updating `domain.mpm.domain.progress`, `status` and `status_message` and tagging Physics areas for redraw, or blocking with `wait=True`. On done it triggers display refresh (Task 6) and, with `auto_surface`, a surface job (Task 7).
  - `MPM_OT_cancel`: cancels the domain's running job.
  - `MPM_OT_free_bake`: deletes `cache_path(domain, quality)` and removes output objects. It refuses while a job runs, with `invoke_confirm`.
  - All of them report `ExportError` and job errors via `self.report({"ERROR"}, msg)` and set `status = "error"`. Errors include the last lines of the log on failure.

- [ ] **Step 1: Write failing tests** in `tests_blender/test_bake.py` (GPU). Use a small scene: domain resolution 16 FINAL, frames 1..4, a cube emitter (jelly), an absolute temp cache dir.
  - `test_bake_wait_writes_cache`: `bpy.ops.mpm.bake(wait=True)` returns `{'FINISHED'}`, `CacheReader(cache_path).available_frames() == [1,2,3,4]`, and `status == "done"`.
  - `test_bake_error_is_reported`: an empty domain (no emitters) gives `{'CANCELLED'}`, `status == "error"`, and a message containing "no emitters".
  - `test_resume`: export with `write_export`, then start `jobs.bake_job(cache, resume=False)`. Poll until the progress frame is ≥ 2, call `job.cancel()`, and wait for it to exit. Then `bpy.ops.mpm.bake(wait=True, resume=True)` gives all 4 frames.
  - `test_free_bake`: removes the cache folder.
  - `test_job_env_has_blender_sys_path`: `bake_job(...)`'s environment PYTHONPATH contains every entry of `sys.path` that exists.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement** `jobs.py` and `operators.py`, and register the operators.
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** with message `feat: bake, resume, cancel and free operators with subprocess jobs`.

---

### Task 6: Playback display

**Files:** create `blender_addon/mpm_blender/display.py` and `tests_blender/test_display.py`; modify `operators.py` and `__init__.py`.

**Interfaces:**
- `refresh_outputs(scene, domain_obj)`: makes the collection `MPM <domain>` and one object `MPM <emitter>` per emitter in the current cache (from `cache.json` emitters), and deletes output objects whose emitter is no longer in the cache. It assigns or creates materials (`MPM <preset label>` from the preset look), sets up the points GN group or a plain mesh according to the effective display and whether a mesh cache exists, then loads the current frame.
- `load_frame(scene)`: the persistent `frame_change_post` handler for all domains in the scene.
  - It clamps the frame to the cache range.
  - A missing frame file is skipped, keeping the last geometry.
  - Points come from the particle cache; meshes from the mesh cache via the manifest.
  - Reads use the PermissionError retry.
  - Any exception is caught and printed, never raised into Blender.
- `@persistent load_post`: re-registers the handler and calls `refresh_outputs` for every domain with a cache.
- `@persistent load_pre`: `jobs.cancel_all()`.
- A bake finishing (Task 5) calls `refresh_outputs`. While a bake runs, the modal timer reloads the current frame when a new frame arrives.

- [ ] **Step 1: Write failing tests** in `tests_blender/test_display.py` (GPU, the small baked scene from Task 5 as a fixture):
  - `test_output_object_per_emitter`: after bake, the collection "MPM Domain" holds "MPM Emitter", and at frame 3 its vertex count equals `len(cache frame 3 pos)`.
  - `test_material_from_preset`: the output object's material is named "MPM Jelly" with base colour ≈ (0.8, 0.15, 0.2), and an existing material with that name isn't modified.
  - `test_scrub_unbaked_frame` (Review Focus): deleting frame 4's npz and setting frame 4 doesn't raise, and the vertex count is unchanged from frame 3.
  - `test_renamed_emitter` (Review Focus): renaming the emitter object after bake makes `frame_set` not raise; after re-bake, the old output object is gone and the new one exists.
  - `test_reload_file` (Review Focus): `save_as_mainfile(tmp)`, then `open_mainfile(tmp)`; the handler is registered and `frame_set(2)` loads geometry.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement** `display.py` and hook it into the operators.
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** with message `feat: playback of baked caches as points or meshes`.

---

### Task 7: Surfacing operators

**Files:** modify `operators.py`, `display.py` and `ui.py`; create `tests_blender/test_surface.py`.

**Interfaces:**
- `write_surface_settings(domain_obj, cache: Path)`: writes `surface_settings.json` from each emitter's surface fields. Emitters with an effective display of POINTS get `enabled=False`.
- `MPM_OT_bake_surface(wait: Bool)`: requires a finished bake (else an error report). It starts a surface job with a progress bar like Bake, then calls `refresh_outputs`.
- `MPM_OT_free_surface`: deletes `<cache>/surface`.
- Auto-surface after a bake when `domain.auto_surface` is set and any emitter's effective display is MESH.

- [ ] **Step 1: Write failing tests** in `tests_blender/test_surface.py` (GPU). Use the small scene with a jelly emitter (AUTO resolves to MESH) and a sand emitter (AUTO resolves to POINTS).
  - `test_auto_surface_after_bake`: after `bake(wait=True)`, `MeshCacheReader(cache).emitters() == ["Jelly"]`; the jelly output object has polygons > 0 at frame 2, and the sand output still uses the points modifier.
  - `test_surface_settings_written`: detail 2 on the jelly emitter shows up in `read_settings(cache)["Jelly"].detail == 2` after `bake_surface(wait=True)`.
  - `test_display_points_override`: jelly with display=POINTS, then bake_surface, leaves the jelly unsurfaced (`emitters() == []`) and the output object back on points.
  - `test_bake_surface_without_bake_errors`.
  - `test_free_surface`: the mesh output falls back to points.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** with message `feat: bake surface operators and auto-surfacing`.

---

### Task 8: Packaging and end-to-end check

**Files:** create `tools/build_extension.py`; modify `blender_addon/mpm_blender/blender_manifest.toml` and `README.md`.

**Interfaces:**
- `tools/build_extension.py [--blender PATH]` runs in this order:
  1. `pip wheel . --no-deps -w blender_addon/mpm_blender/wheels` (builds `mpm_core-0.1.0-py3-none-any.whl`, which contains both packages).
  2. `pip download warp-lang==1.18.0 --no-deps --only-binary=:all: --platform win_amd64` and `--platform manylinux_2_28_x86_64`, into the same folder.
  3. Rewrites the manifest's `wheels = [...]` from the files present.
  4. Runs `blender --command extension build --split-platforms --source-dir blender_addon/mpm_blender --output-dir dist`.
- It prints the produced zip paths. Expected: `dist/mpm_blender-0.1.0-windows_x64.zip` and `...-linux_x64.zip`.

- [ ] **Step 1: Implement** `build_extension.py`; run it. Expected: two zips, each containing the matching Warp wheel and the mpm_core wheel.
- [ ] **Step 2: Install test.** Install into a throwaway Blender user folder (`BLENDER_USER_RESOURCES=out/blender_user`) with `blender --command extension install-file -r user_default -e dist/<windows zip>`. Then, in a headless session with only that extension enabled (no repo paths on `sys.path`), run a script that:
  1. builds the small scene,
  2. calls `bpy.ops.mpm.bake(wait=True)`,
  3. calls `frame_set(3)`,
  4. asserts the output geometry.
  Expected: it works, which proves the shipped wheels are enough.
- [ ] **Step 3: Manual UI check (author).** Install the zip in Blender 5.2 and work through:
  1. Set up the mixed scene from the spec (cream, sand, snow, and a collider).
  2. Bake preview while scrubbing, with the progress bar showing.
  3. Cancel, then resume.
  4. Bake final and Bake Surface.
  5. Render a frame in Cycles.

  Flag this to the author rather than skipping it. The Linux zip on the RTX 4090 is part of the same check.
- [ ] **Step 4: README:** install instructions, the add-on workflow, the dev test commands (`tools/setup_blender_test_env.py`, `tools/blender_tests.py`) and building.
- [ ] **Step 5: Commit** with message `feat: build the Blender extension with bundled wheels`.
