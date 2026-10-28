# MPM Solver Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Blender-independent MPM solver package (`mpm_core`) with a backend interface, a Warp backend, a scene-description format, a cache format, and a bake runner CLI that turns a scene description into a cache.

**Architecture:** `mpm_core` is pure Python + numpy. `mpm_core.solver` defines the `Solver` ABC; `mpm_core.backends.warp` implements it with NVIDIA Warp (the only place `warp` is imported). `mpm_core.bake` reads a scene description from disk, drives a solver, writes per-frame cache files, progress, and checkpoints. The Blender add-on (a later plan) only writes scene descriptions, launches `python -m mpm_core.bake`, and reads the cache.

**Tech Stack:** Python ≥ 3.11, numpy, warp-lang 1.18, pytest. Dev venv at `.venv` (created with `uv venv --python 3.12 .venv`, already exists with warp-lang 1.18.0 and numpy).

**Spec:** `docs/superpowers/specs/2026-10-28-mpm-blender-solver-design.md`

**Out of scope for this plan (next plan):** everything in Blender: add-on UI, object roles, scene export from bpy, launching the runner, display, surfacing, render materials, preset tweak UI.

## Global Constraints

- Python ≥ 3.11; no dependency beyond numpy and warp-lang (pytest for dev only). TOML presets via stdlib `tomllib`.
- `mpm_core` never imports `bpy`. Only `mpm_core/backends/warp/` imports `warp`.
- Windows x64 and Linux x64, NVIDIA GPUs (RTX 5080 Windows, RTX 4090 Linux). All tests must pass on both.
- Units: metres, seconds; z-up; gravity comes from the scene description.
- Cache and scene formats are backend-independent and documented in code docstrings.
- Particles resting on a collider or domain wall sit on its surface within **0.25 × dx** (spec: "within a fraction of a grid cell").
- Materials do not visibly sag at frame 1: a slab at rest moves less than **0.5 × dx** in 24 frames.
- A bake producing NaN/inf stops with a message naming the frame and writes no frame for it.
- Materials share one grid (sticky inter-material contact); the `Solver` interface must not need changes to add inter-material contact later.
- Commit after every task. Commit messages end with `Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>`.

## Review Focus

- **Emitter mesh that isn't closed** (common in Blender: a plane or a mesh with a hole used as a fill): expect a clear `ValueError` naming the problem, not silent zero particles. Test in Task 2.
- **Non-cubic domains** (a wide, flat box): grid dimensions per axis, walls at the box faces, contact accuracy holds. Test in Task 3.
- **Scaled colliders and emitters** (Blender objects are routinely scaled, including non-uniformly): collision surface and sampled volume match the scaled shape. Tests in Task 2 and Task 5.
- **Emitter partly outside the domain:** particles outside are dropped with a warning in progress, never indexed out of the grid. Test in Task 6.
- **Resume after the scene was changed:** resume refuses with a clear error instead of continuing a stale simulation. Test in Task 6.

## File Structure

```
pyproject.toml                         package metadata, deps, pytest config
src/mpm_core/__init__.py
src/mpm_core/materials.py              MaterialParams, Look, Preset, preset loading
src/mpm_core/presets/*.toml            whipped_cream, snow_packed, snow_powder, sand_dry, jelly
src/mpm_core/geometry.py               mesh_is_closed, sample_mesh_volume, transform_points
src/mpm_core/scene.py                  Domain, MeshData, Emitter, Collider, SceneDescription, read/write, hash
src/mpm_core/solver.py                 Solver ABC, ParticleData, AdvanceStats, ColliderGeometry
src/mpm_core/backends/__init__.py      backend registry: register_backend, create_solver
src/mpm_core/backends/warp/__init__.py registers "warp"
src/mpm_core/backends/warp/kernels.py  P2G, grid update, G2P, constitutive + plasticity funcs
src/mpm_core/backends/warp/colliders.py mesh-collider grid and particle projection kernels
src/mpm_core/backends/warp/solver.py   WarpSolver(Solver)
src/mpm_core/cache.py                  CacheWriter, CacheReader, checkpoints
src/mpm_core/bake.py                   bake() + CLI entry (python -m mpm_core.bake)
tests/conftest.py                      gpu marker, device fixture, small scene helpers
tests/test_materials.py, test_geometry.py, test_scene.py, test_warp_basic.py,
tests/test_warp_materials.py, test_warp_colliders.py, test_cache.py, test_bake.py
examples/scene_mix.py, examples/scene_ball.py   spike scenes as scene descriptions
examples/bench.py                      timing benchmark
tools/blender_view_cache.py            minimal Blender viewer for the new cache (dev tool)
```

The spike (`spike/mpm_spike.py`) has working kernel math for MLS-MPM, Hencky/von Mises, Drucker-Prager, and Stomakhin snow. Port the math; do not copy its structure.

---

### Task 1: Package scaffold, materials and presets

**Files:**
- Create: `pyproject.toml`, `src/mpm_core/__init__.py`, `src/mpm_core/materials.py`, `src/mpm_core/presets/{whipped_cream,snow_packed,snow_powder,sand_dry,jelly}.toml`, `tests/conftest.py`, `tests/test_materials.py`

**Interfaces:**
- Produces:
  - `MODELS = ("elastic", "von_mises", "drucker_prager", "snow")`
  - `@dataclass(frozen=True) MaterialParams(name: str, model: str, youngs_modulus: float, poisson_ratio: float, density: float, yield_stress: float = 0.0, friction_angle_deg: float = 0.0, hardening: float = 0.0, critical_compression: float = 0.0, critical_stretch: float = 0.0, max_hardening: float = 5.0)` with `lame() -> tuple[float, float]` returning `(mu, lam)`, `to_dict() -> dict`, `from_dict(d) -> MaterialParams` (classmethod). Validates in `__post_init__`: model in `MODELS`, `youngs_modulus > 0`, `0 <= poisson_ratio < 0.5`, `density > 0`; else `ValueError`.
  - `@dataclass(frozen=True) Look(display: str, base_color: tuple[float, float, float], roughness: float, subsurface: float)`; `display` in `("mesh", "points")`.
  - `@dataclass(frozen=True) Preset(id: str, label: str, material: MaterialParams, look: Look)`
  - `list_presets() -> list[str]` (sorted ids), `load_preset(preset_id: str) -> Preset` (`KeyError` for unknown id).

Preset values (TOML: top-level `label`, tables `[material]` and `[look]`):

| id | label | model | E | ν | ρ | other | display | base_color | rough | sss |
|---|---|---|---|---|---|---|---|---|---|---|
| whipped_cream | Whipped cream | von_mises | 1.0e5 | 0.3 | 400 | yield_stress 2500 | mesh | (0.97, 0.95, 0.90) | 0.5 | 0.8 |
| snow_packed | Snow, packed | snow | 1.4e5 | 0.2 | 400 | hardening 10, crit_compression 2.5e-2, crit_stretch 4.5e-3 | mesh | (0.85, 0.9, 1.0) | 0.3 | 0.3 |
| snow_powder | Snow, powder | snow | 4.0e4 | 0.2 | 200 | hardening 10, crit_compression 3.0e-2, crit_stretch 2.5e-3 | points | (0.9, 0.93, 1.0) | 0.4 | 0.2 |
| sand_dry | Sand, dry | drucker_prager | 3.5e5 | 0.3 | 2200 | friction_angle_deg 30 | points | (0.76, 0.6, 0.38) | 0.9 | 0.0 |
| jelly | Jelly | elastic | 5.0e4 | 0.3 | 1000 | — | mesh | (0.8, 0.15, 0.2) | 0.15 | 0.5 |

- [ ] **Step 1: Write `pyproject.toml`** with project `mpm-core`, src layout, `requires-python = ">=3.11"`, dependencies `numpy`, `warp-lang>=1.18`, optional `dev = ["pytest"]`, package data `mpm_core/presets/*.toml`, pytest marker `gpu: needs a CUDA device`. Install: `uv pip install --python .venv -e ".[dev]"`. Expected: installs without error.

- [ ] **Step 2: Write `tests/conftest.py`**: fixture `device` returning `"cuda:0"`, and auto-skip of `gpu`-marked tests when no CUDA device is found (detect via `warp.get_cuda_device_count()` inside the conftest, imported lazily so non-gpu tests don't need it).

- [ ] **Step 3: Write failing tests in `tests/test_materials.py`**

```python
def test_list_presets():
    assert list_presets() == ["jelly", "sand_dry", "snow_packed", "snow_powder", "whipped_cream"]

def test_load_cream():
    p = load_preset("whipped_cream")
    assert p.label == "Whipped cream"
    assert p.material.model == "von_mises" and p.material.yield_stress == 2500
    assert p.look.display == "mesh"

def test_lame():
    mu, lam = MaterialParams("m", "elastic", 1.0e5, 0.3, 1000).lame()
    assert mu == pytest.approx(1.0e5 / 2.6) and lam == pytest.approx(1.0e5 * 0.3 / (1.3 * 0.4))

def test_roundtrip_dict():
    m = load_preset("sand_dry").material
    assert MaterialParams.from_dict(m.to_dict()) == m

@pytest.mark.parametrize("kw", [{"model": "goo"}, {"poisson_ratio": 0.5}, {"youngs_modulus": 0}, {"density": -1}])
def test_invalid(kw):
    base = dict(name="m", model="elastic", youngs_modulus=1e5, poisson_ratio=0.3, density=1000)
    with pytest.raises(ValueError):
        MaterialParams(**{**base, **kw})

def test_unknown_preset():
    with pytest.raises(KeyError):
        load_preset("custard")
```

- [ ] **Step 4: Run** `.venv/Scripts/python -m pytest tests/test_materials.py -v` (Linux: `.venv/bin/python`). Expected: FAIL (import error).

- [ ] **Step 5: Implement `materials.py` and the five TOML files** per the table. Load TOML with `importlib.resources`.

- [ ] **Step 6: Run** the same command. Expected: all PASS.

- [ ] **Step 7: Commit** `pyproject.toml src/mpm_core tests/conftest.py tests/test_materials.py` with message `feat: mpm_core package with material presets`.

---

### Task 2: Geometry and scene description

**Files:**
- Create: `src/mpm_core/geometry.py`, `src/mpm_core/scene.py`, `tests/test_geometry.py`, `tests/test_scene.py`

**Interfaces:**
- Consumes: `MaterialParams.to_dict/from_dict` (Task 1).
- Produces (`geometry.py`):
  - `mesh_is_closed(tris: np.ndarray) -> bool`: every undirected edge is shared by exactly two triangles.
  - `transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray`: (4,4) applied to (N,3), float32 out.
  - `sample_mesh_volume(verts: np.ndarray, tris: np.ndarray, spacing: float, seed: int) -> np.ndarray`: (N,3) float32 points inside a closed mesh (verts already in world space), lattice at `spacing` aligned to world origin, each point jittered uniformly by ±0.25·spacing with `np.random.default_rng(seed)`. Raises `ValueError("emitter mesh is not closed ...")` if `not mesh_is_closed(tris)`. Inside test: z-ray parity per (x, y) lattice column (for each triangle, find the columns inside its xy projection and the z of the hit; sort hits per column; points between odd/even pairs are inside). Must handle ~500k points from a ~10k triangle mesh in under 2 s.
- Produces (`scene.py`):
  - `WALL_SIDES = ("x-", "x+", "y-", "y+", "z-", "z+")`
  - `@dataclass Domain(min: tuple3, max: tuple3, resolution: int, walls: dict[str, bool] = all True, wall_friction: float = 0.5)` with properties `size`, `dx = max(size) / resolution`, `grid_shape -> tuple[int,int,int] = ceil(size / dx)` per axis.
  - `@dataclass MeshData(verts: np.ndarray (V,3) f32, tris: np.ndarray (T,3) int32)`
  - `@dataclass Emitter(name: str, kind: str, material: int, mesh: MeshData, transforms: np.ndarray (F,4,4) f32, velocity: tuple3 = (0,0,0), use_object_velocity: bool = False, frame_start: int = 1, frame_end: int = 1)`; `kind` in `("fill", "inflow")`. Fill emitters emit once at the scene's `frame_start`.
  - `@dataclass Collider(name: str, mesh: MeshData, transforms: np.ndarray (F,4,4) f32, friction: float = 0.5, closed: bool = True)`
  - `@dataclass SceneDescription(domain: Domain, materials: list[MaterialParams], emitters: list[Emitter], colliders: list[Collider], gravity: tuple3 = (0, 0, -9.81), fps: float = 24.0, frame_start: int = 1, frame_end: int = 250, seed: int = 0, particles_per_cell: int = 8)`. `transforms[i]` is the object's world matrix at frame `frame_start + i`; length must be `frame_end - frame_start + 1`. `particles_per_cell` must be a perfect cube (1, 8, 27). Validation raises `ValueError`.
  - `write_scene(scene: SceneDescription, directory: Path) -> None`: writes `scene.json` (format `version: 1`) and one `.npy` per array (`emitter_{i}_verts.npy`, `emitter_{i}_tris.npy`, `emitter_{i}_xf.npy`, same for `collider_{i}`).
  - `read_scene(directory: Path) -> SceneDescription`
  - `scene_hash(directory: Path) -> str`: sha256 over `scene.json` and all referenced `.npy` files, in a fixed order.

- [ ] **Step 1: Write failing tests in `tests/test_geometry.py`**

```python
# helpers: unit_cube_mesh() -> (verts, tris) for [0,1]^3 (12 tris); icosphere(radius, subdivisions) -> (verts, tris)
def test_closed():
    v, t = unit_cube_mesh()
    assert mesh_is_closed(t) and not mesh_is_closed(t[:-1])

def test_cube_count():
    v, t = unit_cube_mesh()
    pts = sample_mesh_volume(v * 0.2, t, 0.01, seed=0)
    assert abs(len(pts) - 8000) <= 8000 * 0.02
    assert pts.min() >= -0.0025 and pts.max() <= 0.2025

def test_sphere_volume():
    v, t = icosphere(0.1, 4)
    pts = sample_mesh_volume(v, t, 0.005, seed=0)
    assert len(pts) == pytest.approx(4 / 3 * np.pi * 0.1**3 / 0.005**3, rel=0.03)

def test_deterministic():
    v, t = icosphere(0.1, 3)
    assert np.array_equal(sample_mesh_volume(v, t, 0.01, 7), sample_mesh_volume(v, t, 0.01, 7))

def test_open_mesh_raises():
    v, t = unit_cube_mesh()
    with pytest.raises(ValueError, match="not closed"):
        sample_mesh_volume(v, t[:-2], 0.01, 0)

def test_scaled_mesh_volume():  # Review Focus: scaled emitters
    v, t = unit_cube_mesh()
    m = np.diag([0.2, 0.1, 0.4, 1.0]).astype(np.float32)
    pts = sample_mesh_volume(transform_points(m, v), t, 0.01, 0)
    assert len(pts) == pytest.approx(0.2 * 0.1 * 0.4 / 0.01**3, rel=0.03)
```

- [ ] **Step 2: Write failing tests in `tests/test_scene.py`**

```python
def test_domain_grid():
    d = Domain(min=(0, 0, 0), max=(1.0, 0.25, 0.5), resolution=64)
    assert d.dx == pytest.approx(1 / 64) and d.grid_shape == (64, 16, 32)

def test_roundtrip(tmp_path):
    scene = small_scene()  # 1 fill emitter (cube), 1 collider (plane, closed=False), frames 1..10
    write_scene(scene, tmp_path)
    back = read_scene(tmp_path)
    assert back.domain == scene.domain and back.materials == scene.materials
    assert np.array_equal(back.emitters[0].transforms, scene.emitters[0].transforms)
    assert back.colliders[0].closed is False

def test_hash_changes(tmp_path):
    write_scene(small_scene(), tmp_path)
    h = scene_hash(tmp_path)
    write_scene(small_scene(gravity=(0, 0, -5)), tmp_path)
    assert scene_hash(tmp_path) != h

def test_wrong_transform_count():
    with pytest.raises(ValueError):
        small_scene(emitter_frames=5)  # frames 1..10 need 10 transforms

def test_bad_particles_per_cell():
    with pytest.raises(ValueError):
        small_scene(particles_per_cell=6)
```

`small_scene(**overrides)` lives in `tests/conftest.py` so later tasks reuse it.

- [ ] **Step 3: Run** `.venv/Scripts/python -m pytest tests/test_geometry.py tests/test_scene.py -v`. Expected: FAIL (imports).
- [ ] **Step 4: Implement `geometry.py` and `scene.py`.**
- [ ] **Step 5: Run** the same. Expected: all PASS; `test_sphere_volume` under 2 s.
- [ ] **Step 6: Commit** with message `feat: scene description format and mesh volume sampling`.

---

### Task 3: Solver interface and Warp backend (elastic, walls, adaptive dt, state)

**Files:**
- Create: `src/mpm_core/solver.py`, `src/mpm_core/backends/__init__.py`, `src/mpm_core/backends/warp/{__init__,kernels,solver}.py`, `tests/test_warp_basic.py`

**Interfaces:**
- Consumes: `Domain`, `MeshData` (Task 2), `MaterialParams` (Task 1).
- Produces (`solver.py`):
  - `@dataclass ParticleData(positions: (N,3) f32, velocities: (N,3) f32, material_ids: (N,) int32, emitter_ids: (N,) int32)`
  - `@dataclass AdvanceStats(substeps: int, dt: float, removed: int)`
  - `@dataclass ColliderGeometry(mesh: MeshData, closed: bool, friction: float)`
  - `class Solver(ABC)`, constructor `(domain: Domain, materials: Sequence[MaterialParams], gravity: tuple3, particles_per_cell: int = 8, device: str | None = None)`, abstract methods:
    - `add_particles(positions, velocities, material_ids, emitter_ids) -> None`
    - `set_colliders(colliders: Sequence[ColliderGeometry]) -> None`
    - `set_collider_transforms(start: np.ndarray, end: np.ndarray) -> None`: (C,4,4) world matrices at the start and end of the next `advance`.
    - `advance(frame_dt: float) -> AdvanceStats`
    - `particle_count` (property) `-> int`
    - `get_particles() -> ParticleData`
    - `is_occupied(points: np.ndarray) -> np.ndarray` (bool, M): material exists within half a cell, as of the last substep.
    - `save_state() -> dict[str, np.ndarray]`, `load_state(state: dict[str, np.ndarray]) -> None`
  - Particle volume is `dx³ / particles_per_cell`. The bake runner samples at the matching spacing `dx / cbrt(particles_per_cell)`.
  - Materials are passed as one table and all share one grid (sticky contact). Adding inter-material contact later must stay internal to a backend; don't add per-material grid concepts to the interface now.
- Produces (`backends/__init__.py`): `register_backend(name: str, factory: Callable[..., Solver])`, `create_solver(backend: str, **kwargs) -> Solver` (`KeyError` lists known backends). `"warp"` is registered lazily on first `create_solver("warp")`, so importing `mpm_core.backends` doesn't import warp.
- `WarpSolver` behaviour:
  - Dense grid of `domain.grid_shape` plus 3 padding cells on every side. Domain walls lie exactly on the domain box faces.
  - Walls: grid nodes beyond an enabled wall get no-penetration with Coulomb friction `domain.wall_friction`. Particles past an enabled wall are projected back onto the wall plane in G2P, which gives the 0.25·dx contact requirement.
  - Open sides: particles that move more than one cell past an open side are removed at the end of `advance` (reported in `AdvanceStats.removed`).
  - Material table on the GPU, indexed by per-particle material id (as in the spike). In this task only `"elastic"` (fixed-corotated) is supported; other models raise `NotImplementedError` until Task 4.
  - Adaptive dt, computed once per `advance` call:
    - `c = max over materials present of sqrt((lam + 2 mu) * hmax / rho)`, with `hmax = max_hardening` for snow and 1 otherwise.
    - `dt = min(0.4 * dx / (c + vmax), frame_dt)`, where `vmax` is the max particle speed, read from a GPU reduction once per `advance`.
    - `substeps = ceil(frame_dt / dt)`.
  - CUDA graph capture is the builder's choice.

- [ ] **Step 1: Write failing tests in `tests/test_warp_basic.py`** (all `@pytest.mark.gpu`). Helper `cube_particles(lo, hi, dx)` samples a box at spacing dx/2 using `sample_mesh_volume`. Use `JELLY = load_preset("jelly").material`.

```python
def test_free_fall(device):
    d = Domain((0, 0, 0), (1, 1, 1), 64)
    s = create_solver("warp", domain=d, materials=[JELLY], gravity=(0, 0, -9.81), device=device)
    p = cube_particles((0.45, 0.45, 0.6), (0.55, 0.55, 0.7), d.dx)
    s.add_particles(p, np.zeros_like(p), np.zeros(len(p), np.int32), np.zeros(len(p), np.int32))
    z0 = p[:, 2].mean()
    for _ in range(6):
        s.advance(1 / 24)
    assert z0 - s.get_particles().positions[:, 2].mean() == pytest.approx(0.5 * 9.81 * 0.25**2, rel=0.02)

def test_rests_on_floor(device):  # contact accuracy + no sag (10 cm: jelly's own elastic squash stays < 0.5 dx)
    d = Domain((0, 0, 0), (1, 1, 1), 64)
    s = solver_with_cube(device, d, lo=(0.4, 0.4, 0.0), hi=(0.5, 0.5, 0.1))
    start = s.get_particles().positions.copy()
    for _ in range(24):
        s.advance(1 / 24)
    pos = s.get_particles().positions
    assert pos[:, 2].min() >= -0.25 * d.dx
    assert np.abs(pos - start).max() < 0.5 * d.dx

def test_non_cubic_domain(device):  # Review Focus: flat, wide domain
    d = Domain((-0.5, -0.2, 0.0), (0.5, 0.05, 0.5), 64)
    assert d.grid_shape == (64, 16, 32)
    s = solver_with_cube(device, d, lo=(-0.05, -0.1, 0.3), hi=(0.05, 0.0, 0.4))
    for _ in range(24):
        s.advance(1 / 24)
    pos = s.get_particles().positions
    assert pos[:, 2].min() >= -0.25 * d.dx and np.isfinite(pos).all()

def test_open_side_removes(device):
    d = Domain((0, 0, 0), (1, 1, 1), 32, walls={**ALL_WALLS, "z-": False})
    s = solver_with_cube(device, d, lo=(0.4, 0.4, 0.1), hi=(0.6, 0.6, 0.2))
    removed = sum(s.advance(1 / 24).removed for _ in range(24))
    assert s.particle_count == 0 and removed > 0

def test_state_roundtrip(device):
    d = Domain((0, 0, 0), (1, 1, 1), 32)
    a = solver_with_cube(device, d, lo=(0.4, 0.4, 0.3), hi=(0.6, 0.6, 0.5))
    a.advance(1 / 24)
    b = create_solver("warp", domain=d, materials=[JELLY], gravity=(0, 0, -9.81), device=device)
    b.load_state(a.save_state())
    a.advance(1 / 24)
    b.advance(1 / 24)
    assert np.allclose(a.get_particles().positions, b.get_particles().positions, atol=1e-5)

def test_is_occupied(device):
    d = Domain((0, 0, 0), (1, 1, 1), 32)
    s = solver_with_cube(device, d, lo=(0.4, 0.4, 0.0), hi=(0.6, 0.6, 0.2))
    s.advance(1 / 24)
    assert s.is_occupied(np.array([[0.5, 0.5, 0.1], [0.9, 0.9, 0.9]], np.float32)).tolist() == [True, False]

def test_unknown_backend():
    with pytest.raises(KeyError, match="warp"):
        create_solver("nope", domain=None, materials=[], gravity=(0, 0, 0))
```

`solver_with_cube` and `ALL_WALLS` go in `tests/conftest.py`.

- [ ] **Step 2: Run** `.venv/Scripts/python -m pytest tests/test_warp_basic.py -v`. Expected: FAIL (imports).
- [ ] **Step 3: Implement `solver.py` and `backends/__init__.py`.**
- [ ] **Step 4: Implement the Warp backend.** MLS-MPM/APIC with quadratic B-splines, ported from `spike/mpm_spike.py` (`p2g`, `grid_update`, `g2p`), extended to a non-cubic grid with padding and an origin offset, plus wall planes, particle wall projection, removal and the adaptive dt above.
- [ ] **Step 5: Run** the same. Expected: all PASS.
- [ ] **Step 6: Commit** with message `feat: solver interface and Warp backend core`.

---

### Task 4: Plasticity models and preset behaviour

**Files:**
- Modify: `src/mpm_core/backends/warp/kernels.py`, `src/mpm_core/backends/warp/solver.py`
- Create: `tests/test_warp_materials.py`

**Interfaces:**
- Consumes: Task 3 solver, Task 1 presets.
- Produces: `WarpSolver` supports all four `MODELS`:
  - `von_mises`: Hencky elasticity, von Mises return mapping on the log-strain deviator at `yield_stress / (2 mu)`.
  - `drucker_prager`: Klar et al. 2016, with `alpha = sqrt(2/3) * 2 sin(phi) / (3 - sin(phi))`. Positive volumetric log-strain projects to stress-free.
  - `snow`: fixed-corotated with Stomakhin hardening `h = min(exp(hardening * (1 - Jp)), max_hardening)`, and singular values clamped to `[1 - critical_compression, 1 + critical_stretch]`.
  - Math as in `spike/mpm_spike.py` (`project_von_mises`, `project_drucker_prager`, the snow branch).

- [ ] **Step 1: Write failing tests in `tests/test_warp_materials.py`** (all `gpu`; resolution 64 for speed)

```python
@pytest.mark.parametrize("preset", list_presets())
def test_slab_at_rest(device, preset):  # spec: no sag at frame 1
    d = Domain((0, 0, 0), (0.5, 0.5, 0.5), 64)
    s = solver_with_box(device, d, load_preset(preset).material, lo=(0, 0, 0), hi=(0.5, 0.5, 0.07))  # wall to wall
    start = s.get_particles().positions.copy()
    for _ in range(24):
        s.advance(1 / 24)
    assert np.abs(s.get_particles().positions - start).max() < 0.5 * d.dx

def test_sand_column_collapses(device):
    d = Domain((0, 0, 0), (1, 1, 1), 64)
    s = solver_with_box(device, d, load_preset("sand_dry").material, lo=(0.44, 0.44, 0), hi=(0.56, 0.56, 0.33))
    for _ in range(48):
        s.advance(1 / 24)
    pos = s.get_particles().positions
    assert pos[:, 2].max() < 0.6 * 0.33
    assert 0.12 < pos[:, 0].max() - 0.5 < 0.45

def test_cream_holds_shape(device):
    d = Domain((0, 0, 0), (1, 1, 1), 64)
    s = solver_with_sphere(device, d, load_preset("whipped_cream").material, center=(0.5, 0.5, 0.11), radius=0.11)
    for _ in range(48):
        s.advance(1 / 24)
    z = s.get_particles().positions[:, 2]
    assert z.max() - z.min() > 0.5 * 0.22

def test_jelly_recovers(device):  # elastic: drops, bounces, comes back to ~original height
    d = Domain((0, 0, 0), (1, 1, 1), 64)
    s = solver_with_box(device, d, load_preset("jelly").material, lo=(0.4, 0.4, 0.2), hi=(0.6, 0.6, 0.4))
    for _ in range(72):
        s.advance(1 / 24)
    z = s.get_particles().positions[:, 2]
    assert z.max() - z.min() == pytest.approx(0.2, rel=0.15)

def test_mixed_materials_one_domain(device):
    d = Domain((0, 0, 0), (1, 1, 1), 64)
    s = solver_with_presets(device, d, [("sand_dry", (0.3, 0.3, 0), (0.5, 0.5, 0.1)),
                                         ("snow_packed", (0.5, 0.5, 0), (0.7, 0.7, 0.1))])
    for _ in range(24):
        s.advance(1 / 24)
    assert np.isfinite(s.get_particles().positions).all()
```

`solver_with_box`, `solver_with_sphere` and `solver_with_presets` go in `tests/conftest.py`.

- [ ] **Step 2: Run** `.venv/Scripts/python -m pytest tests/test_warp_materials.py -v`. Expected: FAIL (`NotImplementedError` for non-elastic models).
- [ ] **Step 3: Implement the three plasticity models.** If `test_slab_at_rest` fails for a preset, choose the mitigation (spec leaves it to the builder: e.g. a gravity ramp over the first 0.1 s, or preset tuning). Record the choice in a comment at the dt/gravity code.
- [ ] **Step 4: Run** the same. Expected: all PASS.
- [ ] **Step 5: Commit** with message `feat: von Mises, Drucker-Prager and snow plasticity`.

---

### Task 5: Mesh colliders

**Files:**
- Create: `src/mpm_core/backends/warp/colliders.py`, `tests/test_warp_colliders.py`
- Modify: `src/mpm_core/backends/warp/solver.py`

**Interfaces:**
- Consumes: `ColliderGeometry`, `set_colliders`, `set_collider_transforms` (Task 3).
- Produces: `WarpSolver` collider support.
  - One `wp.Mesh` per collider, in local space.
  - Within an `advance`, each collider's transform at substep time `t` is the linear interpolation of the start and end matrices. Its velocity at a point is `(end - start) / frame_dt` applied to the point's local coordinates.
  - **Grid:** for nodes with mass, query signed distance in local space, then scale back to world. Nodes inside get frictional contact against the collider velocity (as the spike's ball contact, using `ColliderGeometry.friction`).
  - **Particles:** after advection in G2P, particles inside are projected to the surface along the normal. This gives the 0.25·dx requirement.
  - Inside test: closed meshes use the sign from `wp.mesh_query_point_sign_normal`. Open meshes (`closed=False`) count as inside only on the back side and within `2·dx` of the surface.
  - Colliders are cleared and replaced by `set_colliders`. With no colliders set, no collider work runs.

- [ ] **Step 1: Write failing tests in `tests/test_warp_colliders.py`** (all `gpu`, resolution 64, domain unit cube, jelly unless stated). Helpers: `plane_mesh(size)` (2 tris, normal +z, open), `unit_cube_mesh()` centered at origin.

```python
def test_rests_on_open_plane(device):
    s = jelly_cube(device, lo=(0.4, 0.4, 0.45), hi=(0.6, 0.6, 0.55))
    s.set_colliders([ColliderGeometry(plane_mesh(0.8), closed=False, friction=0.5)])
    xf = translate(0.5, 0.5, 0.3)
    s.set_collider_transforms(xf[None], xf[None])
    run(s, 36)
    zmin = s.get_particles().positions[:, 2].min()
    assert 0.3 - 0.25 * DX <= zmin <= 0.3 + 1.0 * DX

def test_scaled_box_collider(device):  # Review Focus: scaled colliders
    s = jelly_cube(device, lo=(0.45, 0.45, 0.4), hi=(0.55, 0.55, 0.5))
    s.set_colliders([ColliderGeometry(unit_cube_mesh(), closed=True, friction=0.5)])
    xf = translate(0.5, 0.5, 0.0) @ scale(0.6, 0.6, 0.4)  # top face at z = 0.2
    s.set_collider_transforms(xf[None], xf[None])
    run(s, 36)
    assert s.get_particles().positions[:, 2].min() == pytest.approx(0.2, abs=1.0 * DX)

@pytest.mark.parametrize("friction,slides", [(0.0, True), (1.0, False)])
def test_tilted_plane_friction(device, friction, slides):
    s = jelly_cube_on_plane_tilted(device, degrees=30, friction=friction)
    start = s.get_particles().positions.mean(0)
    run(s, 24)
    moved = np.linalg.norm(s.get_particles().positions.mean(0) - start)
    assert (moved > 0.05) if slides else (moved < 0.01)

def test_moving_box_clears_sand(device):
    s = sand_layer(device, lo=(0.1, 0.3, 0), hi=(0.9, 0.7, 0.06))
    s.set_colliders([ColliderGeometry(unit_cube_mesh(), closed=True, friction=0.5)])
    box = lambda x: translate(x, 0.5, 0.05) @ scale(0.1, 0.5, 0.1)
    for f in range(24):  # 0.3 m/s along x
        s.set_collider_transforms(box(0.15 + 0.3 * f / 24)[None], box(0.15 + 0.3 * (f + 1) / 24)[None])
        s.advance(1 / 24)
    inside = points_in_box(s.get_particles().positions, box(0.45))
    assert inside.sum() <= 0.001 * s.particle_count
```

`translate`, `scale`, `run`, `DX`, `jelly_cube`, `jelly_cube_on_plane_tilted`, `sand_layer` and `points_in_box` go in this test file or in `conftest.py`.

- [ ] **Step 2: Run** `.venv/Scripts/python -m pytest tests/test_warp_colliders.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement `colliders.py`** and wire it into the grid update and G2P.
- [ ] **Step 4: Run** this file and `tests/test_warp_basic.py tests/test_warp_materials.py`. Expected: all PASS.
- [ ] **Step 5: Commit** with message `feat: animated mesh colliders with friction`.

---

### Task 6: Cache and bake runner

**Files:**
- Create: `src/mpm_core/cache.py`, `src/mpm_core/bake.py` (the CLI is `python -m mpm_core.bake`), `tests/test_cache.py`, `tests/test_bake.py`

**Interfaces:**
- Consumes: everything above.
- Cache directory layout (documented in the `cache.py` module docstring):
  ```
  <cache_dir>/scene/            scene description (written by the add-on, read by the runner)
  <cache_dir>/cache.json        {"version": 1, "emitters": [names], "fps", "frame_start", "frame_end", "dx", "backend"}
  <cache_dir>/frames/frame_NNNN.npz   per emitter i: pos_i (n,3) f32, vel_i (n,3) f32; written to .tmp then os.replace
  <cache_dir>/state/checkpoint.npz + checkpoint.json {"frame", "scene_hash"}   overwritten after every frame
  <cache_dir>/progress.json     {"status": "running|done|cancelled|error", "frame", "frame_end", "particles", "ms_per_frame", "message", "warnings": []}
  <cache_dir>/cancel            presence requests a cancel; the runner deletes it when it stops
  ```
- `cache.py`:
  - `CacheWriter(cache_dir: Path, emitter_names: list[str], fps: float, frame_start: int, frame_end: int, dx: float, backend: str)`
    - `write_frame(frame: int, particles: ParticleData) -> None`: splits by emitter id.
  - `CacheReader(cache_dir: Path)`
    - attributes: `.emitter_names`, `.fps`, `.frame_start`, `.frame_end`, `.dx`
    - `available_frames() -> list[int]`
    - `read_frame(frame: int) -> dict[str, tuple[np.ndarray, np.ndarray]]`: emitter name → `(positions, velocities)`.
  - `write_checkpoint(cache_dir, frame: int, state: dict, scene_hash: str) -> None`
  - `read_checkpoint(cache_dir) -> tuple[int, dict, str] | None`
- `bake.py`:
  - `@dataclass BakeResult(status: str, frames_written: int, message: str = "")`
  - `class SceneChangedError(RuntimeError)`
  - `bake(cache_dir: Path, *, backend: str = "warp", device: str | None = None, resume: bool = False, should_cancel: Callable[[], bool] | None = None) -> BakeResult`. `should_cancel` is checked after each frame, alongside the cancel file.
  - CLI `python -m mpm_core.bake CACHE_DIR [--backend B] [--device D] [--resume]`: exit code 0 on done or cancelled, 1 on error. Prints one line per frame.
- Bake loop behaviour:
  - Fill emitters are sampled at `frame_start` with spacing `dx / cbrt(particles_per_cell)` and seed `scene.seed + emitter index`.
  - Inflow emitters are re-sampled each active frame, in world space at that frame's transform, with seed `scene.seed + 1000 * frame + index`. Points where `is_occupied` is true are dropped. Velocity is `emitter.velocity`, plus object velocity if `use_object_velocity`.
  - Points outside the domain box are dropped, and the warning `"<emitter>: N particles outside the domain were dropped"` is added to `progress.warnings` (once per emitter).
  - Frame `frame_start` holds the initial state. Each later frame is one `advance(1 / fps)`.
  - After every frame: NaN check on positions, then `write_frame`, `write_checkpoint`, update progress.
  - On NaN: status `error`, message containing `"frame N"`. No file is written for frame N.
  - On resume: load the checkpoint. If its `scene_hash` differs from the current one, raise `SceneChangedError`. Otherwise continue from checkpoint frame + 1.

- [ ] **Step 1: Write failing tests in `tests/test_cache.py`** (no GPU)

```python
def test_write_read_frame(tmp_path):
    w = CacheWriter(tmp_path, ["a", "b"], 24, 1, 10, 0.01, "warp")
    pd = ParticleData(np.random.rand(5, 3).astype(np.float32), np.zeros((5, 3), np.float32),
                      np.zeros(5, np.int32), np.array([0, 1, 1, 0, 1], np.int32))
    w.write_frame(3, pd)
    r = CacheReader(tmp_path)
    assert r.available_frames() == [3] and r.emitter_names == ["a", "b"]
    f = r.read_frame(3)
    assert f["a"][0].shape == (2, 3) and f["b"][0].shape == (3, 3)

def test_no_tmp_left(tmp_path): ...       # after write_frame, no *.tmp files remain in frames/
def test_checkpoint_roundtrip(tmp_path): ...  # write_checkpoint then read_checkpoint returns same frame, arrays, hash
```

- [ ] **Step 2: Write failing tests in `tests/test_bake.py`.** Non-GPU tests use a `FakeSolver` registered as backend `"fake"` in the test module. It moves particles by gravity, and can be told to return NaN at a given frame. GPU tests use a small real scene (`small_scene` from conftest at resolution 32).

```python
def test_bake_writes_all_frames(tmp_path):  # fake
    make_cache_dir(tmp_path, small_scene(frame_end=6))
    res = bake(tmp_path, backend="fake")
    assert res.status == "done" and CacheReader(tmp_path).available_frames() == [1, 2, 3, 4, 5, 6]
    assert json.loads((tmp_path / "progress.json").read_text())["status"] == "done"

def test_nan_stops(tmp_path):  # fake, NaN at frame 4
    make_cache_dir(tmp_path, small_scene(frame_end=6))
    res = bake(tmp_path, backend="fake_nan4")
    assert res.status == "error" and "frame 4" in res.message
    assert CacheReader(tmp_path).available_frames() == [1, 2, 3]

def test_cancel_and_resume(tmp_path):  # fake
    make_cache_dir(tmp_path, small_scene(frame_end=8))
    calls = iter(range(100))
    res = bake(tmp_path, backend="fake", should_cancel=lambda: next(calls) >= 2)
    assert res.status == "cancelled"
    n = len(CacheReader(tmp_path).available_frames())
    assert bake(tmp_path, backend="fake", resume=True).status == "done"
    assert CacheReader(tmp_path).available_frames() == list(range(1, 9)) and n < 8

def test_cancel_file(tmp_path): ...  # touch <cache>/cancel before bake -> status "cancelled", cancel file removed

def test_resume_refuses_changed_scene(tmp_path):  # Review Focus
    make_cache_dir(tmp_path, small_scene(frame_end=8))
    bake(tmp_path, backend="fake", should_cancel=lambda: True)
    write_scene(small_scene(frame_end=8, gravity=(0, 0, -5)), tmp_path / "scene")
    with pytest.raises(SceneChangedError):
        bake(tmp_path, backend="fake", resume=True)

def test_emitter_outside_domain_warns(tmp_path):  # Review Focus; fake
    make_cache_dir(tmp_path, small_scene(emitter_offset=(0.9, 0, 0)))  # cube half outside x+
    bake(tmp_path, backend="fake")
    warnings = json.loads((tmp_path / "progress.json").read_text())["warnings"]
    assert any("outside the domain were dropped" in w for w in warnings)
    pos = CacheReader(tmp_path).read_frame(1)["emitter_0"][0]
    assert (pos[:, 0] <= 1.0).all()

def test_empty_scene(tmp_path):  # fake: no emitters -> frames written, zero particles, done

@pytest.mark.gpu
def test_inflow_emits_while_active(tmp_path):
    make_cache_dir(tmp_path, small_scene(kind="inflow", velocity=(0, 0, -1), emit_frames=(1, 10), frame_end=20))
    assert bake(tmp_path).status == "done"
    r = CacheReader(tmp_path)
    counts = [len(r.read_frame(f)["emitter_0"][0]) for f in range(1, 21)]
    assert counts[9] > counts[1] and counts[10:] == [counts[10]] * 10

@pytest.mark.gpu
def test_cli(tmp_path):
    make_cache_dir(tmp_path, small_scene(frame_end=3))
    out = subprocess.run([sys.executable, "-m", "mpm_core.bake", str(tmp_path)], capture_output=True, text=True)
    assert out.returncode == 0 and CacheReader(tmp_path).available_frames() == [1, 2, 3]
```

`make_cache_dir(path, scene)` writes the scene to `path / "scene"`. The `small_scene` overrides used here (`emitter_offset`, `kind`, `velocity`, `emit_frames`) are added to `small_scene` in conftest.

- [ ] **Step 3: Run** `.venv/Scripts/python -m pytest tests/test_cache.py tests/test_bake.py -v`. Expected: FAIL.
- [ ] **Step 4: Implement `cache.py` and `bake.py`.**
- [ ] **Step 5: Run** the full suite: `.venv/Scripts/python -m pytest -v`. Expected: all PASS.
- [ ] **Step 6: Commit** with message `feat: cache format and bake runner with progress, cancel, resume`.

---

### Task 7: Example scenes, benchmark, dev viewer, Linux check

**Files:**
- Create: `examples/scene_mix.py`, `examples/scene_ball.py`, `examples/bench.py`, `tools/blender_view_cache.py`, `README.md`

**Interfaces:**
- Consumes: `write_scene`, `bake`, `CacheReader`.
- `examples/scene_mix.py OUT_DIR [--res 128]` and `examples/scene_ball.py OUT_DIR [--res 128]` write the spike's two scenes as scene descriptions into `OUT_DIR/scene`, using icosphere/box meshes, presets, and floor walls only. For the ball scene, the ball is a closed icosphere collider whose per-frame transforms roll it at 0.2 m/s with the matching rotation.
- `examples/bench.py OUT_DIR`: runs `bake`, prints particle count, ms/frame average, and total time.
- `tools/blender_view_cache.py`: run from Blender's Text Editor with `CACHE_DIR` set at the top of the file. Creates one point object per emitter (mesh + Geometry Nodes Mesh to Points, radius 0.3·dx), and loads frames through a `frame_change_post` handler using `CacheReader` logic. To avoid importing the package in Blender, read the npz/json directly. It's a dev tool, superseded by the add-on.

- [ ] **Step 1: Write the example scripts and `bench.py`.**
- [ ] **Step 2: Run** `.venv/Scripts/python examples/scene_mix.py out/mix` then `.venv/Scripts/python examples/bench.py out/mix`. Expected: completes; with about 200k particles at 128³, average ms/frame ≤ 430 on the RTX 5080 (spike baseline at a fixed 5e-5 dt; adaptive dt should beat it). Record the numbers in `README.md`. If it's slower, investigate before continuing (graph capture, dt choice).
- [ ] **Step 3: Run** the ball scene the same way. Expected: completes, ms/frame recorded, no particles inside the ball at any frame. Check with a short snippet over `CacheReader` frames against the ball transforms.
- [ ] **Step 4: Open each cache with `tools/blender_view_cache.py` in Blender 5.2** and render frames 1, 48 and 96 headless. Verify visually: materials sit on the floor plane (no particles below it), the ball ploughs the snow, and the cream holds a dome.
- [ ] **Step 5: Write `README.md`**: what `mpm_core` is, setup (`uv venv`, `uv pip install -e ".[dev]"`), running tests, running examples, cache layout pointer, benchmark numbers.
- [ ] **Step 6: Linux check (author's machine with the RTX 4090):** clone, set up the venv, run `python -m pytest -v` and `examples/bench.py out/mix`. Expected: all tests pass. Add the 4090 numbers to `README.md`. This step needs the author; flag it when you reach it rather than skipping it.
- [ ] **Step 7: Add `out/` to `.gitignore`. Commit** `examples tools README.md .gitignore` with message `feat: example scenes, benchmark and dev cache viewer`.
