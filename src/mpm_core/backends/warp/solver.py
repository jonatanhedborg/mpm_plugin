"""WarpSolver: dense-grid MLS-MPM on the GPU with NVIDIA Warp."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import warp as wp

from mpm_core.backends.warp import colliders as CK
from mpm_core.backends.warp import kernels as K
from mpm_core.materials import MaterialParams
from mpm_core.scene import Domain
from mpm_core.solver import AdvanceStats, ColliderGeometry, ParticleData, Solver

PAD = 3  # padding cells around the domain so stencils never leave the grid
CFL = 0.4
COLLIDER_THICKNESS = 2.0  # open-mesh colliders are solid this many cells behind their surface
_STATE_KEYS = ("x", "v", "C", "F", "Jp", "mat", "emitter")


class WarpSolver(Solver):
    def __init__(self, domain: Domain, materials: Sequence[MaterialParams], gravity: tuple[float, float, float],
                 particles_per_cell: int = 8, device: str | None = None):
        super().__init__(domain, materials, gravity, particles_per_cell, device)
        wp.config.quiet = True
        wp.init()
        self.device = device or str(wp.get_preferred_device())
        for m in self.materials:
            if m.model not in K.MODEL_IDS:
                raise NotImplementedError(f"material model {m.model!r} is not supported by the Warp backend yet")

        self.dx = domain.dx
        self.inv_dx = 1.0 / self.dx
        self.p_vol = self.dx**3 / particles_per_cell
        self.dims = tuple(s + 2 * PAD for s in domain.grid_shape)
        self.origin = np.asarray(domain.min, np.float64) - PAD * self.dx
        self._wall_lo = wp.vec3(*domain.min)
        self._wall_hi = wp.vec3(*domain.max)
        self._lo_on = wp.vec3i(*(int(domain.walls[f"{a}-"]) for a in "xyz"))
        self._hi_on = wp.vec3i(*(int(domain.walls[f"{a}+"]) for a in "xyz"))
        self._clamp_lo = wp.vec3(*(self.origin + self.dx))
        self._clamp_hi = wp.vec3(*(self.origin + (np.asarray(self.dims) - 2) * self.dx))
        self._origin_wp = wp.vec3(*self.origin)
        # last node index at/below each lo wall and first at/above each hi wall
        self._wall_lo_node = wp.vec3i(*(int(math.floor((lo - o) / self.dx + 1e-3))
                                        for lo, o in zip(domain.min, self.origin)))
        self._wall_hi_node = wp.vec3i(*(int(math.ceil((hi - o) / self.dx - 1e-3))
                                        for hi, o in zip(domain.max, self.origin)))
        self._gravity_wp = wp.vec3(*self.gravity)

        mus, lams = zip(*(m.lame() for m in self.materials)) if self.materials else ((), ())
        d = self.device
        self._density = np.array([m.density for m in self.materials], np.float32)
        # P-wave speed per material, including the worst-case hardening for snow
        self._wave_speed = np.array([
            math.sqrt((lam + 2 * mu) * (m.max_hardening if m.model == "snow" else 1.0) / m.density)
            for m, mu, lam in zip(self.materials, mus, lams)], np.float64)
        self.m_model = wp.array([K.MODEL_IDS[m.model] for m in self.materials], dtype=int, device=d)
        self.m_mu = wp.array(mus, dtype=float, device=d)
        self.m_lam = wp.array(lams, dtype=float, device=d)
        self.m_yield = wp.array([m.yield_stress for m in self.materials], dtype=float, device=d)
        self.m_alpha = wp.array([_dp_alpha(m.friction_angle_deg) for m in self.materials], dtype=float, device=d)
        self.m_hardening = wp.array([m.hardening for m in self.materials], dtype=float, device=d)
        self.m_max_hardening = wp.array([m.max_hardening for m in self.materials], dtype=float, device=d)
        self.m_crit_c = wp.array([m.critical_compression for m in self.materials], dtype=float, device=d)
        self.m_crit_s = wp.array([m.critical_stretch for m in self.materials], dtype=float, device=d)

        self.grid_v = wp.zeros(self.dims, dtype=wp.vec3, device=d)
        self.grid_m = wp.zeros(self.dims, dtype=float, device=d)
        self._vmax = wp.zeros(1, dtype=float, device=d)
        self._meshes: list[wp.Mesh] = []
        self._frame_dt = 1.0
        self._upload({k: v for k, v in _empty_state().items()})

    # -- particle storage ---------------------------------------------------

    def _upload(self, s: dict[str, np.ndarray]) -> None:
        d = self.device
        self.x = wp.array(s["x"], dtype=wp.vec3, device=d)
        self.v = wp.array(s["v"], dtype=wp.vec3, device=d)
        self.C = wp.array(s["C"], dtype=wp.mat33, device=d)
        self.F = wp.array(s["F"], dtype=wp.mat33, device=d)
        self.Jp = wp.array(s["Jp"], dtype=float, device=d)
        self.mat = wp.array(s["mat"], dtype=int, device=d)
        self.emitter = wp.array(s["emitter"], dtype=int, device=d)
        self.p_mass = wp.array(self._density[s["mat"]] * self.p_vol if len(s["mat"]) else np.zeros(0, np.float32),
                               dtype=float, device=d)
        self._present = np.unique(s["mat"])
        self._n = len(s["x"])

    def _download(self) -> dict[str, np.ndarray]:
        return {"x": self.x.numpy(), "v": self.v.numpy(), "C": self.C.numpy(), "F": self.F.numpy(),
                "Jp": self.Jp.numpy(), "mat": self.mat.numpy(), "emitter": self.emitter.numpy()}

    def add_particles(self, positions, velocities, material_ids, emitter_ids) -> None:
        n = len(positions)
        if n == 0:
            return
        material_ids = np.asarray(material_ids, np.int32)
        if material_ids.min() < 0 or material_ids.max() >= len(self.materials):
            raise ValueError("material id out of range")
        new = {
            "x": np.asarray(positions, np.float32).reshape(n, 3),
            "v": np.asarray(velocities, np.float32).reshape(n, 3),
            "C": np.zeros((n, 3, 3), np.float32),
            "F": np.tile(np.eye(3, dtype=np.float32), (n, 1, 1)),
            "Jp": np.ones(n, np.float32),
            "mat": material_ids,
            "emitter": np.asarray(emitter_ids, np.int32),
        }
        old = self._download()
        self._upload({k: np.concatenate([old[k], new[k]]) for k in _STATE_KEYS})

    @property
    def particle_count(self) -> int:
        return self._n

    def get_particles(self) -> ParticleData:
        return ParticleData(self.x.numpy(), self.v.numpy(), self.mat.numpy(), self.emitter.numpy())

    def save_state(self) -> dict[str, np.ndarray]:
        return self._download()

    def load_state(self, state: dict[str, np.ndarray]) -> None:
        self._upload({k: np.asarray(state[k]) for k in _STATE_KEYS})

    # -- colliders ------------------------------------------------------------

    def set_colliders(self, colliders: Sequence[ColliderGeometry]) -> None:
        d = self.device
        self._meshes = [wp.Mesh(points=wp.array(c.mesh.verts, dtype=wp.vec3, device=d),
                                indices=wp.array(c.mesh.tris.ravel(), dtype=int, device=d)) for c in colliders]
        self._c_ids = wp.array([m.id for m in self._meshes], dtype=wp.uint64, device=d)
        self._c_closed = wp.array([int(c.closed) for c in colliders], dtype=int, device=d)
        self._c_friction = wp.array([c.friction for c in colliders], dtype=float, device=d)
        self._c_lo = wp.array([c.mesh.verts.min(0) for c in colliders], dtype=wp.vec3, device=d)
        self._c_hi = wp.array([c.mesh.verts.max(0) for c in colliders], dtype=wp.vec3, device=d)
        identity = np.tile(np.eye(4, dtype=np.float32), (len(colliders), 1, 1))
        self.set_collider_transforms(identity, identity)

    def set_collider_transforms(self, start: np.ndarray, end: np.ndarray) -> None:
        start = np.asarray(start, np.float32).reshape(-1, 4, 4)
        end = np.asarray(end, np.float32).reshape(-1, 4, 4)
        if len(start) != len(self._meshes) or len(end) != len(self._meshes):
            raise ValueError(f"expected {len(self._meshes)} collider transforms")
        d = self.device
        poses = []
        for a, b in zip(start, end):
            pose = CK.ColliderPose()
            (pose.t0, q0, pose.S0), (pose.t1, q1, pose.S1) = _decompose(a), _decompose(b)
            if np.dot(q0, q1) < 0:
                q1 = -q1  # slerp the short way round
            pose.q0, pose.q1 = wp.quat(*q0), wp.quat(*q1)
            poses.append(pose)
        self._c_poses = wp.array(poses, dtype=CK.ColliderPose, device=d)
        # AABB early-out margin in local units: a few cells, scaled by the largest inverse scale
        inv_scale = [max(1.0 / max(np.linalg.svd(m[:3, :3], compute_uv=False).min(), 1e-12) for m in (a, b))
                     for a, b in zip(start, end)]
        self._c_margin = wp.array([3.0 * self.dx * k for k in inv_scale], dtype=float, device=d)

    # -- stepping ---------------------------------------------------------------

    def _choose_dt(self, frame_dt: float) -> tuple[float, int]:
        self._vmax.zero_()
        wp.launch(K.max_speed, self._n, inputs=[self.v, self._vmax], device=self.device)
        vmax = float(self._vmax.numpy()[0])
        if not math.isfinite(vmax):
            vmax = 0.0  # already blown up; keep stepping so the NaN reaches the positions and gets reported
        c = float(self._wave_speed[self._present].max())
        dt = min(CFL * self.dx / (c + vmax), frame_dt)
        steps = math.ceil(frame_dt / dt - 1e-9)
        return frame_dt / steps, steps

    def _substep(self, dt: float, s: float) -> None:
        d = self.device
        self.grid_v.zero_()
        self.grid_m.zero_()
        wp.launch(K.p2g, self._n, inputs=[self.x, self.v, self.C, self.F, self.Jp, self.mat, self.p_mass,
                                          self.m_model, self.m_mu, self.m_lam, self.m_hardening,
                                          self.m_max_hardening, self.p_vol, self._origin_wp,
                                          self.inv_dx, self.dx, dt, self.grid_v, self.grid_m], device=d)
        wp.launch(K.grid_update, self.dims, inputs=[self.grid_v, self.grid_m, dt, self._gravity_wp,
                                                    self._wall_lo_node, self._wall_hi_node, self._lo_on,
                                                    self._hi_on, self.domain.wall_friction], device=d)
        if self._meshes:
            wp.launch(CK.grid_colliders, self.dims, inputs=[
                self.grid_v, self.grid_m, self._origin_wp, self.dx, self._c_ids, self._c_poses,
                self._c_closed, self._c_friction, self._c_lo, self._c_hi, self._c_margin, COLLIDER_THICKNESS * self.dx,
                self.dx, s, 1.0 / self._frame_dt], device=d)
        wp.launch(K.g2p, self._n, inputs=[self.x, self.v, self.C, self.F, self.Jp, self.mat, self.m_model,
                                          self.m_mu, self.m_lam, self.m_yield, self.m_alpha, self.m_crit_c,
                                          self.m_crit_s, self.grid_v, self._origin_wp,
                                          self.inv_dx, self.dx, dt, self._wall_lo, self._wall_hi, self._lo_on,
                                          self._hi_on, self._clamp_lo, self._clamp_hi], device=d)
        if self._meshes:
            wp.launch(CK.project_particles, self._n, inputs=[
                self.x, self.v, self._c_ids, self._c_poses, self._c_closed, self._c_lo, self._c_hi,
                self._c_margin, COLLIDER_THICKNESS * self.dx, s, 1.0 / self._frame_dt], device=d)

    def advance(self, frame_dt: float) -> AdvanceStats:
        if self._n == 0:
            return AdvanceStats(0, frame_dt, 0)
        self._frame_dt = frame_dt
        dt, steps = self._choose_dt(frame_dt)
        for k in range(steps):
            self._substep(dt, (k + 1) / steps)
        return AdvanceStats(steps, dt, self._remove_escaped())

    def _remove_escaped(self) -> int:
        """Drop particles more than one cell past an open side."""
        dom = self.domain
        open_lo = [not dom.walls[f"{a}-"] for a in "xyz"]
        open_hi = [not dom.walls[f"{a}+"] for a in "xyz"]
        if not any(open_lo) and not any(open_hi):
            return 0
        x = self.x.numpy()
        out = np.zeros(len(x), bool)
        for a in range(3):
            if open_lo[a]:
                out |= x[:, a] < dom.min[a] - self.dx
            if open_hi[a]:
                out |= x[:, a] > dom.max[a] + self.dx
        removed = int(out.sum())
        if removed:
            s = self._download()
            self._upload({k: v[~out] for k, v in s.items()})
        return removed

    # -- queries ----------------------------------------------------------------

    def is_occupied(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, np.float32).reshape(-1, 3)
        if len(points) == 0:
            return np.zeros(0, bool)
        d = self.device
        grid_vol = wp.zeros(self.dims, dtype=float, device=d)
        wp.launch(K.scatter_volume, self._n, inputs=[self.x, self.p_vol, self._origin_wp, self.inv_dx, grid_vol],
                  device=d)
        out = wp.zeros(len(points), dtype=int, device=d)
        wp.launch(K.query_fill, len(points), inputs=[wp.array(points, dtype=wp.vec3, device=d), grid_vol,
                                                     self._origin_wp, self.inv_dx, self.dx**3, out], device=d)
        return out.numpy().astype(bool)


def _decompose(m: np.ndarray) -> tuple[wp.vec3, np.ndarray, wp.mat33]:
    """World matrix -> (translation, rotation quaternion xyzw, symmetric stretch) with M = [R S | t]."""
    A = np.asarray(m[:3, :3], np.float64)
    U, sig, Vt = np.linalg.svd(A)
    if np.linalg.det(U @ Vt) < 0:  # mirrored: keep R a proper rotation, put the flip into S
        U[:, -1] = -U[:, -1]
        sig[-1] = -sig[-1]
    R = U @ Vt
    S = Vt.T @ np.diag(sig) @ Vt
    return wp.vec3(*m[:3, 3]), _quat_from_matrix(R), wp.mat33(*S.ravel())


def _quat_from_matrix(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion (x, y, z, w)."""
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.copysign(np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def _dp_alpha(friction_angle_deg: float) -> float:
    s = math.sin(math.radians(friction_angle_deg))
    return math.sqrt(2 / 3) * 2 * s / (3 - s)


def _empty_state() -> dict[str, np.ndarray]:
    return {"x": np.zeros((0, 3), np.float32), "v": np.zeros((0, 3), np.float32),
            "C": np.zeros((0, 3, 3), np.float32), "F": np.zeros((0, 3, 3), np.float32),
            "Jp": np.zeros(0, np.float32), "mat": np.zeros(0, np.int32), "emitter": np.zeros(0, np.int32)}
