"""THROWAWAY SPIKE: 3D MLS-MPM on a dense grid with NVIDIA Warp.

Three materials in one domain:
  cream - Hencky elasticity + von Mises plasticity (holds shape, flows when pushed)
  sand  - Hencky elasticity + Drucker-Prager plasticity (Klar et al. 2016)
  snow  - fixed-corotated + Stomakhin et al. 2013 hardening

Domain is the unit cube, z-up. Writes one positions .npy per frame to the cache dir.

    .venv\\Scripts\\python spike\\mpm_spike.py --frames 96 --res 128
"""

import argparse
import math
import os
import time

import numpy as np
import warp as wp

MODEL_VON_MISES = wp.constant(0)
MODEL_DRUCKER_PRAGER = wp.constant(1)
MODEL_SNOW = wp.constant(2)

BOUND = wp.constant(3)  # boundary thickness in cells


@wp.func
def bspline_w(fx: float, i: int) -> float:
    if i == 0:
        return 0.5 * (1.5 - fx) * (1.5 - fx)
    if i == 1:
        return 0.75 - (fx - 1.0) * (fx - 1.0)
    return 0.5 * (fx - 0.5) * (fx - 0.5)


@wp.func
def safe_log3(s: wp.vec3) -> wp.vec3:
    return wp.vec3(
        wp.log(wp.max(s[0], 1.0e-4)),
        wp.log(wp.max(s[1], 1.0e-4)),
        wp.log(wp.max(s[2], 1.0e-4)),
    )


@wp.func
def exp3(e: wp.vec3) -> wp.vec3:
    return wp.vec3(wp.exp(e[0]), wp.exp(e[1]), wp.exp(e[2]))


@wp.kernel
def p2g(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    C: wp.array(dtype=wp.mat33),
    F: wp.array(dtype=wp.mat33),
    Jp: wp.array(dtype=float),
    mat: wp.array(dtype=int),
    p_mass: wp.array(dtype=float),
    m_model: wp.array(dtype=int),
    m_mu: wp.array(dtype=float),
    m_lam: wp.array(dtype=float),
    p_vol: float,
    inv_dx: float,
    dx: float,
    dt: float,
    grid_v: wp.array3d(dtype=wp.vec3),
    grid_m: wp.array3d(dtype=float),
):
    p = wp.tid()
    m = mat[p]
    model = m_model[m]
    mu = m_mu[m]
    lam = m_lam[m]

    Fp = F[p]
    U = wp.mat33()
    sig = wp.vec3()
    V = wp.mat33()
    wp.svd3(Fp, U, sig, V)

    tau = wp.mat33()
    if model == MODEL_SNOW:
        h = wp.min(wp.exp(10.0 * (1.0 - Jp[p])), 5.0)
        mu = mu * h
        lam = lam * h
        R = U * wp.transpose(V)
        J = sig[0] * sig[1] * sig[2]
        tau = 2.0 * mu * (Fp - R) * wp.transpose(Fp) + wp.identity(n=3, dtype=float) * (lam * J * (J - 1.0))
    else:
        eps = safe_log3(sig)
        tr = eps[0] + eps[1] + eps[2]
        t = 2.0 * mu * eps + wp.vec3(lam * tr)
        tau = U * wp.diag(t) * wp.transpose(U)

    pm = p_mass[p]
    affine = (-dt * p_vol * 4.0 * inv_dx * inv_dx) * tau + pm * C[p]

    xp = x[p] * inv_dx
    base = wp.vec3i(int(xp[0] - 0.5), int(xp[1] - 0.5), int(xp[2] - 0.5))
    fx = xp - wp.vec3(float(base[0]), float(base[1]), float(base[2]))
    mv = pm * v[p]

    for i in range(3):
        wi = bspline_w(fx[0], i)
        for j in range(3):
            wj = bspline_w(fx[1], j)
            for k in range(3):
                w = wi * wj * bspline_w(fx[2], k)
                dpos = (wp.vec3(float(i), float(j), float(k)) - fx) * dx
                wp.atomic_add(grid_v, base[0] + i, base[1] + j, base[2] + k, w * (mv + affine * dpos))
                wp.atomic_add(grid_m, base[0] + i, base[1] + j, base[2] + k, w * pm)


@wp.kernel
def grid_update(
    grid_v: wp.array3d(dtype=wp.vec3),
    grid_m: wp.array3d(dtype=float),
    n: int,
    dx: float,
    dt: float,
    gravity: float,
    floor_friction: float,
    ball_c: wp.array(dtype=wp.vec3),
    ball_v: wp.vec3,
    ball_w: wp.vec3,
    ball_r: float,
    ball_friction: float,
):
    i, j, k = wp.tid()
    m = grid_m[i, j, k]
    if m <= 0.0:
        return
    vel = grid_v[i, j, k] / m
    vel = vel + wp.vec3(0.0, 0.0, -dt * gravity)

    # kinematic ball: frictional contact against the ball's surface velocity
    if ball_r > 0.0:
        d = wp.vec3(float(i), float(j), float(k)) * dx - ball_c[0]
        dist = wp.length(d)
        if dist < ball_r and dist > 0.0:
            nrm = d / dist
            vcol = ball_v + wp.cross(ball_w, nrm * ball_r)
            rel = vel - vcol
            vn = wp.dot(rel, nrm)
            if vn < 0.0:
                rt = rel - vn * nrm
                rt_len = wp.length(rt)
                if rt_len > 0.0:
                    rt = rt * wp.max(0.0, 1.0 - ball_friction * (-vn) / rt_len)
                vel = vcol + rt

    # floor: no penetration + Coulomb friction
    if k < BOUND and vel[2] < 0.0:
        vn = vel[2]
        vt = wp.vec3(vel[0], vel[1], 0.0)
        vt_len = wp.length(vt)
        if vt_len > 0.0:
            vt = vt * wp.max(0.0, 1.0 - floor_friction * (-vn) / vt_len)
        vel = vt
    # other walls: free slip
    if k > n - BOUND and vel[2] > 0.0:
        vel = wp.vec3(vel[0], vel[1], 0.0)
    if i < BOUND and vel[0] < 0.0:
        vel = wp.vec3(0.0, vel[1], vel[2])
    if i > n - BOUND and vel[0] > 0.0:
        vel = wp.vec3(0.0, vel[1], vel[2])
    if j < BOUND and vel[1] < 0.0:
        vel = wp.vec3(vel[0], 0.0, vel[2])
    if j > n - BOUND and vel[1] > 0.0:
        vel = wp.vec3(vel[0], 0.0, vel[2])

    grid_v[i, j, k] = vel


@wp.func
def project_von_mises(sig: wp.vec3, mu: float, yield_stress: float) -> wp.vec3:
    eps = safe_log3(sig)
    tr = eps[0] + eps[1] + eps[2]
    dev = eps - wp.vec3(tr / 3.0)
    n = wp.length(dev)
    limit = yield_stress / (2.0 * mu)
    if n > limit:
        eps = wp.vec3(tr / 3.0) + dev * (limit / n)
        return exp3(eps)
    return sig


@wp.func
def project_drucker_prager(sig: wp.vec3, mu: float, lam: float, alpha: float) -> wp.vec3:
    eps = safe_log3(sig)
    tr = eps[0] + eps[1] + eps[2]
    if tr >= 0.0:
        return wp.vec3(1.0)  # separation: stress-free
    dev = eps - wp.vec3(tr / 3.0)
    n = wp.length(dev)
    if n < 1.0e-10:
        return sig
    dgamma = n + (3.0 * lam + 2.0 * mu) / (2.0 * mu) * tr * alpha
    if dgamma <= 0.0:
        return sig
    return exp3(eps - dev * (dgamma / n))


@wp.kernel
def g2p(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    C: wp.array(dtype=wp.mat33),
    F: wp.array(dtype=wp.mat33),
    Jp: wp.array(dtype=float),
    mat: wp.array(dtype=int),
    m_model: wp.array(dtype=int),
    m_mu: wp.array(dtype=float),
    m_lam: wp.array(dtype=float),
    m_yield: wp.array(dtype=float),
    m_alpha: wp.array(dtype=float),
    n: int,
    inv_dx: float,
    dx: float,
    dt: float,
    grid_v: wp.array3d(dtype=wp.vec3),
):
    p = wp.tid()
    xp = x[p] * inv_dx
    base = wp.vec3i(int(xp[0] - 0.5), int(xp[1] - 0.5), int(xp[2] - 0.5))
    fx = xp - wp.vec3(float(base[0]), float(base[1]), float(base[2]))

    new_v = wp.vec3()
    new_C = wp.mat33()
    for i in range(3):
        wi = bspline_w(fx[0], i)
        for j in range(3):
            wj = bspline_w(fx[1], j)
            for k in range(3):
                w = wi * wj * bspline_w(fx[2], k)
                dpos = (wp.vec3(float(i), float(j), float(k)) - fx) * dx
                gv = grid_v[base[0] + i, base[1] + j, base[2] + k]
                new_v = new_v + w * gv
                new_C = new_C + (4.0 * inv_dx * inv_dx * w) * wp.outer(gv, dpos)

    v[p] = new_v
    C[p] = new_C
    lo = 1.0 * dx
    hi = float(n - 2) * dx
    newx = x[p] + dt * new_v
    x[p] = wp.vec3(wp.clamp(newx[0], lo, hi), wp.clamp(newx[1], lo, hi), wp.clamp(newx[2], lo, hi))

    Fn = (wp.identity(n=3, dtype=float) + dt * new_C) * F[p]
    U = wp.mat33()
    sig = wp.vec3()
    V = wp.mat33()
    wp.svd3(Fn, U, sig, V)

    m = mat[p]
    model = m_model[m]
    new_sig = sig
    if model == MODEL_VON_MISES:
        new_sig = project_von_mises(sig, m_mu[m], m_yield[m])
    elif model == MODEL_DRUCKER_PRAGER:
        new_sig = project_drucker_prager(sig, m_mu[m], m_lam[m], m_alpha[m])
    else:
        new_sig = wp.vec3(
            wp.clamp(sig[0], 1.0 - 2.5e-2, 1.0 + 4.5e-3),
            wp.clamp(sig[1], 1.0 - 2.5e-2, 1.0 + 4.5e-3),
            wp.clamp(sig[2], 1.0 - 2.5e-2, 1.0 + 4.5e-3),
        )
        Jp[p] = Jp[p] * (sig[0] * sig[1] * sig[2]) / (new_sig[0] * new_sig[1] * new_sig[2])

    F[p] = U * wp.diag(new_sig) * wp.transpose(V)


@wp.kernel
def move_ball(ball_c: wp.array(dtype=wp.vec3), ball_v: wp.vec3, dt: float):
    ball_c[0] = ball_c[0] + ball_v * dt


# ---------------------------------------------------------------------------
# Host side

# name -> (model, E, nu, density, yield_stress, friction_angle_deg)
MATERIALS = {
    "cream": (0, 1.0e5, 0.3, 400.0, 2500.0, 0.0),
    "sand": (1, 3.5e5, 0.3, 2200.0, 0.0, 30.0),
    "snow": (2, 1.4e5, 0.2, 400.0, 0.0, 0.0),
}
MAT_NAMES = list(MATERIALS)


def sample_sphere(center, radius, spacing, rng):
    r = np.arange(-radius, radius + spacing, spacing)
    g = np.stack(np.meshgrid(r, r, r, indexing="ij"), -1).reshape(-1, 3)
    g = g[np.linalg.norm(g, axis=1) <= radius]
    g += rng.uniform(-0.25, 0.25, g.shape) * spacing
    return g + np.asarray(center)


def sample_box(lo, hi, spacing, rng):
    axes = [np.arange(a, b, spacing) for a, b in zip(lo, hi)]
    g = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    g += rng.uniform(-0.25, 0.25, g.shape) * spacing
    return g


class Solver:
    def __init__(self, res, device="cuda:0"):
        self.n = res
        self.dx = 1.0 / res
        self.inv_dx = float(res)
        self.p_vol = (self.dx * 0.5) ** 3
        self.device = device
        self.chunks = []
        self.ball = None  # (center, velocity, radius); rolls on the floor along its velocity
        self.time = 0.0

    def set_ball(self, center, vel, radius):
        self.ball = (np.asarray(center, np.float32), np.asarray(vel, np.float32), float(radius))

    def ball_state(self):
        """Host-side ball pose for display: center and roll angle about +y."""
        c, v, r = self.ball
        return c + v * self.time, float(v[0]) * self.time / r

    def add(self, pos, material, vel=(0.0, 0.0, 0.0)):
        self.chunks.append((pos.astype(np.float32), MAT_NAMES.index(material), np.asarray(vel, np.float32)))

    def build(self):
        pos = np.concatenate([c[0] for c in self.chunks])
        mat = np.concatenate([np.full(len(c[0]), c[1], np.int32) for c in self.chunks])
        vel = np.concatenate([np.tile(c[2], (len(c[0]), 1)) for c in self.chunks])
        N = len(pos)
        d = self.device

        model, mu, lam, rho, yld, alpha = [], [], [], [], [], []
        for name in MAT_NAMES:
            mdl, E, nu, dens, ys, phi = MATERIALS[name]
            model.append(mdl)
            mu.append(E / (2 * (1 + nu)))
            lam.append(E * nu / ((1 + nu) * (1 - 2 * nu)))
            rho.append(dens)
            yld.append(ys)
            s = math.sin(math.radians(phi))
            alpha.append(math.sqrt(2 / 3) * 2 * s / (3 - s))

        self.mat_np = mat
        self.x = wp.array(pos, dtype=wp.vec3, device=d)
        self.v = wp.array(vel, dtype=wp.vec3, device=d)
        self.C = wp.zeros(N, dtype=wp.mat33, device=d)
        self.F = wp.array(np.tile(np.eye(3, dtype=np.float32), (N, 1, 1)), dtype=wp.mat33, device=d)
        self.Jp = wp.ones(N, dtype=float, device=d)
        self.mat = wp.array(mat, dtype=int, device=d)
        self.p_mass = wp.array(np.asarray(rho, np.float32)[mat] * self.p_vol, dtype=float, device=d)
        self.m_model = wp.array(model, dtype=int, device=d)
        self.m_mu = wp.array(mu, dtype=float, device=d)
        self.m_lam = wp.array(lam, dtype=float, device=d)
        self.m_yield = wp.array(yld, dtype=float, device=d)
        self.m_alpha = wp.array(alpha, dtype=float, device=d)

        n = self.n
        self.grid_v = wp.zeros((n, n, n), dtype=wp.vec3, device=d)
        self.grid_m = wp.zeros((n, n, n), dtype=float, device=d)
        self.N = N
        self.graph = None

        if self.ball is not None:
            c, v, r = self.ball
            self.ball_c = wp.array([c], dtype=wp.vec3, device=d)
            self.ball_v = wp.vec3(*v)
            self.ball_w = wp.vec3(0.0, float(v[0]) / r, 0.0)  # rolling without slipping along x
            self.ball_r = r
        else:
            self.ball_c = wp.zeros(1, dtype=wp.vec3, device=d)
            self.ball_v = self.ball_w = wp.vec3()
            self.ball_r = 0.0

    def _substep(self, dt):
        n = self.n
        self.grid_v.zero_()
        self.grid_m.zero_()
        wp.launch(p2g, self.N, inputs=[self.x, self.v, self.C, self.F, self.Jp, self.mat, self.p_mass,
                                       self.m_model, self.m_mu, self.m_lam, self.p_vol, self.inv_dx, self.dx, dt,
                                       self.grid_v, self.grid_m], device=self.device)
        wp.launch(grid_update, (n, n, n), inputs=[self.grid_v, self.grid_m, n, self.dx, dt, 9.81, 0.5, self.ball_c,
                                                  self.ball_v, self.ball_w, self.ball_r, 1.0], device=self.device)
        wp.launch(g2p, self.N, inputs=[self.x, self.v, self.C, self.F, self.Jp, self.mat, self.m_model, self.m_mu,
                                       self.m_lam, self.m_yield, self.m_alpha, n, self.inv_dx, self.dx, dt,
                                       self.grid_v], device=self.device)
        wp.launch(move_ball, 1, inputs=[self.ball_c, self.ball_v, dt], device=self.device)

    def advance(self, frame_dt, dt_max):
        self.time += frame_dt
        steps = math.ceil(frame_dt / dt_max)
        dt = frame_dt / steps
        if self.graph is None:
            self._substep(dt)  # warm-up: compile kernels outside capture
            with wp.ScopedCapture(device=self.device) as cap:
                self._substep(dt)
            self.graph = cap.graph
            steps -= 2
        for _ in range(steps):
            wp.capture_launch(self.graph)
        return steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--frames", type=int, default=96)
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--dt", type=float, default=5e-5)
    ap.add_argument("--scene", choices=["mix", "ball"], default="mix")
    ap.add_argument("--out", help="cache dir (default: spike/cache_<scene>)")
    args = ap.parse_args()
    args.out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), f"cache_{args.scene}")

    wp.init()
    s = Solver(args.res)
    spacing = s.dx * 0.5
    rng = np.random.default_rng(0)
    if args.scene == "mix":
        # cream blob dropped on the left
        s.add(sample_sphere((0.22, 0.5, 0.3), 0.11, spacing, rng), "cream")
        # sand column in the middle, collapses into a pile
        s.add(sample_box((0.44, 0.44, 0.03), (0.56, 0.56, 0.36), spacing, rng), "sand")
        # snowball thrown at the sand column from the right
        s.add(sample_sphere((0.82, 0.5, 0.3), 0.08, spacing, rng), "snow", vel=(-2.0, 0.0, 0.5))
    else:
        # snow layer on the floor; a kinematic ball rolls through it along +x
        floor = 2.5 * s.dx  # where particles actually rest (see boundary note in blender_load.py)
        s.add(sample_box((0.2, 0.25, floor), (0.97, 0.75, floor + 0.07), spacing, rng), "snow")
        radius = 0.08
        s.set_ball((0.1, 0.5, floor + radius), (0.2, 0.0, 0.0), radius)
    s.build()

    os.makedirs(args.out, exist_ok=True)
    for f in os.listdir(args.out):
        if f.startswith("frame_") or f == "ball.npy":
            os.remove(os.path.join(args.out, f))
    np.save(os.path.join(args.out, "material.npy"), s.mat_np)
    with open(os.path.join(args.out, "meta.txt"), "w") as f:
        f.write(f"res={args.res}\nfps={args.fps}\nmaterials={','.join(MAT_NAMES)}\n")
    counts = {name: int((s.mat_np == i).sum()) for i, name in enumerate(MAT_NAMES)}
    print(f"particles: {s.N:,} {counts}  grid: {args.res}^3")

    np.save(os.path.join(args.out, "frame_0001.npy"), s.x.numpy())
    ball_track = []  # per frame: x, y, z, radius, roll angle
    if s.ball is not None:
        c, a = s.ball_state()
        ball_track.append([*c, s.ball[2], a])
    t_total = 0.0
    for frame in range(2, args.frames + 1):
        t0 = time.perf_counter()
        steps = s.advance(1.0 / args.fps, args.dt)
        pos = s.x.numpy()  # syncs
        t = time.perf_counter() - t0
        t_total += t
        if not np.isfinite(pos).all():
            raise SystemExit(f"frame {frame}: NaN/inf in positions, sim blew up")
        np.save(os.path.join(args.out, f"frame_{frame:04d}.npy"), pos)
        if s.ball is not None:
            c, a = s.ball_state()
            ball_track.append([*c, s.ball[2], a])
            np.save(os.path.join(args.out, "ball.npy"), np.asarray(ball_track, np.float32))
        print(f"frame {frame:4d}  {steps} substeps  {t * 1000:7.1f} ms")
    print(f"total sim time {t_total:.1f}s, {t_total / (args.frames - 1) * 1000:.0f} ms/frame avg")


if __name__ == "__main__":
    main()
