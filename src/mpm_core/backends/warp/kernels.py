"""MLS-MPM kernels: quadratic B-splines, APIC transfers, per-particle material table.

Grid node (i, j, k) sits at ``origin + (i, j, k) * dx``.
"""

import warp as wp

MODEL_ELASTIC = wp.constant(0)

MODEL_IDS = {"elastic": 0}


@wp.func
def bspline_w(fx: float, i: int) -> float:
    if i == 0:
        return 0.5 * (1.5 - fx) * (1.5 - fx)
    if i == 1:
        return 0.75 - (fx - 1.0) * (fx - 1.0)
    return 0.5 * (fx - 0.5) * (fx - 0.5)


@wp.func
def stencil_base(xp: wp.vec3) -> wp.vec3i:
    return wp.vec3i(int(wp.floor(xp[0] - 0.5)), int(wp.floor(xp[1] - 0.5)), int(wp.floor(xp[2] - 0.5)))


@wp.func
def fixed_corotated_tau(F: wp.mat33, mu: float, lam: float) -> wp.mat33:
    U = wp.mat33()
    sig = wp.vec3()
    V = wp.mat33()
    wp.svd3(F, U, sig, V)
    R = U * wp.transpose(V)
    J = sig[0] * sig[1] * sig[2]
    return 2.0 * mu * (F - R) * wp.transpose(F) + wp.identity(n=3, dtype=float) * (lam * J * (J - 1.0))


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
    origin: wp.vec3,
    inv_dx: float,
    dx: float,
    dt: float,
    grid_v: wp.array3d(dtype=wp.vec3),
    grid_m: wp.array3d(dtype=float),
):
    p = wp.tid()
    m = mat[p]
    tau = fixed_corotated_tau(F[p], m_mu[m], m_lam[m])

    pm = p_mass[p]
    affine = (-dt * p_vol * 4.0 * inv_dx * inv_dx) * tau + pm * C[p]
    xp = (x[p] - origin) * inv_dx
    base = stencil_base(xp)
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


@wp.func
def wall_contact(vel: wp.vec3, axis: int, friction: float) -> wp.vec3:
    """Remove the velocity component into the wall; Coulomb friction on the rest."""
    vn = wp.abs(vel[axis])
    vt = vel
    vt[axis] = 0.0
    vt_len = wp.length(vt)
    if vt_len > 0.0:
        vt = vt * wp.max(0.0, 1.0 - friction * vn / vt_len)
    return vt


@wp.kernel
def grid_update(
    grid_v: wp.array3d(dtype=wp.vec3),
    grid_m: wp.array3d(dtype=float),
    origin: wp.vec3,
    dx: float,
    dt: float,
    gravity: wp.vec3,
    wall_lo: wp.vec3,
    wall_hi: wp.vec3,
    lo_on: wp.vec3i,
    hi_on: wp.vec3i,
    wall_friction: float,
):
    i, j, k = wp.tid()
    m = grid_m[i, j, k]
    if m <= 0.0:
        return
    vel = grid_v[i, j, k] / m + dt * gravity
    pos = origin + wp.vec3(float(i), float(j), float(k)) * dx
    for a in range(3):
        if lo_on[a] != 0 and pos[a] <= wall_lo[a] and vel[a] < 0.0:
            vel = wall_contact(vel, a, wall_friction)
        if hi_on[a] != 0 and pos[a] >= wall_hi[a] and vel[a] > 0.0:
            vel = wall_contact(vel, a, wall_friction)
    grid_v[i, j, k] = vel


@wp.kernel
def g2p(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    C: wp.array(dtype=wp.mat33),
    F: wp.array(dtype=wp.mat33),
    grid_v: wp.array3d(dtype=wp.vec3),
    origin: wp.vec3,
    inv_dx: float,
    dx: float,
    dt: float,
    wall_lo: wp.vec3,
    wall_hi: wp.vec3,
    lo_on: wp.vec3i,
    hi_on: wp.vec3i,
    clamp_lo: wp.vec3,
    clamp_hi: wp.vec3,
):
    p = wp.tid()
    xp = (x[p] - origin) * inv_dx
    base = stencil_base(xp)
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

    nx = x[p] + dt * new_v
    # project onto enabled walls so resting material sits on the wall plane
    for a in range(3):
        if lo_on[a] != 0 and nx[a] < wall_lo[a]:
            nx[a] = wall_lo[a]
        if hi_on[a] != 0 and nx[a] > wall_hi[a]:
            nx[a] = wall_hi[a]
    # keep the stencil inside the padded grid (only matters past open sides)
    x[p] = wp.vec3(wp.clamp(nx[0], clamp_lo[0], clamp_hi[0]), wp.clamp(nx[1], clamp_lo[1], clamp_hi[1]),
                   wp.clamp(nx[2], clamp_lo[2], clamp_hi[2]))
    F[p] = (wp.identity(n=3, dtype=float) + dt * new_C) * F[p]


@wp.kernel
def max_speed(v: wp.array(dtype=wp.vec3), out: wp.array(dtype=float)):
    p = wp.tid()
    wp.atomic_max(out, 0, wp.length(v[p]))


@wp.kernel
def scatter_volume(x: wp.array(dtype=wp.vec3), p_vol: float, origin: wp.vec3, inv_dx: float,
                   grid_vol: wp.array3d(dtype=float)):
    p = wp.tid()
    xp = (x[p] - origin) * inv_dx
    base = stencil_base(xp)
    fx = xp - wp.vec3(float(base[0]), float(base[1]), float(base[2]))
    for i in range(3):
        wi = bspline_w(fx[0], i)
        for j in range(3):
            wj = bspline_w(fx[1], j)
            for k in range(3):
                wp.atomic_add(grid_vol, base[0] + i, base[1] + j, base[2] + k, wi * wj * bspline_w(fx[2], k) * p_vol)


@wp.kernel
def query_fill(points: wp.array(dtype=wp.vec3), grid_vol: wp.array3d(dtype=float), origin: wp.vec3,
               inv_dx: float, cell_vol: float, out: wp.array(dtype=int)):
    q = wp.tid()
    g = (points[q] - origin) * inv_dx
    i = int(wp.round(g[0]))
    j = int(wp.round(g[1]))
    k = int(wp.round(g[2]))
    if i < 0 or j < 0 or k < 0 or i >= grid_vol.shape[0] or j >= grid_vol.shape[1] or k >= grid_vol.shape[2]:
        out[q] = 0
        return
    out[q] = wp.where(grid_vol[i, j, k] >= 0.5 * cell_vol, 1, 0)
