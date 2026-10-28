"""Mesh colliders: signed distance via wp.Mesh queries in collider local space.

Within one advance() each collider moves from its start to its end pose; ``s`` in [0, 1]
is the substep's position in the frame. Poses are decomposed on the host into
translation, rotation and stretch (M = [R S | t]); the rotation is slerped and the
rest lerped, so a spinning collider keeps its size mid-frame. Collider velocity is
the time derivative of that interpolated motion.

Closed meshes: inside where the query sign is negative.
Open meshes: inside only on the back side and within ``thickness`` of the surface.

Grid nodes get frictional contact within ``band`` in front of the surface (as the
domain walls do for nodes on the wall), so resting material feels friction.
Particles are only projected when actually inside.
"""

import warp as wp


@wp.struct
class ColliderPose:
    t0: wp.vec3
    t1: wp.vec3
    q0: wp.quat
    q1: wp.quat
    S0: wp.mat33
    S1: wp.mat33


@wp.func
def collider_matrix(pose: ColliderPose, s: float) -> wp.mat44:
    A = wp.quat_to_matrix(wp.quat_slerp(pose.q0, pose.q1, s)) * (pose.S0 + s * (pose.S1 - pose.S0))
    t = pose.t0 + s * (pose.t1 - pose.t0)
    return wp.mat44(A[0, 0], A[0, 1], A[0, 2], t[0],
                    A[1, 0], A[1, 1], A[1, 2], t[1],
                    A[2, 0], A[2, 1], A[2, 2], t[2],
                    0.0, 0.0, 0.0, 1.0)


@wp.func
def collider_sdf(mesh: wp.uint64, M: wp.mat44, p: wp.vec3, closed: int, thickness: float,
                 aabb_lo: wp.vec3, aabb_hi: wp.vec3, margin: float) -> wp.vec4:
    """(outward normal xyz, signed distance). Distance > 0 (1e30) when far outside."""
    far = wp.vec4(0.0, 0.0, 1.0, 1.0e30)
    Minv = wp.inverse(M)
    pl = wp.transform_point(Minv, p)
    for a in range(3):
        if pl[a] < aabb_lo[a] - margin or pl[a] > aabb_hi[a] + margin:
            return far
    q = wp.mesh_query_point_sign_normal(mesh, pl, 1.0e6)
    if not q.result:
        return far
    cp = wp.mesh_eval_position(mesh, q.face, q.u, q.v)
    d = wp.transform_vector(M, pl - cp)  # world-space offset from the closest surface point
    dist = wp.length(d)
    n = wp.vec3()
    if dist > 1.0e-9:
        n = d * (q.sign / dist)
    else:
        nl = wp.mesh_eval_face_normal(mesh, q.face)
        n = wp.normalize(wp.transform_vector(wp.transpose(Minv), nl))
    phi = q.sign * dist
    if closed == 0 and q.sign < 0.0 and dist > thickness:
        return far
    return wp.vec4(n[0], n[1], n[2], phi)


@wp.func
def collider_velocity(pose: ColliderPose, M: wp.mat44, s: float, p: wp.vec3, inv_frame_dt: float) -> wp.vec3:
    """Velocity of the collider material point currently at p (central difference in s)."""
    pl = wp.transform_point(wp.inverse(M), p)
    h = 1.0e-3
    dM = (collider_matrix(pose, s + h) - collider_matrix(pose, s - h)) * (inv_frame_dt / (2.0 * h))
    return wp.transform_point(dM, pl)


@wp.kernel
def grid_colliders(
    grid_v: wp.array3d(dtype=wp.vec3),
    grid_m: wp.array3d(dtype=float),
    origin: wp.vec3,
    dx: float,
    meshes: wp.array(dtype=wp.uint64),
    poses: wp.array(dtype=ColliderPose),
    closed: wp.array(dtype=int),
    friction: wp.array(dtype=float),
    aabb_lo: wp.array(dtype=wp.vec3),
    aabb_hi: wp.array(dtype=wp.vec3),
    margin: wp.array(dtype=float),
    thickness: float,
    band: float,
    s: float,
    inv_frame_dt: float,
):
    i, j, k = wp.tid()
    if grid_m[i, j, k] <= 0.0:
        return
    pos = origin + wp.vec3(float(i), float(j), float(k)) * dx
    vel = grid_v[i, j, k]
    for c in range(meshes.shape[0]):
        M = collider_matrix(poses[c], s)
        r = collider_sdf(meshes[c], M, pos, closed[c], thickness, aabb_lo[c], aabb_hi[c], margin[c])
        if r[3] < band:
            n = wp.vec3(r[0], r[1], r[2])
            vcol = collider_velocity(poses[c], M, s, pos, inv_frame_dt)
            rel = vel - vcol
            vn = wp.dot(rel, n)
            if vn < 0.0:
                rt = rel - vn * n
                rt_len = wp.length(rt)
                if rt_len > 0.0:
                    rt = rt * wp.max(0.0, 1.0 - friction[c] * (-vn) / rt_len)
                if r[3] < 0.0:
                    vel = vcol + rt  # behind the surface: no penetration
                else:
                    vel = vcol + rt + vn * n  # in the band: friction only, keep approaching
    grid_v[i, j, k] = vel


@wp.kernel
def project_particles(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    meshes: wp.array(dtype=wp.uint64),
    poses: wp.array(dtype=ColliderPose),
    closed: wp.array(dtype=int),
    aabb_lo: wp.array(dtype=wp.vec3),
    aabb_hi: wp.array(dtype=wp.vec3),
    margin: wp.array(dtype=float),
    thickness: float,
    s: float,
    inv_frame_dt: float,
):
    """Push particles that ended up inside a collider back onto its surface."""
    p = wp.tid()
    pos = x[p]
    vel = v[p]
    for c in range(meshes.shape[0]):
        M = collider_matrix(poses[c], s)
        r = collider_sdf(meshes[c], M, pos, closed[c], thickness, aabb_lo[c], aabb_hi[c], margin[c])
        if r[3] < 0.0:
            n = wp.vec3(r[0], r[1], r[2])
            pos = pos - r[3] * n
            vcol = collider_velocity(poses[c], M, s, pos, inv_frame_dt)
            vn = wp.dot(vel - vcol, n)
            if vn < 0.0:
                vel = vel - vn * n
    x[p] = pos
    v[p] = vel
