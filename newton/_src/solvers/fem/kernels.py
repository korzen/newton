# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp kernels supporting :class:`~newton.solvers.SolverFEM`."""

from __future__ import annotations

import warp as wp

from ...geometry import GeoType, ParticleFlags, ShapeFlags


@wp.kernel
def compute_gravity_force(
    particle_mass: wp.array[float],
    gravity: wp.array[wp.vec3],
    particle_world: wp.array[wp.int32],
    f_out: wp.array[wp.vec3],
):
    """Initialize per-node external force with gravity (mass * g)."""
    tid = wp.tid()
    world_idx = wp.max(particle_world[tid], 0)
    f_out[tid] = particle_mass[tid] * gravity[world_idx]


@wp.kernel
def add_external_force(
    particle_f: wp.array[wp.vec3],
    f_out: wp.array[wp.vec3],
):
    """Accumulate user-supplied particle forces into the external force buffer."""
    tid = wp.tid()
    f_out[tid] = f_out[tid] + particle_f[tid]


@wp.kernel
def scatter_soft_contact_forces(
    soft_contact_count: wp.array[wp.int32],
    soft_contact_particle: wp.array[wp.int32],
    soft_contact_shape: wp.array[wp.int32],
    soft_contact_body_pos: wp.array[wp.vec3],
    soft_contact_body_vel: wp.array[wp.vec3],
    soft_contact_normal: wp.array[wp.vec3],
    shape_body: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    ke: float,
    kd: float,
    mu: float,
    f_out: wp.array[wp.vec3],
):
    """Scatter a Kelvin-Voigt soft-contact response into a per-node force array.

    Mirrors the contact model used by SolverVBD (spring-damper normal with
    Coulomb-clamped tangential viscous friction) but writes directly into a
    nodal force buffer rather than into a per-vertex Hessian.
    """
    tid = wp.tid()
    if tid >= soft_contact_count[0]:
        return

    particle_idx = soft_contact_particle[tid]
    if particle_idx < 0:
        return

    shape_idx = soft_contact_shape[tid]
    body_idx = wp.int32(-1)
    if shape_idx >= 0:
        body_idx = shape_body[shape_idx]

    X_wb = wp.transform_identity()
    com_world = wp.vec3(0.0, 0.0, 0.0)
    body_v = wp.vec3(0.0, 0.0, 0.0)
    body_w = wp.vec3(0.0, 0.0, 0.0)
    if body_idx >= 0:
        X_wb = body_q[body_idx]
        com_world = wp.transform_point(X_wb, body_com[body_idx])
        body_vs = body_qd[body_idx]
        body_w = wp.spatial_top(body_vs)
        body_v = wp.spatial_bottom(body_vs)

    bx = wp.transform_point(X_wb, soft_contact_body_pos[tid])
    n = soft_contact_normal[tid]
    px = particle_q[particle_idx]
    radius = particle_radius[particle_idx]

    penetration = -(wp.dot(n, px - bx) - radius)
    if penetration <= 0.0:
        return

    bv_local = soft_contact_body_vel[tid]
    bv = body_v + wp.cross(body_w, bx - com_world) + wp.transform_vector(X_wb, bv_local)
    pv = particle_qd[particle_idx]
    rel_v = pv - bv
    vn = wp.dot(rel_v, n)

    fn_mag = ke * penetration
    if vn < 0.0:
        fn_mag = fn_mag - kd * vn
    if fn_mag < 0.0:
        fn_mag = 0.0
    f_normal = fn_mag * n

    vt = rel_v - vn * n
    f_tangent = -kd * vt
    f_t_norm = wp.length(f_tangent)
    f_t_max = mu * fn_mag
    if f_t_norm > f_t_max and f_t_norm > 0.0:
        f_tangent = f_tangent * (f_t_max / f_t_norm)

    wp.atomic_add(f_out, particle_idx, f_normal + f_tangent)


@wp.kernel
def build_mass_blocks(
    particle_mass: wp.array[float],
    mass_blocks: wp.array[wp.mat33],
):
    """Build per-node diagonal mat33 blocks for the lumped mass BSR matrix."""
    tid = wp.tid()
    m = particle_mass[tid]
    mass_blocks[tid] = wp.mat33(
        m, 0.0, 0.0,
        0.0, m, 0.0,
        0.0, 0.0, m,
    )


@wp.kernel
def build_dirichlet_projector_blocks(
    particle_mass: wp.array[float],
    particle_flags: wp.array[wp.int32],
    projector_blocks: wp.array[wp.mat33],
):
    """Build identity projector blocks for inactive or zero-mass particles."""
    tid = wp.tid()
    fixed = (particle_flags[tid] & ParticleFlags.ACTIVE) == 0 or particle_mass[tid] == 0.0
    value = float(fixed)
    projector_blocks[tid] = wp.mat33(
        value, 0.0, 0.0,
        0.0, value, 0.0,
        0.0, 0.0, value,
    )


@wp.kernel
def sync_displacement_state(
    rest_positions: wp.array[wp.vec3],
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    u_n: wp.array[wp.vec3],
    v_n: wp.array[wp.vec3],
    u_cur: wp.array[wp.vec3],
):
    """Initialize previous and current displacement fields from the input state."""
    tid = wp.tid()
    u = particle_q[tid] - rest_positions[tid]
    u_n[tid] = u
    v_n[tid] = particle_qd[tid]
    u_cur[tid] = u


@wp.kernel
def refresh_lame_parameters(
    tet_materials: wp.array2d[float],
    use_mu_override: int,
    mu_override: float,
    use_lambda_override: int,
    lambda_override: float,
    mu_e: wp.array[float],
    lambda_e: wp.array[float],
):
    """Refresh per-element Lamé parameters from model materials or scalar overrides."""
    tid = wp.tid()
    if use_mu_override:
        mu_e[tid] = mu_override
    else:
        mu_e[tid] = tet_materials[tid, 0]

    if use_lambda_override:
        lambda_e[tid] = lambda_override
    else:
        lambda_e[tid] = tet_materials[tid, 1]


@wp.kernel
def build_newton_rhs(
    f_ext: wp.array[wp.vec3],
    f_int: wp.array[wp.vec3],
    particle_mass: wp.array[float],
    u_cur: wp.array[wp.vec3],
    u_n: wp.array[wp.vec3],
    v_n: wp.array[wp.vec3],
    dt: float,
    damping: float,
    rhs: wp.array[wp.vec3],
):
    """Right-hand side for a backward-Euler displacement Newton increment."""
    tid = wp.tid()
    m = particle_mass[tid]
    du_inertial = u_cur[tid] - u_n[tid] - dt * v_n[tid]
    du_damped = u_cur[tid] - u_n[tid]
    rhs[tid] = f_ext[tid] - f_int[tid] - (m / (dt * dt)) * du_inertial - (damping * m / dt) * du_damped


@wp.kernel
def accumulate_displacement_delta(
    delta_u: wp.array[wp.vec3],
    particle_mass: wp.array[float],
    particle_flags: wp.array[wp.int32],
    u_dofs: wp.array[wp.vec3],
):
    """Accumulate Newton displacement increments, defensively zeroing fixed DOFs."""
    tid = wp.tid()
    if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0 or particle_mass[tid] == 0.0:
        delta_u[tid] = wp.vec3(0.0, 0.0, 0.0)
        return

    u_dofs[tid] = u_dofs[tid] + delta_u[tid]


@wp.kernel
def write_state_outputs(
    rest_positions: wp.array[wp.vec3],
    u_dofs: wp.array[wp.vec3],
    u_n: wp.array[wp.vec3],
    particle_mass: wp.array[float],
    particle_flags: wp.array[wp.int32],
    dt: float,
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
):
    """Write rest + displacement and backward-Euler velocity to particle state."""
    tid = wp.tid()
    particle_q[tid] = rest_positions[tid] + u_dofs[tid]
    if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0 or particle_mass[tid] == 0.0:
        particle_qd[tid] = wp.vec3(0.0, 0.0, 0.0)
    else:
        particle_qd[tid] = (u_dofs[tid] - u_n[tid]) / dt


@wp.kernel
def project_particles_against_infinite_planes(
    particle_q_in: wp.array[wp.vec3],
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    particle_mass: wp.array[float],
    particle_flags: wp.array[wp.int32],
    particle_radius: wp.array[float],
    particle_world: wp.array[wp.int32],
    shape_transform: wp.array[wp.transform],
    shape_body: wp.array[wp.int32],
    shape_type: wp.array[wp.int32],
    shape_scale: wp.array[wp.vec3],
    shape_flags: wp.array[wp.int32],
    shape_world: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    shape_count: int,
    dt: float,
    relaxation: float,
):
    """Project active particles out of infinite plane shapes and update velocity."""
    tid = wp.tid()
    if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0 or particle_mass[tid] == 0.0:
        particle_qd[tid] = wp.vec3(0.0, 0.0, 0.0)
        return

    q = particle_q[tid]
    radius = particle_radius[tid]
    particle_world_id = particle_world[tid]

    for shape_idx in range(shape_count):
        if (shape_flags[shape_idx] & ShapeFlags.COLLIDE_PARTICLES) == 0:
            continue
        if shape_type[shape_idx] != GeoType.PLANE:
            continue

        scale = shape_scale[shape_idx]
        if scale[0] != 0.0 or scale[1] != 0.0:
            continue

        shape_world_id = shape_world[shape_idx]
        if particle_world_id != -1 and shape_world_id != -1 and particle_world_id != shape_world_id:
            continue

        X_ws = shape_transform[shape_idx]
        body_idx = shape_body[shape_idx]
        if body_idx >= 0:
            X_ws = wp.transform_multiply(body_q[body_idx], X_ws)

        plane_point = wp.transform_get_translation(X_ws)
        normal = wp.normalize(wp.transform_vector(X_ws, wp.vec3(0.0, 0.0, 1.0)))
        signed_distance = wp.dot(normal, q - plane_point) - radius

        if signed_distance < 0.0:
            q = q - relaxation * signed_distance * normal

    particle_q[tid] = q
    particle_qd[tid] = (q - particle_q_in[tid]) / dt


@wp.kernel
def project_particle_drag_constraint(
    particle_q_in: wp.array[wp.vec3],
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    particle_mass: wp.array[float],
    particle_flags: wp.array[wp.int32],
    pick_particle_indices: wp.array[wp.int32],
    pick_particle_weights: wp.array[float],
    pick_particle_target: wp.array[wp.vec3],
    pick_particle_point: wp.array[wp.vec3],
    dt: float,
    relaxation: float,
):
    """XPBD-style projection of a picked particle or surface point to a mouse target."""
    if pick_particle_indices[0] < 0:
        return

    picked = wp.vec3(0.0, 0.0, 0.0)
    denom = float(0.0)

    for k in range(3):
        i = pick_particle_indices[k]
        if i < 0:
            continue

        w = pick_particle_weights[k]
        picked = picked + w * particle_q[i]
        if (particle_flags[i] & ParticleFlags.ACTIVE) != 0 and particle_mass[i] > 0.0:
            denom = denom + w * w / particle_mass[i]

    target = pick_particle_target[0]
    correction = relaxation * (target - picked)

    if denom > 0.0:
        for k in range(3):
            i = pick_particle_indices[k]
            if i < 0:
                continue
            if (particle_flags[i] & ParticleFlags.ACTIVE) == 0 or particle_mass[i] == 0.0:
                particle_qd[i] = wp.vec3(0.0, 0.0, 0.0)
                continue

            w = pick_particle_weights[k]
            q = particle_q[i] + correction * (w / particle_mass[i] / denom)
            particle_q[i] = q
            particle_qd[i] = (q - particle_q_in[i]) / dt

    picked_after = wp.vec3(0.0, 0.0, 0.0)
    for k in range(3):
        i = pick_particle_indices[k]
        if i >= 0:
            picked_after = picked_after + pick_particle_weights[k] * particle_q[i]
    pick_particle_point[0] = picked_after
