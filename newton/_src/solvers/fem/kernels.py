# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp kernels supporting :class:`~newton.solvers.SolverFEM`."""

from __future__ import annotations

import warp as wp


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
def build_rhs(
    f_ext: wp.array[wp.vec3],
    f_int: wp.array[wp.vec3],
    particle_mass: wp.array[float],
    v_prev: wp.array[wp.vec3],
    dt: float,
    rhs: wp.array[wp.vec3],
):
    """Right-hand side of backward-Euler step: M v_prev + dt (f_ext - f_int)."""
    tid = wp.tid()
    rhs[tid] = particle_mass[tid] * v_prev[tid] + dt * (f_ext[tid] - f_int[tid])


@wp.kernel
def integrate_displacement(
    v_new: wp.array[wp.vec3],
    dt: float,
    damping: float,
    u_dofs: wp.array[wp.vec3],
    velocity: wp.array[wp.vec3],
):
    """Apply velocity damping, accumulate displacement, and cache velocity."""
    tid = wp.tid()
    factor = wp.max(1.0 - damping * dt, 0.0)
    v = v_new[tid] * factor
    velocity[tid] = v
    u_dofs[tid] = u_dofs[tid] + dt * v


@wp.kernel
def write_state_outputs(
    rest_positions: wp.array[wp.vec3],
    u_dofs: wp.array[wp.vec3],
    velocity: wp.array[wp.vec3],
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
):
    """Write rest + displacement to particle positions and copy out velocity."""
    tid = wp.tid()
    particle_q[tid] = rest_positions[tid] + u_dofs[tid]
    particle_qd[tid] = velocity[tid]
