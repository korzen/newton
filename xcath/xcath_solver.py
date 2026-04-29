# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Sample-local solver extensions for XCATH vessel containment."""

from __future__ import annotations

import numpy as np
import warp as wp

from newton._src.solvers.xpbd_rod.solver_xpbd_rod import SolverXPBDRod, _RodWorkspace

COLLISION_PROJECTION_STAGE_PRE = "pre"
COLLISION_PROJECTION_STAGE_POST = "post"
_COLLISION_PROJECTION_STAGES = (
    COLLISION_PROJECTION_STAGE_PRE,
    COLLISION_PROJECTION_STAGE_POST,
)
DEFAULT_MESH_EDGE_MAX_TRIANGLES = 64


def compute_smooth_vertex_normals(vertices: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Compute area-weighted vertex normals from triangle topology."""
    vertices = np.asarray(vertices, dtype=np.float32).reshape((-1, 3))
    triangles = np.asarray(indices, dtype=np.int32).reshape((-1, 3))

    normals = np.zeros_like(vertices, dtype=np.float32)
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0).astype(np.float32)

    np.add.at(normals, triangles[:, 0], face_normals)
    np.add.at(normals, triangles[:, 1], face_normals)
    np.add.at(normals, triangles[:, 2], face_normals)

    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1.0e-12
    normals[valid] /= lengths[valid, None]
    normals[~valid] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return normals


@wp.func
def _mesh_face_normal(mesh_id: wp.uint64, face_index: int) -> wp.vec3:
    """Compute an outward face normal for a triangle in ``mesh_id``."""
    mesh = wp.mesh_get(mesh_id)
    i0 = mesh.indices[face_index * 3 + 0]
    i1 = mesh.indices[face_index * 3 + 1]
    i2 = mesh.indices[face_index * 3 + 2]
    v0 = mesh.points[i0]
    v1 = mesh.points[i1]
    v2 = mesh.points[i2]
    return wp.normalize(wp.cross(v1 - v0, v2 - v0))


@wp.func
def _safe_normalize(v: wp.vec3, fallback: wp.vec3) -> wp.vec3:
    length = wp.length(v)
    if length > 1.0e-8:
        return v / length
    return fallback


@wp.func
def _closest_point_triangle(point: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3) -> wp.vec3:
    ab = b - a
    ac = c - a
    ap = point - a
    d1 = wp.dot(ab, ap)
    d2 = wp.dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return a

    bp = point - b
    d3 = wp.dot(ab, bp)
    d4 = wp.dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return b

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return a + ab * v

    cp = point - c
    d5 = wp.dot(ab, cp)
    d6 = wp.dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return c

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return a + ac * w

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + (c - b) * w

    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    return a + ab * v + ac * w


@wp.func
def _closest_point_triangle_barycentric(point: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3) -> wp.vec3:
    ab = b - a
    ac = c - a
    ap = point - a
    d1 = wp.dot(ab, ap)
    d2 = wp.dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return wp.vec3(1.0, 0.0, 0.0)

    bp = point - b
    d3 = wp.dot(ab, bp)
    d4 = wp.dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return wp.vec3(0.0, 1.0, 0.0)

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return wp.vec3(1.0 - v, v, 0.0)

    cp = point - c
    d5 = wp.dot(ab, cp)
    d6 = wp.dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return wp.vec3(0.0, 0.0, 1.0)

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return wp.vec3(1.0 - w, 0.0, w)

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return wp.vec3(0.0, 1.0 - w, w)

    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    return wp.vec3(1.0 - v - w, v, w)


@wp.func
def _mesh_barycentric_normal(
    mesh_id: wp.uint64,
    smooth_normals: wp.array[wp.vec3],
    face_index: int,
    weights: wp.vec3,
    fallback: wp.vec3,
) -> wp.vec3:
    mesh = wp.mesh_get(mesh_id)
    i0 = mesh.indices[face_index * 3 + 0]
    i1 = mesh.indices[face_index * 3 + 1]
    i2 = mesh.indices[face_index * 3 + 2]
    n = smooth_normals[i0] * weights.x + smooth_normals[i1] * weights.y + smooth_normals[i2] * weights.z
    return _safe_normalize(n, fallback)


@wp.func
def _segment_segment_barycentric(a0: wp.vec3, a1: wp.vec3, b0: wp.vec3, b1: wp.vec3) -> wp.vec4:
    d1 = a1 - a0
    d2 = b1 - b0
    r = a0 - b0
    a = wp.dot(d1, d1)
    e = wp.dot(d2, d2)
    f = wp.dot(d2, r)

    s = float(0.0)
    t = float(0.0)
    eps = float(1.0e-8)

    if a <= eps and e <= eps:
        s = 0.0
        t = 0.0
    elif a <= eps:
        s = 0.0
        t = wp.clamp(f / e, 0.0, 1.0)
    else:
        c = wp.dot(d1, r)
        if e <= eps:
            t = 0.0
            s = wp.clamp(-c / a, 0.0, 1.0)
        else:
            b = wp.dot(d1, d2)
            denom = a * e - b * b
            if wp.abs(denom) > eps:
                s = wp.clamp((b * f - c * e) / denom, 0.0, 1.0)
            else:
                s = 0.0

            tnom = b * s + f
            if tnom < 0.0:
                t = 0.0
                s = wp.clamp(-c / a, 0.0, 1.0)
            elif tnom > e:
                t = 1.0
                s = wp.clamp((b - c) / a, 0.0, 1.0)
            else:
                t = tnom / e

    return wp.vec4(1.0 - s, s, 1.0 - t, t)


@wp.kernel
def _track_sliding_kernel(
    predicted_positions: wp.array(dtype=wp.vec3),
    inv_masses: wp.array(dtype=wp.float32),
    track_start: wp.vec3,
    track_dir: wp.vec3,
    track_length: float,
    stiffness: float,
    end_idx: int,
):
    """Project non-tip particles onto the insertion track."""
    i = wp.tid()
    if i == 0 or i >= end_idx:
        return
    if inv_masses[i] <= 0.0:
        return

    pos = predicted_positions[i]
    t = wp.dot(pos - track_start, track_dir)
    if t < 0.0 or t > track_length:
        return

    closest = track_start + track_dir * t
    predicted_positions[i] = pos + (closest - pos) * stiffness


@wp.kernel
def _clear_vec3_kernel(values: wp.array[wp.vec3]):
    i = wp.tid()
    values[i] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def _project_vessel_containment_kernel(
    current_positions: wp.array(dtype=wp.vec3),
    predicted_positions: wp.array(dtype=wp.vec3),
    mesh_id: wp.uint64,
    smooth_normals: wp.array[wp.vec3],
    inv_masses: wp.array(dtype=wp.float32),
    sign_scale: float,
    use_smooth_normals: bool,
    target_phi: float,
    max_dist: float,
):
    """Project predicted positions to the vessel interior signed offset."""
    i = wp.tid()
    if inv_masses[i] <= 0.0:
        return

    pos = predicted_positions[i]

    face_index = int(0)
    face_u = float(0.0)
    face_v = float(0.0)
    sign = float(0.0)

    if not wp.mesh_query_point_sign_normal(mesh_id, pos, max_dist, sign, face_index, face_u, face_v):
        return

    closest = wp.mesh_eval_position(mesh_id, face_index, face_u, face_v)
    diff = pos - closest
    dist = wp.length(diff)

    side = sign * sign_scale
    if wp.abs(side) < 1.0e-6:
        side = sign_scale

    grad = wp.vec3(0.0, 0.0, 0.0)
    if use_smooth_normals:
        weights = wp.vec3(1.0 - face_u - face_v, face_u, face_v)
        face_normal = _mesh_face_normal(mesh_id, face_index)
        grad = _mesh_barycentric_normal(mesh_id, smooth_normals, face_index, weights, face_normal) * sign_scale
        if dist > 1.0e-8 and wp.dot(grad, diff) * side < 0.0:
            grad = -grad
    elif dist > 1.0e-8:
        grad = (diff / dist) * side
    else:
        grad = _mesh_face_normal(mesh_id, face_index) * side

    phi = side * dist
    if phi <= target_phi:
        return

    # Velocity is recomputed by _warp_integrate_positions as (predicted - current) / dt,
    # so we clamp the position-delta's outward component to keep that derived velocity from
    # pointing outward. Writing to velocities here would be overwritten downstream.
    projected = pos - grad * (phi - target_phi)
    disp = projected - current_positions[i]
    outward_disp = wp.dot(disp, grad)
    if outward_disp > 0.0:
        projected = projected - grad * outward_disp
    predicted_positions[i] = projected


@wp.kernel
def _project_mesh_vertex_collision_kernel(
    predicted_positions: wp.array[wp.vec3],
    mesh_id: wp.uint64,
    smooth_normals: wp.array[wp.vec3],
    inv_masses: wp.array[float],
    radius: float,
    query_radius: float,
    sign_scale: float,
    use_smooth_normals: bool,
    max_triangles: int,
    corrections: wp.array[wp.vec3],
):
    """Per-particle vertex-vs-triangle containment.

    Picks the single triangle in the AABB query with maximum penetration and applies
    that correction, instead of summing per-triangle pushes. Summing over-corrects on
    curved surfaces where many tangent triangles each report a similar penetration.
    """
    i = wp.tid()
    if inv_masses[i] <= 0.0:
        return

    p = predicted_positions[i]
    lower = wp.vec3(p.x - query_radius, p.y - query_radius, p.z - query_radius)
    upper = wp.vec3(p.x + query_radius, p.y + query_radius, p.z + query_radius)

    mesh = wp.mesh_get(mesh_id)
    query = wp.mesh_query_aabb(mesh_id, lower, upper)
    face_index = wp.int32(0)
    tri_count = int(0)
    best_pen = float(0.0)
    best_n = wp.vec3(0.0, 0.0, 0.0)

    while wp.mesh_query_aabb_next(query, face_index):
        if tri_count >= max_triangles:
            break
        tri_count = tri_count + 1

        i0 = mesh.indices[face_index * 3 + 0]
        i1 = mesh.indices[face_index * 3 + 1]
        i2 = mesh.indices[face_index * 3 + 2]
        v0 = mesh.points[i0]
        v1 = mesh.points[i1]
        v2 = mesh.points[i2]
        # n_outward points toward the side where the rod must NOT be.
        # sign_scale matches the SDF kernel's convention (flips if mesh winding is reversed).
        face_normal = _safe_normalize(wp.cross(v1 - v0, v2 - v0), wp.vec3(0.0, 0.0, 1.0))
        bar = _closest_point_triangle_barycentric(p, v0, v1, v2)
        n_outward = face_normal * sign_scale
        if use_smooth_normals:
            n_outward = _mesh_barycentric_normal(mesh_id, smooth_normals, face_index, bar, face_normal) * sign_scale

        cp = v0 * bar.x + v1 * bar.y + v2 * bar.z
        signed = wp.dot(p - cp, n_outward)
        penetration = signed + radius
        if penetration > best_pen:
            best_pen = penetration
            best_n = n_outward

    if best_pen > 0.0:
        wp.atomic_add(corrections, i, -best_n * best_pen)


@wp.kernel
def _project_mesh_vertex_collision_kernel_averaged(
    predicted_positions: wp.array[wp.vec3],
    mesh_id: wp.uint64,
    smooth_normals: wp.array[wp.vec3],
    inv_masses: wp.array[float],
    radius: float,
    query_radius: float,
    sign_scale: float,
    use_smooth_normals: bool,
    max_triangles: int,
    corrections: wp.array[wp.vec3],
):
    """Per-particle vertex-vs-triangle containment. """

    i = wp.tid()
    if inv_masses[i] <= 0.0:
        return

    p = predicted_positions[i]
    lower = wp.vec3(p.x - query_radius, p.y - query_radius, p.z - query_radius)
    upper = wp.vec3(p.x + query_radius, p.y + query_radius, p.z + query_radius)

    mesh = wp.mesh_get(mesh_id)
    query = wp.mesh_query_aabb(mesh_id, lower, upper)
    face_index = wp.int32(0)
    tri_count = int(0)
    col_count = int(0)
    delta = wp.vec3(0.0, 0.0, 0.0)


    while wp.mesh_query_aabb_next(query, face_index):
        if tri_count >= max_triangles:
            break

        tri_count = tri_count + 1

        i0 = mesh.indices[face_index * 3 + 0]
        i1 = mesh.indices[face_index * 3 + 1]
        i2 = mesh.indices[face_index * 3 + 2]

        v0 = mesh.points[i0]
        v1 = mesh.points[i1]
        v2 = mesh.points[i2]

        # n_outward points toward the side where the rod must NOT be.
        # sign_scale matches the SDF kernel's convention (flips if mesh winding is reversed).
        face_normal = _safe_normalize(wp.cross(v1 - v0, v2 - v0), wp.vec3(0.0, 0.0, 1.0))
        bar = _closest_point_triangle_barycentric(p, v0, v1, v2)
        n_outward = face_normal * sign_scale
        if use_smooth_normals:
            n_outward = _mesh_barycentric_normal(mesh_id, smooth_normals, face_index, bar, face_normal) * sign_scale

        cp = v0 * bar.x + v1 * bar.y + v2 * bar.z
        signed = wp.dot(p - cp, n_outward)
        penetration = signed + radius
        if penetration > 0.0:
            delta = delta - n_outward * penetration
            col_count += 1

    if col_count > 0:
        wp.atomic_add(corrections, i, delta / float(col_count))




@wp.kernel
def _project_mesh_edge_collision_kernel(
    predicted_positions: wp.array[wp.vec3],
    mesh_id: wp.uint64,
    smooth_normals: wp.array[wp.vec3],
    inv_masses: wp.array[float],
    radius: float,
    query_radius: float,
    sign_scale: float,
    use_smooth_normals: bool,
    max_triangles: int,
    corrections: wp.array[wp.vec3],
):
    """Per-rod-edge: rod-segment versus mesh-edge containment corrections.

    Picks the single mesh edge with maximum penetration and applies that correction
    (analogous to the vertex kernel — summing all overlapping mesh edges over-corrects
    on curved surfaces). Mesh edges are deduped via ``ia < ib`` so each undirected
    edge in a closed manifold is considered from only one adjacent triangle.
    """
    edge = wp.tid()

    p0 = predicted_positions[edge]
    p1 = predicted_positions[edge + 1]
    inv0 = inv_masses[edge]
    inv1 = inv_masses[edge + 1]
    if inv0 <= 0.0 and inv1 <= 0.0:
        return

    lower = wp.vec3(
        wp.min(p0.x, p1.x) - query_radius,
        wp.min(p0.y, p1.y) - query_radius,
        wp.min(p0.z, p1.z) - query_radius,
    )
    upper = wp.vec3(
        wp.max(p0.x, p1.x) + query_radius,
        wp.max(p0.y, p1.y) + query_radius,
        wp.max(p0.z, p1.z) + query_radius,
    )

    mesh = wp.mesh_get(mesh_id)
    query = wp.mesh_query_aabb(mesh_id, lower, upper)
    face_index = wp.int32(0)
    tri_count = int(0)
    best_pen = float(0.0)
    best_n = wp.vec3(0.0, 0.0, 0.0)
    best_bar = wp.vec4(0.0, 0.0, 0.0, 0.0)

    while wp.mesh_query_aabb_next(query, face_index):
        if tri_count >= max_triangles:
            break
        tri_count = tri_count + 1

        i0 = mesh.indices[face_index * 3 + 0]
        i1 = mesh.indices[face_index * 3 + 1]
        i2 = mesh.indices[face_index * 3 + 2]
        v0 = mesh.points[i0]
        v1 = mesh.points[i1]
        v2 = mesh.points[i2]
        face_normal = _safe_normalize(wp.cross(v1 - v0, v2 - v0), wp.vec3(0.0, 0.0, 1.0))

        for mesh_edge in range(3):
            ia = i0
            ib = i1
            ea = v0
            eb = v1
            mesh_normal_a = wp.vec3(1.0, 0.0, 0.0)
            mesh_normal_b = wp.vec3(0.0, 1.0, 0.0)
            if mesh_edge == 1:
                ia = i1
                ib = i2
                ea = v1
                eb = v2
                mesh_normal_a = wp.vec3(0.0, 1.0, 0.0)
                mesh_normal_b = wp.vec3(0.0, 0.0, 1.0)
            elif mesh_edge == 2:
                ia = i2
                ib = i0
                ea = v2
                eb = v0
                mesh_normal_a = wp.vec3(0.0, 0.0, 1.0)
                mesh_normal_b = wp.vec3(1.0, 0.0, 0.0)
            # Dedup: each interior edge is adjacent to two triangles whose local
            # winding visits it in opposite directions. Process only one orientation.
            if ia >= ib:
                continue

            bar = _segment_segment_barycentric(p0, p1, ea, eb)
            rod_point = p0 * bar.x + p1 * bar.y
            mesh_point = ea * bar.z + eb * bar.w
            n_outward = face_normal * sign_scale
            if use_smooth_normals:
                weights = mesh_normal_a * bar.z + mesh_normal_b * bar.w
                n_outward = _mesh_barycentric_normal(
                    mesh_id, smooth_normals, face_index, weights, face_normal
                ) * sign_scale
            signed = wp.dot(rod_point - mesh_point, n_outward)
            penetration = signed + radius
            if penetration > best_pen:
                best_pen = penetration
                best_n = n_outward
                best_bar = bar

    if best_pen > 0.0:
        denom = inv0 * best_bar.x * best_bar.x + inv1 * best_bar.y * best_bar.y
        if denom > 1.0e-8:
            correction = -best_n * (best_pen / denom)
            if inv0 > 0.0:
                wp.atomic_add(corrections, edge, correction * best_bar.x * inv0)
            if inv1 > 0.0:
                wp.atomic_add(corrections, edge + 1, correction * best_bar.y * inv1)


@wp.kernel
def _project_mesh_edge_collision_kernel_averaged(
    predicted_positions: wp.array[wp.vec3],
    mesh_id: wp.uint64,
    smooth_normals: wp.array[wp.vec3],
    inv_masses: wp.array[float],
    radius: float,
    query_radius: float,
    sign_scale: float,
    use_smooth_normals: bool,
    max_triangles: int,
    corrections: wp.array[wp.vec3],
):
    """Per-rod-edge: rod-segment versus mesh-edge containment, averaged.

    Sums every penetrating mesh-edge correction split across the two rod endpoints,
    then writes the arithmetic mean per endpoint. Mesh edges are deduped via
    ``ia < ib`` so each undirected edge in a closed manifold is considered from
    only one adjacent triangle.
    """
    edge = wp.tid()

    p0 = predicted_positions[edge]
    p1 = predicted_positions[edge + 1]
    inv0 = inv_masses[edge]
    inv1 = inv_masses[edge + 1]
    if inv0 <= 0.0 and inv1 <= 0.0:
        return

    lower = wp.vec3(
        wp.min(p0.x, p1.x) - query_radius,
        wp.min(p0.y, p1.y) - query_radius,
        wp.min(p0.z, p1.z) - query_radius,
    )
    upper = wp.vec3(
        wp.max(p0.x, p1.x) + query_radius,
        wp.max(p0.y, p1.y) + query_radius,
        wp.max(p0.z, p1.z) + query_radius,
    )

    mesh = wp.mesh_get(mesh_id)
    query = wp.mesh_query_aabb(mesh_id, lower, upper)
    face_index = wp.int32(0)
    tri_count = int(0)
    col_count = int(0)
    delta_p0 = wp.vec3(0.0, 0.0, 0.0)
    delta_p1 = wp.vec3(0.0, 0.0, 0.0)

    while wp.mesh_query_aabb_next(query, face_index):
        if tri_count >= max_triangles:
            break
        tri_count = tri_count + 1

        i0 = mesh.indices[face_index * 3 + 0]
        i1 = mesh.indices[face_index * 3 + 1]
        i2 = mesh.indices[face_index * 3 + 2]
        v0 = mesh.points[i0]
        v1 = mesh.points[i1]
        v2 = mesh.points[i2]
        face_normal = _safe_normalize(wp.cross(v1 - v0, v2 - v0), wp.vec3(0.0, 0.0, 1.0))

        for mesh_edge in range(3):
            ia = i0
            ib = i1
            ea = v0
            eb = v1
            mesh_normal_a = wp.vec3(1.0, 0.0, 0.0)
            mesh_normal_b = wp.vec3(0.0, 1.0, 0.0)
            if mesh_edge == 1:
                ia = i1
                ib = i2
                ea = v1
                eb = v2
                mesh_normal_a = wp.vec3(0.0, 1.0, 0.0)
                mesh_normal_b = wp.vec3(0.0, 0.0, 1.0)
            elif mesh_edge == 2:
                ia = i2
                ib = i0
                ea = v2
                eb = v0
                mesh_normal_a = wp.vec3(0.0, 0.0, 1.0)
                mesh_normal_b = wp.vec3(1.0, 0.0, 0.0)
            # Dedup: each interior edge is adjacent to two triangles whose local
            # winding visits it in opposite directions. Process only one orientation.
            if ia >= ib:
                continue

            bar = _segment_segment_barycentric(p0, p1, ea, eb)
            rod_point = p0 * bar.x + p1 * bar.y
            mesh_point = ea * bar.z + eb * bar.w
            n_outward = face_normal * sign_scale
            if use_smooth_normals:
                weights = mesh_normal_a * bar.z + mesh_normal_b * bar.w
                n_outward = _mesh_barycentric_normal(
                    mesh_id, smooth_normals, face_index, weights, face_normal
                ) * sign_scale
            signed = wp.dot(rod_point - mesh_point, n_outward)
            penetration = signed + radius
            if penetration > 0.0:
                denom = inv0 * bar.x * bar.x + inv1 * bar.y * bar.y
                if denom > 1.0e-8:
                    correction = -n_outward * (penetration / denom)
                    if inv0 > 0.0:
                        delta_p0 = delta_p0 + correction * bar.x * inv0
                    if inv1 > 0.0:
                        delta_p1 = delta_p1 + correction * bar.y * inv1
                    col_count = col_count + 1

    if col_count > 0:
        inv_count = 1.0 / float(col_count)
        if inv0 > 0.0:
            wp.atomic_add(corrections, edge, delta_p0 * inv_count)
        if inv1 > 0.0:
            wp.atomic_add(corrections, edge + 1, delta_p1 * inv_count)


@wp.kernel
def _apply_mesh_edge_collision_corrections_kernel(
    predicted_positions: wp.array[wp.vec3],
    inv_masses: wp.array[float],
    corrections: wp.array[wp.vec3],
):
    """Apply accumulated mesh-edge collision corrections."""
    i = wp.tid()
    if inv_masses[i] > 0.0:
        predicted_positions[i] = predicted_positions[i] + corrections[i]


@wp.kernel
def _sample_signed_distance_kernel(
    positions: wp.array(dtype=wp.vec3),
    mesh_id: wp.uint64,
    sign_scale: float,
    max_dist: float,
    signed_distance: wp.array(dtype=wp.float32),
    hit: wp.array(dtype=wp.int32),
):
    """Sample signed distance from ``positions`` to ``mesh_id``."""
    i = wp.tid()

    face_index = int(0)
    face_u = float(0.0)
    face_v = float(0.0)
    sign = float(0.0)

    if not wp.mesh_query_point_sign_normal(mesh_id, positions[i], max_dist, sign, face_index, face_u, face_v):
        signed_distance[i] = max_dist
        hit[i] = 0
        return

    closest = wp.mesh_eval_position(mesh_id, face_index, face_u, face_v)
    signed_distance[i] = wp.length(positions[i] - closest) * sign * sign_scale
    hit[i] = 1


def compute_signed_distances(
    positions: wp.array,
    mesh_id: wp.uint64,
    max_dist: float,
    sign_scale: float = 1.0,
    device: wp.Device | None = None,
) -> tuple[wp.array, wp.array]:
    """Sample signed distance for each point in ``positions``."""
    device = device or positions.device
    count = int(positions.shape[0])
    signed_distance = wp.empty(count, dtype=wp.float32, device=device)
    hit = wp.empty(count, dtype=wp.int32, device=device)
    wp.launch(
        _sample_signed_distance_kernel,
        dim=count,
        inputs=[positions, mesh_id, float(sign_scale), float(max_dist)],
        outputs=[signed_distance, hit],
        device=device,
    )
    return signed_distance, hit


def _normalize_collision_projection_stage(stage: str) -> str:
    """Normalize the collision projection stage option."""
    normalized = str(stage).strip().lower()
    if normalized not in _COLLISION_PROJECTION_STAGES:
        raise ValueError(
            f"Unknown collision_projection_stage {stage!r}. "
            f"Expected one of {_COLLISION_PROJECTION_STAGES}"
        )
    return normalized


class XCathRodSolver(SolverXPBDRod):
    """Sample-local XPBD rod solver with track and vessel projections."""

    def __init__(
        self,
        *args,
        collision_mesh: wp.Mesh | None,
        track_start: np.ndarray,
        track_dir: np.ndarray,
        track_length: float,
        tip_num_edges: int,
        particle_radius: float,
        segment_length: float,
        track_stiffness: float = 1.0,
        track_enabled: bool = True,
        collision_enabled: bool = True,
        collision_mesh_normals: wp.array[wp.vec3] | None = None,
        smooth_collision_normals_enabled: bool = False,
        collision_iterations: int = 2,
        collision_pre_constraints_enabled: bool = False,
        collision_post_constraints_enabled: bool = True,
        mesh_edge_collision_enabled: bool = False,
        mesh_edge_collision_radius: float | None = None,
        mesh_edge_collision_query_radius: float | None = None,
        mesh_edge_collision_max_triangles: int = DEFAULT_MESH_EDGE_MAX_TRIANGLES,
        collision_projection_stage: str | None = None,
        sign_scale: float = 1.0,
        target_phi: float | None = None,
        max_dist: float | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        track_start = np.asarray(track_start, dtype=np.float32)
        track_dir = np.asarray(track_dir, dtype=np.float32)
        norm = np.linalg.norm(track_dir)
        if norm <= 0.0:
            raise ValueError("track_dir must be non-zero")

        track_dir = track_dir / norm

        self.collision_mesh = collision_mesh
        self.collision_mesh_normals = collision_mesh_normals
        self.smooth_collision_normals_enabled = bool(smooth_collision_normals_enabled)
        self.track_start = wp.vec3(float(track_start[0]), float(track_start[1]), float(track_start[2]))
        self.track_dir = wp.vec3(float(track_dir[0]), float(track_dir[1]), float(track_dir[2]))
        self.track_length = float(track_length)
        self.tip_num_edges = int(tip_num_edges)
        self.track_stiffness = float(track_stiffness)
        self.track_enabled = bool(track_enabled)
        self.collision_enabled = bool(collision_enabled)
        self.collision_iterations = max(1, int(collision_iterations))
        self.collision_pre_constraints_enabled = bool(collision_pre_constraints_enabled)
        self.collision_post_constraints_enabled = bool(collision_post_constraints_enabled)
        self.mesh_edge_collision_enabled = bool(mesh_edge_collision_enabled)
        self.mesh_edge_collision_radius = (
            float(mesh_edge_collision_radius) if mesh_edge_collision_radius is not None else float(particle_radius)
        )
        self.mesh_edge_collision_query_radius = (
            float(mesh_edge_collision_query_radius)
            if mesh_edge_collision_query_radius is not None
            else self.mesh_edge_collision_radius
        )
        self.mesh_edge_collision_max_triangles = max(1, int(mesh_edge_collision_max_triangles))
        if collision_projection_stage is not None:
            self.set_collision_projection_stage(collision_projection_stage)
        self.sign_scale = float(sign_scale)
        self.target_phi = float(target_phi) if target_phi is not None else -float(particle_radius)
        self.max_dist = float(max_dist) if max_dist is not None else 2.0 * float(particle_radius) + float(segment_length)

        # Dedicated correction buffers per rod for the mesh-edge collision path.
        # Decoupled from ws.pos_corrections_wp (which the constraint solver owns).
        self._mesh_edge_corrections: list[wp.array] = [
            wp.zeros(max(1, ws.num_points), dtype=wp.vec3, device=ws.device) for ws in self._rods
        ]
        self._dummy_collision_mesh_normals: dict[str, wp.array[wp.vec3]] = {}

    def set_collision_mesh(
        self, collision_mesh: wp.Mesh | None, collision_mesh_normals: wp.array[wp.vec3] | None = None
    ) -> None:
        """Swap the vessel mesh used for containment."""
        self.collision_mesh = collision_mesh
        self.collision_mesh_normals = collision_mesh_normals

    def _collision_normals_for_device(self, device: wp.Device) -> tuple[wp.array[wp.vec3], bool]:
        """Return a normals array plus whether kernels should sample it."""
        if self.smooth_collision_normals_enabled and self.collision_mesh_normals is not None:
            return self.collision_mesh_normals, True

        key = str(device)
        if key not in self._dummy_collision_mesh_normals:
            self._dummy_collision_mesh_normals[key] = wp.zeros(1, dtype=wp.vec3, device=device)
        return self._dummy_collision_mesh_normals[key], False

    def set_collision_projection_stage(self, stage: str) -> None:
        """Set whether vessel containment runs before or after rod constraints."""
        normalized = _normalize_collision_projection_stage(stage)
        self.collision_pre_constraints_enabled = normalized == COLLISION_PROJECTION_STAGE_PRE
        self.collision_post_constraints_enabled = normalized == COLLISION_PROJECTION_STAGE_POST

    def _project_track_guidance(self, ws: _RodWorkspace, device: wp.Device) -> None:
        """Project catheter particles toward the insertion track."""
        if self.track_enabled:
            end_idx = max(1, ws.num_points - self.tip_num_edges)
            wp.launch(
                _track_sliding_kernel,
                dim=ws.num_points,
                inputs=[
                    ws.predicted_positions_wp,
                    ws.inv_masses_wp,
                    self.track_start,
                    self.track_dir,
                    float(self.track_length),
                    float(self.track_stiffness),
                    int(end_idx),
                ],
                device=device,
            )

    def _project_vessel_containment(self, rod_idx: int, ws: _RodWorkspace, device: wp.Device) -> None:
        """Project catheter particles to the vessel interior."""
        if self.collision_enabled and self.collision_mesh is not None:
            smooth_normals, use_smooth_normals = self._collision_normals_for_device(device)
            if self.mesh_edge_collision_enabled:
                corrections = self._mesh_edge_corrections[rod_idx]
                for _ in range(self.collision_iterations):
                    wp.launch(_clear_vec3_kernel, dim=ws.num_points, inputs=[corrections], device=device)
                    wp.launch(
                        _project_mesh_vertex_collision_kernel_averaged,
                        dim=ws.num_points,
                        inputs=[
                            ws.predicted_positions_wp,
                            self.collision_mesh.id,
                            smooth_normals,
                            ws.inv_masses_wp,
                            float(self.mesh_edge_collision_radius),
                            float(self.mesh_edge_collision_query_radius),
                            float(self.sign_scale),
                            bool(use_smooth_normals),
                            int(self.mesh_edge_collision_max_triangles),
                            corrections,
                        ],
                        device=device,
                    )
                    wp.launch(
                        _project_mesh_edge_collision_kernel_averaged,
                        dim=ws.num_edges,
                        inputs=[
                            ws.predicted_positions_wp,
                            self.collision_mesh.id,
                            smooth_normals,
                            ws.inv_masses_wp,
                            float(self.mesh_edge_collision_radius),
                            float(self.mesh_edge_collision_query_radius),
                            float(self.sign_scale),
                            bool(use_smooth_normals),
                            int(self.mesh_edge_collision_max_triangles),
                            corrections,
                        ],
                        device=device,
                    )
                    wp.launch(
                        _apply_mesh_edge_collision_corrections_kernel,
                        dim=ws.num_points,
                        inputs=[ws.predicted_positions_wp, ws.inv_masses_wp, corrections],
                        device=device,
                    )
            else:
                for _ in range(self.collision_iterations):
                    wp.launch(
                        _project_vessel_containment_kernel,
                        dim=ws.num_points,
                        inputs=[
                            ws.positions_wp,
                            ws.predicted_positions_wp,
                            self.collision_mesh.id,
                            smooth_normals,
                            ws.inv_masses_wp,
                            float(self.sign_scale),
                            bool(use_smooth_normals),
                            float(self.target_phi),
                            float(self.max_dist),
                        ],
                        device=device,
                    )

    def _project_xcath_predicted_positions(
        self, rod_idx: int, ws: _RodWorkspace, device: wp.Device
    ) -> None:
        """Track guidance always; vessel containment only when the post-stage flag is set."""
        self._project_track_guidance(ws, device)
        if self.collision_post_constraints_enabled:
            self._project_vessel_containment(rod_idx, ws, device)

    def _project_predicted_positions_pre_constraints(
        self,
        rod_idx: int,
        ws: _RodWorkspace,
        dt: float,
        device: wp.Device,
    ) -> None:
        """Apply vessel containment before rod constraints when configured."""
        del dt
        if self.collision_pre_constraints_enabled:
            self._project_vessel_containment(rod_idx, ws, device)

    def _project_predicted_positions_post_constraints(
        self,
        rod_idx: int,
        ws: _RodWorkspace,
        dt: float,
        device: wp.Device,
    ) -> None:
        """Apply XCATH projections after rod constraints when configured."""
        del dt
        self._project_xcath_predicted_positions(rod_idx, ws, device)


__all__ = [
    "XCathRodSolver",
    "COLLISION_PROJECTION_STAGE_POST",
    "COLLISION_PROJECTION_STAGE_PRE",
    "compute_smooth_vertex_normals",
    "compute_signed_distances",
    "_apply_mesh_edge_collision_corrections_kernel",
    "_project_mesh_edge_collision_kernel",
    "_project_mesh_vertex_collision_kernel",
    "_project_vessel_containment_kernel",
    "_sample_signed_distance_kernel",
]
