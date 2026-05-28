# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example VSD Device
#
# This simulation loads a legacy VTK unstructured grid containing tetrahedral
# cells and simulates it as a volumetric soft body.
#
# Command: uv run -m newton.examples vsd_device
#
###########################################################################

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.slicer.vtk_loader import load_vtk_polydata, load_vtk_unstructured_grid

PARTICLE_ACTIVE = wp.constant(int(newton.ParticleFlags.ACTIVE))
MESH_PATH = Path(__file__).resolve().parent / "vsd_device" / "mesh3.1.vtk"
# VESSEL_MESH_PATH = Path(__file__).resolve().parent / "anatomy" / "RVOT1_Alterra_vessel.vtk"
VESSEL_MESH_PATH = Path(__file__).resolve().parent / "anatomy" / "VSD from LV.vtk"
DEFAULT_DEVICE_POSE_SLOTS_PATH = Path.home() / ".cache" / "newton" / "vsd_device_pose_slots.json"
VESSEL_CONTACT_RADIUS_MIN = 0.0
VESSEL_CONTACT_RADIUS_MAX = 0.2
VESSEL_CONTACT_RELAXATION_MIN = 0.0
VESSEL_CONTACT_RELAXATION_MAX = 1.0
VESSEL_CONTACT_ITERATIONS_MIN = 0
VESSEL_CONTACT_ITERATIONS_MAX = 8
DEVICE_COMPRESSION_SCALE_MIN = 0.05
DEVICE_COMPRESSION_SCALE_MAX = 1.0
DEVICE_POSE_SLOT_KEYS = tuple(str(slot) for slot in range(1, 10))
VBD_TET_STIFFNESS_EXPONENT_MIN = 1
VBD_TET_STIFFNESS_EXPONENT_MAX = 10
XPBD_TET_STIFFNESS_EXPONENT_MIN = 1
XPBD_TET_STIFFNESS_EXPONENT_MAX = 10
FEM_TET_STIFFNESS_EXPONENT_MIN = 1
FEM_TET_STIFFNESS_EXPONENT_MAX = 7
TET_STIFFNESS_MULTIPLIER_MIN = 0.0
TET_STIFFNESS_MULTIPLIER_MAX = 10.0
XPBD_DISABLED_TET_COMPLIANCE = 1.0e8
FEM_SOFT_CONTACT_STIFFNESS_EXPONENT_MIN = 1
FEM_SOFT_CONTACT_STIFFNESS_EXPONENT_MAX = 8
FEM_SOFT_CONTACT_STIFFNESS_MULTIPLIER_MIN = 0.0
FEM_SOFT_CONTACT_STIFFNESS_MULTIPLIER_MAX = 10.0
FEM_SOFT_CONTACT_DAMPING_MIN = 0.0
FEM_SOFT_CONTACT_DAMPING_MAX = 500.0
FEM_SOFT_CONTACT_FRICTION_MIN = 0.0
FEM_SOFT_CONTACT_FRICTION_MAX = 2.0
FEM_SOFT_CONTACT_MARGIN_MIN = 0.0
FEM_SOFT_CONTACT_MARGIN_MAX = 0.1
FEM_GLOBAL_DAMPING_MIN = 0.0
FEM_GLOBAL_DAMPING_MAX = 100.0
DEVICE_PARTICLE_RADIUS_MIN = 0.001
DEVICE_PARTICLE_RADIUS_MAX = 0.08
VESSEL_CONTACT_FORCE_COLOR_MAX_MIN = 1.0
VESSEL_CONTACT_FORCE_COLOR_MAX_MAX = 5000.0
VESSEL_CONTACT_FORCE_COLOR_MULTIPLIER_MIN = 0.0
VESSEL_CONTACT_FORCE_COLOR_MULTIPLIER_MAX = 100.0
VESSEL_ALPHA_MIN = 0.0
VESSEL_ALPHA_MAX = 1.0
VESSEL_BASE_COLOR = (0.55, 0.55, 0.55)
VSD_ALPHA_MIN = 0.0
VSD_ALPHA_MAX = 1.0
VSD_CONTACT_FORCE_COLOR_MULTIPLIER_MIN = 0.0
VSD_CONTACT_FORCE_COLOR_MULTIPLIER_MAX = 100.0
VSD_BASE_COLOR = (0.7, 0.6, 0.4)
CONTACT_COLOR_SMOOTHING_PASSES_MIN = 0
CONTACT_COLOR_SMOOTHING_PASSES_MAX = 8
CONTACT_COLOR_SMOOTHING_STRENGTH_MIN = 0.0
CONTACT_COLOR_SMOOTHING_STRENGTH_MAX = 1.0
PLACEMENT_AXIS_BAND_RADIUS_MIN = 0.0
PLACEMENT_AXIS_BAND_RADIUS_MAX = 0.2
PLACEMENT_CLONE_STIFFNESS_MIN = 0.0
PLACEMENT_CLONE_STIFFNESS_MAX = 1000.0
PLACEMENT_CLONE_DAMPING_MIN = 0.0
PLACEMENT_CLONE_DAMPING_MAX = 1.0
PLACEMENT_MAX_CLONE_FORCE_MIN = 0.0
PLACEMENT_MAX_CLONE_FORCE_MAX = 5000.0
MINIMOU_WORKSPACE_SCALE_MIN = 0.01
MINIMOU_WORKSPACE_SCALE_MAX = 1.0
MINIMOU_POSITION_OFFSET_SPEED = 0.005
MINIMOU_ROTATION_OFFSET_SPEED_DEG = 0.5
MINIMOU_DEVICE_FRAME_AXIS_LENGTH = 0.5
WORLD_ORIGIN_AXIS_LENGTH = 0.3
WORLD_ORIGIN_AXIS_THICKNESS = 0.008
WORLD_ORIGIN_CUBE_HALF_EXTENT = 0.018


def load_vtk_unstructured_tet_mesh(path: Path) -> newton.TetMesh:
    """Load tetrahedral cells from a legacy ASCII VTK unstructured grid.

    Thin wrapper around :func:`newton.examples.slicer.vtk_loader.load_vtk_unstructured_grid`
    that extracts the dataset's ``VTK_TETRA`` cells into a :class:`newton.TetMesh`.
    """
    return load_vtk_unstructured_grid(path).to_tet_mesh()


def load_vtk_triangle_mesh(path: Path) -> newton.Mesh:
    """Load triangle geometry from a legacy ASCII VTK surface-capable dataset.

    Supports ``UNSTRUCTURED_GRID`` files by extracting native triangles and
    volumetric boundary triangles, and ``POLYDATA`` files by triangulating
    polygonal surface data.
    """
    dataset_type = read_vtk_dataset_type(path)
    if dataset_type == "POLYDATA":
        return load_vtk_polydata(path).to_mesh()
    if dataset_type != "UNSTRUCTURED_GRID":
        raise ValueError(f"Expected DATASET UNSTRUCTURED_GRID or POLYDATA in '{path}', got {dataset_type!r}.")

    grid = load_vtk_unstructured_grid(path)
    triangle_indices = grid.triangle_indices()
    if triangle_indices.size == 0:
        raise ValueError(f"No triangle surface could be extracted from '{path}'.")
    return newton.Mesh(
        vertices=grid.points,
        indices=triangle_indices.reshape(-1).astype(np.int32),
        compute_inertia=False,
    )


def read_vtk_dataset_type(path: Path) -> str:
    """Read the legacy VTK DATASET type without parsing the full file."""
    with path.open(encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0].upper() == "DATASET":
                return parts[1].upper()
    raise ValueError(f"VTK file '{path}' is missing the DATASET line.")


def compute_triangle_vertex_normals(vertices: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Compute area-independent vertex normals for triangle rendering."""
    triangles = indices.reshape(-1, 3)
    normals = np.zeros_like(vertices, dtype=np.float32)
    tri_vertices = vertices[triangles]
    face_normals = np.cross(tri_vertices[:, 1] - tri_vertices[:, 0], tri_vertices[:, 2] - tri_vertices[:, 0])
    face_lengths = np.linalg.norm(face_normals, axis=1)
    valid_faces = face_lengths > 1.0e-12
    face_normals[valid_faces] /= face_lengths[valid_faces, None]

    for corner in range(3):
        np.add.at(normals, triangles[:, corner], face_normals)

    normal_lengths = np.linalg.norm(normals, axis=1)
    valid_vertices = normal_lengths > 1.0e-12
    normals[valid_vertices] /= normal_lengths[valid_vertices, None]
    normals[~valid_vertices] = np.array((0.0, 0.0, 1.0), dtype=np.float32)
    return normals


def build_triangle_vertex_adjacency(indices: np.ndarray, vertex_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Build compact vertex-neighbor lists from triangle indices."""
    triangles = np.asarray(indices, dtype=np.int32).reshape(-1, 3)
    if vertex_count <= 0 or triangles.size == 0:
        return np.zeros(1, dtype=np.int32), np.empty(0, dtype=np.int32)

    directed_edges = np.empty((triangles.shape[0] * 6, 2), dtype=np.int32)
    directed_edges[0::6] = triangles[:, (0, 1)]
    directed_edges[1::6] = triangles[:, (0, 2)]
    directed_edges[2::6] = triangles[:, (1, 0)]
    directed_edges[3::6] = triangles[:, (1, 2)]
    directed_edges[4::6] = triangles[:, (2, 0)]
    directed_edges[5::6] = triangles[:, (2, 1)]

    order = np.lexsort((directed_edges[:, 1], directed_edges[:, 0]))
    directed_edges = directed_edges[order]
    unique = np.empty(directed_edges.shape[0], dtype=bool)
    unique[0] = True
    unique[1:] = np.any(directed_edges[1:] != directed_edges[:-1], axis=1)
    directed_edges = directed_edges[unique]

    counts = np.bincount(directed_edges[:, 0], minlength=vertex_count).astype(np.int32, copy=False)
    offsets = np.empty(vertex_count + 1, dtype=np.int32)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])

    return offsets, directed_edges[:, 1].astype(np.int32, copy=False)


def normalize_quat_xyzw(quaternion) -> np.ndarray:
    values = np.asarray(quaternion, dtype=np.float32)
    if values.shape != (4,):
        return np.array((0.0, 0.0, 0.0, 1.0), dtype=np.float32)

    norm = float(np.linalg.norm(values))
    if norm < 1.0e-8 or not math.isfinite(norm):
        return np.array((0.0, 0.0, 0.0, 1.0), dtype=np.float32)

    values = values / norm
    if values[3] < 0.0:
        values = -values
    return values


def quat_mul_xyzw(left, right) -> np.ndarray:
    lx, ly, lz, lw = normalize_quat_xyzw(left)
    rx, ry, rz, rw = normalize_quat_xyzw(right)
    return normalize_quat_xyzw(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        )
    )


def quat_inverse_xyzw(quaternion) -> np.ndarray:
    x, y, z, w = normalize_quat_xyzw(quaternion)
    return np.array((-x, -y, -z, w), dtype=np.float32)


def quat_rotate_xyzw(quaternion, vector) -> np.ndarray:
    q = normalize_quat_xyzw(quaternion)
    v = np.asarray(vector, dtype=np.float32)
    q_vec = q[:3]
    t = 2.0 * np.cross(q_vec, v)
    return v + q[3] * t + np.cross(q_vec, t)


@wp.kernel
def project_particles_vs_static_tri_mesh(
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    particle_inv_mass: wp.array[float],
    particle_flags: wp.array[wp.int32],
    mesh_id: wp.uint64,
    contact_radius: float,
    relaxation: float,
    dt: float,
    particle_force_metric: wp.array[float],
    vertex_force_metric: wp.array[float],
):
    particle_idx = wp.tid()
    if contact_radius <= 0.0 or relaxation <= 0.0 or dt <= 0.0:
        return
    if (particle_flags[particle_idx] & PARTICLE_ACTIVE) == 0:
        return
    if particle_inv_mass[particle_idx] <= 0.0:
        return

    x = particle_q[particle_idx]
    query = wp.mesh_query_point_no_sign(mesh_id, x, contact_radius)
    if not query.result:
        return

    closest = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
    delta = x - closest
    dist_sq = wp.length_sq(delta)

    normal = wp.vec3(0.0, 0.0, 1.0)
    dist = float(0.0)
    if dist_sq > 1.0e-16:
        dist = wp.sqrt(dist_sq)
        normal = delta / dist
    else:
        face_normal = wp.mesh_eval_face_normal(mesh_id, query.face)
        face_normal_len_sq = wp.length_sq(face_normal)
        if face_normal_len_sq > 1.0e-16:
            normal = face_normal / wp.sqrt(face_normal_len_sq)

    penetration = contact_radius - dist
    if penetration <= 0.0:
        return

    correction = normal * (penetration * relaxation)
    particle_q[particle_idx] = x + correction
    particle_qd[particle_idx] = particle_qd[particle_idx] + correction / dt

    force = penetration * relaxation / (particle_inv_mass[particle_idx] * dt * dt)
    wp.atomic_add(particle_force_metric, particle_idx, force)

    i0 = wp.mesh_get_index(mesh_id, query.face * 3 + 0)
    i1 = wp.mesh_get_index(mesh_id, query.face * 3 + 1)
    i2 = wp.mesh_get_index(mesh_id, query.face * 3 + 2)
    wp.atomic_add(vertex_force_metric, i0, force)
    wp.atomic_add(vertex_force_metric, i1, force)
    wp.atomic_add(vertex_force_metric, i2, force)


@wp.kernel
def accumulate_vessel_soft_contact_force_metric(
    soft_contact_count: wp.array[wp.int32],
    soft_contact_particle: wp.array[wp.int32],
    soft_contact_shape: wp.array[wp.int32],
    soft_contact_body_pos: wp.array[wp.vec3],
    soft_contact_body_vel: wp.array[wp.vec3],
    soft_contact_normal: wp.array[wp.vec3],
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    vessel_shape_idx: int,
    ke: float,
    kd: float,
    mu: float,
    force_metric: wp.array[float],
):
    tid = wp.tid()
    if tid >= soft_contact_count[0]:
        return
    if soft_contact_shape[tid] != vessel_shape_idx:
        return

    particle_idx = soft_contact_particle[tid]
    if particle_idx < 0:
        return

    bx = soft_contact_body_pos[tid]
    n = soft_contact_normal[tid]
    px = particle_q[particle_idx]
    radius = particle_radius[particle_idx]

    penetration = -(wp.dot(n, px - bx) - radius)
    if penetration <= 0.0:
        return

    rel_v = particle_qd[particle_idx] - soft_contact_body_vel[tid]
    vn = wp.dot(rel_v, n)

    fn_mag = ke * penetration
    if vn < 0.0:
        fn_mag = fn_mag - kd * vn
    if fn_mag < 0.0:
        fn_mag = 0.0

    vt = rel_v - vn * n
    f_tangent = -kd * vt
    f_t_norm = wp.length(f_tangent)
    f_t_max = mu * fn_mag
    if f_t_norm > f_t_max and f_t_norm > 0.0:
        f_tangent = f_tangent * (f_t_max / f_t_norm)

    wp.atomic_add(force_metric, particle_idx, wp.length(fn_mag * n + f_tangent))


@wp.kernel
def accumulate_vessel_vertex_soft_contact_force_metric(
    soft_contact_count: wp.array[wp.int32],
    soft_contact_particle: wp.array[wp.int32],
    soft_contact_shape: wp.array[wp.int32],
    soft_contact_body_pos: wp.array[wp.vec3],
    soft_contact_body_vel: wp.array[wp.vec3],
    soft_contact_normal: wp.array[wp.vec3],
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    vessel_mesh_id: wp.uint64,
    vessel_shape_idx: int,
    query_radius: float,
    ke: float,
    kd: float,
    mu: float,
    force_metric: wp.array[float],
):
    tid = wp.tid()
    if tid >= soft_contact_count[0] or query_radius <= 0.0:
        return
    if soft_contact_shape[tid] != vessel_shape_idx:
        return

    particle_idx = soft_contact_particle[tid]
    if particle_idx < 0:
        return

    bx = soft_contact_body_pos[tid]
    n = soft_contact_normal[tid]
    px = particle_q[particle_idx]
    radius = particle_radius[particle_idx]

    penetration = -(wp.dot(n, px - bx) - radius)
    if penetration <= 0.0:
        return

    rel_v = particle_qd[particle_idx] - soft_contact_body_vel[tid]
    vn = wp.dot(rel_v, n)

    fn_mag = ke * penetration
    if vn < 0.0:
        fn_mag = fn_mag - kd * vn
    if fn_mag < 0.0:
        fn_mag = 0.0

    vt = rel_v - vn * n
    f_tangent = -kd * vt
    f_t_norm = wp.length(f_tangent)
    f_t_max = mu * fn_mag
    if f_t_norm > f_t_max and f_t_norm > 0.0:
        f_tangent = f_tangent * (f_t_max / f_t_norm)

    query = wp.mesh_query_point_no_sign(vessel_mesh_id, bx, query_radius)
    if not query.result:
        return

    force = wp.length(fn_mag * n + f_tangent)
    i0 = wp.mesh_get_index(vessel_mesh_id, query.face * 3 + 0)
    i1 = wp.mesh_get_index(vessel_mesh_id, query.face * 3 + 1)
    i2 = wp.mesh_get_index(vessel_mesh_id, query.face * 3 + 2)

    wp.atomic_add(force_metric, i0, force)
    wp.atomic_add(force_metric, i1, force)
    wp.atomic_add(force_metric, i2, force)


@wp.kernel
def colorize_contact_force_metric(
    force_metric: wp.array[float],
    color_max: float,
    marker_radius: float,
    marker_radii: wp.array[float],
    marker_colors: wp.array[wp.vec3],
):
    particle_idx = wp.tid()
    force = force_metric[particle_idx]
    if force <= 0.0 or color_max <= 0.0:
        marker_radii[particle_idx] = 0.0
        marker_colors[particle_idx] = wp.vec3(0.0, 0.0, 0.0)
        return

    t = wp.min(force / color_max, 1.0)
    if t < 0.5:
        u = 2.0 * t
        marker_colors[particle_idx] = wp.vec3(u, 0.25 + 0.75 * u, 1.0 - u)
    else:
        u = 2.0 * (t - 0.5)
        marker_colors[particle_idx] = wp.vec3(1.0, 1.0 - u, 0.0)
    marker_radii[particle_idx] = marker_radius


@wp.kernel
def colorize_vessel_contact_force_metric(
    force_metric: wp.array[float],
    color_max: float,
    color_multiplier: float,
    base_color: wp.vec4,
    vertex_colors: wp.array[wp.vec4],
):
    vertex_idx = wp.tid()
    force = force_metric[vertex_idx]
    color = base_color

    if force > 0.0 and color_max > 0.0 and color_multiplier > 0.0:
        t = wp.min(force * color_multiplier / color_max, 1.0)
        heat = wp.vec3(0.0, 0.0, 0.0)
        if t < 0.5:
            u = 2.0 * t
            heat = wp.vec3(u, 0.25 + 0.75 * u, 1.0 - u)
        else:
            u = 2.0 * (t - 0.5)
            heat = wp.vec3(1.0, 1.0 - u, 0.0)

        color = wp.vec4(heat.x, heat.y, heat.z, base_color.w)

    vertex_colors[vertex_idx] = color


@wp.kernel
def smooth_vertex_colors_laplacian(
    source_colors: wp.array[wp.vec4],
    neighbor_offsets: wp.array[wp.int32],
    neighbor_indices: wp.array[wp.int32],
    strength: float,
    smoothed_colors: wp.array[wp.vec4],
):
    vertex_idx = wp.tid()
    start = neighbor_offsets[vertex_idx]
    end = neighbor_offsets[vertex_idx + 1]
    neighbor_count = end - start
    source = source_colors[vertex_idx]

    if neighbor_count <= 0 or strength <= 0.0:
        smoothed_colors[vertex_idx] = source
        return

    color_sum = wp.vec4(0.0, 0.0, 0.0, 0.0)
    cursor = start
    while cursor < end:
        color_sum = color_sum + source_colors[neighbor_indices[cursor]]
        cursor = cursor + 1

    t = wp.min(wp.max(strength, 0.0), 1.0)
    average = color_sum / float(neighbor_count)
    smoothed_colors[vertex_idx] = (1.0 - t) * source + t * average


@wp.kernel
def apply_particle_transform_delta(
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    particle_start: int,
    X_delta: wp.transform,
):
    particle_idx = particle_start + wp.tid()
    particle_q[particle_idx] = wp.transform_point(X_delta, particle_q[particle_idx])
    particle_qd[particle_idx] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def apply_particle_rest_shape(
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    particle_rest_q: wp.array[wp.vec3],
    particle_start: int,
    X_ws: wp.transform,
    scale: wp.vec3,
):
    particle_idx = particle_start + wp.tid()
    p_scaled = wp.cw_mul(particle_rest_q[wp.tid()], scale)
    particle_q[particle_idx] = wp.transform_point(X_ws, p_scaled)
    particle_qd[particle_idx] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def update_device_clone_targets(
    rest_local_q: wp.array[wp.vec3],
    target_q: wp.array[wp.vec3],
    prev_target_q: wp.array[wp.vec3],
    X_ws: wp.transform,
    scale: wp.vec3,
    reset_previous: int,
):
    clone_idx = wp.tid()
    target = wp.transform_point(X_ws, wp.cw_mul(rest_local_q[clone_idx], scale))
    target_q[clone_idx] = target
    if reset_previous != 0:
        prev_target_q[clone_idx] = target


@wp.kernel
def update_device_clone_target_kinematics(
    rest_local_q: wp.array[wp.vec3],
    target_q: wp.array[wp.vec3],
    target_qd: wp.array[wp.vec3],
    prev_target_q: wp.array[wp.vec3],
    X_ws_buffer: wp.array[wp.transform],
    scale_buffer: wp.array[wp.vec3],
    dt: float,
    reset_previous: int,
):
    clone_idx = wp.tid()
    X_ws = X_ws_buffer[0]
    scale = scale_buffer[0]
    target = wp.transform_point(X_ws, wp.cw_mul(rest_local_q[clone_idx], scale))
    prev_target = prev_target_q[clone_idx]

    target_q[clone_idx] = target
    if reset_previous != 0 or dt <= 0.0:
        target_qd[clone_idx] = wp.vec3(0.0, 0.0, 0.0)
    else:
        target_qd[clone_idx] = (target - prev_target) / dt
    prev_target_q[clone_idx] = target


@wp.kernel
def apply_device_clone_target_forces(
    particle_indices: wp.array[wp.int32],
    target_q: wp.array[wp.vec3],
    target_qd: wp.array[wp.vec3],
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    particle_f: wp.array[wp.vec3],
    stiffness: float,
    damping: float,
    max_force: float,
):
    clone_idx = wp.tid()
    particle_idx = particle_indices[clone_idx]
    force = stiffness * (target_q[clone_idx] - particle_q[particle_idx])
    force = force + damping * (target_qd[clone_idx] - particle_qd[particle_idx])

    force_len = wp.length(force)
    if max_force <= 0.0:
        force = wp.vec3(0.0, 0.0, 0.0)
    elif force_len > max_force and force_len > 0.0:
        force = force * (max_force / force_len)

    particle_f[particle_idx] = particle_f[particle_idx] + force


class StaticTriMeshParticleProjector:
    """Projects particles away from a static triangle mesh using a Warp mesh BVH."""

    def __init__(self, mesh: newton.Mesh, pos: wp.vec3, scale: float, device):
        offset = np.array([float(pos[0]), float(pos[1]), float(pos[2])], dtype=np.float32)
        world_vertices = mesh.vertices * np.float32(scale) + offset

        self.points = wp.array(world_vertices, dtype=wp.vec3, device=device)
        self.indices = wp.array(mesh.indices, dtype=wp.int32, device=device)
        if mesh.normals is not None and len(mesh.normals) == len(mesh.vertices):
            normals = np.asarray(mesh.normals, dtype=np.float32)
        else:
            normals = compute_triangle_vertex_normals(world_vertices, mesh.indices)
        self.normals = wp.array(normals, dtype=wp.vec3, device=device)
        self.mesh = wp.Mesh(points=self.points, indices=self.indices)

    def project(
        self,
        model: newton.Model,
        state: newton.State,
        contact_radius: float,
        relaxation: float,
        dt: float,
        particle_force_metric: wp.array[float],
        vertex_force_metric: wp.array[float],
    ):
        wp.launch(
            kernel=project_particles_vs_static_tri_mesh,
            dim=model.particle_count,
            inputs=[
                state.particle_q,
                state.particle_qd,
                model.particle_inv_mass,
                model.particle_flags,
                self.mesh.id,
                contact_radius,
                relaxation,
                dt,
            ],
            outputs=[particle_force_metric, vertex_force_metric],
            device=model.device,
        )


class ParticleTransformGizmo:
    """Moves a particle range by the delta of a mutable viewer gizmo transform."""

    def __init__(
        self,
        particle_start: int,
        particle_count: int,
        initial_transform: wp.transform,
        initial_scale: np.ndarray,
        rest_local_q: np.ndarray,
        device,
    ):
        self.particle_start = particle_start
        self.particle_count = particle_count
        self.transform = wp.transform(*initial_transform)
        self._last_transform = wp.transform(*initial_transform)
        self.scale = np.asarray(initial_scale, dtype=np.float32)
        self.rest_local_q = wp.array(rest_local_q, dtype=wp.vec3, device=device)
        self._was_paused = False

    @staticmethod
    def _transform_array(transform: wp.transform) -> np.ndarray:
        return np.asarray(transform, dtype=np.float32)

    def sync_to_particles(self, state: newton.State):
        particle_q = state.particle_q.numpy()
        particle_slice = particle_q[self.particle_start : self.particle_start + self.particle_count]
        centroid = np.mean(particle_slice, axis=0)
        self.transform[:] = wp.transform(
            wp.vec3(float(centroid[0]), float(centroid[1]), float(centroid[2])),
            wp.transform_get_rotation(self.transform),
        )
        self._last_transform[:] = self.transform

    def set_scale(self, scale: np.ndarray):
        self.scale = np.asarray(scale, dtype=np.float32)

    def apply_delta_if_changed(self, model: newton.Model, states: tuple[newton.State, ...]):
        current = self._transform_array(self.transform)
        previous = self._transform_array(self._last_transform)
        transform_changed = not np.allclose(current, previous, rtol=0.0, atol=1.0e-6)

        if transform_changed:
            X_delta = wp.transform_multiply(self.transform, wp.transform_inverse(self._last_transform))
            for state in states:
                wp.launch(
                    kernel=apply_particle_transform_delta,
                    dim=self.particle_count,
                    inputs=[
                        state.particle_q,
                        state.particle_qd,
                        self.particle_start,
                        X_delta,
                    ],
                    device=model.device,
                )
            self._last_transform[:] = self.transform

    def apply_rest_shape(self, model: newton.Model, states: tuple[newton.State, ...], scale: np.ndarray):
        scale_vec = wp.vec3(float(scale[0]), float(scale[1]), float(scale[2]))
        for state in states:
            wp.launch(
                kernel=apply_particle_rest_shape,
                dim=self.particle_count,
                inputs=[
                    state.particle_q,
                    state.particle_qd,
                    self.rest_local_q,
                    self.particle_start,
                    self.transform,
                    scale_vec,
                ],
                device=model.device,
            )


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.solver_type = args.solver
        self.sim_time = 0.0
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 8
        self.sim_dt = self.frame_dt / self.sim_substeps
        self._needs_graph_recapture = False
        self._gravity_key_was_down = False
        self._control_handle_key_was_down = False
        self._reset_offset_key_was_down = False
        self._camera_input_space_key_was_down = False
        self._gravity_enabled = True
        self._reset_device_key_was_down = False
        self._compress_device_key_was_down = False
        self._device_pose_slot_key_was_down = dict.fromkeys(DEVICE_POSE_SLOT_KEYS, False)
        self._device_pose_slots_path = Path(args.device_pose_slots_file).expanduser()
        self._device_pose_slots: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._load_device_pose_slots_from_file()
        self.vessel_contact_radius = min(
            max(float(args.vessel_contact_radius), VESSEL_CONTACT_RADIUS_MIN),
            VESSEL_CONTACT_RADIUS_MAX,
        )
        self.vessel_contact_relaxation = min(
            max(float(args.vessel_contact_relaxation), VESSEL_CONTACT_RELAXATION_MIN),
            VESSEL_CONTACT_RELAXATION_MAX,
        )
        self.vessel_contact_iterations = min(
            max(int(args.vessel_contact_iterations), VESSEL_CONTACT_ITERATIONS_MIN),
            VESSEL_CONTACT_ITERATIONS_MAX,
        )
        self.device_compression = self._clamp_device_compression(args.device_compression)

        if self.solver_type not in {"xpbd", "vbd", "fem"}:
            raise ValueError("The VSD device example only supports the XPBD, VBD, and FEM solvers.")

        self.device_gizmo_space = self._validate_device_gizmo_space(getattr(args, "device_gizmo_space", "world"))
        requested_interactive_placement = bool(getattr(args, "interactive_placement", False))
        self.interactive_placement_enabled = requested_interactive_placement and self.solver_type == "fem"
        if requested_interactive_placement and self.solver_type != "fem":
            print("Interactive placement is available only with --solver fem; ignoring --interactive-placement.")
        self.placement_drive_axes = self._parse_placement_drive_axes(
            getattr(args, "placement_drive_axes", ("x", "y", "z"))
        )
        self.placement_axis_band_radius = min(
            max(float(getattr(args, "placement_axis_band_radius", 0.05)), PLACEMENT_AXIS_BAND_RADIUS_MIN),
            PLACEMENT_AXIS_BAND_RADIUS_MAX,
        )
        self.placement_clone_stiffness = min(
            max(float(getattr(args, "placement_clone_stiffness", 1000.0)), PLACEMENT_CLONE_STIFFNESS_MIN),
            PLACEMENT_CLONE_STIFFNESS_MAX,
        )
        self.placement_clone_damping = min(
            max(float(getattr(args, "placement_clone_damping", 0.0)), PLACEMENT_CLONE_DAMPING_MIN),
            PLACEMENT_CLONE_DAMPING_MAX,
        )
        self.placement_max_clone_force = min(
            max(float(getattr(args, "placement_max_clone_force", 1000.0)), PLACEMENT_MAX_CLONE_FORCE_MIN),
            PLACEMENT_MAX_CLONE_FORCE_MAX,
        )
        self.show_placement_clone_targets = True
        self.show_placement_driving_particles = True

        self.minimou_controller: object | None = None
        self.minimou_device_index = self._validate_minimou_device_index(args.minimou_device_index)
        self.minimou_enabled = False
        self.minimou_power_enabled = bool(args.minimou_power)
        self.minimou_control_enabled = bool(args.minimou_control)
        self.minimou_workspace_scale = self._clamp_minimou_workspace_scale(args.minimou_workspace_scale)
        self.minimou_workspace_follows_camera = bool(args.minimou_workspace_follows_camera)
        self.minimou_workspace_pos_offset = self._vec3_arg(
            args.minimou_workspace_pos_offset,
            "MiniMou workspace position offset",
        )
        self.minimou_workspace_rot_offset_deg = self._vec3_arg(
            args.minimou_workspace_rot_offset,
            "MiniMou workspace rotation offset",
        )
        self.minimou_last_sample: dict | None = None
        self.minimou_status = "Disconnected"
        self.minimou_error = ""
        self.minimou_takeover_requested = True
        self.minimou_anchor_device_transform: wp.transform | None = None
        self.minimou_anchor_handle_transform: wp.transform | None = None
        self.show_minimou_device_frame = True
        self._minimou_enable_on_start = bool(args.minimou_enabled or args.minimou_power or args.minimou_control)

        self.fem_soft_vessel_contact_enabled = bool(args.fem_soft_vessel_contact)
        self.fem_soft_contact_stiffness_multiplier = min(
            max(float(args.fem_soft_contact_stiffness_multiplier), FEM_SOFT_CONTACT_STIFFNESS_MULTIPLIER_MIN),
            FEM_SOFT_CONTACT_STIFFNESS_MULTIPLIER_MAX,
        )
        self.fem_soft_contact_stiffness_exponent = min(
            max(int(args.fem_soft_contact_stiffness_exponent), FEM_SOFT_CONTACT_STIFFNESS_EXPONENT_MIN),
            FEM_SOFT_CONTACT_STIFFNESS_EXPONENT_MAX,
        )
        self.fem_soft_contact_kd = min(
            max(float(args.fem_soft_contact_damping), FEM_SOFT_CONTACT_DAMPING_MIN),
            FEM_SOFT_CONTACT_DAMPING_MAX,
        )
        self.fem_soft_contact_mu = min(
            max(float(args.fem_soft_contact_friction), FEM_SOFT_CONTACT_FRICTION_MIN),
            FEM_SOFT_CONTACT_FRICTION_MAX,
        )
        self.fem_soft_contact_margin = min(
            max(float(args.fem_soft_contact_margin), FEM_SOFT_CONTACT_MARGIN_MIN),
            FEM_SOFT_CONTACT_MARGIN_MAX,
        )
        self.fem_global_damping = min(
            max(float(args.fem_global_damping), FEM_GLOBAL_DAMPING_MIN),
            FEM_GLOBAL_DAMPING_MAX,
        )
        self.device_particle_radius = min(
            max(float(args.device_particle_radius), DEVICE_PARTICLE_RADIUS_MIN),
            DEVICE_PARTICLE_RADIUS_MAX,
        )
        self.show_vessel_contact_markers = bool(args.show_vessel_contact_markers)
        self.vessel_contact_force_color_max = min(
            max(float(args.vessel_contact_force_color_max), VESSEL_CONTACT_FORCE_COLOR_MAX_MIN),
            VESSEL_CONTACT_FORCE_COLOR_MAX_MAX,
        )
        self.vessel_contact_force_color_multiplier = min(
            max(float(args.vessel_contact_force_color_multiplier), VESSEL_CONTACT_FORCE_COLOR_MULTIPLIER_MIN),
            VESSEL_CONTACT_FORCE_COLOR_MULTIPLIER_MAX,
        )
        self.vessel_alpha = min(max(float(args.vessel_alpha), VESSEL_ALPHA_MIN), VESSEL_ALPHA_MAX)
        self.vessel_transparent_surface = bool(args.vessel_transparent_surface)
        self.vsd_contact_force_color_multiplier = min(
            max(float(args.vsd_contact_force_color_multiplier), VSD_CONTACT_FORCE_COLOR_MULTIPLIER_MIN),
            VSD_CONTACT_FORCE_COLOR_MULTIPLIER_MAX,
        )
        self.vsd_alpha = min(max(float(args.vsd_alpha), VSD_ALPHA_MIN), VSD_ALPHA_MAX)
        self.vsd_transparent_surface = bool(args.vsd_transparent_surface)
        self.contact_color_smoothing_passes = min(
            max(int(args.contact_color_smoothing_passes), CONTACT_COLOR_SMOOTHING_PASSES_MIN),
            CONTACT_COLOR_SMOOTHING_PASSES_MAX,
        )
        self.contact_color_smoothing_strength = min(
            max(float(args.contact_color_smoothing_strength), CONTACT_COLOR_SMOOTHING_STRENGTH_MIN),
            CONTACT_COLOR_SMOOTHING_STRENGTH_MAX,
        )
        self._soft_contact_count_cached = 0

        self.iterations = 1
        self.fp64 = bool(getattr(args, "fp64", False))

        if self.solver_type == "xpbd":
            self.tet_stiffness_multiplier = 1.0
            self.tet_stiffness_exponent = 1
            self.tet_stiffness_exponent_min = XPBD_TET_STIFFNESS_EXPONENT_MIN
            self.tet_stiffness_exponent_max = XPBD_TET_STIFFNESS_EXPONENT_MAX
        elif self.solver_type == "fem":
            self.tet_stiffness_multiplier = 1.0
            self.tet_stiffness_exponent = 5
            self.tet_stiffness_exponent_min = FEM_TET_STIFFNESS_EXPONENT_MIN
            self.tet_stiffness_exponent_max = FEM_TET_STIFFNESS_EXPONENT_MAX
        else:
            self.tet_stiffness_multiplier = 1.0
            self.tet_stiffness_exponent = 5
            self.tet_stiffness_exponent_min = VBD_TET_STIFFNESS_EXPONENT_MIN
            self.tet_stiffness_exponent_max = VBD_TET_STIFFNESS_EXPONENT_MAX
        self.tet_stiffness = self._compute_tet_stiffness()

        builder = newton.ModelBuilder()
        builder.add_ground_plane()

        tet_mesh = load_vtk_unstructured_tet_mesh(MESH_PATH)
        vessel_mesh = load_vtk_triangle_mesh(VESSEL_MESH_PATH)
        self.mesh_scale = 0.05 * float(args.scale)
        self.anatomy_scale = 0.05 * float(args.anatomy_scale)
        ox, oy, oz = (float(v) for v in args.offset)
        self.mesh_pos = wp.vec3(0.0 + ox, 0.0 + oy, 0.45 + oz)

        self.vessel_shape_idx = builder.add_shape_mesh(
            body=-1,
            xform=wp.transform(self.mesh_pos, wp.quat_identity()),
            mesh=vessel_mesh,
            scale=(self.anatomy_scale, self.anatomy_scale, self.anatomy_scale),
            cfg=newton.ModelBuilder.ShapeConfig(
                density=0.0,
                has_shape_collision=False,
                has_particle_collision=self._use_fem_soft_vessel_contact(),
                ke=self._compute_fem_soft_contact_stiffness(),
                kd=self.fem_soft_contact_kd,
                mu=self.fem_soft_contact_mu,
            ),
            color=VESSEL_BASE_COLOR,
            label="rvot_alterra_vessel",
        )

        self.device_particle_start = builder.particle_count
        builder.add_soft_mesh(
            pos=self.mesh_pos,
            rot=wp.quat_identity(),
            scale=self.mesh_scale,
            vel=wp.vec3(0.0, 0.0, 0.0),
            mesh=tet_mesh,
            density=1.0e3,
            k_mu=self.tet_stiffness,
            k_lambda=self.tet_stiffness,
            k_damp=1.0e-4,
            particle_radius=self.device_particle_radius,
        )
        self.device_particle_count = builder.particle_count - self.device_particle_start

        # Color the mesh for VBD solver
        builder.color()

        self.model = builder.finalize()
        self._simulation_particle_flags = self.model.particle_flags
        self._set_vessel_model_shape_visible(False)
        if self.solver_type == "fem":
            self.model.soft_contact_ke = self._compute_fem_soft_contact_stiffness()
            self.model.soft_contact_kd = self.fem_soft_contact_kd
            self.model.soft_contact_mu = self.fem_soft_contact_mu
        else:
            self.model.soft_contact_ke = 1.0e2
            self.model.soft_contact_kd = 0
            self.model.soft_contact_mu = 1.0
        self._gravity_default = self.model.gravity.numpy().copy()
        device_particle_q = self.model.particle_q.numpy()[
            self.device_particle_start : self.device_particle_start + self.device_particle_count
        ]
        device_rest_center = np.mean(device_particle_q, axis=0)
        device_rest_local_q = device_particle_q - device_rest_center
        self.device_rest_local_q_np = np.asarray(device_rest_local_q, dtype=np.float32)

        self.vessel_particle_projector = StaticTriMeshParticleProjector(
            mesh=vessel_mesh,
            pos=self.mesh_pos,
            scale=self.anatomy_scale,
            device=self.model.device,
        )
        self.device_transform_gizmo = ParticleTransformGizmo(
            self.device_particle_start,
            self.device_particle_count,
            wp.transform(self.mesh_pos, wp.quat_identity()),
            self.device_compression,
            device_rest_local_q,
            self.model.device,
        )

        self.solver = self._create_solver()

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase="explicit",
            soft_contact_margin=self._collision_soft_contact_margin(),
        )
        self.contacts = self.collision_pipeline.contacts()
        self.vessel_contact_force_metric = wp.zeros(self.model.particle_count, dtype=float, device=self.model.device)
        self.vessel_contact_marker_radii = wp.zeros(self.model.particle_count, dtype=float, device=self.model.device)
        self.vessel_contact_marker_colors = wp.zeros(self.model.particle_count, dtype=wp.vec3, device=self.model.device)
        self.vessel_vertex_force_metric = wp.zeros(
            len(self.vessel_particle_projector.points), dtype=float, device=self.model.device
        )
        self.vessel_vertex_colors = wp.zeros(
            len(self.vessel_particle_projector.points), dtype=wp.vec4, device=self.model.device
        )
        self.vessel_vertex_colors_scratch = wp.zeros(
            len(self.vessel_particle_projector.points), dtype=wp.vec4, device=self.model.device
        )
        self.vessel_surface_indices_np = np.asarray(vessel_mesh.indices, dtype=np.int32)
        self.vessel_color_neighbor_offsets: wp.array[wp.int32] | None = None
        self.vessel_color_neighbor_indices: wp.array[wp.int32] | None = None
        device_surface_indices = tet_mesh.surface_tri_indices.astype(np.int32) + self.device_particle_start
        self.vsd_surface_indices_np = device_surface_indices
        self.vsd_surface_indices = wp.array(device_surface_indices, dtype=wp.int32, device=self.model.device)
        self.vsd_surface_vertex_colors = wp.zeros(
            self.model.particle_count, dtype=wp.vec4, device=self.model.device
        )
        self.vsd_surface_vertex_colors_scratch = wp.zeros(
            self.model.particle_count, dtype=wp.vec4, device=self.model.device
        )
        self.vsd_color_neighbor_offsets: wp.array[wp.int32] | None = None
        self.vsd_color_neighbor_indices: wp.array[wp.int32] | None = None
        self.placement_particle_indices_np = np.empty(0, dtype=np.int32)
        self.placement_particle_indices: wp.array[wp.int32] | None = None
        self.placement_rest_local_q: wp.array[wp.vec3] | None = None
        self.placement_target_q: wp.array[wp.vec3] | None = None
        self.placement_target_qd: wp.array[wp.vec3] | None = None
        self.placement_prev_target_q: wp.array[wp.vec3] | None = None
        self.placement_render_particle_flags: wp.array[wp.int32] | None = None
        self.placement_clone_count = 0
        self.placement_control_transform = wp.array(
            [self.device_transform_gizmo.transform],
            dtype=wp.transform,
            device=self.model.device,
        )
        self.placement_control_scale = wp.array(
            [self._device_compression_vec3()],
            dtype=wp.vec3,
            device=self.model.device,
        )
        self._last_placement_control_transform = np.empty(0, dtype=np.float32)
        self._last_placement_control_scale = np.empty(0, dtype=np.float32)
        self._sync_placement_control_buffers(force=True)
        self._refresh_placement_clone_buffers()
        self._create_world_origin_axes_buffers()

        self.viewer.set_model(self.model)
        self._connect_particle_drag_projection()
        self._register_contact_panel()
        self._register_placement_panel()
        self._register_minimou_panel()
        if self._minimou_enable_on_start:
            self._set_minimou_enabled(True)
        if self.minimou_control_enabled:
            self._request_minimou_takeover()
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(wp.vec3(1.8, -2.0, 1.1), -18.0, 132.0)

        self.capture()

    def __del__(self):
        try:
            self._disconnect_minimou()
        except Exception:
            pass

    def _connect_particle_drag_projection(self):
        if self.solver_type != "fem" or not hasattr(self.solver, "set_particle_drag_constraint"):
            return

        picking = getattr(self.viewer, "picking", None)
        if picking is None or not hasattr(picking, "pick_particle_indices"):
            return

        self.solver.set_particle_drag_constraint(
            picking.pick_particle_indices,
            picking.pick_particle_weights,
            picking.pick_particle_target,
            picking.pick_particle_point,
        )

    def _register_contact_panel(self):
        if hasattr(self.viewer, "register_ui_callback"):
            self.viewer.register_ui_callback(lambda ui, ex=self: ex.contact_gui(ui), position="free")

    def _register_placement_panel(self):
        if hasattr(self.viewer, "register_ui_callback"):
            self.viewer.register_ui_callback(lambda ui, ex=self: ex.placement_gui(ui), position="free")

    def _register_minimou_panel(self):
        if hasattr(self.viewer, "register_ui_callback"):
            self.viewer.register_ui_callback(lambda ui, ex=self: ex.minimou_gui(ui), position="free")

    def _create_world_origin_axes_buffers(self):
        length = float(WORLD_ORIGIN_AXIS_LENGTH)
        half_length = 0.5 * length
        self.world_origin_axis_xforms = wp.array(
            [
                wp.transform(wp.vec3(half_length, 0.0, 0.0), wp.quat_identity()),
                wp.transform(
                    wp.vec3(0.0, half_length, 0.0),
                    wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.5 * math.pi),
                ),
                wp.transform(
                    wp.vec3(0.0, 0.0, half_length),
                    wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -0.5 * math.pi),
                ),
            ],
            dtype=wp.transform,
            device=self.model.device,
        )
        self.world_origin_axis_colors = wp.array(
            [
                wp.vec3(1.0, 0.05, 0.05),
                wp.vec3(0.05, 1.0, 0.05),
                wp.vec3(0.05, 0.25, 1.0),
            ],
            dtype=wp.vec3,
            device=self.model.device,
        )
        self.world_origin_cube_xform = wp.array(
            [wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())],
            dtype=wp.transform,
            device=self.model.device,
        )
        self.world_origin_cube_color = wp.array(
            [wp.vec3(1.0, 1.0, 1.0)],
            dtype=wp.vec3,
            device=self.model.device,
        )

    def _interactive_placement_available(self) -> bool:
        return self.solver_type == "fem"

    def _use_interactive_placement(self) -> bool:
        return self._interactive_placement_available() and self.interactive_placement_enabled

    def _set_interactive_placement_enabled(self, enabled: bool):
        self.interactive_placement_enabled = bool(enabled) and self._interactive_placement_available()
        if self.interactive_placement_enabled and not self.viewer.is_paused():
            self.device_transform_gizmo.sync_to_particles(self.state_0)
            if self.minimou_control_enabled:
                self._request_minimou_takeover()
        self._reset_placement_clone_targets()
        self._sync_placement_control_buffers(force=True)
        self._mark_graph_recapture()

    @staticmethod
    def _validate_minimou_device_index(value) -> int:
        if isinstance(value, bool):
            raise ValueError("MiniMou device index must be a non-negative integer.")
        result = int(value)
        if result < 0:
            raise ValueError("MiniMou device index must be a non-negative integer.")
        return result

    @staticmethod
    def _clamp_minimou_workspace_scale(value) -> float:
        return min(max(float(value), MINIMOU_WORKSPACE_SCALE_MIN), MINIMOU_WORKSPACE_SCALE_MAX)

    @staticmethod
    def _vec3_arg(values, name: str) -> np.ndarray:
        result = np.asarray(values, dtype=np.float32)
        if result.shape != (3,):
            raise ValueError(f"{name} must contain exactly three values.")
        return result

    def _set_minimou_enabled(self, enabled: bool):
        if not enabled:
            self._disconnect_minimou()
            self.minimou_enabled = False
            self.minimou_control_enabled = False
            self.minimou_status = "Disconnected"
            return

        if self.minimou_controller is not None:
            self.minimou_enabled = True
            return

        try:
            from newton.examples.slicer.follou import MiniMouController

            self.minimou_controller = MiniMouController(
                device_index=self.minimou_device_index,
                scale=self.minimou_workspace_scale,
            )
            self.minimou_enabled = True
            self.minimou_error = ""
            self.minimou_status = "Connected"
            if self.minimou_power_enabled:
                self._set_minimou_power_enabled(True)
            if self.minimou_control_enabled:
                self._request_minimou_takeover()
        except Exception as exc:
            self.minimou_controller = None
            self.minimou_enabled = False
            self.minimou_error = str(exc)
            self.minimou_status = "Connection failed"

    def _disconnect_minimou(self):
        controller, self.minimou_controller = self.minimou_controller, None
        self.minimou_anchor_device_transform = None
        self.minimou_anchor_handle_transform = None
        self.minimou_last_sample = None
        if controller is None:
            return

        close = getattr(controller, "close", None)
        if callable(close):
            try:
                close()
            except Exception as exc:
                self.minimou_error = str(exc)

    def _set_minimou_power_enabled(self, enabled: bool):
        self.minimou_power_enabled = bool(enabled)
        if self.minimou_power_enabled and self.minimou_controller is None:
            self._set_minimou_enabled(True)

        if self.minimou_controller is None:
            return

        try:
            self.minimou_controller.set_power(self.minimou_power_enabled)
            self.minimou_error = ""
        except Exception as exc:
            self.minimou_error = str(exc)
            self.minimou_status = "Power command failed"

    def _set_minimou_control_enabled(self, enabled: bool):
        self.minimou_control_enabled = bool(enabled)
        if not self.minimou_control_enabled:
            self.minimou_anchor_device_transform = None
            self.minimou_anchor_handle_transform = None
            return

        if not self.minimou_enabled:
            self._set_minimou_enabled(True)
        if not self.minimou_enabled:
            self.minimou_control_enabled = False
            return

        self._request_minimou_takeover()

    def _toggle_minimou_control_handle(self):
        self._set_minimou_control_enabled(not self.minimou_control_enabled)
        state = "enabled" if self.minimou_control_enabled else "disabled"
        print(f"Control Handle {state}.")

    def _set_minimou_workspace_follows_camera(self, enabled: bool):
        self.minimou_workspace_follows_camera = bool(enabled)
        if self.minimou_control_enabled:
            self._request_minimou_takeover()

    def _toggle_minimou_camera_input_space(self):
        self._set_minimou_workspace_follows_camera(not self.minimou_workspace_follows_camera)
        state = "enabled" if self.minimou_workspace_follows_camera else "disabled"
        print(f"Camera Input Space {state}.")

    def _request_minimou_takeover(self):
        self.minimou_takeover_requested = True
        self.minimou_anchor_device_transform = None
        self.minimou_anchor_handle_transform = None
        if self.minimou_control_enabled:
            self.minimou_status = "Takeover pending"

    def _minimou_control_active(self) -> bool:
        return self.minimou_enabled and self.minimou_control_enabled and self.minimou_controller is not None

    def _poll_minimou(self) -> dict | None:
        if self.minimou_controller is None:
            return None

        try:
            if hasattr(self.minimou_controller, "scale"):
                self.minimou_controller.scale = self.minimou_workspace_scale
            sample = self.minimou_controller.poll()
        except Exception as exc:
            self.minimou_error = str(exc)
            self.minimou_status = "Poll failed"
            return None

        if not sample.get("valid", False):
            self.minimou_status = "Invalid sample"
            return None

        self.minimou_last_sample = sample
        if self.minimou_takeover_requested:
            self.minimou_status = "Takeover pending"
        else:
            self.minimou_status = "Tracking" if self.minimou_control_enabled else "Connected"
        return sample

    def _camera_input_space_transform(self) -> wp.transform | None:
        camera = getattr(self.viewer, "camera", None)
        if camera is None:
            return None

        try:
            position = np.asarray(camera.pos, dtype=np.float32)
            right = np.asarray(camera.get_right(), dtype=np.float32)
            forward = np.asarray(camera.get_front(), dtype=np.float32)
            up = np.asarray(camera.get_up(), dtype=np.float32)
        except (AttributeError, TypeError, ValueError):
            return None

        if position.shape != (3,) or right.shape != (3,) or forward.shape != (3,) or up.shape != (3,):
            return None
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(right + forward + up)):
            return None

        right_norm = float(np.linalg.norm(right))
        forward_norm = float(np.linalg.norm(forward))
        up_norm = float(np.linalg.norm(up))
        if min(right_norm, forward_norm, up_norm) < 1.0e-8:
            return None

        right /= right_norm
        forward /= forward_norm
        up /= up_norm
        rotation = wp.quat_from_matrix(
            wp.matrix_from_cols(
                wp.vec3(float(right[0]), float(right[1]), float(right[2])),
                wp.vec3(float(forward[0]), float(forward[1]), float(forward[2])),
                wp.vec3(float(up[0]), float(up[1]), float(up[2])),
            )
        )
        return wp.transform(
            wp.vec3(float(position[0]), float(position[1]), float(position[2])),
            rotation,
        )

    def _minimou_rotation_offset(self) -> wp.quat:
        roll, pitch, yaw = (math.radians(float(value)) for value in self.minimou_workspace_rot_offset_deg)
        return wp.quat_rpy(roll, pitch, yaw)

    def _minimou_sample_transform(self, sample: dict) -> wp.transform:
        position = np.asarray(sample["position"], dtype=np.float32)
        rotation = np.asarray(sample["rotation"], dtype=np.float32)
        local_position = wp.vec3(
            float(position[0] + self.minimou_workspace_pos_offset[0]),
            float(position[1] + self.minimou_workspace_pos_offset[1]),
            float(position[2] + self.minimou_workspace_pos_offset[2]),
        )
        local_rotation = wp.mul(
            wp.quat(float(rotation[0]), float(rotation[1]), float(rotation[2]), float(rotation[3])),
            self._minimou_rotation_offset(),
        )

        if not self.minimou_workspace_follows_camera:
            return wp.transform(local_position, local_rotation)

        camera_transform = self._camera_input_space_transform()
        if camera_transform is None:
            return wp.transform(local_position, local_rotation)

        _camera_position, camera_rotation = self._transform_components(camera_transform)
        world_position = wp.transform_point(camera_transform, local_position)
        world_rotation = quat_mul_xyzw(quat_inverse_xyzw(camera_rotation), local_rotation)
        return wp.transform(
            world_position,
            wp.quat(
                float(world_rotation[0]),
                float(world_rotation[1]),
                float(world_rotation[2]),
                float(world_rotation[3]),
            ),
        )

    @staticmethod
    def _transform_components(transform: wp.transform) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(transform, dtype=np.float32)
        return values[:3], normalize_quat_xyzw(values[3:])

    def _capture_minimou_takeover(self, device_transform: wp.transform):
        self.minimou_anchor_device_transform = wp.transform(*device_transform)
        self.minimou_anchor_handle_transform = wp.transform(*self.device_transform_gizmo.transform)
        self.minimou_takeover_requested = False
        self.minimou_status = "Tracking"

    def _reset_minimou_handle_offset(self):
        if not self._minimou_control_active():
            return

        sample = self._poll_minimou()
        if sample is None:
            return

        device_transform = self._minimou_sample_transform(sample)
        self.device_transform_gizmo.transform[:] = device_transform
        self.minimou_anchor_device_transform = wp.transform(*device_transform)
        self.minimou_anchor_handle_transform = wp.transform(*device_transform)
        self.minimou_takeover_requested = False
        self.minimou_status = "Offset reset"

    def _update_minimou_controlled_gizmo(self):
        if not self._minimou_control_active():
            return

        sample = self._poll_minimou()
        if sample is None:
            return

        device_transform = self._minimou_sample_transform(sample)
        if (
            self.minimou_takeover_requested
            or self.minimou_anchor_device_transform is None
            or self.minimou_anchor_handle_transform is None
        ):
            self._capture_minimou_takeover(device_transform)
            return

        device_pos, device_rot = self._transform_components(device_transform)
        anchor_device_pos, anchor_device_rot = self._transform_components(self.minimou_anchor_device_transform)
        anchor_handle_pos, anchor_handle_rot = self._transform_components(self.minimou_anchor_handle_transform)

        target_pos = anchor_handle_pos + (device_pos - anchor_device_pos)
        target_rot = quat_mul_xyzw(quat_mul_xyzw(device_rot, quat_inverse_xyzw(anchor_device_rot)), anchor_handle_rot)
        self.device_transform_gizmo.transform[:] = wp.transform(
            wp.vec3(float(target_pos[0]), float(target_pos[1]), float(target_pos[2])),
            wp.quat(float(target_rot[0]), float(target_rot[1]), float(target_rot[2]), float(target_rot[3])),
        )

    def _refresh_placement_clone_buffers(self):
        enabled_axes = tuple(axis for axis, enabled in enumerate(self.placement_drive_axes) if enabled)
        local_q = self.device_rest_local_q_np
        radius_sq = self.placement_axis_band_radius * self.placement_axis_band_radius
        mask = np.zeros(local_q.shape[0], dtype=bool)

        if radius_sq > 0.0:
            for axis in enabled_axes:
                other_axes = tuple(i for i in range(3) if i != axis)
                axis_dist_sq = local_q[:, other_axes[0]] * local_q[:, other_axes[0]]
                axis_dist_sq += local_q[:, other_axes[1]] * local_q[:, other_axes[1]]
                mask |= axis_dist_sq <= radius_sq

        local_indices = np.nonzero(mask)[0].astype(np.int32)
        particle_indices = (local_indices + self.device_particle_start).astype(np.int32)
        rest_local_q = local_q[local_indices].astype(np.float32, copy=False)

        self.placement_clone_count = int(local_indices.shape[0])
        self.placement_particle_indices_np = particle_indices
        self.placement_particle_indices = wp.array(particle_indices, dtype=wp.int32, device=self.model.device)
        self.placement_rest_local_q = wp.array(rest_local_q, dtype=wp.vec3, device=self.model.device)
        self.placement_target_q = wp.zeros(self.placement_clone_count, dtype=wp.vec3, device=self.model.device)
        self.placement_target_qd = wp.zeros(self.placement_clone_count, dtype=wp.vec3, device=self.model.device)
        self.placement_prev_target_q = wp.zeros(self.placement_clone_count, dtype=wp.vec3, device=self.model.device)
        self._refresh_placement_render_particle_flags()
        self._reset_placement_clone_targets()
        self._mark_graph_recapture()

    def _refresh_placement_render_particle_flags(self):
        self.placement_render_particle_flags = None
        if self._simulation_particle_flags is None or self.placement_clone_count == 0:
            return

        flags = self._simulation_particle_flags.numpy().copy()
        flags[self.placement_particle_indices_np] &= ~int(newton.ParticleFlags.ACTIVE)
        self.placement_render_particle_flags = wp.array(flags, dtype=wp.int32, device=self.model.device)

    def _device_compression_vec3(self) -> wp.vec3:
        return wp.vec3(
            float(self.device_compression[0]),
            float(self.device_compression[1]),
            float(self.device_compression[2]),
        )

    def _sync_placement_control_buffers(self, *, force: bool = False):
        current_transform = np.asarray(self.device_transform_gizmo.transform, dtype=np.float32)
        current_scale = np.asarray(self.device_compression, dtype=np.float32)
        transform_changed = (
            force
            or current_transform.shape != self._last_placement_control_transform.shape
            or not np.array_equal(current_transform, self._last_placement_control_transform)
        )
        scale_changed = (
            force
            or current_scale.shape != self._last_placement_control_scale.shape
            or not np.array_equal(current_scale, self._last_placement_control_scale)
        )

        if transform_changed:
            self.placement_control_transform.assign([self.device_transform_gizmo.transform])
            self._last_placement_control_transform = current_transform.copy()
        if scale_changed:
            self.placement_control_scale.assign([self._device_compression_vec3()])
            self._last_placement_control_scale = current_scale.copy()

    def _reset_placement_clone_targets(self):
        self._sync_placement_control_buffers(force=True)
        self._update_placement_clone_target_kinematics(reset_previous=True)

    def _update_placement_clone_targets(self, reset_previous: bool):
        if self.placement_clone_count == 0:
            return

        scale_vec = self._device_compression_vec3()
        wp.launch(
            kernel=update_device_clone_targets,
            dim=self.placement_clone_count,
            inputs=[
                self.placement_rest_local_q,
                self.placement_target_q,
                self.placement_prev_target_q,
                self.device_transform_gizmo.transform,
                scale_vec,
                int(reset_previous),
            ],
            device=self.model.device,
        )

    def _update_placement_clone_target_kinematics(self, reset_previous: bool):
        if self.placement_clone_count == 0:
            return

        wp.launch(
            kernel=update_device_clone_target_kinematics,
            dim=self.placement_clone_count,
            inputs=[
                self.placement_rest_local_q,
                self.placement_target_q,
                self.placement_target_qd,
                self.placement_prev_target_q,
                self.placement_control_transform,
                self.placement_control_scale,
                self.frame_dt,
                int(reset_previous),
            ],
            device=self.model.device,
        )

    def _apply_interactive_placement_forces(self):
        if not self._use_interactive_placement() or self.placement_clone_count == 0:
            return

        wp.launch(
            kernel=apply_device_clone_target_forces,
            dim=self.placement_clone_count,
            inputs=[
                self.placement_particle_indices,
                self.placement_target_q,
                self.placement_target_qd,
                self.state_0.particle_q,
                self.state_0.particle_qd,
                self.state_0.particle_f,
                self.placement_clone_stiffness,
                self.placement_clone_damping,
                self.placement_max_clone_force,
            ],
            device=self.model.device,
        )

    def _use_fem_soft_vessel_contact(self) -> bool:
        return self.solver_type == "fem" and self.fem_soft_vessel_contact_enabled

    def _use_vessel_particle_projection(self) -> bool:
        return not self._use_fem_soft_vessel_contact() and self.vessel_contact_iterations > 0

    def _compute_fem_soft_contact_stiffness(self) -> float:
        return self.fem_soft_contact_stiffness_multiplier * 10.0**self.fem_soft_contact_stiffness_exponent

    def _collision_soft_contact_margin(self) -> float:
        if self._use_fem_soft_vessel_contact():
            return self.fem_soft_contact_margin
        return 0.01

    def _set_vessel_particle_collision_enabled(self, enabled: bool):
        flags = self.model.shape_flags.numpy()
        particle_collision = int(newton.ShapeFlags.COLLIDE_PARTICLES)
        if enabled:
            flags[self.vessel_shape_idx] |= particle_collision
        else:
            flags[self.vessel_shape_idx] &= ~particle_collision
        self.model.shape_flags.assign(flags)

    def _set_vessel_model_shape_visible(self, visible: bool):
        flags = self.model.shape_flags.numpy()
        visible_flag = int(newton.ShapeFlags.VISIBLE)
        if visible:
            flags[self.vessel_shape_idx] |= visible_flag
        else:
            flags[self.vessel_shape_idx] &= ~visible_flag
        self.model.shape_flags.assign(flags)

    def _set_fem_soft_vessel_contact_enabled(self, enabled: bool):
        self.fem_soft_vessel_contact_enabled = bool(enabled)
        self._set_vessel_particle_collision_enabled(self._use_fem_soft_vessel_contact())
        self._apply_fem_soft_contact_settings()
        self._mark_graph_recapture()

        if not self._use_fem_soft_vessel_contact() and self.vessel_contact_iterations == 0:
            self.vessel_contact_iterations = 1
            self._mark_graph_recapture()

    def _apply_fem_soft_contact_settings(self):
        if self.solver_type != "fem":
            return

        contact_ke = self._compute_fem_soft_contact_stiffness()
        self.model.soft_contact_ke = contact_ke
        self.model.soft_contact_kd = self.fem_soft_contact_kd
        self.model.soft_contact_mu = self.fem_soft_contact_mu

        if self.model.shape_material_ke is not None:
            shape_ke = self.model.shape_material_ke.numpy()
            shape_kd = self.model.shape_material_kd.numpy()
            shape_mu = self.model.shape_material_mu.numpy()
            shape_ke[self.vessel_shape_idx] = contact_ke
            shape_kd[self.vessel_shape_idx] = self.fem_soft_contact_kd
            shape_mu[self.vessel_shape_idx] = self.fem_soft_contact_mu
            self.model.shape_material_ke.assign(shape_ke)
            self.model.shape_material_kd.assign(shape_kd)
            self.model.shape_material_mu.assign(shape_mu)

        if hasattr(self.solver, "_contact_ke"):
            self.solver._contact_ke = contact_ke
            self.solver._contact_kd = self.fem_soft_contact_kd
            self.solver._contact_mu = self.fem_soft_contact_mu
        self._mark_graph_recapture()

    def _apply_fem_global_damping(self):
        if self.solver_type != "fem":
            return
        if hasattr(self.solver, "k_damp"):
            self.solver.k_damp = self.fem_global_damping
        else:
            self.solver._k_damp = self.fem_global_damping
        self._mark_graph_recapture()

    def _apply_device_particle_radius(self):
        particle_radius = self.model.particle_radius.numpy()
        particle_radius[self.device_particle_start : self.device_particle_start + self.device_particle_count] = (
            self.device_particle_radius
        )
        self.model.particle_radius.assign(particle_radius)

    def _create_solver(self):
        if self.solver_type == "vbd":
            return newton.solvers.SolverVBD(
                model=self.model,
                iterations=self.iterations,
                particle_enable_self_contact=False,
                particle_enable_tile_solve=False,
            )

        if self.solver_type == "fem":
            return newton.solvers.SolverFEM(
                model=self.model,
                iterations=self.iterations,
                k_damp=self.fem_global_damping,
                plane_contact_projection_iterations=1,
                particle_drag_projection_iterations=1,
                fp64=self.fp64,
            )

        return newton.solvers.SolverXPBD(
            model=self.model,
            iterations=self.iterations,
            soft_body_relaxation=self._xpbd_tet_compliance(),
        )

    def _xpbd_tet_compliance(self) -> float:
        if self.tet_stiffness <= 0.0:
            return XPBD_DISABLED_TET_COMPLIANCE
        return 1.0 / self.tet_stiffness

    def _compute_tet_stiffness(self) -> float:
        return self.tet_stiffness_multiplier * 10.0**self.tet_stiffness_exponent

    def _apply_tet_stiffness(self):
        self.tet_stiffness = self._compute_tet_stiffness()

        if self.model.tet_materials is not None:
            tet_materials = self.model.tet_materials.numpy()
            tet_materials[:, 0] = self.tet_stiffness
            tet_materials[:, 1] = self.tet_stiffness
            self.model.tet_materials.assign(tet_materials)

        if hasattr(self.solver, "soft_body_relaxation"):
            self.solver.soft_body_relaxation = self._xpbd_tet_compliance()
        if hasattr(self.solver, "refresh_material_parameters"):
            self.solver.refresh_material_parameters()

    def _mark_graph_recapture(self):
        self._needs_graph_recapture = True

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None
        self._needs_graph_recapture = False

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # apply forces to the model
            self.viewer.apply_forces(self.state_0)

            self.collision_pipeline.collide(
                self.state_0,
                self.contacts,
                soft_contact_margin=self._collision_soft_contact_margin(),
            )
            self._apply_interactive_placement_forces()
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            if self._use_vessel_particle_projection():
                self._project_vessel_particle_contacts(self.state_1)

            # swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _project_vessel_particle_contacts(self, state: newton.State):
        self.vessel_contact_force_metric.zero_()
        self.vessel_vertex_force_metric.zero_()
        for _ in range(self.vessel_contact_iterations):
            self.vessel_particle_projector.project(
                self.model,
                state,
                self.vessel_contact_radius,
                self.vessel_contact_relaxation,
                self.sim_dt,
                self.vessel_contact_force_metric,
                self.vessel_vertex_force_metric,
            )

    def step(self):
        self._update_minimou_controlled_gizmo()
        if self._interactive_placement_available():
            self._sync_placement_control_buffers()
            if self._use_interactive_placement():
                self._update_placement_clone_target_kinematics(reset_previous=False)

        if self._needs_graph_recapture:
            self.capture()

        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def test_final(self):
        particle_q = self.state_0.particle_q.numpy()
        assert np.all(np.isfinite(particle_q)), "particle positions must remain finite"

        min_pos = np.min(particle_q, axis=0)
        max_pos = np.max(particle_q, axis=0)
        bbox_size = np.linalg.norm(max_pos - min_pos)

        assert bbox_size < 5.0, f"Bounding box exploded: size={bbox_size:.2f}"
        assert min_pos[2] > -0.5, f"Excessive ground penetration: z_min={min_pos[2]:.4f}"

    def render(self):
        self._handle_global_keys()
        self._update_device_transform_gizmo()
        self.viewer.begin_frame(self.sim_time)
        if self._should_show_device_transform_gizmo():
            self.viewer.log_gizmo(
                "vsd_device",
                self.device_transform_gizmo.transform,
                space=self.device_gizmo_space,
            )
        self._render_world_origin_axes()
        self._render_minimou_device_frame()
        self._update_vessel_contact_force_metrics()
        show_triangles = self.viewer.show_triangles
        self.viewer.show_triangles = False
        try:
            self._log_state_with_particle_visibility()
        finally:
            self.viewer.show_triangles = show_triangles
        self.viewer.log_contacts(self.contacts, self.state_0)
        self._render_vessel_contact_mesh()
        self._render_vsd_contact_mesh()
        self._render_vessel_contact_markers()
        self._render_placement_clone_targets()
        self.viewer.end_frame()

    def _log_state_with_particle_visibility(self):
        if not self._should_hide_placement_driving_particles():
            self.viewer.log_state(self.state_0)
            return

        particle_flags = self.model.particle_flags
        self.model.particle_flags = self.placement_render_particle_flags
        try:
            self.viewer.log_state(self.state_0)
        finally:
            self.model.particle_flags = particle_flags

    def _should_hide_placement_driving_particles(self) -> bool:
        return (
            not self.show_placement_driving_particles
            and self._use_interactive_placement()
            and self.placement_clone_count > 0
            and self.placement_render_particle_flags is not None
        )

    def _update_vessel_contact_force_metrics(self):
        if not self._use_fem_soft_vessel_contact():
            if not self._use_vessel_particle_projection():
                self.vessel_contact_force_metric.zero_()
                self.vessel_vertex_force_metric.zero_()
            return

        self.vessel_contact_force_metric.zero_()
        self.vessel_vertex_force_metric.zero_()
        if self.contacts is None or self.contacts.soft_contact_max <= 0:
            return

        wp.launch(
            kernel=accumulate_vessel_soft_contact_force_metric,
            dim=self.contacts.soft_contact_max,
            inputs=[
                self.contacts.soft_contact_count,
                self.contacts.soft_contact_particle,
                self.contacts.soft_contact_shape,
                self.contacts.soft_contact_body_pos,
                self.contacts.soft_contact_body_vel,
                self.contacts.soft_contact_normal,
                self.state_0.particle_q,
                self.state_0.particle_qd,
                self.model.particle_radius,
                self.vessel_shape_idx,
                self._compute_fem_soft_contact_stiffness(),
                self.fem_soft_contact_kd,
                self.fem_soft_contact_mu,
            ],
            outputs=[self.vessel_contact_force_metric],
            device=self.model.device,
        )
        wp.launch(
            kernel=accumulate_vessel_vertex_soft_contact_force_metric,
            dim=self.contacts.soft_contact_max,
            inputs=[
                self.contacts.soft_contact_count,
                self.contacts.soft_contact_particle,
                self.contacts.soft_contact_shape,
                self.contacts.soft_contact_body_pos,
                self.contacts.soft_contact_body_vel,
                self.contacts.soft_contact_normal,
                self.state_0.particle_q,
                self.state_0.particle_qd,
                self.model.particle_radius,
                self.vessel_particle_projector.mesh.id,
                self.vessel_shape_idx,
                self._vessel_contact_color_query_radius(),
                self._compute_fem_soft_contact_stiffness(),
                self.fem_soft_contact_kd,
                self.fem_soft_contact_mu,
            ],
            outputs=[self.vessel_vertex_force_metric],
            device=self.model.device,
        )

    def _vessel_base_color(self) -> wp.vec4:
        return wp.vec4(
            float(VESSEL_BASE_COLOR[0]),
            float(VESSEL_BASE_COLOR[1]),
            float(VESSEL_BASE_COLOR[2]),
            self._vessel_surface_alpha(),
        )

    def _vsd_base_color(self) -> wp.vec4:
        return wp.vec4(
            float(VSD_BASE_COLOR[0]),
            float(VSD_BASE_COLOR[1]),
            float(VSD_BASE_COLOR[2]),
            self._vsd_surface_alpha(),
        )

    def _vessel_surface_alpha(self) -> float:
        return self.vessel_alpha if self.vessel_transparent_surface else 1.0

    def _vsd_surface_alpha(self) -> float:
        return self.vsd_alpha if self.vsd_transparent_surface else 1.0

    def _smooth_surface_vertex_colors(
        self,
        vertex_colors: wp.array[wp.vec4],
        scratch_colors: wp.array[wp.vec4],
        surface_indices: np.ndarray,
        vertex_count: int,
        neighbor_offsets_attr: str,
        neighbor_indices_attr: str,
    ) -> wp.array[wp.vec4]:
        passes = self.contact_color_smoothing_passes
        strength = self.contact_color_smoothing_strength
        if passes <= 0 or strength <= 0.0:
            return vertex_colors

        neighbor_offsets, neighbor_indices = self._get_surface_color_adjacency(
            surface_indices,
            vertex_count,
            neighbor_offsets_attr,
            neighbor_indices_attr,
        )
        source = vertex_colors
        destination = scratch_colors
        for _ in range(passes):
            wp.launch(
                kernel=smooth_vertex_colors_laplacian,
                dim=len(vertex_colors),
                inputs=[source, neighbor_offsets, neighbor_indices, strength],
                outputs=[destination],
                device=self.model.device,
            )
            source, destination = destination, source

        return source

    def _get_surface_color_adjacency(
        self,
        surface_indices: np.ndarray,
        vertex_count: int,
        neighbor_offsets_attr: str,
        neighbor_indices_attr: str,
    ) -> tuple[wp.array[wp.int32], wp.array[wp.int32]]:
        neighbor_offsets = getattr(self, neighbor_offsets_attr)
        neighbor_indices = getattr(self, neighbor_indices_attr)
        if neighbor_offsets is None or neighbor_indices is None:
            offsets_np, indices_np = build_triangle_vertex_adjacency(surface_indices, vertex_count)
            neighbor_offsets = wp.array(offsets_np, dtype=wp.int32, device=self.model.device)
            neighbor_indices = wp.array(indices_np, dtype=wp.int32, device=self.model.device)
            setattr(self, neighbor_offsets_attr, neighbor_offsets)
            setattr(self, neighbor_indices_attr, neighbor_indices)

        return neighbor_offsets, neighbor_indices

    def _vessel_contact_color_query_radius(self) -> float:
        return max(self.fem_soft_contact_margin, self.device_particle_radius, 1.0e-4)

    def _render_vsd_contact_mesh(self):
        mesh_name = "/vsd/device_contact_mesh"

        wp.launch(
            kernel=colorize_vessel_contact_force_metric,
            dim=self.model.particle_count,
            inputs=[
                self.vessel_contact_force_metric,
                self.vessel_contact_force_color_max,
                self.vsd_contact_force_color_multiplier,
                self._vsd_base_color(),
            ],
            outputs=[self.vsd_surface_vertex_colors],
            device=self.model.device,
        )
        vertex_colors = self._smooth_surface_vertex_colors(
            self.vsd_surface_vertex_colors,
            self.vsd_surface_vertex_colors_scratch,
            self.vsd_surface_indices_np,
            self.model.particle_count,
            "vsd_color_neighbor_offsets",
            "vsd_color_neighbor_indices",
        )

        self.viewer.log_mesh(
            mesh_name,
            self.state_0.particle_q,
            self.vsd_surface_indices,
            hidden=False,
            backface_culling=True,
            color=(1.0, 1.0, 1.0, 1.0),
            roughness=0.7,
            vertex_colors=vertex_colors,
            transparent=self.vsd_transparent_surface and self.vsd_alpha < 0.999,
        )

    def _render_vessel_contact_mesh(self):
        mesh_name = "/vsd/vessel_contact_mesh"

        wp.launch(
            kernel=colorize_vessel_contact_force_metric,
            dim=len(self.vessel_vertex_colors),
            inputs=[
                self.vessel_vertex_force_metric,
                self.vessel_contact_force_color_max,
                self.vessel_contact_force_color_multiplier,
                self._vessel_base_color(),
            ],
            outputs=[self.vessel_vertex_colors],
            device=self.model.device,
        )
        vertex_colors = self._smooth_surface_vertex_colors(
            self.vessel_vertex_colors,
            self.vessel_vertex_colors_scratch,
            self.vessel_surface_indices_np,
            len(self.vessel_particle_projector.points),
            "vessel_color_neighbor_offsets",
            "vessel_color_neighbor_indices",
        )

        self.viewer.log_mesh(
            mesh_name,
            self.vessel_particle_projector.points,
            self.vessel_particle_projector.indices,
            normals=self.vessel_particle_projector.normals,
            hidden=False,
            backface_culling=True,
            color=(1.0, 1.0, 1.0, 1.0),
            roughness=0.8,
            vertex_colors=vertex_colors,
            transparent=self.vessel_transparent_surface and self.vessel_alpha < 0.999,
        )

    def _render_vessel_contact_markers(self):
        marker_name = "/vsd/vessel_contact_markers"
        if not self.show_vessel_contact_markers or not (
            self._use_fem_soft_vessel_contact() or self._use_vessel_particle_projection()
        ):
            self.viewer.log_points(marker_name, points=None, hidden=True)
            return
        if self._use_fem_soft_vessel_contact() and (self.contacts is None or self.contacts.soft_contact_max <= 0):
            self.viewer.log_points(marker_name, points=None, hidden=True)
            return

        wp.launch(
            kernel=colorize_contact_force_metric,
            dim=self.model.particle_count,
            inputs=[
                self.vessel_contact_force_metric,
                self.vessel_contact_force_color_max,
                1.35 * self.device_particle_radius,
            ],
            outputs=[self.vessel_contact_marker_radii, self.vessel_contact_marker_colors],
            device=self.model.device,
        )
        self.viewer.log_points(
            marker_name,
            points=self.state_0.particle_q,
            radii=self.vessel_contact_marker_radii,
            colors=self.vessel_contact_marker_colors,
            hidden=False,
        )

    def _render_placement_clone_targets(self):
        marker_name = "/vsd/placement_clone_targets"
        if (
            not self.show_placement_clone_targets
            or not self._use_interactive_placement()
            or self.placement_clone_count == 0
        ):
            self.viewer.log_points(marker_name, points=None, hidden=True)
            return

        self._update_placement_clone_targets(reset_previous=False)
        self.viewer.log_points(
            marker_name,
            points=self.placement_target_q,
            radii=max(0.35 * self.device_particle_radius, 0.003),
            colors=(0.0, 0.85, 1.0),
            hidden=False,
        )

    def _render_world_origin_axes(self):
        if not hasattr(self.viewer, "log_shapes"):
            return

        self.viewer.log_shapes(
            "/vsd/world_origin_axes",
            newton.GeoType.BOX,
            (
                0.5 * WORLD_ORIGIN_AXIS_LENGTH,
                WORLD_ORIGIN_AXIS_THICKNESS,
                WORLD_ORIGIN_AXIS_THICKNESS,
            ),
            self.world_origin_axis_xforms,
            self.world_origin_axis_colors,
        )
        self.viewer.log_shapes(
            "/vsd/world_origin_cube",
            newton.GeoType.BOX,
            WORLD_ORIGIN_CUBE_HALF_EXTENT,
            self.world_origin_cube_xform,
            self.world_origin_cube_color,
        )

    def _render_minimou_device_frame(self):
        frame_name = "/vsd/minimou_device_frame"
        if not hasattr(self.viewer, "log_arrows"):
            return

        if not self.show_minimou_device_frame or not self.minimou_enabled or self.minimou_controller is None:
            self.viewer.log_arrows(frame_name, None, None, None)
            return

        if not self._minimou_control_active():
            self._poll_minimou()

        if self.minimou_last_sample is None:
            self.viewer.log_arrows(frame_name, None, None, None)
            return

        device_transform = self._minimou_sample_transform(self.minimou_last_sample)
        position, rotation = self._transform_components(device_transform)
        axis_length = float(MINIMOU_DEVICE_FRAME_AXIS_LENGTH)
        axes = np.asarray(
            (
                (axis_length, 0.0, 0.0),
                (0.0, axis_length, 0.0),
                (0.0, 0.0, axis_length),
            ),
            dtype=np.float32,
        )
        starts_np = np.repeat(position[None, :], 3, axis=0)
        ends_np = starts_np + np.asarray([quat_rotate_xyzw(rotation, axis) for axis in axes], dtype=np.float32)
        colors_np = np.asarray(
            (
                (1.0, 0.1, 0.1),
                (0.1, 1.0, 0.1),
                (0.1, 0.35, 1.0),
            ),
            dtype=np.float32,
        )
        starts = wp.array(starts_np, dtype=wp.vec3, device=self.model.device)
        ends = wp.array(ends_np, dtype=wp.vec3, device=self.model.device)
        colors = wp.array(colors_np, dtype=wp.vec3, device=self.model.device)
        self.viewer.log_arrows(frame_name, starts, ends, colors)

    def _handle_global_keys(self):
        gravity_down = bool(self.viewer.is_key_down("g"))
        if gravity_down and not self._gravity_key_was_down:
            self._toggle_gravity()
        self._gravity_key_was_down = gravity_down

        control_handle_down = bool(self.viewer.is_key_down("v"))
        if control_handle_down and not self._control_handle_key_was_down:
            self._toggle_minimou_control_handle()
        self._control_handle_key_was_down = control_handle_down

        reset_offset_down = bool(self.viewer.is_key_down("o"))
        if reset_offset_down and not self._reset_offset_key_was_down:
            self._reset_minimou_handle_offset()
        self._reset_offset_key_was_down = reset_offset_down

        camera_input_space_down = bool(self.viewer.is_key_down("x"))
        if camera_input_space_down and not self._camera_input_space_key_was_down:
            self._toggle_minimou_camera_input_space()
        self._camera_input_space_key_was_down = camera_input_space_down

    def _toggle_gravity(self):
        self._gravity_enabled = not self._gravity_enabled
        if self._gravity_enabled:
            self.model.set_gravity(self._gravity_default)
        else:
            self.model.set_gravity(np.zeros_like(self._gravity_default))

        self.solver.notify_model_changed(newton.solvers.SolverNotifyFlags.MODEL_PROPERTIES)
        state = "enabled" if self._gravity_enabled else "disabled"
        print(f"Gravity {state}.")

    def _should_show_device_transform_gizmo(self) -> bool:
        return hasattr(self.viewer, "log_gizmo") and (self.viewer.is_paused() or self._use_interactive_placement())

    def _update_device_transform_gizmo(self):
        paused = self.viewer.is_paused()
        if not paused:
            self.device_transform_gizmo._was_paused = False
            self._reset_device_key_was_down = False
            self._compress_device_key_was_down = False
            for key in DEVICE_POSE_SLOT_KEYS:
                self._device_pose_slot_key_was_down[key] = False
            return

        if not self.device_transform_gizmo._was_paused:
            self.device_transform_gizmo.sync_to_particles(self.state_0)
            if self.minimou_control_enabled:
                self._request_minimou_takeover()

        self._update_minimou_controlled_gizmo()
        self.device_transform_gizmo.apply_delta_if_changed(self.model, (self.state_0, self.state_1))
        self._handle_pause_device_keys()
        if self._use_interactive_placement():
            self._reset_placement_clone_targets()
        self.device_transform_gizmo._was_paused = True

    @staticmethod
    def _validate_device_gizmo_space(value: str) -> str:
        space = str(value).lower()
        if space not in {"world", "local"}:
            raise ValueError("Device gizmo space must be 'world' or 'local'.")
        return space

    @staticmethod
    def _parse_placement_drive_axes(values) -> list[bool]:
        axis_enabled = [False, False, False]
        axis_to_index = {"x": 0, "y": 1, "z": 2}
        for value in values:
            axis = str(value).lower()
            if axis not in axis_to_index:
                raise ValueError("Placement drive axes must be chosen from X, Y, and Z.")
            axis_enabled[axis_to_index[axis]] = True
        return axis_enabled

    @staticmethod
    def _clamp_device_compression(values) -> np.ndarray:
        compression = np.asarray(values, dtype=np.float32)
        if compression.shape != (3,):
            raise ValueError("Device compression must contain exactly three scale values.")
        return np.clip(compression, DEVICE_COMPRESSION_SCALE_MIN, DEVICE_COMPRESSION_SCALE_MAX)

    def _handle_pause_device_keys(self):
        reset_down = bool(self.viewer.is_key_down("r"))
        compress_down = bool(self.viewer.is_key_down("c"))

        if reset_down and not self._reset_device_key_was_down:
            self.device_compression[:] = 1.0
            self.device_transform_gizmo.set_scale(self.device_compression)
            self.device_transform_gizmo.apply_rest_shape(
                self.model,
                (self.state_0, self.state_1),
                self.device_compression,
            )

        if compress_down and not self._compress_device_key_was_down:
            self.device_transform_gizmo.apply_rest_shape(
                self.model,
                (self.state_0, self.state_1),
                self.device_compression,
            )

        self._reset_device_key_was_down = reset_down
        self._compress_device_key_was_down = compress_down
        self._handle_device_pose_slot_keys()

    def _handle_device_pose_slot_keys(self):
        save_down = bool(self.viewer.is_key_down("ctrl"))

        for key in DEVICE_POSE_SLOT_KEYS:
            slot_down = bool(self.viewer.is_key_down(key))
            if slot_down and not self._device_pose_slot_key_was_down[key]:
                slot = int(key)
                if save_down:
                    self._save_device_pose_slot(slot)
                else:
                    self._load_device_pose_slot(slot)

            self._device_pose_slot_key_was_down[key] = slot_down

    def _save_device_pose_slot(self, slot: int):
        transform = np.asarray(self.device_transform_gizmo.transform, dtype=np.float32).copy()
        compression = self.device_compression.copy()
        self._device_pose_slots[slot] = (transform, compression)
        if self._write_device_pose_slots_file():
            print(f"Saved VSD device pose slot {slot} to '{self._device_pose_slots_path}'.")
        else:
            print(f"Saved VSD device pose slot {slot} in memory only.")

    def _load_device_pose_slot(self, slot: int):
        pose = self._device_pose_slots.get(slot)
        if pose is None:
            print(f"No VSD device pose saved in slot {slot}.")
            return

        transform, compression = pose
        self.device_transform_gizmo.transform[:] = wp.transform(*transform)
        self.device_transform_gizmo._last_transform[:] = self.device_transform_gizmo.transform
        self.device_compression[:] = compression
        self.device_transform_gizmo.set_scale(self.device_compression)
        self.device_transform_gizmo.apply_rest_shape(
            self.model,
            (self.state_0, self.state_1),
            self.device_compression,
        )
        if self.minimou_control_enabled:
            self._request_minimou_takeover()
        print(f"Loaded VSD device pose slot {slot}.")

    def _load_device_pose_slots_from_file(self):
        if not self._device_pose_slots_path.exists():
            return

        try:
            data = json.loads(self._device_pose_slots_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("slot file must contain a JSON object")

            slots_data = data.get("slots", data)
            if not isinstance(slots_data, dict):
                raise ValueError("slot file 'slots' field must contain a JSON object")

            loaded_count = 0

            for key in DEVICE_POSE_SLOT_KEYS:
                slot_data = slots_data.get(key)
                if slot_data is None:
                    continue

                transform = np.asarray(slot_data["transform"], dtype=np.float32)
                compression = self._clamp_device_compression(slot_data["compression"])
                if transform.shape != (7,):
                    raise ValueError(f"slot {key} transform must contain 7 values")

                self._device_pose_slots[int(key)] = (transform, compression)
                loaded_count += 1

        except (KeyError, OSError, TypeError, ValueError) as exc:
            print(f"Could not load VSD device pose slots from '{self._device_pose_slots_path}': {exc}")
            return

        if loaded_count > 0:
            print(f"Loaded {loaded_count} VSD device pose slot(s) from '{self._device_pose_slots_path}'.")

    def _write_device_pose_slots_file(self) -> bool:
        slots_data = {}
        for slot, (transform, compression) in sorted(self._device_pose_slots.items()):
            slots_data[str(slot)] = {
                "transform": transform.astype(float).tolist(),
                "compression": compression.astype(float).tolist(),
            }

        data = {
            "version": 1,
            "slots": slots_data,
        }

        try:
            self._device_pose_slots_path.parent.mkdir(parents=True, exist_ok=True)
            text = json.dumps(data, indent=2, sort_keys=True) + "\n"
            self._device_pose_slots_path.write_text(text, encoding="utf-8")
        except OSError as exc:
            print(f"Could not write VSD device pose slots to '{self._device_pose_slots_path}': {exc}")
            return False

        return True

    def _placement_gizmo_space_gui(self, ui):
        ui.text("Gizmo Space")
        if hasattr(ui, "radio_button"):
            if ui.radio_button("World##placement_gizmo_space", self.device_gizmo_space == "world"):
                self.device_gizmo_space = "world"
            if hasattr(ui, "same_line"):
                ui.same_line()
            if ui.radio_button("Local##placement_gizmo_space", self.device_gizmo_space == "local"):
                self.device_gizmo_space = "local"
            return

        changed, local_space = ui.checkbox("Local Gizmo Space", self.device_gizmo_space == "local")
        if changed:
            self.device_gizmo_space = "local" if local_space else "world"

    def minimou_gui(self, ui):
        if not hasattr(ui, "begin"):
            return

        ui.set_next_window_pos(ui.ImVec2(1060.0, 20.0), ui.Cond_.appearing)
        ui.set_next_window_size(ui.ImVec2(360.0, 340.0), ui.Cond_.appearing)
        if not ui.begin("MiniMou Input"):
            ui.end()
            return

        if hasattr(ui, "input_int"):
            index_disabled = self.minimou_enabled and hasattr(ui, "begin_disabled")
            if index_disabled:
                ui.begin_disabled()
            changed, value = ui.input_int("Device Index", self.minimou_device_index, 1, 1)
            if changed:
                self.minimou_device_index = self._validate_minimou_device_index(value)
            if index_disabled:
                ui.end_disabled()

        changed, enabled = ui.checkbox("Enable MiniMou", self.minimou_enabled)
        if changed:
            self._set_minimou_enabled(bool(enabled))

        changed, power = ui.checkbox("Power Device", self.minimou_power_enabled)
        if changed:
            self._set_minimou_power_enabled(bool(power))

        changed, control = ui.checkbox("Control Handle", self.minimou_control_enabled)
        if changed:
            self._set_minimou_control_enabled(bool(control))

        changed, follows_camera = ui.checkbox("Camera Input Space", self.minimou_workspace_follows_camera)
        if changed:
            self._set_minimou_workspace_follows_camera(bool(follows_camera))

        if self.minimou_control_enabled and ui.button("Reset Offset"):
            self._reset_minimou_handle_offset()

        changed, show_frame = ui.checkbox("Device XYZ Frame", self.show_minimou_device_frame)
        if changed:
            self.show_minimou_device_frame = bool(show_frame)

        if ui.button("Takeover Current Handle"):
            self._request_minimou_takeover()

        ui.separator()
        changed, value = ui.slider_float(
            "Workspace Scale",
            self.minimou_workspace_scale,
            MINIMOU_WORKSPACE_SCALE_MIN,
            MINIMOU_WORKSPACE_SCALE_MAX,
            "%.3f",
        )
        if changed:
            self.minimou_workspace_scale = self._clamp_minimou_workspace_scale(value)
            if self.minimou_control_enabled:
                self._request_minimou_takeover()

        changed, value = ui.drag_float3(
            "Pos Offset",
            self.minimou_workspace_pos_offset.tolist(),
            MINIMOU_POSITION_OFFSET_SPEED,
            -10.0,
            10.0,
            "%.3f",
        )
        if changed:
            self.minimou_workspace_pos_offset = self._vec3_arg(value, "MiniMou workspace position offset")
            if self.minimou_control_enabled:
                self._request_minimou_takeover()

        changed, value = ui.drag_float3(
            "Rot Offset XYZ Deg",
            self.minimou_workspace_rot_offset_deg.tolist(),
            MINIMOU_ROTATION_OFFSET_SPEED_DEG,
            -180.0,
            180.0,
            "%.1f",
        )
        if changed:
            self.minimou_workspace_rot_offset_deg = self._vec3_arg(value, "MiniMou workspace rotation offset")
            if self.minimou_control_enabled:
                self._request_minimou_takeover()

        ui.separator()
        ui.text(f"Status: {self.minimou_status}")
        if self.minimou_error:
            ui.text(f"Error: {self.minimou_error}")
        if self.minimou_last_sample is not None:
            grip = float(self.minimou_last_sample.get("grip", 0.0))
            button = bool(self.minimou_last_sample.get("button", False))
            raw_pos = np.asarray(self.minimou_last_sample.get("raw_position", (0.0, 0.0, 0.0)), dtype=np.float32)
            pos = np.asarray(self.minimou_last_sample.get("position", (0.0, 0.0, 0.0)), dtype=np.float32)
            raw_rot = np.asarray(self.minimou_last_sample.get("raw_rotation", (0.0, 0.0, 0.0, 1.0)), dtype=np.float32)
            rot = np.asarray(self.minimou_last_sample.get("rotation", (0.0, 0.0, 0.0, 1.0)), dtype=np.float32)
            ui.text(f"Raw Pos: {raw_pos[0]:.3f}, {raw_pos[1]:.3f}, {raw_pos[2]:.3f}")
            ui.text(f"Mapped Pos: {pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}")
            ui.text(f"Raw Quat: {raw_rot[0]:.3f}, {raw_rot[1]:.3f}, {raw_rot[2]:.3f}, {raw_rot[3]:.3f}")
            ui.text(f"Mapped Quat: {rot[0]:.3f}, {rot[1]:.3f}, {rot[2]:.3f}, {rot[3]:.3f}")
            ui.text(f"Grip: {grip:.2f}  Button: {button}")
        if self.minimou_control_enabled and not self.viewer.is_paused() and not self._use_interactive_placement():
            ui.text("Running control requires Interactive Placement.")

        ui.end()

    def placement_gui(self, ui):
        if not hasattr(ui, "begin"):
            return

        ui.set_next_window_pos(ui.ImVec2(700.0, 20.0), ui.Cond_.appearing)
        ui.set_next_window_size(ui.ImVec2(340.0, 300.0), ui.Cond_.appearing)
        if not ui.begin("VSD Placement"):
            ui.end()
            return

        self._placement_gizmo_space_gui(ui)
        ui.separator()

        placement_disabled = not self._interactive_placement_available()
        if placement_disabled and hasattr(ui, "begin_disabled"):
            ui.begin_disabled()

        changed, enabled = ui.checkbox("Interactive Placement", self.interactive_placement_enabled)
        if changed:
            self._set_interactive_placement_enabled(bool(enabled))

        axis_changed = False
        ui.text("Drive Axes")
        for axis, label in enumerate(("X", "Y", "Z")):
            changed, enabled = ui.checkbox(f"{label}##placement_drive_axis_{label}", self.placement_drive_axes[axis])
            if changed:
                self.placement_drive_axes[axis] = bool(enabled)
                axis_changed = True
            if axis < 2 and hasattr(ui, "same_line"):
                ui.same_line()
        if axis_changed:
            self._refresh_placement_clone_buffers()

        changed, value = ui.slider_float(
            "Axis Band Radius",
            self.placement_axis_band_radius,
            PLACEMENT_AXIS_BAND_RADIUS_MIN,
            PLACEMENT_AXIS_BAND_RADIUS_MAX,
            "%.3f",
        )
        if changed:
            self.placement_axis_band_radius = min(
                max(float(value), PLACEMENT_AXIS_BAND_RADIUS_MIN),
                PLACEMENT_AXIS_BAND_RADIUS_MAX,
            )
            self._refresh_placement_clone_buffers()

        changed, value = ui.slider_float(
            "Clone Stiffness",
            self.placement_clone_stiffness,
            PLACEMENT_CLONE_STIFFNESS_MIN,
            PLACEMENT_CLONE_STIFFNESS_MAX,
            "%.0f",
        )
        if changed:
            self.placement_clone_stiffness = min(
                max(float(value), PLACEMENT_CLONE_STIFFNESS_MIN),
                PLACEMENT_CLONE_STIFFNESS_MAX,
            )
            self._mark_graph_recapture()

        changed, value = ui.slider_float(
            "Clone Damping",
            self.placement_clone_damping,
            PLACEMENT_CLONE_DAMPING_MIN,
            PLACEMENT_CLONE_DAMPING_MAX,
            "%.1f",
        )
        if changed:
            self.placement_clone_damping = min(
                max(float(value), PLACEMENT_CLONE_DAMPING_MIN),
                PLACEMENT_CLONE_DAMPING_MAX,
            )
            self._mark_graph_recapture()

        changed, value = ui.slider_float(
            "Max Clone Force",
            self.placement_max_clone_force,
            PLACEMENT_MAX_CLONE_FORCE_MIN,
            PLACEMENT_MAX_CLONE_FORCE_MAX,
            "%.0f",
        )
        if changed:
            self.placement_max_clone_force = min(
                max(float(value), PLACEMENT_MAX_CLONE_FORCE_MIN),
                PLACEMENT_MAX_CLONE_FORCE_MAX,
            )
            self._mark_graph_recapture()

        changed, show_targets = ui.checkbox("Clone Targets", self.show_placement_clone_targets)
        if changed:
            self.show_placement_clone_targets = bool(show_targets)

        changed, show_particles = ui.checkbox("Driving Particles", self.show_placement_driving_particles)
        if changed:
            self.show_placement_driving_particles = bool(show_particles)

        ui.text(f"Selected Particles: {self.placement_clone_count}")

        if placement_disabled and hasattr(ui, "end_disabled"):
            ui.end_disabled()
        ui.end()

    def _projection_contact_gui(self, ui, disabled: bool):
        if disabled and hasattr(ui, "begin_disabled"):
            ui.begin_disabled()

        changed, value = ui.slider_float(
            "Projection Radius",
            self.vessel_contact_radius,
            VESSEL_CONTACT_RADIUS_MIN,
            VESSEL_CONTACT_RADIUS_MAX,
            "%.3f",
        )
        if changed:
            self.vessel_contact_radius = min(
                max(float(value), VESSEL_CONTACT_RADIUS_MIN),
                VESSEL_CONTACT_RADIUS_MAX,
            )
            self._mark_graph_recapture()

        changed, value = ui.slider_float(
            "Projection Relaxation",
            self.vessel_contact_relaxation,
            VESSEL_CONTACT_RELAXATION_MIN,
            VESSEL_CONTACT_RELAXATION_MAX,
            "%.2f",
        )
        if changed:
            self.vessel_contact_relaxation = min(
                max(float(value), VESSEL_CONTACT_RELAXATION_MIN),
                VESSEL_CONTACT_RELAXATION_MAX,
            )
            self._mark_graph_recapture()

        changed, value = ui.slider_int(
            "Projection Iterations",
            self.vessel_contact_iterations,
            VESSEL_CONTACT_ITERATIONS_MIN,
            VESSEL_CONTACT_ITERATIONS_MAX,
            "%d",
        )
        if changed:
            self.vessel_contact_iterations = min(
                max(int(value), VESSEL_CONTACT_ITERATIONS_MIN),
                VESSEL_CONTACT_ITERATIONS_MAX,
            )
            self._mark_graph_recapture()

        if disabled and hasattr(ui, "end_disabled"):
            ui.end_disabled()

    def _contact_model_gui(self, ui):
        if self.solver_type != "fem":
            ui.text("Contact Model: PBD Projection")
            return

        fem_soft_contact = self._use_fem_soft_vessel_contact()
        if hasattr(ui, "radio_button"):
            if ui.radio_button("FEM Soft Contact", fem_soft_contact):
                self._set_fem_soft_vessel_contact_enabled(True)
                fem_soft_contact = True
            if hasattr(ui, "same_line"):
                ui.same_line()
            if ui.radio_button("PBD Projection", not fem_soft_contact):
                self._set_fem_soft_vessel_contact_enabled(False)
            return

        changed, enabled = ui.checkbox("FEM Soft Contact", self.fem_soft_vessel_contact_enabled)
        if changed:
            self._set_fem_soft_vessel_contact_enabled(bool(enabled))

    def contact_gui(self, ui):
        if not hasattr(ui, "begin"):
            return

        ui.set_next_window_pos(ui.ImVec2(325.0, 20.0), ui.Cond_.appearing)
        ui.set_next_window_size(ui.ImVec2(360.0, 540.0), ui.Cond_.appearing)
        if not ui.begin("VSD Vessel Contact"):
            ui.end()
            return

        self._contact_model_gui(ui)
        ui.separator()

        if self.solver_type == "fem":
            soft_contact_disabled = not self._use_fem_soft_vessel_contact()
            if soft_contact_disabled and hasattr(ui, "begin_disabled"):
                ui.begin_disabled()

            changed, value = ui.slider_float(
                "Soft Margin",
                self.fem_soft_contact_margin,
                FEM_SOFT_CONTACT_MARGIN_MIN,
                FEM_SOFT_CONTACT_MARGIN_MAX,
                "%.3f",
            )
            if changed:
                self.fem_soft_contact_margin = min(
                    max(float(value), FEM_SOFT_CONTACT_MARGIN_MIN),
                    FEM_SOFT_CONTACT_MARGIN_MAX,
                )
                self._mark_graph_recapture()

            changed, value = ui.slider_int(
                "Contact Stiffness 10^n",
                self.fem_soft_contact_stiffness_exponent,
                FEM_SOFT_CONTACT_STIFFNESS_EXPONENT_MIN,
                FEM_SOFT_CONTACT_STIFFNESS_EXPONENT_MAX,
                "%d",
            )
            if changed:
                self.fem_soft_contact_stiffness_exponent = min(
                    max(int(value), FEM_SOFT_CONTACT_STIFFNESS_EXPONENT_MIN),
                    FEM_SOFT_CONTACT_STIFFNESS_EXPONENT_MAX,
                )
                self._apply_fem_soft_contact_settings()

            changed, value = ui.slider_float(
                "Contact Stiffness Mult",
                self.fem_soft_contact_stiffness_multiplier,
                FEM_SOFT_CONTACT_STIFFNESS_MULTIPLIER_MIN,
                FEM_SOFT_CONTACT_STIFFNESS_MULTIPLIER_MAX,
                "%.2f",
            )
            if changed:
                self.fem_soft_contact_stiffness_multiplier = min(
                    max(float(value), FEM_SOFT_CONTACT_STIFFNESS_MULTIPLIER_MIN),
                    FEM_SOFT_CONTACT_STIFFNESS_MULTIPLIER_MAX,
                )
                self._apply_fem_soft_contact_settings()

            changed, value = ui.slider_float(
                "Contact Damping",
                self.fem_soft_contact_kd,
                FEM_SOFT_CONTACT_DAMPING_MIN,
                FEM_SOFT_CONTACT_DAMPING_MAX,
                "%.1f",
            )
            if changed:
                self.fem_soft_contact_kd = min(
                    max(float(value), FEM_SOFT_CONTACT_DAMPING_MIN),
                    FEM_SOFT_CONTACT_DAMPING_MAX,
                )
                self._apply_fem_soft_contact_settings()

            changed, value = ui.slider_float(
                "Contact Friction",
                self.fem_soft_contact_mu,
                FEM_SOFT_CONTACT_FRICTION_MIN,
                FEM_SOFT_CONTACT_FRICTION_MAX,
                "%.2f",
            )
            if changed:
                self.fem_soft_contact_mu = min(
                    max(float(value), FEM_SOFT_CONTACT_FRICTION_MIN),
                    FEM_SOFT_CONTACT_FRICTION_MAX,
                )
                self._apply_fem_soft_contact_settings()

            if self.contacts is not None and self.viewer.is_paused():
                self._soft_contact_count_cached = int(self.contacts.soft_contact_count.numpy()[0])
            ui.text(f"Soft contacts: {self._soft_contact_count_cached}")
            if soft_contact_disabled and hasattr(ui, "end_disabled"):
                ui.end_disabled()
            ui.separator()

        changed, value = ui.slider_float(
            "Device Particle Radius",
            self.device_particle_radius,
            DEVICE_PARTICLE_RADIUS_MIN,
            DEVICE_PARTICLE_RADIUS_MAX,
            "%.3f",
        )
        if changed:
            self.device_particle_radius = min(
                max(float(value), DEVICE_PARTICLE_RADIUS_MIN),
                DEVICE_PARTICLE_RADIUS_MAX,
            )
            self._apply_device_particle_radius()

        changed, show_markers = ui.checkbox("Contact Markers", self.show_vessel_contact_markers)
        if changed:
            self.show_vessel_contact_markers = bool(show_markers)

        changed, value = ui.slider_float(
            "Marker Force Max",
            self.vessel_contact_force_color_max,
            VESSEL_CONTACT_FORCE_COLOR_MAX_MIN,
            VESSEL_CONTACT_FORCE_COLOR_MAX_MAX,
            "%.0f",
        )
        if changed:
            self.vessel_contact_force_color_max = min(
                max(float(value), VESSEL_CONTACT_FORCE_COLOR_MAX_MIN),
                VESSEL_CONTACT_FORCE_COLOR_MAX_MAX,
            )

        changed, value = ui.slider_float(
            "Vessel Color Mult",
            self.vessel_contact_force_color_multiplier,
            VESSEL_CONTACT_FORCE_COLOR_MULTIPLIER_MIN,
            VESSEL_CONTACT_FORCE_COLOR_MULTIPLIER_MAX,
            "%.1f",
        )
        if changed:
            self.vessel_contact_force_color_multiplier = min(
                max(float(value), VESSEL_CONTACT_FORCE_COLOR_MULTIPLIER_MIN),
                VESSEL_CONTACT_FORCE_COLOR_MULTIPLIER_MAX,
            )

        changed, value = ui.slider_float(
            "Vessel Alpha",
            self.vessel_alpha,
            VESSEL_ALPHA_MIN,
            VESSEL_ALPHA_MAX,
            "%.2f",
        )
        if changed:
            self.vessel_alpha = min(max(float(value), VESSEL_ALPHA_MIN), VESSEL_ALPHA_MAX)

        changed, transparent = ui.checkbox("Vessel Transparent", self.vessel_transparent_surface)
        if changed:
            self.vessel_transparent_surface = bool(transparent)

        changed, value = ui.slider_float(
            "VSD Color Mult",
            self.vsd_contact_force_color_multiplier,
            VSD_CONTACT_FORCE_COLOR_MULTIPLIER_MIN,
            VSD_CONTACT_FORCE_COLOR_MULTIPLIER_MAX,
            "%.1f",
        )
        if changed:
            self.vsd_contact_force_color_multiplier = min(
                max(float(value), VSD_CONTACT_FORCE_COLOR_MULTIPLIER_MIN),
                VSD_CONTACT_FORCE_COLOR_MULTIPLIER_MAX,
            )

        changed, value = ui.slider_float(
            "VSD Alpha",
            self.vsd_alpha,
            VSD_ALPHA_MIN,
            VSD_ALPHA_MAX,
            "%.2f",
        )
        if changed:
            self.vsd_alpha = min(max(float(value), VSD_ALPHA_MIN), VSD_ALPHA_MAX)

        changed, transparent = ui.checkbox("VSD Transparent", self.vsd_transparent_surface)
        if changed:
            self.vsd_transparent_surface = bool(transparent)

        changed, value = ui.slider_int(
            "Color Smooth Passes",
            self.contact_color_smoothing_passes,
            CONTACT_COLOR_SMOOTHING_PASSES_MIN,
            CONTACT_COLOR_SMOOTHING_PASSES_MAX,
            "%d",
        )
        if changed:
            self.contact_color_smoothing_passes = min(
                max(int(value), CONTACT_COLOR_SMOOTHING_PASSES_MIN),
                CONTACT_COLOR_SMOOTHING_PASSES_MAX,
            )

        changed, value = ui.slider_float(
            "Color Smooth Strength",
            self.contact_color_smoothing_strength,
            CONTACT_COLOR_SMOOTHING_STRENGTH_MIN,
            CONTACT_COLOR_SMOOTHING_STRENGTH_MAX,
            "%.2f",
        )
        if changed:
            self.contact_color_smoothing_strength = min(
                max(float(value), CONTACT_COLOR_SMOOTHING_STRENGTH_MIN),
                CONTACT_COLOR_SMOOTHING_STRENGTH_MAX,
            )

        ui.separator()
        self._projection_contact_gui(ui, disabled=self._use_fem_soft_vessel_contact())
        ui.end()

    def gui(self, ui):
        changed, value = ui.slider_int("Substeps", self.sim_substeps, 1, 32, "%d")
        if changed:
            self.sim_substeps = max(1, int(value))
            self.sim_dt = self.frame_dt / self.sim_substeps
            self._mark_graph_recapture()

        changed, value = ui.slider_int("Constraint Iterations", self.iterations, 1, 64, "%d")
        if changed:
            self.iterations = max(1, int(value))
            self.solver.iterations = self.iterations
            self._mark_graph_recapture()

        if self.solver_type == "fem":
            changed, value = ui.slider_float(
                "FEM Global Damping",
                self.fem_global_damping,
                FEM_GLOBAL_DAMPING_MIN,
                FEM_GLOBAL_DAMPING_MAX,
                "%.1f",
            )
            if changed:
                self.fem_global_damping = min(
                    max(float(value), FEM_GLOBAL_DAMPING_MIN),
                    FEM_GLOBAL_DAMPING_MAX,
                )
                self._apply_fem_global_damping()

        if hasattr(ui, "input_int"):
            changed, value = ui.input_int("Stiffness 10^n", self.tet_stiffness_exponent, 1, 1)
        else:
            changed, value = ui.slider_int(
                "Stiffness 10^n",
                self.tet_stiffness_exponent,
                self.tet_stiffness_exponent_min,
                self.tet_stiffness_exponent_max,
                "%d",
            )
        if changed:
            self.tet_stiffness_exponent = min(
                max(int(value), self.tet_stiffness_exponent_min),
                self.tet_stiffness_exponent_max,
            )
            self._apply_tet_stiffness()
            self._mark_graph_recapture()

        for axis, label in enumerate(("X", "Y", "Z")):
            changed, value = ui.slider_float(
                f"Device Compression {label}",
                float(self.device_compression[axis]),
                DEVICE_COMPRESSION_SCALE_MIN,
                DEVICE_COMPRESSION_SCALE_MAX,
                "%.2f",
            )
            if changed:
                self.device_compression[axis] = min(
                    max(float(value), DEVICE_COMPRESSION_SCALE_MIN),
                    DEVICE_COMPRESSION_SCALE_MAX,
                )
                self.device_transform_gizmo.set_scale(self.device_compression)
                self._reset_placement_clone_targets()

        changed, value = ui.slider_float(
            "Stiffness Multiplier",
            self.tet_stiffness_multiplier,
            TET_STIFFNESS_MULTIPLIER_MIN,
            TET_STIFFNESS_MULTIPLIER_MAX,
            "%.2f",
        )
        if changed:
            self.tet_stiffness_multiplier = min(
                max(float(value), TET_STIFFNESS_MULTIPLIER_MIN),
                TET_STIFFNESS_MULTIPLIER_MAX,
            )
            self._apply_tet_stiffness()
            self._mark_graph_recapture()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--solver",
            help="Type of solver",
            type=str,
            choices=["xpbd", "vbd", "fem"],
            default="vbd",
        )
        parser.add_argument(
            "--fp64",
            help="Request double precision for the FEM solver (currently falls back to fp32 with a warning).",
            action="store_true",
        )
        parser.add_argument(
            "--fem-global-damping",
            help="FEM global mass-proportional viscous damping coefficient [1/s].",
            type=float,
            default=1.0,
        )
        parser.add_argument(
            "--scale",
            help="Scale factor applied to the physics mesh vertices before building the model.",
            type=float,
            default=1.0,
        )
        parser.add_argument(
            "--anatomy-scale",
            help="Scale factor applied to the static anatomy mesh before building the model.",
            type=float,
            default=1.0,
        )
        parser.add_argument(
            "--offset",
            help="Translation vector (x y z) [m] added to the initial mesh position.",
            type=float,
            nargs=3,
            metavar=("X", "Y", "Z"),
            default=[0.0, 0.0, 0.0],
        )
        parser.add_argument(
            "--vessel-contact-radius",
            help="PBD clearance radius [m] for projecting VSD particles away from vessel triangles.",
            type=float,
            default=0.02,
        )
        parser.add_argument(
            "--vessel-contact-relaxation",
            help="Relaxation factor for static vessel particle-triangle projection.",
            type=float,
            default=0.8,
        )
        parser.add_argument(
            "--vessel-contact-iterations",
            help="Number of particle-triangle projection passes after each solver substep.",
            type=int,
            default=2,
        )
        parser.add_argument(
            "--fem-soft-vessel-contact",
            help="Enable FEM soft particle-shape contacts against the vessel mesh.",
            action=argparse.BooleanOptionalAction,
            default=True,
        )
        parser.add_argument(
            "--fem-soft-contact-margin",
            help="FEM soft-contact detection margin [m] for particle-shape contacts.",
            type=float,
            default=0.03,
        )
        parser.add_argument(
            "--fem-soft-contact-stiffness-exponent",
            help="Exponent n for FEM soft-contact stiffness 10^n [N/m].",
            type=int,
            default=3,
        )
        parser.add_argument(
            "--fem-soft-contact-stiffness-multiplier",
            help="Multiplier for FEM soft-contact stiffness.",
            type=float,
            default=1.0,
        )
        parser.add_argument(
            "--fem-soft-contact-damping",
            help="FEM soft-contact damping [N*s/m].",
            type=float,
            default=0.0,
        )
        parser.add_argument(
            "--fem-soft-contact-friction",
            help="FEM soft-contact Coulomb friction coefficient.",
            type=float,
            default=0.4,
        )
        parser.add_argument(
            "--device-particle-radius",
            help="VSD device particle contact/render radius [m].",
            type=float,
            default=0.02,
        )
        parser.add_argument(
            "--show-vessel-contact-markers",
            help="Show per-particle vessel contact force markers.",
            action=argparse.BooleanOptionalAction,
            default=False,
        )
        parser.add_argument(
            "--vessel-contact-force-color-max",
            help="Force magnitude [N] mapped to the hottest vessel contact color.",
            type=float,
            default=1000.0,
        )
        parser.add_argument(
            "--vessel-contact-force-color-multiplier",
            help="Multiplier applied before mapping vessel contact force to vertex color.",
            type=float,
            default=10.0,
        )
        parser.add_argument(
            "--vessel-alpha",
            help="Vessel mesh opacity alpha in [0, 1].",
            type=float,
            default=0.45,
        )
        parser.add_argument(
            "--vessel-transparent-surface",
            help="Render the vessel surface through the transparent mesh path.",
            action=argparse.BooleanOptionalAction,
            default=True,
        )
        parser.add_argument(
            "--vsd-contact-force-color-multiplier",
            help="Multiplier applied before mapping VSD contact force to surface color.",
            type=float,
            default=10.0,
        )
        parser.add_argument(
            "--vsd-alpha",
            help="VSD device surface opacity alpha in [0, 1].",
            type=float,
            default=1.0,
        )
        parser.add_argument(
            "--vsd-transparent-surface",
            help="Render the VSD surface through the transparent mesh path.",
            action=argparse.BooleanOptionalAction,
            default=True,
        )
        parser.add_argument(
            "--contact-color-smoothing-passes",
            help="Laplacian smoothing passes applied to vessel and VSD vertex colors.",
            type=int,
            default=0,
        )
        parser.add_argument(
            "--contact-color-smoothing-strength",
            help="Per-pass Laplacian smoothing strength for vessel and VSD vertex colors.",
            type=float,
            default=0.5,
        )
        parser.add_argument(
            "--device-gizmo-space",
            help="Handle orientation space for the VSD transform gizmo.",
            type=str,
            choices=["world", "local"],
            default="world",
        )
        parser.add_argument(
            "--interactive-placement",
            help="Drive FEM device particles with soft transform-gizmo clone targets.",
            action=argparse.BooleanOptionalAction,
            default=False,
        )
        parser.add_argument(
            "--placement-drive-axes",
            help="Local device axes used to select interactive placement particles.",
            type=str,
            nargs="*",
            metavar="AXIS",
            default=["x", "y", "z"],
        )
        parser.add_argument(
            "--placement-axis-band-radius",
            help="Local-axis selection band radius [m] for interactive placement particles.",
            type=float,
            default=0.05,
        )
        parser.add_argument(
            "--placement-clone-stiffness",
            help="Interactive placement clone spring stiffness [N/m].",
            type=float,
            default=1000.0,
        )
        parser.add_argument(
            "--placement-clone-damping",
            help="Interactive placement clone damping [N*s/m].",
            type=float,
            default=0.0,
        )
        parser.add_argument(
            "--placement-max-clone-force",
            help="Maximum force [N] applied by each interactive placement clone.",
            type=float,
            default=1000.0,
        )
        parser.add_argument(
            "--device-compression",
            help="Initial local X Y Z compression scales for the VSD device in pause mode.",
            type=float,
            nargs=3,
            metavar=("X", "Y", "Z"),
            default=[1.0, 1.0, 1.0],
        )
        parser.add_argument(
            "--device-pose-slots-file",
            help="JSON file used to persist pause-mode VSD device pose slots.",
            type=str,
            default=str(DEFAULT_DEVICE_POSE_SLOTS_PATH),
        )
        parser.add_argument(
            "--minimou-enabled",
            help="Connect to a MiniMou device when the example starts.",
            action=argparse.BooleanOptionalAction,
            default=False,
        )
        parser.add_argument(
            "--minimou-power",
            help="Send the MiniMou power-on command when the device is connected.",
            action=argparse.BooleanOptionalAction,
            default=False,
        )
        parser.add_argument(
            "--minimou-control",
            help="Drive the VSD handle gizmo from the MiniMou device.",
            action=argparse.BooleanOptionalAction,
            default=False,
        )
        parser.add_argument(
            "--minimou-device-index",
            help="Zero-based MiniMou device index from the Follou device scan.",
            type=int,
            default=0,
        )
        parser.add_argument(
            "--minimou-workspace-scale",
            help="Scale applied to MiniMou workspace position deltas.",
            type=float,
            default=0.001,
        )
        parser.add_argument(
            "--minimou-workspace-follows-camera",
            help="Make the MiniMou input space move and rotate with the viewer camera.",
            action=argparse.BooleanOptionalAction,
            default=False,
        )
        parser.add_argument(
            "--minimou-workspace-pos-offset",
            help="MiniMou workspace position offset [m].",
            type=float,
            nargs=3,
            metavar=("X", "Y", "Z"),
            default=[0.0, 0.0, 0.0],
        )
        parser.add_argument(
            "--minimou-workspace-rot-offset",
            help="MiniMou workspace rotation offset as XYZ Euler angles [deg].",
            type=float,
            nargs=3,
            metavar=("X", "Y", "Z"),
            default=[0.0, 0.0, 0.0],
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
