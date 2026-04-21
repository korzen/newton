# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Sample-local solver extensions for XCATH vessel containment."""

from __future__ import annotations

import numpy as np
import warp as wp

from newton._src.solvers.xpbd_rod.solver_xpbd_rod import SolverXPBDRod, _RodWorkspace


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
def _project_vessel_containment_kernel(
    current_positions: wp.array(dtype=wp.vec3),
    predicted_positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    mesh_id: wp.uint64,
    inv_masses: wp.array(dtype=wp.float32),
    sign_scale: float,
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
    if dist > 1.0e-8:
        grad = (diff / dist) * side
    else:
        grad = _mesh_face_normal(mesh_id, face_index) * side

    phi = side * dist
    if phi <= target_phi:
        return

    projected = pos - grad * (phi - target_phi)
    disp = projected - current_positions[i]
    outward_disp = wp.dot(disp, grad)
    if outward_disp > 0.0:
        projected = projected - grad * outward_disp
    predicted_positions[i] = projected

    vel = velocities[i]
    outward_speed = wp.dot(vel, grad)
    if outward_speed > 0.0:
        velocities[i] = vel - grad * outward_speed


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
        collision_iterations: int = 2,
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
        self.track_start = wp.vec3(float(track_start[0]), float(track_start[1]), float(track_start[2]))
        self.track_dir = wp.vec3(float(track_dir[0]), float(track_dir[1]), float(track_dir[2]))
        self.track_length = float(track_length)
        self.tip_num_edges = int(tip_num_edges)
        self.track_stiffness = float(track_stiffness)
        self.track_enabled = bool(track_enabled)
        self.collision_enabled = bool(collision_enabled)
        self.collision_iterations = max(1, int(collision_iterations))
        self.sign_scale = float(sign_scale)
        self.target_phi = float(target_phi) if target_phi is not None else -float(particle_radius)
        self.max_dist = float(max_dist) if max_dist is not None else 2.0 * float(particle_radius) + float(segment_length)

    def set_collision_mesh(self, collision_mesh: wp.Mesh | None) -> None:
        """Swap the vessel mesh used for containment."""
        self.collision_mesh = collision_mesh

    def _project_predicted_positions(
        self,
        rod_idx: int,
        ws: _RodWorkspace,
        dt: float,
        device: wp.Device,
    ) -> None:
        """Apply XCATH track guidance and vessel containment to predicted positions."""
        del rod_idx, dt

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

        if self.collision_enabled and self.collision_mesh is not None:
            for _ in range(self.collision_iterations):
                wp.launch(
                    _project_vessel_containment_kernel,
                    dim=ws.num_points,
                    inputs=[
                        ws.positions_wp,
                        ws.predicted_positions_wp,
                        ws.velocities_wp,
                        self.collision_mesh.id,
                        ws.inv_masses_wp,
                        float(self.sign_scale),
                        float(self.target_phi),
                        float(self.max_dist),
                    ],
                    device=device,
                )


__all__ = [
    "XCathRodSolver",
    "compute_signed_distances",
    "_project_vessel_containment_kernel",
    "_sample_signed_distance_kernel",
]
