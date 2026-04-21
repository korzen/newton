# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for XCATH vessel containment projection."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.solvers
from newton.solvers import xpbd_rod
from xcath.xcath_solver import XCathRodSolver, _project_vessel_containment_kernel, compute_signed_distances

_WARP_CACHE_DIR = Path(__file__).resolve().parents[2] / ".warp_cache"
_WARP_CACHE_DIR.mkdir(exist_ok=True)
wp.config.kernel_cache_dir = str(_WARP_CACHE_DIR)


def _make_capped_cylinder_mesh(
    radius: float = 0.2,
    length: float = 2.0,
    segments: int = 64,
    device: wp.Device | None = None,
) -> wp.Mesh:
    """Create a closed cylinder mesh aligned with the x-axis."""
    device = device or wp.get_device()

    vertices: list[list[float]] = []
    for x in (0.0, length):
        for seg in range(segments):
            theta = 2.0 * np.pi * seg / segments
            vertices.append([x, radius * np.cos(theta), radius * np.sin(theta)])

    left_center = len(vertices)
    vertices.append([0.0, 0.0, 0.0])
    right_center = len(vertices)
    vertices.append([length, 0.0, 0.0])

    indices: list[int] = []
    for seg in range(segments):
        next_seg = (seg + 1) % segments

        a = seg
        d = next_seg
        b = segments + seg
        c = segments + next_seg

        indices.extend([a, d, c])
        indices.extend([a, c, b])

        indices.extend([left_center, next_seg, seg])
        indices.extend([right_center, segments + seg, segments + next_seg])

    vertices_np = np.asarray(vertices, dtype=np.float32)
    indices_np = np.asarray(indices, dtype=np.int32)
    return wp.Mesh(
        points=wp.array(vertices_np, dtype=wp.vec3, device=device),
        indices=wp.array(indices_np, dtype=wp.int32, device=device),
    )


def _make_single_point_arrays(
    position: np.ndarray,
    velocity: np.ndarray,
    device: wp.Device,
) -> tuple[wp.array, wp.array, wp.array, wp.array]:
    """Create single-point Warp arrays for projection tests."""
    current = wp.array(np.asarray([position], dtype=np.float32), dtype=wp.vec3, device=device)
    predicted = wp.array(np.asarray([position], dtype=np.float32), dtype=wp.vec3, device=device)
    velocities = wp.array(np.asarray([velocity], dtype=np.float32), dtype=wp.vec3, device=device)
    inv_masses = wp.array(np.asarray([1.0], dtype=np.float32), dtype=wp.float32, device=device)
    return current, predicted, velocities, inv_masses


def _make_rod_model(
    num_points: int = 8,
    spacing: float = 0.05,
) -> tuple[newton.Model, np.ndarray]:
    """Create a straight rod model aligned with the x-axis."""
    builder = newton.ModelBuilder()
    newton.solvers.SolverXPBDRod.register_custom_attributes(builder)

    positions = np.zeros((num_points, 3), dtype=np.float32)
    positions[:, 0] = 0.25 + np.arange(num_points, dtype=np.float32) * spacing

    xpbd_rod.add_elastic_rod(
        builder,
        positions=positions,
        radius=0.02,
        particle_mass=0.05,
        bend_stiffness=0.1,
        twist_stiffness=0.1,
        young_modulus=1.0e4,
        torsion_modulus=1.0e4,
        lock_root=True,
        lock_root_rotation=True,
    )

    model = builder.finalize()
    model.gravity = wp.array([[0.0, 0.0, 0.0]], dtype=wp.vec3, device=model.device)
    return model, positions


class TestXCathVesselProjectionKernel(unittest.TestCase):
    """Unit tests for the XCATH vessel projection kernel."""

    def setUp(self):
        self.device = wp.get_device()
        self.vessel_radius = 0.2
        self.particle_radius = 0.02
        self.target_phi = -self.particle_radius
        self.max_dist = 2.0 * self.particle_radius + 0.05
        self.mesh = _make_capped_cylinder_mesh(radius=self.vessel_radius, device=self.device)

    def _project(self, position: np.ndarray, velocity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        current, predicted, velocities, inv_masses = _make_single_point_arrays(position, velocity, self.device)
        wp.launch(
            _project_vessel_containment_kernel,
            dim=1,
            inputs=[
                current,
                predicted,
                velocities,
                self.mesh.id,
                inv_masses,
                self.target_phi,
                self.max_dist,
            ],
            device=self.device,
        )
        return predicted.numpy()[0], velocities.numpy()[0]

    @staticmethod
    def _radial_phi(point: np.ndarray, vessel_radius: float) -> float:
        return float(np.linalg.norm(point[1:]) - vessel_radius)

    def test_outside_point_projects_to_target_phi(self):
        projected, _ = self._project(np.array([0.5, 0.25, 0.0], dtype=np.float32), np.zeros(3, dtype=np.float32))
        self.assertAlmostEqual(self._radial_phi(projected, self.vessel_radius), self.target_phi, places=3)

    def test_inside_near_wall_projects_to_target_phi(self):
        projected, _ = self._project(np.array([0.5, 0.19, 0.0], dtype=np.float32), np.zeros(3, dtype=np.float32))
        self.assertAlmostEqual(self._radial_phi(projected, self.vessel_radius), self.target_phi, places=3)

    def test_deep_inside_point_is_unchanged(self):
        point = np.array([0.5, 0.05, 0.02], dtype=np.float32)
        projected, _ = self._project(point, np.zeros(3, dtype=np.float32))
        np.testing.assert_allclose(projected, point, atol=1.0e-6)

    def test_outward_velocity_component_is_removed(self):
        velocity = np.array([0.3, 0.5, 0.4], dtype=np.float32)
        _, projected_velocity = self._project(np.array([0.5, 0.25, 0.0], dtype=np.float32), velocity)
        np.testing.assert_allclose(projected_velocity, np.array([0.3, 0.0, 0.4], dtype=np.float32), atol=1.0e-6)


class TestXCathVesselContainmentIntegration(unittest.TestCase):
    """Integration tests for XCATH vessel containment inside the rod solver."""

    def test_solver_keeps_dynamic_particles_inside_closed_tube(self):
        model, positions = _make_rod_model()
        device = model.device
        mesh = _make_capped_cylinder_mesh(device=device)

        track_start = positions[0]
        track_dir = positions[-1] - positions[0]
        track_length = float(np.linalg.norm(track_dir))
        track_dir = track_dir / track_length

        solver = XCathRodSolver(
            model=model,
            linear_damping=0.0,
            angular_damping=0.0,
            solver_backend="block_thomas",
            floor_z=None,
            collision_mesh=mesh,
            track_start=track_start,
            track_dir=track_dir,
            track_length=track_length,
            tip_num_edges=2,
            particle_radius=0.02,
            segment_length=0.05,
            track_stiffness=1.0,
            track_enabled=True,
            collision_enabled=True,
            collision_iterations=2,
        )

        ws = solver._rods[0]
        velocities = np.zeros((positions.shape[0], 3), dtype=np.float32)
        velocities[1:, 1] = 25.0
        ws.velocities_wp.assign(wp.array(velocities, dtype=wp.vec3, device=device))

        state_in = model.state()
        state_out = model.state()
        control = model.control()
        contacts = model.contacts()

        for _ in range(20):
            solver.step(state_in, state_out, control, contacts, 0.01)
            state_in, state_out = state_out, state_in

        particle_q = state_in.particle_q.numpy()
        self.assertTrue(np.all(np.isfinite(particle_q)))

        signed_distance_wp, hit_wp = compute_signed_distances(
            state_in.particle_q,
            mesh.id,
            max_dist=1.0,
            device=device,
        )
        signed_distance = signed_distance_wp.numpy()
        hit = hit_wp.numpy().astype(bool)
        inv_mass = model.particle_inv_mass.numpy()
        dynamic = inv_mass > 0.0

        self.assertTrue(np.all(hit[dynamic]), "All dynamic particles should stay within mesh query range")
        self.assertTrue(
            np.all(signed_distance[dynamic] <= solver.target_phi + 5.0e-3),
            "Dynamic particles must remain inside the vessel containment offset",
        )
        np.testing.assert_allclose(particle_q[0], track_start, atol=1.0e-6)


if __name__ == "__main__":
    unittest.main()
