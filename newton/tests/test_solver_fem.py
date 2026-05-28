# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _build_single_tet_model(
    device,
    *,
    gravity: float = -9.81,
    k_mu: float = 1.0e3,
    k_lambda: float = 1.0e3,
    fixed_particles: tuple[int, ...] = (),
    pos_z: float = 0.0,
    add_ground: bool = False,
    particle_radius: float | None = None,
):
    builder = newton.ModelBuilder(gravity=gravity)
    if add_ground:
        builder.add_ground_plane()
    builder.add_soft_mesh(
        pos=(0.0, 0.0, pos_z),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=(0.0, 0.0, 0.0),
        vertices=[
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ],
        indices=[0, 1, 2, 3],
        density=1000.0,
        k_mu=k_mu,
        k_lambda=k_lambda,
        k_damp=0.0,
        particle_radius=particle_radius,
        add_surface_mesh_edges=False,
    )

    for particle in fixed_particles:
        builder.particle_mass[particle] = 0.0

    return builder.finalize(device=device)


def _build_fixed_grid_model(device, *, gravity: float = -9.81, k_mu: float = 1.0e3, k_lambda: float = 1.0e3):
    builder = newton.ModelBuilder(gravity=gravity)
    builder.add_soft_grid(
        pos=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=1,
        dim_y=1,
        dim_z=1,
        cell_x=1.0,
        cell_y=1.0,
        cell_z=1.0,
        density=1000.0,
        k_mu=k_mu,
        k_lambda=k_lambda,
        k_damp=0.0,
        fix_bottom=True,
        add_surface_mesh_edges=False,
    )
    return builder.finalize(device=device)


def _step_once(model, *, iterations: int, dt: float, plane_projection_iterations: int = 0):
    solver = newton.solvers.SolverFEM(
        model,
        iterations=iterations,
        cg_tol=1.0e-7,
        cg_max_iters=128,
        plane_contact_projection_iterations=plane_projection_iterations,
    )
    state_in = model.state()
    state_out = model.state()
    solver.step(state_in, state_out, None, None, dt)
    return state_out.particle_q.numpy(), state_out.particle_qd.numpy(), solver


class TestSolverFEM(unittest.TestCase):
    pass


def test_fem_rest_stability(test, device):
    model = _build_single_tet_model(device, gravity=0.0)
    rest = model.particle_q.numpy()

    q, qd, _solver = _step_once(model, iterations=3, dt=0.05)

    np.testing.assert_allclose(q, rest, rtol=0.0, atol=1.0e-7)
    np.testing.assert_allclose(qd, np.zeros_like(qd), rtol=0.0, atol=1.0e-7)


def test_fem_gravity_preserves_fixed_particles(test, device):
    fixed = (0, 1, 2)
    model = _build_single_tet_model(device, fixed_particles=fixed)
    rest = model.particle_q.numpy()

    q, qd, _solver = _step_once(model, iterations=4, dt=0.1)

    np.testing.assert_array_equal(q[list(fixed)], rest[list(fixed)])
    np.testing.assert_array_equal(qd[list(fixed)], np.zeros((len(fixed), 3), dtype=np.float32))
    test.assertLess(q[3, 2], rest[3, 2])
    test.assertTrue(np.all(np.isfinite(q)))


def test_fem_newton_iterations_update_displacement(test, device):
    model = _build_fixed_grid_model(device)

    q_one, _qd_one, _solver_one = _step_once(model, iterations=1, dt=0.2)
    q_many, _qd_many, _solver_many = _step_once(model, iterations=4, dt=0.2)

    test.assertGreater(float(np.linalg.norm(q_many - q_one)), 1.0e-4)
    test.assertTrue(np.all(np.isfinite(q_many)))


def test_fem_material_refresh_changes_deformation(test, device):
    model = _build_single_tet_model(device, k_mu=100.0, k_lambda=100.0, fixed_particles=(0, 1, 2))
    solver = newton.solvers.SolverFEM(model, iterations=4, cg_tol=1.0e-7, cg_max_iters=128)
    state_in = model.state()

    soft_out = model.state()
    solver.step(state_in, soft_out, None, None, 0.1)
    soft_q = soft_out.particle_q.numpy()

    tet_materials = model.tet_materials.numpy()
    tet_materials[:, 0] = 1.0e5
    tet_materials[:, 1] = 1.0e5
    model.tet_materials.assign(tet_materials)
    solver.refresh_material_parameters()

    stiff_out = model.state()
    solver.step(state_in, stiff_out, None, None, 0.1)
    stiff_q = stiff_out.particle_q.numpy()

    test.assertGreater(float(np.linalg.norm(stiff_q - soft_q)), 1.0e-3)
    test.assertGreater(stiff_q[3, 2], soft_q[3, 2])


def test_fem_global_damping_damps_velocity(test, device):
    model = _build_single_tet_model(device, gravity=0.0, k_mu=0.0, k_lambda=0.0)
    state_in = model.state()
    velocity = np.zeros((model.particle_count, 3), dtype=np.float32)
    velocity[:, 0] = 1.0
    state_in.particle_qd.assign(velocity)

    solver = newton.solvers.SolverFEM(model, iterations=1, k_damp=10.0, cg_tol=1.0e-7, cg_max_iters=128)
    test.assertEqual(solver.k_damp, 10.0)
    with test.assertRaises(ValueError):
        solver.k_damp = -1.0

    solver.k_damp = 5.0
    state_out = model.state()
    solver.step(state_in, state_out, None, None, 0.1)
    qd = state_out.particle_qd.numpy()

    expected_x = 1.0 / (1.0 + solver.k_damp * 0.1)
    np.testing.assert_allclose(qd[:, 0], expected_x, rtol=1.0e-5, atol=1.0e-6)
    np.testing.assert_allclose(qd[:, 1:], 0.0, rtol=0.0, atol=1.0e-7)
    test.assertLess(float(np.linalg.norm(qd)), float(np.linalg.norm(velocity)))


def test_fem_plane_projection_clamps_ground_penetration(test, device):
    radius = 0.1
    model = _build_single_tet_model(device, gravity=0.0, pos_z=-0.25, add_ground=True, particle_radius=radius)

    q, qd, _solver = _step_once(model, iterations=1, dt=0.1, plane_projection_iterations=1)

    np.testing.assert_allclose(q[:3, 2], radius, rtol=0.0, atol=1.0e-6)
    test.assertGreater(q[3, 2], radius)
    test.assertTrue(np.all(qd[:3, 2] >= -1.0e-6))


def test_fem_plane_projection_preserves_fixed_particles(test, device):
    radius = 0.1
    fixed = (0,)
    model = _build_single_tet_model(
        device,
        gravity=0.0,
        pos_z=-0.25,
        add_ground=True,
        particle_radius=radius,
        fixed_particles=fixed,
    )
    rest = model.particle_q.numpy()

    q, qd, _solver = _step_once(model, iterations=1, dt=0.1, plane_projection_iterations=1)

    np.testing.assert_array_equal(q[list(fixed)], rest[list(fixed)])
    np.testing.assert_array_equal(qd[list(fixed)], np.zeros((len(fixed), 3), dtype=np.float32))
    np.testing.assert_allclose(q[1:3, 2], radius, rtol=0.0, atol=1.0e-6)


def test_fem_plane_projection_disabled_preserves_penetration(test, device):
    radius = 0.1
    model = _build_single_tet_model(device, gravity=0.0, pos_z=-0.25, add_ground=True, particle_radius=radius)

    q, qd, _solver = _step_once(model, iterations=1, dt=0.1, plane_projection_iterations=0)
    rest = model.particle_q.numpy()

    np.testing.assert_allclose(q, rest, rtol=0.0, atol=1.0e-7)
    np.testing.assert_allclose(qd, np.zeros_like(qd), rtol=0.0, atol=1.0e-7)
    test.assertLess(q[0, 2], radius)


def test_fem_particle_drag_projection_moves_picked_point(test, device):
    model = _build_single_tet_model(device, gravity=0.0)
    solver = newton.solvers.SolverFEM(
        model,
        iterations=1,
        cg_tol=1.0e-7,
        cg_max_iters=128,
        particle_drag_projection_iterations=1,
    )
    target = wp.array([wp.vec3(1.0 / 3.0, 1.0 / 3.0, 0.4)], dtype=wp.vec3, device=device)
    picked_point = wp.zeros(1, dtype=wp.vec3, device=device)
    solver.set_particle_drag_constraint(
        wp.array([0, 1, 2], dtype=wp.int32, device=device),
        wp.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=float, device=device),
        target,
        picked_point,
    )

    state_in = model.state()
    state_out = model.state()
    solver.step(state_in, state_out, None, None, 0.1)

    np.testing.assert_allclose(picked_point.numpy()[0], target.numpy()[0], rtol=0.0, atol=1.0e-6)
    qd = state_out.particle_qd.numpy()
    test.assertGreater(float(np.linalg.norm(qd[:3])), 0.0)


def test_fem_particle_drag_projection_disabled_preserves_state(test, device):
    model = _build_single_tet_model(device, gravity=0.0)
    solver = newton.solvers.SolverFEM(model, iterations=1, cg_tol=1.0e-7, cg_max_iters=128)
    target = wp.array([wp.vec3(1.0 / 3.0, 1.0 / 3.0, 0.4)], dtype=wp.vec3, device=device)
    picked_point = wp.zeros(1, dtype=wp.vec3, device=device)
    solver.set_particle_drag_constraint(
        wp.array([0, 1, 2], dtype=wp.int32, device=device),
        wp.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=float, device=device),
        target,
        picked_point,
    )

    state_in = model.state()
    state_out = model.state()
    solver.step(state_in, state_out, None, None, 0.1)

    np.testing.assert_allclose(state_out.particle_q.numpy(), model.particle_q.numpy(), rtol=0.0, atol=1.0e-7)
    np.testing.assert_allclose(state_out.particle_qd.numpy(), 0.0, rtol=0.0, atol=1.0e-7)


def test_fem_cuda_graph_capture_no_fixed_particles(test, device):
    device = wp.get_device(device)
    if not device.is_cuda:
        test.skipTest("CUDA graph capture requires a CUDA device")

    model = _build_single_tet_model(device, gravity=-9.81)
    solver = newton.solvers.SolverFEM(model, iterations=1, cg_tol=1.0e-7, cg_max_iters=128)
    state_in = model.state()
    state_out = model.state()

    with wp.ScopedCapture(device=device) as capture:
        solver.step(state_in, state_out, None, None, 0.05)
    wp.capture_launch(capture.graph)

    q = state_out.particle_q.numpy()
    test.assertTrue(np.all(np.isfinite(q)))
    test.assertLess(q[:, 2].min(), model.particle_q.numpy()[:, 2].min())


devices = get_test_devices(mode="basic")
add_function_test(TestSolverFEM, "test_fem_rest_stability", test_fem_rest_stability, devices=devices)
add_function_test(
    TestSolverFEM,
    "test_fem_gravity_preserves_fixed_particles",
    test_fem_gravity_preserves_fixed_particles,
    devices=devices,
)
add_function_test(
    TestSolverFEM,
    "test_fem_newton_iterations_update_displacement",
    test_fem_newton_iterations_update_displacement,
    devices=devices,
)
add_function_test(
    TestSolverFEM,
    "test_fem_material_refresh_changes_deformation",
    test_fem_material_refresh_changes_deformation,
    devices=devices,
)
add_function_test(
    TestSolverFEM,
    "test_fem_global_damping_damps_velocity",
    test_fem_global_damping_damps_velocity,
    devices=devices,
)
add_function_test(
    TestSolverFEM,
    "test_fem_plane_projection_clamps_ground_penetration",
    test_fem_plane_projection_clamps_ground_penetration,
    devices=devices,
)
add_function_test(
    TestSolverFEM,
    "test_fem_plane_projection_preserves_fixed_particles",
    test_fem_plane_projection_preserves_fixed_particles,
    devices=devices,
)
add_function_test(
    TestSolverFEM,
    "test_fem_plane_projection_disabled_preserves_penetration",
    test_fem_plane_projection_disabled_preserves_penetration,
    devices=devices,
)
add_function_test(
    TestSolverFEM,
    "test_fem_particle_drag_projection_moves_picked_point",
    test_fem_particle_drag_projection_moves_picked_point,
    devices=devices,
)
add_function_test(
    TestSolverFEM,
    "test_fem_particle_drag_projection_disabled_preserves_state",
    test_fem_particle_drag_projection_disabled_preserves_state,
    devices=devices,
)
add_function_test(
    TestSolverFEM,
    "test_fem_cuda_graph_capture_no_fixed_particles",
    test_fem_cuda_graph_capture_no_fixed_particles,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
