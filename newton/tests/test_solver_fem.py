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
):
    builder = newton.ModelBuilder(gravity=gravity)
    builder.add_soft_mesh(
        pos=(0.0, 0.0, 0.0),
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


def _step_once(model, *, iterations: int, dt: float):
    solver = newton.solvers.SolverFEM(model, iterations=iterations, cg_tol=1.0e-7, cg_max_iters=128)
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


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
