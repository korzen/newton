# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for elastic rod constraints integrated within SolverXPBD."""

import unittest

import numpy as np

import newton
import newton.solvers
from newton.solvers import xpbd_rod


def _make_rod_model_xpbd(
    num_points=20,
    spacing=0.05,
    height=1.0,
    particle_mass=0.05,
    bend_stiffness=0.1,
    twist_stiffness=0.1,
    young_modulus=1.0e4,
    torsion_modulus=1.0e4,
    gravity=(0.0, 0.0, -9.81),
    add_ground=False,
):
    builder = newton.ModelBuilder()
    newton.solvers.SolverXPBD.register_custom_attributes(builder)

    positions = np.zeros((num_points, 3), dtype=np.float32)
    for i in range(num_points):
        positions[i, 0] = i * spacing
        positions[i, 2] = height

    xpbd_rod.add_elastic_rod(
        builder,
        positions=positions,
        particle_mass=particle_mass,
        bend_stiffness=bend_stiffness,
        twist_stiffness=twist_stiffness,
        young_modulus=young_modulus,
        torsion_modulus=torsion_modulus,
    )

    if add_ground:
        builder.add_ground_plane()

    model = builder.finalize()
    if gravity != (0.0, 0.0, -9.81):
        model.set_gravity(gravity)
    return model


def _step_n(solver, model, n_steps, dt=0.001):
    s0 = model.state()
    s1 = model.state()
    ctrl = model.control()
    cont = model.contacts()
    for _ in range(n_steps):
        solver.step(s0, s1, ctrl, cont, dt)
        s0, s1 = s1, s0
    return s0


class TestSolverXPBDRodIntegrated(unittest.TestCase):
    """Test elastic rod constraints running inside SolverXPBD."""

    def test_rod_under_gravity(self):
        """Rod tip descends under gravity when using SolverXPBD."""
        model = _make_rod_model_xpbd()
        solver = newton.solvers.SolverXPBD(
            model=model,
            iterations=2,
            rod_linear_damping=0.01,
            rod_angular_damping=0.01,
        )
        state = _step_n(solver, model, 600)
        q = state.particle_q.numpy()
        self.assertFalse(np.any(np.isnan(q)), "NaN in particle positions")
        self.assertLess(q[-1, 2], 1.0, f"Tip should descend: z={q[-1, 2]:.4f}")
        self.assertAlmostEqual(q[0, 2], 1.0, places=2, msg=f"Root should be fixed: z={q[0, 2]:.4f}")

    def test_zero_gravity_rest(self):
        """Rod stays at rest with zero gravity when using SolverXPBD."""
        model = _make_rod_model_xpbd(num_points=10, gravity=(0.0, 0.0, 0.0))
        solver = newton.solvers.SolverXPBD(model=model, iterations=2)
        state = _step_n(solver, model, 100)
        q = state.particle_q.numpy()
        self.assertFalse(np.any(np.isnan(q)), "NaN in particle positions")
        max_dev = np.max(np.abs(q[:, 2] - 1.0))
        self.assertLess(max_dev, 0.01, f"Max deviation from rest: {max_dev}")

    def test_no_nan_inf(self):
        """No NaN or Inf after N steps."""
        model = _make_rod_model_xpbd()
        solver = newton.solvers.SolverXPBD(
            model=model,
            iterations=2,
            rod_linear_damping=0.01,
            rod_angular_damping=0.01,
        )
        state = _step_n(solver, model, 200)
        q = state.particle_q.numpy()
        self.assertFalse(np.any(np.isnan(q)), "NaN detected")
        self.assertFalse(np.any(np.isinf(q)), "Inf detected")

    def test_multi_rod(self):
        """Two rods in one model using SolverXPBD don't interfere."""
        builder = newton.ModelBuilder()
        newton.solvers.SolverXPBD.register_custom_attributes(builder)

        for rod_idx in range(2):
            pos = np.zeros((10, 3), dtype=np.float32)
            for i in range(10):
                pos[i, 0] = i * 0.05
                pos[i, 1] = rod_idx * 0.5
                pos[i, 2] = 1.0
            xpbd_rod.add_elastic_rod(
                builder,
                positions=pos,
                particle_mass=0.05,
                bend_stiffness=0.1,
                twist_stiffness=0.1,
                young_modulus=1.0e4,
                torsion_modulus=1.0e4,
            )

        model = builder.finalize()
        solver = newton.solvers.SolverXPBD(
            model=model,
            iterations=2,
            rod_linear_damping=0.01,
            rod_angular_damping=0.01,
        )
        self.assertIsNotNone(solver._rod_constraints)
        self.assertEqual(len(solver._rod_constraints._rods), 2)

        state = _step_n(solver, model, 100)
        q = state.particle_q.numpy()
        self.assertFalse(np.any(np.isnan(q)), "NaN in multi-rod")
        self.assertEqual(q.shape[0], 20)

    def test_rod_with_ground_plane(self):
        """Rod interacts with a ground plane via XPBD contacts."""
        model = _make_rod_model_xpbd(
            height=0.3,
            bend_stiffness=0.01,
            add_ground=True,
        )
        solver = newton.solvers.SolverXPBD(
            model=model,
            iterations=2,
            rod_linear_damping=0.01,
            rod_angular_damping=0.01,
        )

        state = _step_n(solver, model, 500)
        q = state.particle_q.numpy()
        self.assertFalse(np.any(np.isnan(q)), "NaN in particle positions")
        self.assertTrue(np.all(np.isfinite(q)), "Non-finite values in particle positions")

    def test_no_rod_data_no_crash(self):
        """SolverXPBD works normally when model has no rod data."""
        builder = newton.ModelBuilder()
        builder.add_particle(pos=(0.0, 0.0, 1.0), vel=(0.0, 0.0, 0.0), mass=1.0)
        model = builder.finalize()
        solver = newton.solvers.SolverXPBD(model=model, iterations=2)
        self.assertIsNone(solver._rod_constraints)
        state = _step_n(solver, model, 10)
        q = state.particle_q.numpy()
        self.assertFalse(np.any(np.isnan(q)))

    def test_stiffness_ordering(self):
        """Stiffer rods droop less than softer ones under gravity."""
        builder = newton.ModelBuilder()
        newton.solvers.SolverXPBD.register_custom_attributes(builder)

        stiffness_values = [0.001, 1.0]
        for i, bs in enumerate(stiffness_values):
            pos = np.zeros((20, 3), dtype=np.float32)
            for j in range(20):
                pos[j, 0] = j * 0.05
                pos[j, 1] = i * 0.5
                pos[j, 2] = 1.0
            xpbd_rod.add_elastic_rod(
                builder,
                positions=pos,
                particle_mass=0.05,
                bend_stiffness=bs,
                twist_stiffness=0.1,
                young_modulus=1.0e4,
                torsion_modulus=1.0e4,
            )

        model = builder.finalize()
        solver = newton.solvers.SolverXPBD(
            model=model,
            iterations=2,
            rod_linear_damping=0.01,
            rod_angular_damping=0.01,
        )

        state = _step_n(solver, model, 600)
        q = state.particle_q.numpy()

        soft_tip_z = q[19, 2]
        stiff_tip_z = q[39, 2]
        self.assertLess(
            soft_tip_z, stiff_tip_z + 0.05,
            f"Soft rod tip z={soft_tip_z:.4f} should be lower than stiff rod tip z={stiff_tip_z:.4f}",
        )


if __name__ == "__main__":
    unittest.main()
