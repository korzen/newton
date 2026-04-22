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

###########################################################################
# Example Elastic Rod XPBD
#
# A horizontal Cosserat elastic rod fixed at one end, bending under gravity.
# Uses the base XPBD solver with particle positions plus material-frame
# quaternions stored in the xpbd custom namespace.
#
# Command: uv run -m newton.examples elastic_rod_xpbd
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.solvers
from newton.examples.elastic_rod.rod_mesher import RodMesher
from newton.solvers import xpbd_rod


@wp.kernel
def _apply_floor_collision(
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    floor_z: float,
) -> None:
    tid = wp.tid()
    position = particle_q[tid]
    if position[2] < floor_z:
        particle_q[tid] = wp.vec3(position[0], position[1], floor_z)
        velocity = particle_qd[tid]
        if velocity[2] < 0.0:
            particle_qd[tid] = wp.vec3(velocity[0], velocity[1], 0.0)


class Example:
    def __init__(self, viewer, args=None):
        self.viewer = viewer
        self.sim_time = 0.0
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 8
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.bend_stiffness = 0.5
        self.twist_stiffness = 0.5
        self.young_modulus = 1.0e6
        self.torsion_modulus = 1.0e6
        self.rest_bend_d1 = 0.0
        self.rest_bend_d2 = 0.0
        self.rest_twist = 0.0
        self.rest_length = 0.05
        self.lock_root = True
        self.lock_root_rotation = True
        self.floor_z = 0.0

        num_points = 128
        spacing = 0.05

        builder = newton.ModelBuilder()
        newton.solvers.SolverXPBD.register_custom_attributes(builder)

        positions = np.zeros((num_points, 3), dtype=np.float32)
        for i in range(num_points):
            positions[i, 0] = i * spacing
            positions[i, 2] = 1.0

        xpbd_rod.add_elastic_rod(
            builder,
            positions=positions,
            radius=0.005,
            particle_mass=0.05,
            bend_stiffness=self.bend_stiffness,
            twist_stiffness=self.twist_stiffness,
            young_modulus=self.young_modulus,
            torsion_modulus=self.torsion_modulus,
            lock_root=self.lock_root,
            lock_root_rotation=self.lock_root_rotation,
        )

        builder.add_ground_plane()

        self.model = builder.finalize()
        # Disable the generic particle hash-grid/contact path so this example
        # tracks the standalone rod example more closely.
        self.model.particle_max_radius = 0.0
        self.model.particle_grid = None
        self.solver = newton.solvers.SolverXPBD(
            model=self.model,
            iterations=1,
            angular_damping=0.001,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        self._compact_start = int(self.model.xpbd.particle_start.numpy()[0])
        self._particle_count = int(self.model.xpbd.particle_count.numpy()[0])
        self._edge_start = int(self.model.xpbd.edge_start.numpy()[0])
        self._edge_count = int(self.model.xpbd.edge_count.numpy()[0])
        particle_indices = self.model.xpbd.particle_index.numpy()[
            self._compact_start : self._compact_start + self._particle_count
        ]
        self._particle_start = int(particle_indices[0])

        self.show_directors = False
        self.director_scale = 0.03
        device = self.model.device
        self._director_starts = wp.zeros(self._edge_count * 3, dtype=wp.vec3, device=device)
        self._director_ends = wp.zeros(self._edge_count * 3, dtype=wp.vec3, device=device)
        self._director_colors = wp.zeros(self._edge_count * 3, dtype=wp.vec3, device=device)

        self._mesher = RodMesher(
            num_points=self._particle_count,
            radius=0.01,
            resolution=8,
            smoothing=3,
            device=device,
        )

        particle_inv_mass = self.model.particle_inv_mass.numpy()
        quat_inv_mass = self.model.xpbd.quat_inv_mass.numpy()
        orientation = self.state_0.xpbd.orientation.numpy()
        self._root_q = orientation[self._compact_start].copy()
        self._free_inv_mass = float(particle_inv_mass[self._particle_start + 1]) if self._particle_count > 1 else 1.0
        self._free_quat_inv_mass = (
            float(quat_inv_mass[self._compact_start + 1]) if self._particle_count > 1 else 1.0
        )

        self.graph = None
        if device.is_cuda and wp.is_mempool_enabled(device):
            with wp.ScopedCapture(device=device) as capture:
                self._simulate_substeps()
            self.graph = capture.graph
            print("CUDA graph captured for simulation substeps")
        else:
            print("CUDA graph not available, using standard kernel launches")

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True
        if hasattr(self.viewer, "register_ui_callback"):
            self.viewer.register_ui_callback(self._render_ui, position="side")

    def _copy_float(self, array: wp.array, value: float, offset: int) -> None:
        wp.copy(
            dest=array,
            src=wp.array(np.array([value], dtype=np.float32), dtype=wp.float32, device=self.model.device),
            dest_offset=offset,
            count=1,
        )

    def _copy_quat(self, q: np.ndarray) -> None:
        q_array = wp.array(
            [wp.quat(float(q[0]), float(q[1]), float(q[2]), float(q[3]))],
            dtype=wp.quat,
            device=self.model.device,
        )
        wp.copy(dest=self.state_0.xpbd.orientation, src=q_array, dest_offset=self._compact_start, count=1)
        wp.copy(dest=self.state_1.xpbd.orientation, src=q_array, dest_offset=self._compact_start, count=1)

    def _zero_root_velocities(self) -> None:
        zero_vec = wp.array([wp.vec3(0.0, 0.0, 0.0)], dtype=wp.vec3, device=self.model.device)
        wp.copy(dest=self.state_0.particle_qd, src=zero_vec, dest_offset=self._particle_start, count=1)
        wp.copy(dest=self.state_1.particle_qd, src=zero_vec, dest_offset=self._particle_start, count=1)
        wp.copy(dest=self.state_0.xpbd.angular_velocity, src=zero_vec, dest_offset=self._compact_start, count=1)
        wp.copy(dest=self.state_1.xpbd.angular_velocity, src=zero_vec, dest_offset=self._compact_start, count=1)

    def _rotate_root(self) -> None:
        if not self.lock_root or not self.lock_root_rotation:
            return

        rotate_speed = 1.5
        dx = 0.0
        dz = 0.0

        if hasattr(self.viewer, "is_key_down"):
            if self.viewer.is_key_down("i"):
                dz -= rotate_speed * self.frame_dt
            if self.viewer.is_key_down("k"):
                dz += rotate_speed * self.frame_dt
            if self.viewer.is_key_down("j"):
                dx -= rotate_speed * self.frame_dt
            if self.viewer.is_key_down("l"):
                dx += rotate_speed * self.frame_dt

        if dx == 0.0 and dz == 0.0:
            return

        qx = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), float(dx))
        qz = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(dz))
        q_cur = wp.quat(float(self._root_q[0]), float(self._root_q[1]), float(self._root_q[2]), float(self._root_q[3]))
        q_wp = wp.normalize(wp.mul(qz, wp.mul(qx, q_cur)))
        q = np.array([q_wp[0], q_wp[1], q_wp[2], q_wp[3]], dtype=np.float32)
        self._root_q = q
        self._copy_quat(q)
        self._zero_root_velocities()

    def _simulate_substeps(self) -> None:
        for _ in range(self.sim_substeps):
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            wp.launch(
                _apply_floor_collision,
                dim=self._particle_count,
                inputs=[
                    self.state_1.particle_q[self._particle_start : self._particle_start + self._particle_count],
                    self.state_1.particle_qd[self._particle_start : self._particle_start + self._particle_count],
                    self.floor_z,
                ],
                device=self.model.device,
            )
            self.state_0, self.state_1 = self.state_1, self.state_0

    def simulate(self) -> None:
        self._rotate_root()
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self._simulate_substeps()

    def step(self) -> None:
        self.simulate()
        self.sim_time += self.frame_dt

    def _build_director_lines(self) -> None:
        wp.launch(
            xpbd_rod.compute_director_lines_kernel,
            dim=self._edge_count * 3,
            inputs=[
                self.state_0.particle_q[self._particle_start : self._particle_start + self._particle_count],
                self.state_0.xpbd.orientation[self._compact_start : self._compact_start + self._particle_count],
                self._edge_count,
                self.director_scale,
            ],
            outputs=[
                self._director_starts,
                self._director_ends,
                self._director_colors,
            ],
        )

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)

        self._mesher.update(self.state_0.particle_q[self._particle_start : self._particle_start + self._particle_count])
        self.viewer.log_mesh(
            "/rod_mesh_0",
            self._mesher.vertices,
            self._mesher.indices,
            self._mesher.normals,
            self._mesher.uvs,
        )

        if self.show_directors:
            self._build_director_lines()
            self.viewer.log_lines(
                "/rod_directors_0",
                self._director_starts,
                self._director_ends,
                self._director_colors,
                width=0.005,
            )
        else:
            self.viewer.log_lines("/rod_directors_0", None, None, None, hidden=True)

        self.viewer.end_frame()

    def _update_rod_stiffness(self) -> None:
        bend_stiffness = np.full(
            (self._edge_count, 3),
            [self.bend_stiffness, self.bend_stiffness, self.twist_stiffness],
            dtype=np.float32,
        )
        self.model.xpbd.bend_stiffness.assign(wp.array(bend_stiffness, dtype=wp.vec3, device=self.model.device))
        self.model.xpbd.young_modulus.assign(
            wp.array(np.array([self.young_modulus], dtype=np.float32), dtype=wp.float32, device=self.model.device)
        )
        self.model.xpbd.torsion_modulus.assign(
            wp.array(np.array([self.torsion_modulus], dtype=np.float32), dtype=wp.float32, device=self.model.device)
        )

    def _update_root_lock(self) -> None:
        root_inv_mass = 0.0 if self.lock_root else self._free_inv_mass
        self._copy_float(self.model.particle_inv_mass, root_inv_mass, self._particle_start)
        if self.solver._rod_projector is not None:
            self._copy_float(self.solver._rod_projector.inv_masses, root_inv_mass, self._compact_start)

        root_quat_inv_mass = 0.0 if self.lock_root_rotation else self._free_quat_inv_mass
        self._copy_float(self.model.xpbd.quat_inv_mass, root_quat_inv_mass, self._compact_start)

        root_orientation = self.state_0.xpbd.orientation.numpy()[self._compact_start]
        self._root_q = root_orientation.copy()
        self._zero_root_velocities()

    def _update_rest_lengths(self) -> None:
        rest_lengths = np.full(self._edge_count, self.rest_length, dtype=np.float32)
        self.model.xpbd.rest_length.assign(wp.array(rest_lengths, dtype=wp.float32, device=self.model.device))

    def _update_rest_darboux(self) -> None:
        rest_darboux = np.full(
            (self._edge_count, 3),
            [self.rest_bend_d1, self.rest_bend_d2, self.rest_twist],
            dtype=np.float32,
        )
        self.model.xpbd.rest_darboux.assign(wp.array(rest_darboux, dtype=wp.vec3, device=self.model.device))

    def _render_ui(self, imgui) -> None:
        changed_lock, self.lock_root = imgui.checkbox("Lock Root Position", self.lock_root)
        changed_lock_rot, self.lock_root_rotation = imgui.checkbox("Lock Root Rotation", self.lock_root_rotation)
        if changed_lock or changed_lock_rot:
            self._update_root_lock()

        imgui.separator()
        changed_bend, self.bend_stiffness = imgui.slider_float("Bend Stiffness", self.bend_stiffness, 0.0, 1.0)
        changed_twist, self.twist_stiffness = imgui.slider_float("Twist Stiffness", self.twist_stiffness, 0.0, 1.0)
        if changed_bend or changed_twist:
            self._update_rod_stiffness()

        imgui.separator()
        changed_E, self.young_modulus = imgui.input_float("Young Modulus [Pa]", self.young_modulus, format="%.1f")
        changed_G, self.torsion_modulus = imgui.input_float("Torsion Modulus [Pa]", self.torsion_modulus, format="%.1f")
        if changed_E or changed_G:
            self._update_rod_stiffness()

        imgui.separator()
        changed_rl, self.rest_length = imgui.slider_float("Rest Length", self.rest_length, 0.01, 0.1)
        if changed_rl:
            self._update_rest_lengths()

        imgui.separator()
        changed_d1, self.rest_bend_d1 = imgui.slider_float("Rest Bend d1", self.rest_bend_d1, -0.1, 0.1)
        changed_d2, self.rest_bend_d2 = imgui.slider_float("Rest Bend d2", self.rest_bend_d2, -0.1, 0.1)
        changed_tw, self.rest_twist = imgui.slider_float("Rest Twist", self.rest_twist, -0.1, 0.1)
        if changed_d1 or changed_d2 or changed_tw:
            self._update_rest_darboux()

        imgui.separator()
        _, self.show_directors = imgui.checkbox("Show Material Frames", self.show_directors)
        _, self.director_scale = imgui.slider_float("Director Scale", self.director_scale, 0.01, 0.1)

    def test_final(self) -> None:
        particle_q = self.state_0.particle_q.numpy()
        orientation_q = self.state_0.xpbd.orientation.numpy()
        assert np.all(np.isfinite(particle_q)), "Particle positions must stay finite"
        assert np.all(np.isfinite(orientation_q)), "Rod orientations must stay finite"


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer=viewer, args=args)
    newton.examples.run(example, args)
