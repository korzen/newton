# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Cable Drop XPBD
#
# A minimal XPBD cable debugging scene:
# - one straight cable built with ModelBuilder.add_rod()
# - one pinned segment
# - gravity + ground contact
# - no twist drive, no obstacles, no junctions
#
# Command: python -m newton.examples cable_drop_xpbd
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples


@wp.kernel
def _apply_rest_rotation_offset(
    cable_joint_ids: wp.array[int],
    base_rest_rotation: wp.array[wp.quat],
    rest_offset: wp.quat,
    rest_rotation_out: wp.array[wp.quat],
):
    tid = wp.tid()
    joint_id = cable_joint_ids[tid]
    rest_rotation_out[joint_id] = wp.normalize(wp.mul(rest_offset, base_rest_rotation[joint_id]))


@wp.kernel
def _apply_cable_material_controls(
    cable_joint_ids: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_dof_dim: wp.array2d[int],
    stretch_stiffness: float,
    stretch_damping: float,
    bend_stiffness: float,
    bend_damping: float,
    segment_length: float,
    joint_target_ke: wp.array[float],
    joint_target_kd: wp.array[float],
):
    tid = wp.tid()
    joint_id = cable_joint_ids[tid]
    axis_start = joint_qd_start[joint_id]
    lin_axis_count = joint_dof_dim[joint_id, 0]
    ang_axis_count = joint_dof_dim[joint_id, 1]
    inv_segment_length = 1.0 / wp.max(segment_length, 1.0e-9)

    if lin_axis_count > 0:
        stretch_idx = axis_start
        joint_target_ke[stretch_idx] = stretch_stiffness * inv_segment_length
        joint_target_kd[stretch_idx] = stretch_damping

    if ang_axis_count > 0:
        bend_idx = axis_start + lin_axis_count
        joint_target_ke[bend_idx] = bend_stiffness * inv_segment_length
        joint_target_kd[bend_idx] = bend_damping


class Example:
    def __init__(
        self,
        viewer,
        args,
        segments: int = 64,
        length: float = 1.0,
        height: float = 1.0,
        radius: float = 0.02,
        bend_stiffness: float = 1.0e1,
        bend_damping: float = 0.0,
        stretch_stiffness: float = 1.0e6,
        stretch_damping: float = 0.0,
        iterations: int = 8,
    ):
        self.viewer = viewer
        self.args = args

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 8
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.length = float(length)
        self.height = float(height)
        self.radius = float(radius)
        self.segments = int(segments)
        self.segment_length = self.length / max(self.segments, 1)
        self.bend_stiffness = float(bend_stiffness)
        self.bend_damping = float(bend_damping)
        self.stretch_stiffness = float(stretch_stiffness)
        self.stretch_damping = float(stretch_damping)
        self.bend_stiffness_log10 = self._value_to_log10(self.bend_stiffness)
        self.stretch_stiffness_log10 = self._value_to_log10(self.stretch_stiffness)

        builder = newton.ModelBuilder()
        builder.default_shape_cfg.ke = 1.0e4
        builder.default_shape_cfg.kd = 1.0e-1
        builder.default_shape_cfg.mu = 0.5

        points, quaternions = newton.utils.create_straight_cable_points_and_quaternions(
            start=wp.vec3(0.0, 0.0, self.height),
            direction=wp.vec3(1.0, 0.0, 0.0),
            length=self.length,
            num_segments=self.segments,
            twist_total=0.0,
        )

        self.rod_bodies, self.rod_joints = builder.add_rod(
            positions=points,
            quaternions=quaternions,
            radius=self.radius,
            bend_stiffness=self.bend_stiffness,
            bend_damping=self.bend_damping,
            stretch_stiffness=self.stretch_stiffness,
            stretch_damping=self.stretch_damping,
            label="xpbd_drop_cable",
        )

        self.pinned_body = self.rod_bodies[0]
        builder.body_flags[self.pinned_body] = int(newton.BodyFlags.KINEMATIC)

        builder.add_ground_plane()
        builder.color()

        sim_device = wp.get_device(args.device) if args.device else None
        self.model = builder.finalize(device=sim_device)
        self.model.set_gravity((0.0, 0.0, -9.81))

        self.solver = newton.solvers.SolverXPBD(
            self.model,
            iterations=int(iterations),
            angular_damping=0.1,
        )

        


        self._cable_joint_ids = np.where(self.model.joint_type.numpy() == int(newton.JointType.CABLE))[0].astype(
            np.int32
        )
        self._cable_joint_ids_wp = wp.array(self._cable_joint_ids, dtype=int, device=self.model.device)
        self._base_cable_rest_rotation = wp.clone(self.solver.cable_joint_rest_rotation)

        self.rest_bend_x_deg = float(getattr(args, "rest_bend_x", 0.0))
        self.rest_bend_y_deg = float(getattr(args, "rest_bend_y", 0.0))
        self.rest_bend_z_deg = float(getattr(args, "rest_bend_z", 0.0))
        self._apply_rest_bend_offset()
        self._apply_cable_material_controls()

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        if self.state_0.body_q is None:
            raise RuntimeError("Body state is not available.")
        self._pinned_body_q0 = self.state_0.body_q.numpy()[self.pinned_body].copy()

        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(
                pos=wp.vec3(2.2, -2.0, 1.4),
                pitch=-10.0,
                yaw=135.0,
            )

        self.capture()

    @staticmethod
    def _value_to_log10(value: float) -> float:
        return float(np.log10(max(float(value), 1.0e-12)))

    def capture(self):
        if self.solver.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def _make_rest_offset_quaternion(self) -> wp.quat:
        x_rad = np.deg2rad(self.rest_bend_x_deg)
        y_rad = np.deg2rad(self.rest_bend_y_deg)
        z_rad = np.deg2rad(self.rest_bend_z_deg)

        qx = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), float(x_rad))
        qy = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), float(y_rad))
        qz = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(z_rad))
        return wp.normalize(wp.mul(qz, wp.mul(qy, qx)))

    def _apply_rest_bend_offset(self) -> None:
        if self._cable_joint_ids_wp.shape[0] == 0:
            return
        wp.launch(
            kernel=_apply_rest_rotation_offset,
            dim=self._cable_joint_ids_wp.shape[0],
            inputs=[
                self._cable_joint_ids_wp,
                self._base_cable_rest_rotation,
                self._make_rest_offset_quaternion(),
            ],
            outputs=[self.solver.cable_joint_rest_rotation],
            device=self.model.device,
        )

    def _apply_cable_material_controls(self) -> None:
        if self._cable_joint_ids_wp.shape[0] == 0:
            return
        wp.launch(
            kernel=_apply_cable_material_controls,
            dim=self._cable_joint_ids_wp.shape[0],
            inputs=[
                self._cable_joint_ids_wp,
                self.model.joint_qd_start,
                self.model.joint_dof_dim,
                self.stretch_stiffness,
                self.stretch_damping,
                self.bend_stiffness,
                self.bend_damping,
                self.segment_length,
            ],
            outputs=[self.model.joint_target_ke, self.model.joint_target_kd],
            device=self.model.device,
        )

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(
                self.state_0,
                self.state_1,
                self.control,
                self.contacts,
                self.sim_dt,
            )
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def gui(self, ui):
        changed_bend_ke, self.bend_stiffness_log10 = ui.slider_float(
            "Bend Stiffness [log10]", self.bend_stiffness_log10, -6.0, 12.0, "%.2f"
        )
        changed_stretch_ke, self.stretch_stiffness_log10 = ui.slider_float(
            "Stretch Stiffness [log10]", self.stretch_stiffness_log10, -6.0, 12.0, "%.2f"
        )
        changed_bend_kd, self.bend_damping = ui.slider_float(
            "Bend Damping", self.bend_damping, 0.0, 1.0, "%.4f"
        )
        changed_stretch_kd, self.stretch_damping = ui.slider_float(
            "Stretch Damping", self.stretch_damping, 0.0, 1.0, "%.4f"
        )
        if changed_bend_ke or changed_stretch_ke or changed_bend_kd or changed_stretch_kd:
            self.bend_stiffness = float(10.0**self.bend_stiffness_log10)
            self.stretch_stiffness = float(10.0**self.stretch_stiffness_log10)
            self._apply_cable_material_controls()

        ui.separator()
        ui.text("Rest bend offset applied to all cable joints")
        changed_x, self.rest_bend_x_deg = ui.slider_float("Rest X [deg]", self.rest_bend_x_deg, -180.0, 180.0)
        changed_y, self.rest_bend_y_deg = ui.slider_float("Rest Y [deg]", self.rest_bend_y_deg, -180.0, 180.0)
        changed_z, self.rest_bend_z_deg = ui.slider_float("Rest Z [deg]", self.rest_bend_z_deg, -180.0, 180.0)
        if changed_x or changed_y or changed_z:
            self._apply_rest_bend_offset()

        if ui.button("Reset Rest Bend"):
            self.rest_bend_x_deg = 0.0
            self.rest_bend_y_deg = 0.0
            self.rest_bend_z_deg = 0.0
            self._apply_rest_bend_offset()

    def test_final(self):
        if self.state_0.body_q is None or self.state_0.body_qd is None:
            raise RuntimeError("Body state is not available.")

        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()

        if not np.isfinite(body_q).all():
            raise ValueError("Non-finite values in cable body transforms.")
        if not np.isfinite(body_qd).all():
            raise ValueError("Non-finite values in cable body velocities.")

        pinned_q = body_q[self.pinned_body]
        if np.max(np.abs(pinned_q[:3] - self._pinned_body_q0[:3])) > 1.0e-5:
            raise ValueError("Pinned cable segment moved unexpectedly.")

        positions = body_q[:, :3]
        z_positions = positions[:, 2]
        if np.min(z_positions) > self.height - 0.2:
            raise ValueError("Cable did not visibly drop under gravity.")
        if np.min(z_positions) < -0.25:
            raise ValueError("Cable penetrated the ground too much.")
        if np.max(np.abs(positions)) > 10.0:
            raise ValueError("Cable drifted out of bounds.")

        rod_positions = positions[self.rod_bodies]
        consecutive_distances = np.linalg.norm(np.diff(rod_positions, axis=0), axis=1)
        if np.max(consecutive_distances) > 3.0 * self.segment_length:
            raise ValueError("Cable segments separated too far from one another.")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--segments", type=int, default=32, help="Number of cable segments.")
        parser.add_argument("--length", type=float, default=1.2, help="Cable length [m].")
        parser.add_argument("--height", type=float, default=1.0, help="Pinned cable start height [m].")
        parser.add_argument("--radius", type=float, default=0.02, help="Cable capsule radius [m].")
        parser.add_argument("--bend-stiffness", type=float, default=1.0e6, help="Cable bend stiffness.")
        parser.add_argument("--bend-damping", type=float, default=0.0, help="Cable bend damping.")
        parser.add_argument("--stretch-stiffness", type=float, default=1.0e9, help="Cable stretch stiffness.")
        parser.add_argument("--stretch-damping", type=float, default=0.0, help="Cable stretch damping.")
        parser.add_argument("--iterations", type=int, default=8, help="XPBD solver iterations per substep.")
        parser.add_argument("--rest-bend-x", type=float, default=0.0, help="Initial rest bend rotation around X [deg].")
        parser.add_argument("--rest-bend-y", type=float, default=0.0, help="Initial rest bend rotation around Y [deg].")
        parser.add_argument("--rest-bend-z", type=float, default=0.0, help="Initial rest bend rotation around Z [deg].")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)

    example = Example(
        viewer,
        args,
        segments=args.segments,
        length=args.length,
        height=args.height,
        radius=args.radius,
        bend_stiffness=args.bend_stiffness,
        bend_damping=args.bend_damping,
        stretch_stiffness=args.stretch_stiffness,
        stretch_damping=args.stretch_damping,
        iterations=args.iterations,
    )

    newton.examples.run(example, args)
