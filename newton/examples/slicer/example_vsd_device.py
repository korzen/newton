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

import json
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.slicer.vtk_loader import load_vtk_unstructured_grid

PARTICLE_ACTIVE = wp.constant(int(newton.ParticleFlags.ACTIVE))
MESH_PATH = Path(__file__).resolve().parent / "vsd_device" / "mesh3.1.vtk"
VESSEL_MESH_PATH = Path(__file__).resolve().parent / "anatomy" / "RVOT1_Alterra_vessel.vtk"
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


def load_vtk_unstructured_tet_mesh(path: Path) -> newton.TetMesh:
    """Load tetrahedral cells from a legacy ASCII VTK unstructured grid.

    Thin wrapper around :func:`newton.examples.slicer.vtk_loader.load_vtk_unstructured_grid`
    that extracts the dataset's ``VTK_TETRA`` cells into a :class:`newton.TetMesh`.
    """
    return load_vtk_unstructured_grid(path).to_tet_mesh()


def load_vtk_unstructured_triangle_mesh(path: Path) -> newton.Mesh:
    """Load triangle geometry from a legacy ASCII VTK unstructured grid.

    Native triangle cells are used directly, while volumetric cells contribute
    their boundary triangles via
    :meth:`newton.examples.slicer.vtk_loader.VtkUnstructuredGrid.triangle_indices`.
    """
    grid = load_vtk_unstructured_grid(path)
    triangle_indices = grid.triangle_indices()
    if triangle_indices.size == 0:
        raise ValueError(f"No triangle surface could be extracted from '{path}'.")

    return newton.Mesh(
        vertices=grid.points,
        indices=triangle_indices.reshape(-1).astype(np.int32),
        compute_inertia=False,
    )


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


class StaticTriMeshParticleProjector:
    """Projects particles away from a static triangle mesh using a Warp mesh BVH."""

    def __init__(self, mesh: newton.Mesh, pos: wp.vec3, scale: float, device):
        offset = np.array([float(pos[0]), float(pos[1]), float(pos[2])], dtype=np.float32)
        world_vertices = mesh.vertices * np.float32(scale) + offset

        self.points = wp.array(world_vertices, dtype=wp.vec3, device=device)
        self.indices = wp.array(mesh.indices, dtype=wp.int32, device=device)
        self.mesh = wp.Mesh(points=self.points, indices=self.indices)

    def project(self, model: newton.Model, state: newton.State, contact_radius: float, relaxation: float, dt: float):
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

        self.iterations = 4 if self.solver_type == "fem" else 8
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
        vessel_mesh = load_vtk_unstructured_triangle_mesh(VESSEL_MESH_PATH)
        self.mesh_scale = 0.05 * float(args.scale)
        self.anatomy_scale = 0.05 * float(args.anatomy_scale)
        ox, oy, oz = (float(v) for v in args.offset)
        self.mesh_pos = wp.vec3(0.0 + ox, 0.0 + oy, 0.45 + oz)

        builder.add_shape_mesh(
            body=-1,
            xform=wp.transform(self.mesh_pos, wp.quat_identity()),
            mesh=vessel_mesh,
            scale=(self.anatomy_scale, self.anatomy_scale, self.anatomy_scale),
            cfg=newton.ModelBuilder.ShapeConfig(
                density=0.0,
                has_shape_collision=False,
                has_particle_collision=False,
            ),
            color=(0.55, 0.18, 0.16),
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
            particle_radius=0.02,
        )
        self.device_particle_count = builder.particle_count - self.device_particle_start

        # Color the mesh for VBD solver
        builder.color()

        self.model = builder.finalize()
        self.model.soft_contact_ke = 1.0e2
        self.model.soft_contact_kd = 0
        self.model.soft_contact_mu = 1.0
        self._gravity_default = self.model.gravity.numpy().copy()
        device_particle_q = self.model.particle_q.numpy()[
            self.device_particle_start : self.device_particle_start + self.device_particle_count
        ]
        device_rest_center = np.mean(device_particle_q, axis=0)
        device_rest_local_q = device_particle_q - device_rest_center

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

        self.contacts = self.model.contacts()

        self.viewer.set_model(self.model)
        self._connect_particle_drag_projection()
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(wp.vec3(1.8, -2.0, 1.1), -18.0, 132.0)

        self.capture()

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
        # The FEM solver assembles BSR matrices and runs an iterative CG
        # solve with data-dependent residual checks each step; neither is
        # capturable into a static CUDA graph.
        if wp.get_device().is_cuda and self.solver_type != "fem":
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

            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self._project_vessel_particle_contacts(self.state_1)

            # swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _project_vessel_particle_contacts(self, state: newton.State):
        for _ in range(self.vessel_contact_iterations):
            self.vessel_particle_projector.project(
                self.model,
                state,
                self.vessel_contact_radius,
                self.vessel_contact_relaxation,
                self.sim_dt,
            )

    def step(self):
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
            self.viewer.log_gizmo("vsd_device", self.device_transform_gizmo.transform)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def _handle_global_keys(self):
        gravity_down = bool(self.viewer.is_key_down("g"))
        if gravity_down and not self._gravity_key_was_down:
            self._toggle_gravity()
        self._gravity_key_was_down = gravity_down

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
        return hasattr(self.viewer, "log_gizmo") and self.viewer.is_paused()

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

        self.device_transform_gizmo.apply_delta_if_changed(self.model, (self.state_0, self.state_1))
        self._handle_pause_device_keys()
        self.device_transform_gizmo._was_paused = True

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

        changed, value = ui.slider_float(
            "Vessel Contact Radius",
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
            "Vessel Contact Relaxation",
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
            "Vessel Contact Iterations",
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
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
