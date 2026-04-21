# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import warp as wp

from ...core.types import override
from ...sim import Contacts, Control, Model, ModelBuilder, State
from ..flags import SolverNotifyFlags
from ..solver import SolverBase
from .kernels import (
    accumulate_weighted_contact_impulse,
    apply_body_delta_velocities,
    apply_body_deltas,
    apply_joint_forces,
    apply_particle_deltas,
    apply_particle_shape_restitution,
    apply_rigid_restitution,
    bending_constraint,
    convert_contact_impulse_to_force,
    copy_kinematic_body_state_kernel,
    solve_body_contact_positions,
    solve_body_joints,
    solve_particle_particle_contacts,
    solve_particle_shape_contacts,
    # solve_simple_body_joints,
    solve_springs,
    solve_tetrahedra,
    update_body_velocities,
)

# ---------------------------------------------------------------------------
# Elastic rod imports (reuse workspace classes and kernels from xpbd_rod)
# ---------------------------------------------------------------------------
from ..xpbd_rod.solver_xpbd_rod import _BatchedRodWorkspace, _RodWorkspace
from ..xpbd_rod.constants import (
    BAND_LDAB,
    BLOCK_DIM,
    DIRECT_SOLVE_BACKENDS,
    DIRECT_SOLVE_BANDED_CHOLESKY,
    DIRECT_SOLVE_BLOCK_JACOBI,
    DIRECT_SOLVE_BLOCK_THOMAS,
    DIRECT_SOLVE_SPLIT_THOMAS,
    TILE,
)
from ..xpbd_rod.kernels_assembly import (
    _warp_assemble_darboux_blocks,
    _warp_assemble_jmjt_banded,
    _warp_assemble_jmjt_blocks,
    _warp_assemble_jmjt_blocks_batched,
    _warp_assemble_jmjt_dense,
    _warp_assemble_stretch_blocks,
    _warp_compute_inv_inertia_world_batched,
    _warp_pad_diagonal,
)
from ..xpbd_rod.kernels_collision import (
    _warp_compute_corrections_parallel,
    _warp_compute_corrections_parallel_batched,
    _warp_compute_inv_inertia_world,
    _warp_merge_delta_lambda,
    _warp_zero_2d as _rod_zero_2d,
    _warp_zero_float as _rod_zero_float,
    _warp_zero_vec3 as _rod_zero_vec3,
)
from ..xpbd_rod.kernels_constraints import (
    _warp_build_rhs,
    _warp_build_rhs_darboux,
    _warp_build_rhs_stretch,
    _warp_compute_jacobians_batched,
    _warp_compute_jacobians_direct,
    _warp_prepare_compliance,
    _warp_prepare_compliance_batched,
    _warp_update_constraints_batched_v2,
    _warp_update_constraints_direct,
)
from ..xpbd_rod.kernels_integration import (
    _warp_integrate_rotations,
    _warp_integrate_rotations_batched,
    _warp_predict_rotations,
    _warp_predict_rotations_batched,
)
from ..xpbd_rod.kernels_solvers import (
    _warp_block_thomas_solve,
    _warp_block_thomas_solve_3x3,
    _warp_block_thomas_solve_batched,
    _warp_cholesky_solve_tile,
    _warp_solve_blocks_jacobi,
    _warp_spbsv_u11_1rhs,
)


# ---------------------------------------------------------------------------
# Glue kernels for integrating rod corrections into the XPBD particle deltas
# ---------------------------------------------------------------------------


@wp.func
def _quat_correction(q: wp.quat, dtheta: wp.vec3) -> wp.quat:
    """Apply a small angular correction *dtheta* to quaternion *q*."""
    norm_sq = dtheta[0] * dtheta[0] + dtheta[1] * dtheta[1] + dtheta[2] * dtheta[2]
    if norm_sq < 1.0e-20:
        return q
    cx = 0.5 * (q[3] * dtheta[0] + q[2] * dtheta[1] - q[1] * dtheta[2])
    cy = 0.5 * (-q[2] * dtheta[0] + q[3] * dtheta[1] + q[0] * dtheta[2])
    cz = 0.5 * (q[1] * dtheta[0] - q[0] * dtheta[1] + q[3] * dtheta[2])
    cw = 0.5 * (-q[0] * dtheta[0] - q[1] * dtheta[1] - q[2] * dtheta[2])
    return wp.normalize(wp.quat(q[0] + cx, q[1] + cy, q[2] + cz, q[3] + cw))


@wp.kernel
def _add_rod_corrections_to_particle_deltas(
    pos_corrections: wp.array(dtype=wp.vec3),
    particle_deltas: wp.array(dtype=wp.vec3),
    particle_start: int,
    count: int,
):
    """Add rod position corrections into the global particle_deltas array."""
    tid = wp.tid()
    if tid < count:
        wp.atomic_add(particle_deltas, particle_start + tid, pos_corrections[tid])


@wp.kernel
def _apply_rod_orientation_corrections(
    predicted_orientations: wp.array(dtype=wp.quat),
    rot_corrections: wp.array(dtype=wp.vec3),
    count: int,
):
    """Apply rotation corrections to rod predicted orientations (orientation-only)."""
    tid = wp.tid()
    if tid < count:
        predicted_orientations[tid] = _quat_correction(predicted_orientations[tid], rot_corrections[tid])


# ---------------------------------------------------------------------------
# _ElasticRodConstraints -- component that manages rod state inside SolverXPBD
# ---------------------------------------------------------------------------


class _ElasticRodConstraints:
    """Manages Cosserat elastic rod constraint solving within :class:`SolverXPBD`.

    This is an internal component; users interact with it indirectly through
    :meth:`SolverXPBD.register_custom_attributes` and the ``rod_*`` constructor
    parameters.
    """

    def __init__(
        self,
        model: Model,
        solver_backend: str = DIRECT_SOLVE_BLOCK_THOMAS,
        linear_damping: float = 0.0,
        angular_damping: float = 0.0,
    ):
        if solver_backend not in DIRECT_SOLVE_BACKENDS:
            raise ValueError(
                f"Unknown rod solver backend {solver_backend!r}. "
                f"Expected one of {DIRECT_SOLVE_BACKENDS}"
            )

        self.solver_backend = solver_backend
        self.linear_damping = linear_damping
        self.angular_damping = angular_damping

        device = model.device

        self._rods: list[_RodWorkspace] = []
        self._rod_particle_starts: list[int] = []
        self._batched_ws: _BatchedRodWorkspace | None = None

        rod_data = model.xpbd_rod
        rod_num_points = rod_data["rod_num_points"]
        rod_particle_starts = rod_data["rod_particle_start"]
        rod_young_moduli = rod_data["rod_young_modulus"]
        rod_torsion_moduli = rod_data["rod_torsion_modulus"]

        all_orientations = rod_data["orientations"]
        all_quat_inv_masses = rod_data["quat_inv_masses"]
        all_rest_lengths = rod_data["rest_lengths"]
        all_rest_darboux = rod_data["rest_darboux"]
        all_bend_stiffness = rod_data["bend_stiffness"]

        orient_cursor = 0
        edge_cursor = 0

        for rod_idx in range(len(rod_num_points)):
            np_ = rod_num_points[rod_idx]
            ne = np_ - 1
            ps = rod_particle_starts[rod_idx]

            ws = _RodWorkspace(np_, ne, device)
            ws.young_modulus = rod_young_moduli[rod_idx]
            ws.torsion_modulus = rod_torsion_moduli[rod_idx]

            wp.copy(dest=ws.positions_wp, src=model.particle_q, dest_offset=0, src_offset=ps, count=np_)
            wp.copy(dest=ws.predicted_positions_wp, src=model.particle_q, dest_offset=0, src_offset=ps, count=np_)
            wp.copy(dest=ws.inv_masses_wp, src=model.particle_inv_mass, dest_offset=0, src_offset=ps, count=np_)

            orient_slice = np.array(all_orientations[orient_cursor : orient_cursor + np_], dtype=np.float32)
            ws.orientations_wp.assign(wp.array(orient_slice, dtype=wp.quat, device=device))
            ws.predicted_orientations_wp.assign(wp.array(orient_slice, dtype=wp.quat, device=device))
            ws.prev_orientations_wp.assign(wp.array(orient_slice, dtype=wp.quat, device=device))

            qim_slice = np.array(all_quat_inv_masses[orient_cursor : orient_cursor + np_], dtype=np.float32)
            ws.quat_inv_masses_wp.assign(wp.array(qim_slice, dtype=wp.float32, device=device))

            rl_slice = np.array(all_rest_lengths[edge_cursor : edge_cursor + ne], dtype=np.float32)
            ws.rest_lengths_wp.assign(wp.array(rl_slice, dtype=wp.float32, device=device))

            rd_slice = np.array(all_rest_darboux[edge_cursor : edge_cursor + ne], dtype=np.float32)
            ws.rest_darboux_wp.assign(wp.array(rd_slice, dtype=wp.vec3, device=device))

            bs_slice = np.array(all_bend_stiffness[edge_cursor : edge_cursor + ne], dtype=np.float32)
            ws.bend_stiffness_wp.assign(wp.array(bs_slice, dtype=wp.vec3, device=device))

            if model.gravity is not None:
                g = model.gravity.numpy()
                ws.gravity = wp.vec3(float(g[0][0]), float(g[0][1]), float(g[0][2]))

            orient_cursor += np_
            edge_cursor += ne
            self._rods.append(ws)

        self._rod_particle_starts = list(rod_particle_starts) if rod_num_points else []

        self._batched_ws = None
        if len(self._rods) > 1 and self.solver_backend == DIRECT_SOLVE_BLOCK_THOMAS:
            self._batched_ws = _BatchedRodWorkspace(self._rods, device)

    # -- Phase 2: predict orientations + prepare constraints ----------------

    def predict_orientations(self, particle_q: wp.array, dt: float, device: wp.Device) -> None:
        """Predict rod orientations and sync positions from *particle_q*."""
        if self._batched_ws is not None:
            self._predict_orientations_batched(particle_q, dt, device)
            return

        for rod_idx, ws in enumerate(self._rods):
            if ws.num_edges == 0:
                continue
            ps = self._rod_particle_starts[rod_idx]

            wp.copy(dest=ws.predicted_positions_wp, src=particle_q, dest_offset=0, src_offset=ps, count=ws.num_points)

            wp.launch(
                _warp_predict_rotations,
                dim=ws.num_points,
                inputs=[
                    ws.orientations_wp,
                    ws.angular_velocities_wp,
                    ws.torques_wp,
                    ws.quat_inv_masses_wp,
                    float(dt),
                    float(self.angular_damping),
                    ws.predicted_orientations_wp,
                ],
                device=device,
            )

            wp.launch(_rod_zero_float, dim=ws.n_dofs, inputs=[ws.lambda_sum_wp], device=device)
            wp.launch(
                _warp_prepare_compliance,
                dim=ws.num_edges,
                inputs=[
                    ws.rest_lengths_wp,
                    ws.bend_stiffness_wp,
                    float(ws.young_modulus),
                    float(ws.torsion_modulus),
                    float(dt),
                    ws.compliance_wp,
                ],
                device=device,
            )

    def _predict_orientations_batched(self, particle_q: wp.array, dt: float, device: wp.Device) -> None:
        bws = self._batched_ws
        tp = bws.total_particles
        te = bws.total_edges
        td = bws.total_dofs

        for rod_idx, ws in enumerate(self._rods):
            ps = self._rod_particle_starts[rod_idx]
            po = bws.rod_offsets_cpu[rod_idx]
            wp.copy(dest=bws.predicted_positions, src=particle_q, dest_offset=po, src_offset=ps, count=ws.num_points)

        wp.launch(
            _warp_predict_rotations_batched,
            dim=tp,
            inputs=[
                bws.orientations,
                bws.angular_velocities,
                bws.torques,
                bws.quat_inv_masses,
                float(dt),
                float(self.angular_damping),
                bws.predicted_orientations,
            ],
            device=device,
        )

        wp.launch(_rod_zero_float, dim=td, inputs=[bws.lambda_sum], device=device)
        wp.launch(
            _warp_prepare_compliance_batched,
            dim=te,
            inputs=[
                bws.rest_lengths,
                bws.bend_stiffness,
                bws.edge_rod_id,
                bws.young_modulus,
                bws.torsion_modulus,
                float(dt),
                bws.compliance,
            ],
            device=device,
        )

    # -- Phase 3: solve constraints inside iteration loop -------------------

    def solve_constraints(
        self, particle_q: wp.array, particle_deltas: wp.array, dt: float, device: wp.Device
    ) -> None:
        """Project rod constraints and feed corrections into *particle_deltas*."""
        if self._batched_ws is not None:
            self._solve_constraints_batched(particle_q, particle_deltas, dt, device)
            return

        for rod_idx, ws in enumerate(self._rods):
            if ws.num_edges == 0:
                continue
            ps = self._rod_particle_starts[rod_idx]

            wp.copy(
                dest=ws.predicted_positions_wp, src=particle_q, dest_offset=0, src_offset=ps, count=ws.num_points
            )

            self._project_direct(ws, device)

            wp.launch(
                _add_rod_corrections_to_particle_deltas,
                dim=ws.num_points,
                inputs=[ws.pos_corrections_wp, particle_deltas, int(ps), int(ws.num_points)],
                device=device,
            )

            wp.launch(
                _apply_rod_orientation_corrections,
                dim=ws.num_points,
                inputs=[ws.predicted_orientations_wp, ws.rot_corrections_wp, int(ws.num_points)],
                device=device,
            )

    def _solve_constraints_batched(
        self, particle_q: wp.array, particle_deltas: wp.array, dt: float, device: wp.Device
    ) -> None:
        bws = self._batched_ws
        tp = bws.total_particles

        for rod_idx, ws in enumerate(self._rods):
            ps = self._rod_particle_starts[rod_idx]
            po = bws.rod_offsets_cpu[rod_idx]
            wp.copy(dest=bws.predicted_positions, src=particle_q, dest_offset=po, src_offset=ps, count=ws.num_points)

        self._project_direct_batched(bws, device)

        for rod_idx, ws in enumerate(self._rods):
            ps = self._rod_particle_starts[rod_idx]
            po = bws.rod_offsets_cpu[rod_idx]
            # pos_corrections is a contiguous slice of the batched workspace
            wp.launch(
                _add_rod_corrections_to_particle_deltas,
                dim=ws.num_points,
                inputs=[bws.pos_corrections[po:], particle_deltas, int(ps), int(ws.num_points)],
                device=device,
            )

        wp.launch(
            _apply_rod_orientation_corrections,
            dim=tp,
            inputs=[bws.predicted_orientations, bws.rot_corrections, int(tp)],
            device=device,
        )

    # -- Phase 4: integrate orientations ------------------------------------

    def integrate_orientations(self, dt: float, device: wp.Device) -> None:
        """Integrate rod orientations after the XPBD iteration loop."""
        if self._batched_ws is not None:
            bws = self._batched_ws
            wp.launch(
                _warp_integrate_rotations_batched,
                dim=bws.total_particles,
                inputs=[
                    bws.orientations,
                    bws.predicted_orientations,
                    bws.prev_orientations,
                    bws.angular_velocities,
                    bws.quat_inv_masses,
                    float(dt),
                ],
                device=device,
            )
            return

        for ws in self._rods:
            if ws.num_edges == 0:
                continue
            wp.launch(
                _warp_integrate_rotations,
                dim=ws.num_points,
                inputs=[
                    ws.orientations_wp,
                    ws.predicted_orientations_wp,
                    ws.prev_orientations_wp,
                    ws.angular_velocities_wp,
                    ws.quat_inv_masses_wp,
                    float(dt),
                ],
                device=device,
            )

    # -- Projection helpers (reused from SolverXPBDRod) ---------------------

    def _project_direct(self, ws: _RodWorkspace, device: wp.Device) -> None:
        if ws.num_edges == 0:
            return

        wp.launch(
            _warp_update_constraints_direct,
            dim=ws.num_edges,
            inputs=[
                ws.predicted_positions_wp,
                ws.predicted_orientations_wp,
                ws.rest_lengths_wp,
                ws.rest_darboux_wp,
                ws.constraint_values_wp,
            ],
            device=device,
        )

        wp.launch(
            _warp_compute_jacobians_direct,
            dim=ws.num_edges,
            inputs=[ws.predicted_orientations_wp, ws.rest_lengths_wp, ws.jacobian_pos_wp, ws.jacobian_rot_wp],
            device=device,
        )

        wp.launch(
            _warp_compute_inv_inertia_world,
            dim=ws.num_points,
            inputs=[
                ws.predicted_orientations_wp,
                ws.quat_inv_masses_wp,
                ws.inv_inertia_local_diag,
                ws.inv_inertia_wp,
            ],
            device=device,
        )

        n_dofs = ws.n_dofs
        delta_lambda = self._solve_system(ws, n_dofs, device)

        wp.launch(_rod_zero_vec3, dim=ws.num_points, inputs=[ws.pos_corrections_wp], device=device)
        wp.launch(_rod_zero_vec3, dim=ws.num_points, inputs=[ws.rot_corrections_wp], device=device)
        wp.launch(_rod_zero_float, dim=1, inputs=[ws._delta_lambda_max_wp], device=device)
        wp.launch(_rod_zero_float, dim=1, inputs=[ws._correction_max_wp], device=device)

        wp.launch(
            _warp_compute_corrections_parallel,
            dim=ws.num_edges,
            inputs=[
                ws.predicted_positions_wp,
                ws.inv_masses_wp,
                ws.quat_inv_masses_wp,
                ws.inv_inertia_wp,
                ws.jacobian_pos_wp,
                ws.jacobian_rot_wp,
                delta_lambda,
                ws.lambda_sum_wp,
                int(ws.num_edges),
                ws.pos_corrections_wp,
                ws.rot_corrections_wp,
                ws._delta_lambda_max_wp,
                ws._correction_max_wp,
            ],
            device=device,
        )

    def _project_direct_batched(self, bws: _BatchedRodWorkspace, device: wp.Device) -> None:
        tp = bws.total_particles
        te = bws.total_edges
        td = bws.total_dofs

        wp.launch(
            _warp_update_constraints_batched_v2,
            dim=te,
            inputs=[
                bws.predicted_positions,
                bws.predicted_orientations,
                bws.rest_lengths,
                bws.rest_darboux,
                bws.rod_offsets,
                bws.edge_offsets,
                bws.edge_rod_id,
                bws.constraint_values,
            ],
            device=device,
        )

        wp.launch(
            _warp_compute_jacobians_batched,
            dim=te,
            inputs=[
                bws.predicted_orientations,
                bws.rest_lengths,
                bws.rod_offsets,
                bws.edge_offsets,
                bws.edge_rod_id,
                bws.jacobian_pos,
                bws.jacobian_rot,
            ],
            device=device,
        )

        wp.launch(
            _warp_compute_inv_inertia_world_batched,
            dim=tp,
            inputs=[
                bws.predicted_orientations,
                bws.quat_inv_masses,
                bws.inv_inertia_local_diag,
                bws.particle_rod_id,
                bws.inv_inertia,
            ],
            device=device,
        )

        wp.launch(
            _warp_assemble_jmjt_blocks_batched,
            dim=te,
            inputs=[
                bws.jacobian_pos,
                bws.jacobian_rot,
                bws.compliance,
                bws.inv_masses,
                bws.inv_inertia,
                bws.rod_offsets,
                bws.edge_offsets,
                bws.edge_rod_id,
                bws.diag_blocks,
                bws.offdiag_blocks,
            ],
            device=device,
        )

        wp.launch(
            _warp_build_rhs,
            dim=td,
            inputs=[bws.constraint_values, bws.compliance, bws.lambda_sum, int(td), bws.rhs],
            device=device,
        )

        wp.launch(
            _warp_block_thomas_solve_batched,
            dim=bws.n_rods,
            inputs=[
                bws.diag_blocks,
                bws.offdiag_blocks,
                bws.rhs,
                bws.edge_offsets,
                int(bws.n_rods),
                bws.c_blocks,
                bws.d_prime,
                bws.delta_lambda,
            ],
            device=device,
        )

        wp.launch(_rod_zero_vec3, dim=tp, inputs=[bws.pos_corrections], device=device)
        wp.launch(_rod_zero_vec3, dim=tp, inputs=[bws.rot_corrections], device=device)
        wp.launch(_rod_zero_float, dim=1, inputs=[bws._delta_lambda_max], device=device)
        wp.launch(_rod_zero_float, dim=1, inputs=[bws._correction_max], device=device)

        wp.launch(
            _warp_compute_corrections_parallel_batched,
            dim=te,
            inputs=[
                bws.predicted_positions,
                bws.inv_masses,
                bws.quat_inv_masses,
                bws.inv_inertia,
                bws.jacobian_pos,
                bws.jacobian_rot,
                bws.delta_lambda,
                bws.lambda_sum,
                bws.rod_offsets,
                bws.edge_offsets,
                bws.edge_rod_id,
                bws.pos_corrections,
                bws.rot_corrections,
                bws._delta_lambda_max,
                bws._correction_max,
            ],
            device=device,
        )

    def _solve_system(self, ws: _RodWorkspace, n_dofs: int, device: wp.Device) -> wp.array:
        if self.solver_backend == DIRECT_SOLVE_SPLIT_THOMAS:
            return self._solve_split_thomas(ws, device)

        if self.solver_backend == DIRECT_SOLVE_BLOCK_JACOBI:
            wp.launch(
                _warp_assemble_jmjt_blocks,
                dim=ws.num_edges,
                inputs=[
                    ws.jacobian_pos_wp, ws.jacobian_rot_wp, ws.compliance_wp,
                    ws.inv_masses_wp, ws.inv_inertia_wp, int(ws.num_edges),
                    ws.diag_blocks_wp, ws.offdiag_blocks_wp,
                ],
                device=device,
            )
            wp.launch(
                _warp_build_rhs,
                dim=n_dofs,
                inputs=[ws.constraint_values_wp, ws.compliance_wp, ws.lambda_sum_wp, int(n_dofs), ws.rhs_wp],
                device=device,
            )
            wp.launch(
                _warp_solve_blocks_jacobi,
                dim=ws.num_edges,
                inputs=[ws.diag_blocks_wp, ws.rhs_wp, ws.delta_lambda_wp, int(ws.num_edges)],
                device=device,
            )
            return ws.delta_lambda_wp

        if self.solver_backend == DIRECT_SOLVE_BANDED_CHOLESKY:
            wp.launch(
                _rod_zero_2d,
                dim=BAND_LDAB * max(1, n_dofs),
                inputs=[ws.ab_wp, int(BAND_LDAB), int(max(1, n_dofs))],
                device=device,
            )
            wp.launch(
                _warp_assemble_jmjt_banded,
                dim=ws.num_edges,
                inputs=[
                    ws.jacobian_pos_wp, ws.jacobian_rot_wp, ws.compliance_wp,
                    ws.inv_masses_wp, ws.inv_inertia_wp, int(n_dofs), ws.ab_wp,
                ],
                device=device,
            )
            wp.launch(
                _warp_build_rhs,
                dim=n_dofs,
                inputs=[ws.constraint_values_wp, ws.compliance_wp, ws.lambda_sum_wp, int(n_dofs), ws.rhs_wp],
                device=device,
            )
            wp.launch(_warp_spbsv_u11_1rhs, dim=1, inputs=[int(n_dofs), ws.ab_wp, ws.rhs_wp], device=device)
            return ws.rhs_wp

        if n_dofs <= TILE:
            wp.launch(
                _rod_zero_2d, dim=TILE * TILE, inputs=[ws.A_wp, int(TILE), int(TILE)], device=device
            )
            wp.launch(
                _warp_assemble_jmjt_dense,
                dim=ws.num_edges,
                inputs=[
                    ws.jacobian_pos_wp, ws.jacobian_rot_wp, ws.compliance_wp,
                    ws.inv_masses_wp, ws.inv_inertia_wp, int(n_dofs), ws.A_wp,
                ],
                device=device,
            )
            wp.launch(
                _warp_build_rhs,
                dim=TILE,
                inputs=[ws.constraint_values_wp, ws.compliance_wp, ws.lambda_sum_wp, int(n_dofs), ws.rhs_tile_wp],
                device=device,
            )
            if n_dofs < TILE:
                wp.launch(_warp_pad_diagonal, dim=TILE, inputs=[ws.A_wp, int(n_dofs), int(TILE)], device=device)
            wp.launch_tiled(
                _warp_cholesky_solve_tile,
                dim=[1, 1],
                inputs=[ws.A_wp, ws.rhs_tile_wp],
                outputs=[ws.delta_lambda_tile_wp],
                block_dim=BLOCK_DIM,
                device=device,
            )
            return ws.delta_lambda_tile_wp

        wp.launch(
            _warp_assemble_jmjt_blocks,
            dim=ws.num_edges,
            inputs=[
                ws.jacobian_pos_wp, ws.jacobian_rot_wp, ws.compliance_wp,
                ws.inv_masses_wp, ws.inv_inertia_wp, int(ws.num_edges),
                ws.diag_blocks_wp, ws.offdiag_blocks_wp,
            ],
            device=device,
        )
        wp.launch(
            _warp_build_rhs,
            dim=n_dofs,
            inputs=[ws.constraint_values_wp, ws.compliance_wp, ws.lambda_sum_wp, int(n_dofs), ws.rhs_wp],
            device=device,
        )
        wp.launch(
            _warp_block_thomas_solve,
            dim=1,
            inputs=[
                ws.diag_blocks_wp, ws.offdiag_blocks_wp, ws.rhs_wp,
                int(ws.num_edges), ws.c_blocks_wp, ws.d_prime_wp, ws.delta_lambda_wp,
            ],
            device=device,
        )
        return ws.delta_lambda_wp

    def _solve_split_thomas(self, ws: _RodWorkspace, device: wp.Device) -> wp.array:
        n = ws.num_edges
        if ws._split_stretch_diag_wp is None:
            ws._split_stretch_diag_wp = wp.zeros(n * 9, dtype=wp.float32, device=device)
            ws._split_stretch_offdiag_wp = wp.zeros(n * 9, dtype=wp.float32, device=device)
            ws._split_stretch_rhs_wp = wp.zeros(n * 3, dtype=wp.float32, device=device)
            ws._split_stretch_c_blocks_wp = wp.zeros(n * 9, dtype=wp.float32, device=device)
            ws._split_stretch_d_prime_wp = wp.zeros(n * 3, dtype=wp.float32, device=device)
            ws._split_stretch_delta_lambda_wp = wp.zeros(n * 3, dtype=wp.float32, device=device)
            ws._split_darboux_diag_wp = wp.zeros(n * 9, dtype=wp.float32, device=device)
            ws._split_darboux_offdiag_wp = wp.zeros(n * 9, dtype=wp.float32, device=device)
            ws._split_darboux_rhs_wp = wp.zeros(n * 3, dtype=wp.float32, device=device)
            ws._split_darboux_c_blocks_wp = wp.zeros(n * 9, dtype=wp.float32, device=device)
            ws._split_darboux_d_prime_wp = wp.zeros(n * 3, dtype=wp.float32, device=device)
            ws._split_darboux_delta_lambda_wp = wp.zeros(n * 3, dtype=wp.float32, device=device)

        wp.launch(
            _warp_assemble_stretch_blocks, dim=n,
            inputs=[
                ws.jacobian_pos_wp, ws.jacobian_rot_wp, ws.compliance_wp,
                ws.inv_masses_wp, ws.inv_inertia_wp, int(n),
                ws._split_stretch_diag_wp, ws._split_stretch_offdiag_wp,
            ],
            device=device,
        )
        wp.launch(
            _warp_assemble_darboux_blocks, dim=n,
            inputs=[
                ws.jacobian_rot_wp, ws.compliance_wp, ws.inv_inertia_wp, int(n),
                ws._split_darboux_diag_wp, ws._split_darboux_offdiag_wp,
            ],
            device=device,
        )
        wp.launch(
            _warp_build_rhs_stretch, dim=n,
            inputs=[ws.constraint_values_wp, ws.compliance_wp, ws.lambda_sum_wp, int(n), ws._split_stretch_rhs_wp],
            device=device,
        )
        wp.launch(
            _warp_build_rhs_darboux, dim=n,
            inputs=[ws.constraint_values_wp, ws.compliance_wp, ws.lambda_sum_wp, int(n), ws._split_darboux_rhs_wp],
            device=device,
        )
        wp.launch(
            _warp_block_thomas_solve_3x3, dim=1,
            inputs=[
                ws._split_stretch_diag_wp, ws._split_stretch_offdiag_wp, ws._split_stretch_rhs_wp, int(n),
                ws._split_stretch_c_blocks_wp, ws._split_stretch_d_prime_wp, ws._split_stretch_delta_lambda_wp,
            ],
            device=device,
        )
        wp.launch(
            _warp_block_thomas_solve_3x3, dim=1,
            inputs=[
                ws._split_darboux_diag_wp, ws._split_darboux_offdiag_wp, ws._split_darboux_rhs_wp, int(n),
                ws._split_darboux_c_blocks_wp, ws._split_darboux_d_prime_wp, ws._split_darboux_delta_lambda_wp,
            ],
            device=device,
        )
        wp.launch(
            _warp_merge_delta_lambda, dim=n,
            inputs=[ws._split_stretch_delta_lambda_wp, ws._split_darboux_delta_lambda_wp, ws.delta_lambda_wp, int(n)],
            device=device,
        )
        return ws.delta_lambda_wp


class SolverXPBD(SolverBase):
    """An implicit integrator using eXtended Position-Based Dynamics (XPBD) for rigid and soft body simulation.

    References:
        - Miles Macklin, Matthias Müller, and Nuttapong Chentanez. 2016. XPBD: position-based simulation of compliant constrained dynamics. In Proceedings of the 9th International Conference on Motion in Games (MIG '16). Association for Computing Machinery, New York, NY, USA, 49-54. https://doi.org/10.1145/2994258.2994272
        - Matthias Müller, Miles Macklin, Nuttapong Chentanez, Stefan Jeschke, and Tae-Yong Kim. 2020. Detailed rigid body simulation with extended position based dynamics. In Proceedings of the ACM SIGGRAPH/Eurographics Symposium on Computer Animation (SCA '20). Eurographics Association, Goslar, DEU, Article 10, 1-12. https://doi.org/10.1111/cgf.14105

    After constructing :class:`Model`, :class:`State`, and :class:`Control` (optional) objects, this time-integrator
    may be used to advance the simulation state forward in time.

    Limitations:
        **Momentum conservation** -- When ``rigid_contact_con_weighting`` is
        enabled (the default), each body's positional correction is divided by
        its number of active contacts.  This improves convergence for stacking
        scenarios but means the solver does not conserve momentum at contacts.
        Reported per-contact forces (see :meth:`update_contacts`) are
        approximate: for contacts between two dynamic bodies the force is
        computed using the harmonic mean of the two bodies' contact counts,
        which is symmetric but not exact.

    Joint limitations:
        - Supported joint types: PRISMATIC, REVOLUTE, BALL, FIXED, FREE, DISTANCE, D6.
          CABLE joints are not supported.
        - :attr:`~newton.Model.joint_enabled`,
          :attr:`~newton.Model.joint_target_ke`/:attr:`~newton.Model.joint_target_kd`, and
          :attr:`~newton.Control.joint_f` are supported.
          Joint limits are enforced as hard positional constraints (``joint_limit_ke``/``joint_limit_kd`` are not used).
        - :attr:`~newton.Model.joint_armature`, :attr:`~newton.Model.joint_friction`,
          :attr:`~newton.Model.joint_effort_limit`, :attr:`~newton.Model.joint_velocity_limit`,
          and :attr:`~newton.Model.joint_target_mode` are not supported.
        - Equality and mimic constraints are not supported.

        See :ref:`Joint feature support` for the full comparison across solvers.

    Example
    -------

    .. code-block:: python

        solver = newton.solvers.SolverXPBD(model)

        # simulation loop
        for i in range(100):
            solver.step(state_in, state_out, control, contacts, dt)
            state_in, state_out = state_out, state_in

    """

    def __init__(
        self,
        model: Model,
        iterations: int = 2,
        soft_body_relaxation: float = 0.9,
        soft_contact_relaxation: float = 0.9,
        joint_linear_relaxation: float = 0.7,
        joint_angular_relaxation: float = 0.4,
        joint_linear_compliance: float = 0.0,
        joint_angular_compliance: float = 0.0,
        rigid_contact_relaxation: float = 0.8,
        rigid_contact_con_weighting: bool = True,
        angular_damping: float = 0.0,
        enable_restitution: bool = False,
        rod_solver_backend: str = DIRECT_SOLVE_BLOCK_THOMAS,
        rod_linear_damping: float = 0.0,
        rod_angular_damping: float = 0.0,
    ):
        super().__init__(model=model)
        self.iterations = iterations

        self.soft_body_relaxation = soft_body_relaxation
        self.soft_contact_relaxation = soft_contact_relaxation

        self.joint_linear_relaxation = joint_linear_relaxation
        self.joint_angular_relaxation = joint_angular_relaxation
        self.joint_linear_compliance = joint_linear_compliance
        self.joint_angular_compliance = joint_angular_compliance

        self.rigid_contact_relaxation = rigid_contact_relaxation
        self.rigid_contact_con_weighting = rigid_contact_con_weighting

        self.angular_damping = angular_damping

        self.enable_restitution = enable_restitution

        self.compute_body_velocity_from_position_delta = False

        self._init_kinematic_state()

        # helper variables to track constraint resolution vars
        self._particle_delta_counter = 0
        self._body_delta_counter = 0

        if model.particle_count > 1 and model.particle_grid is not None:
            # reserve space for the particle hash grid
            with wp.ScopedDevice(model.device):
                model.particle_grid.reserve(model.particle_count)

        # Elastic rod constraint component (if model has rod data)
        self._rod_constraints: _ElasticRodConstraints | None = None
        if hasattr(model, "xpbd_rod") and model.xpbd_rod.get("rod_num_points"):
            self._rod_constraints = _ElasticRodConstraints(
                model=model,
                solver_backend=rod_solver_backend,
                linear_damping=rod_linear_damping,
                angular_damping=rod_angular_damping,
            )

    @classmethod
    def register_custom_attributes(cls, builder: ModelBuilder) -> None:
        """Register rod-specific data storage on the builder.

        Must be called before adding rods and before
        :meth:`~newton.ModelBuilder.finalize`.
        """
        builder._xpbd_rod_data = {
            "rod_num_points": [],
            "rod_particle_start": [],
            "rod_young_modulus": [],
            "rod_torsion_modulus": [],
            "orientations": [],
            "quat_inv_masses": [],
            "rest_lengths": [],
            "rest_darboux": [],
            "bend_stiffness": [],
        }

        original_finalize = builder.finalize

        def _finalize_with_rod_data(*args, **kwargs):
            model = original_finalize(*args, **kwargs)
            model.xpbd_rod = builder._xpbd_rod_data
            return model

        builder.finalize = _finalize_with_rod_data

    @override
    def notify_model_changed(self, flags: int) -> None:
        if flags & (SolverNotifyFlags.BODY_PROPERTIES | SolverNotifyFlags.BODY_INERTIAL_PROPERTIES):
            self._refresh_kinematic_state()

    def copy_kinematic_body_state(self, model: Model, state_in: State, state_out: State):
        if model.body_count == 0:
            return
        wp.launch(
            kernel=copy_kinematic_body_state_kernel,
            dim=model.body_count,
            inputs=[model.body_flags, state_in.body_q, state_in.body_qd],
            outputs=[state_out.body_q, state_out.body_qd],
            device=model.device,
        )

    def _apply_particle_deltas(
        self,
        model: Model,
        state_in: State,
        state_out: State,
        particle_deltas: wp.array,
        dt: float,
    ):
        if state_in.requires_grad:
            particle_q = state_out.particle_q
            # allocate new particle arrays so gradients can be tracked correctly without overwriting
            new_particle_q = wp.empty_like(state_out.particle_q)
            new_particle_qd = wp.empty_like(state_out.particle_qd)
            self._particle_delta_counter += 1
        else:
            if self._particle_delta_counter == 0:
                particle_q = state_out.particle_q
                new_particle_q = state_in.particle_q
                new_particle_qd = state_in.particle_qd
            else:
                particle_q = state_in.particle_q
                new_particle_q = state_out.particle_q
                new_particle_qd = state_out.particle_qd
            self._particle_delta_counter = 1 - self._particle_delta_counter

        wp.launch(
            kernel=apply_particle_deltas,
            dim=model.particle_count,
            inputs=[
                self.particle_q_init,
                particle_q,
                model.particle_flags,
                particle_deltas,
                dt,
                model.particle_max_velocity,
            ],
            outputs=[new_particle_q, new_particle_qd],
            device=model.device,
        )

        if state_in.requires_grad:
            state_out.particle_q = new_particle_q
            state_out.particle_qd = new_particle_qd

        return new_particle_q, new_particle_qd

    def _apply_body_deltas(
        self,
        model: Model,
        state_in: State,
        state_out: State,
        body_deltas: wp.array,
        dt: float,
        rigid_contact_inv_weight: wp.array = None,
    ):
        with wp.ScopedTimer("apply_body_deltas", False):
            if state_in.requires_grad:
                body_q = state_out.body_q
                body_qd = state_out.body_qd
                new_body_q = wp.clone(body_q)
                new_body_qd = wp.clone(body_qd)
                self._body_delta_counter += 1
            else:
                if self._body_delta_counter == 0:
                    body_q = state_out.body_q
                    body_qd = state_out.body_qd
                    new_body_q = state_in.body_q
                    new_body_qd = state_in.body_qd
                else:
                    body_q = state_in.body_q
                    body_qd = state_in.body_qd
                    new_body_q = state_out.body_q
                    new_body_qd = state_out.body_qd
                self._body_delta_counter = 1 - self._body_delta_counter

            wp.launch(
                kernel=apply_body_deltas,
                dim=model.body_count,
                inputs=[
                    body_q,
                    body_qd,
                    model.body_com,
                    model.body_inertia,
                    self.body_inv_mass_effective,
                    self.body_inv_inertia_effective,
                    body_deltas,
                    rigid_contact_inv_weight,
                    dt,
                ],
                outputs=[
                    new_body_q,
                    new_body_qd,
                ],
                device=model.device,
            )

            if state_in.requires_grad:
                state_out.body_q = new_body_q
                state_out.body_qd = new_body_qd

        return new_body_q, new_body_qd

    @override
    def step(self, state_in: State, state_out: State, control: Control, contacts: Contacts, dt: float) -> None:
        requires_grad = state_in.requires_grad
        self._particle_delta_counter = 0
        self._body_delta_counter = 0

        model = self.model

        particle_q = None
        particle_qd = None
        particle_deltas = None

        body_q = None
        body_qd = None
        body_q_init = None
        body_qd_init = None
        body_deltas = None

        rigid_contact_inv_weight = None

        contact_impulse = None
        contact_impulse_iter = None

        if contacts:
            if self.rigid_contact_con_weighting:
                rigid_contact_inv_weight = wp.zeros(model.body_count, dtype=float, device=model.device)
            rigid_contact_inv_weight_init = None

            if contacts.force is not None:
                contact_impulse = wp.zeros(contacts.rigid_contact_max, dtype=wp.spatial_vector, device=model.device)
                contact_impulse_iter = wp.zeros(
                    contacts.rigid_contact_max, dtype=wp.spatial_vector, device=model.device
                )

        if control is None:
            control = model.control(clone_variables=False)

        with wp.ScopedTimer("simulate", False):
            if model.particle_count:
                particle_q = state_out.particle_q
                particle_qd = state_out.particle_qd

                self.particle_q_init = wp.clone(state_in.particle_q)
                if self.enable_restitution:
                    self.particle_qd_init = wp.clone(state_in.particle_qd)
                particle_deltas = wp.empty_like(state_out.particle_qd)

                self.integrate_particles(model, state_in, state_out, dt)

                if self._rod_constraints is not None:
                    self._rod_constraints.predict_orientations(state_out.particle_q, dt, model.device)

                # Build/update the particle hash grid for particle-particle contact queries
                if model.particle_count > 1 and model.particle_grid is not None:
                    # Search radius must cover the maximum interaction distance used by the contact query
                    search_radius = model.particle_max_radius * 2.0 + model.particle_cohesion
                    with wp.ScopedDevice(model.device):
                        model.particle_grid.build(state_out.particle_q, radius=search_radius)

            if model.body_count:
                body_q = state_out.body_q
                body_qd = state_out.body_qd

                if self.compute_body_velocity_from_position_delta or self.enable_restitution:
                    body_q_init = wp.clone(state_in.body_q)
                    body_qd_init = wp.clone(state_in.body_qd)

                body_deltas = wp.empty_like(state_out.body_qd)

                body_f_tmp = state_in.body_f
                if model.joint_count:
                    # Avoid accumulating joint_f into the persistent state body_f buffer.
                    body_f_tmp = wp.clone(state_in.body_f)
                    wp.launch(
                        kernel=apply_joint_forces,
                        dim=model.joint_count,
                        inputs=[
                            state_in.body_q,
                            model.body_com,
                            model.joint_type,
                            model.joint_enabled,
                            model.joint_parent,
                            model.joint_child,
                            model.joint_X_p,
                            model.joint_X_c,
                            model.joint_qd_start,
                            model.joint_dof_dim,
                            model.joint_axis,
                            control.joint_f,
                        ],
                        outputs=[body_f_tmp],
                        device=model.device,
                    )

                if body_f_tmp is state_in.body_f:
                    self.integrate_bodies(model, state_in, state_out, dt, self.angular_damping)
                else:
                    body_f_prev = state_in.body_f
                    state_in.body_f = body_f_tmp
                    self.integrate_bodies(model, state_in, state_out, dt, self.angular_damping)
                    state_in.body_f = body_f_prev

            spring_constraint_lambdas = None
            if model.spring_count:
                spring_constraint_lambdas = wp.empty_like(model.spring_rest_length)
            edge_constraint_lambdas = None
            if model.edge_count:
                edge_constraint_lambdas = wp.empty_like(model.edge_rest_angle)

            for i in range(self.iterations):
                with wp.ScopedTimer(f"iteration_{i}", False):
                    if model.body_count:
                        if requires_grad and i > 0:
                            body_deltas = wp.zeros_like(body_deltas)
                        else:
                            body_deltas.zero_()

                    if model.particle_count:
                        if requires_grad and i > 0:
                            particle_deltas = wp.zeros_like(particle_deltas)
                        else:
                            particle_deltas.zero_()

                        # particle-rigid body contacts (besides ground plane)
                        if model.shape_count:
                            wp.launch(
                                kernel=solve_particle_shape_contacts,
                                dim=contacts.soft_contact_max,
                                inputs=[
                                    particle_q,
                                    particle_qd,
                                    model.particle_inv_mass,
                                    model.particle_radius,
                                    model.particle_flags,
                                    body_q,
                                    body_qd,
                                    model.body_com,
                                    self.body_inv_mass_effective,
                                    self.body_inv_inertia_effective,
                                    model.shape_body,
                                    model.shape_material_mu,
                                    model.soft_contact_mu,
                                    model.particle_adhesion,
                                    contacts.soft_contact_count,
                                    contacts.soft_contact_particle,
                                    contacts.soft_contact_shape,
                                    contacts.soft_contact_body_pos,
                                    contacts.soft_contact_body_vel,
                                    contacts.soft_contact_normal,
                                    contacts.soft_contact_max,
                                    dt,
                                    self.soft_contact_relaxation,
                                ],
                                # outputs
                                outputs=[particle_deltas, body_deltas],
                                device=model.device,
                            )

                        if model.particle_max_radius > 0.0 and model.particle_count > 1:
                            # assert model.particle_grid.reserved, "model.particle_grid must be built, see HashGrid.build()"
                            assert model.particle_grid is not None
                            wp.launch(
                                kernel=solve_particle_particle_contacts,
                                dim=model.particle_count,
                                inputs=[
                                    model.particle_grid.id,
                                    particle_q,
                                    particle_qd,
                                    model.particle_inv_mass,
                                    model.particle_radius,
                                    model.particle_flags,
                                    model.particle_mu,
                                    model.particle_cohesion,
                                    model.particle_max_radius,
                                    dt,
                                    self.soft_contact_relaxation,
                                ],
                                outputs=[particle_deltas],
                                device=model.device,
                            )

                        # distance constraints
                        if model.spring_count:
                            spring_constraint_lambdas.zero_()
                            wp.launch(
                                kernel=solve_springs,
                                dim=model.spring_count,
                                inputs=[
                                    particle_q,
                                    particle_qd,
                                    model.particle_inv_mass,
                                    model.spring_indices,
                                    model.spring_rest_length,
                                    model.spring_stiffness,
                                    model.spring_damping,
                                    dt,
                                    spring_constraint_lambdas,
                                ],
                                outputs=[particle_deltas],
                                device=model.device,
                            )

                        # bending constraints
                        if model.edge_count:
                            edge_constraint_lambdas.zero_()
                            wp.launch(
                                kernel=bending_constraint,
                                dim=model.edge_count,
                                inputs=[
                                    particle_q,
                                    particle_qd,
                                    model.particle_inv_mass,
                                    model.edge_indices,
                                    model.edge_rest_angle,
                                    model.edge_bending_properties,
                                    dt,
                                    edge_constraint_lambdas,
                                ],
                                outputs=[particle_deltas],
                                device=model.device,
                            )

                        # tetrahedral FEM
                        if model.tet_count:
                            wp.launch(
                                kernel=solve_tetrahedra,
                                dim=model.tet_count,
                                inputs=[
                                    particle_q,
                                    particle_qd,
                                    model.particle_inv_mass,
                                    model.tet_indices,
                                    model.tet_poses,
                                    model.tet_activations,
                                    model.tet_materials,
                                    dt,
                                    self.soft_body_relaxation,
                                ],
                                outputs=[particle_deltas],
                                device=model.device,
                            )

                        if self._rod_constraints is not None:
                            self._rod_constraints.solve_constraints(
                                particle_q, particle_deltas, dt, model.device
                            )

                        particle_q, particle_qd = self._apply_particle_deltas(
                            model, state_in, state_out, particle_deltas, dt
                        )

                    # handle rigid bodies
                    # ----------------------------

                    # Solve rigid contact constraints
                    if model.body_count and contacts is not None:
                        if self.rigid_contact_con_weighting:
                            rigid_contact_inv_weight.zero_()

                        if contact_impulse_iter is not None:
                            contact_impulse_iter.zero_()

                        wp.launch(
                            kernel=solve_body_contact_positions,
                            dim=contacts.rigid_contact_max,
                            inputs=[
                                body_q,
                                body_qd,
                                model.body_flags,
                                model.body_com,
                                self.body_inv_mass_effective,
                                self.body_inv_inertia_effective,
                                model.shape_body,
                                contacts.rigid_contact_count,
                                contacts.rigid_contact_point0,
                                contacts.rigid_contact_point1,
                                contacts.rigid_contact_offset0,
                                contacts.rigid_contact_offset1,
                                contacts.rigid_contact_normal,
                                contacts.rigid_contact_margin0,
                                contacts.rigid_contact_margin1,
                                contacts.rigid_contact_shape0,
                                contacts.rigid_contact_shape1,
                                model.shape_material_mu,
                                model.shape_material_mu_torsional,
                                model.shape_material_mu_rolling,
                                self.rigid_contact_relaxation,
                                dt,
                            ],
                            outputs=[
                                body_deltas,
                                rigid_contact_inv_weight,
                                contact_impulse_iter,
                            ],
                            device=model.device,
                        )

                        if contact_impulse_iter is not None:
                            wp.launch(
                                kernel=accumulate_weighted_contact_impulse,
                                dim=contacts.rigid_contact_max,
                                inputs=[
                                    contacts.rigid_contact_count,
                                    contact_impulse_iter,
                                    contacts.rigid_contact_shape0,
                                    contacts.rigid_contact_shape1,
                                    model.shape_body,
                                    rigid_contact_inv_weight,
                                ],
                                outputs=[contact_impulse],
                                device=model.device,
                            )

                        # if model.rigid_contact_count.numpy()[0] > 0:
                        #     print("rigid_contact_count:", model.rigid_contact_count.numpy().flatten())
                        #     # print("rigid_active_contact_distance:", rigid_active_contact_distance.numpy().flatten())
                        #     # print("rigid_active_contact_point0:", rigid_active_contact_point0.numpy().flatten())
                        #     # print("rigid_active_contact_point1:", rigid_active_contact_point1.numpy().flatten())
                        #     print("body_deltas:", body_deltas.numpy().flatten())

                        # print(rigid_active_contact_distance.numpy().flatten())

                        if self.enable_restitution and i == 0:
                            # remember contact constraint weighting from the first iteration
                            if self.rigid_contact_con_weighting:
                                rigid_contact_inv_weight_init = wp.clone(rigid_contact_inv_weight)
                            else:
                                rigid_contact_inv_weight_init = None

                        body_q, body_qd = self._apply_body_deltas(
                            model, state_in, state_out, body_deltas, dt, rigid_contact_inv_weight
                        )

                    if model.joint_count:
                        if requires_grad:
                            body_deltas = wp.zeros_like(body_deltas)
                        else:
                            body_deltas.zero_()

                        wp.launch(
                            kernel=solve_body_joints,
                            dim=model.joint_count,
                            inputs=[
                                body_q,
                                body_qd,
                                model.body_com,
                                self.body_inv_mass_effective,
                                self.body_inv_inertia_effective,
                                model.joint_type,
                                model.joint_enabled,
                                model.joint_parent,
                                model.joint_child,
                                model.joint_X_p,
                                model.joint_X_c,
                                model.joint_limit_lower,
                                model.joint_limit_upper,
                                model.joint_qd_start,
                                model.joint_dof_dim,
                                model.joint_axis,
                                control.joint_target_pos,
                                control.joint_target_vel,
                                model.joint_target_ke,
                                model.joint_target_kd,
                                self.joint_linear_compliance,
                                self.joint_angular_compliance,
                                self.joint_angular_relaxation,
                                self.joint_linear_relaxation,
                                dt,
                            ],
                            outputs=[body_deltas],
                            device=model.device,
                        )

                        body_q, body_qd = self._apply_body_deltas(model, state_in, state_out, body_deltas, dt)

            self._contact_impulse = contact_impulse
            self._contact_impulse_capacity = contacts.rigid_contact_max if contacts is not None else 0
            self._last_dt = dt

            if self._rod_constraints is not None:
                self._rod_constraints.integrate_orientations(dt, model.device)

            if model.particle_count:
                if particle_q.ptr != state_out.particle_q.ptr:
                    state_out.particle_q.assign(particle_q)
                    state_out.particle_qd.assign(particle_qd)

            if model.body_count:
                if body_q.ptr != state_out.body_q.ptr:
                    state_out.body_q.assign(body_q)
                    state_out.body_qd.assign(body_qd)

            # update body velocities from position changes
            if self.compute_body_velocity_from_position_delta and model.body_count and not requires_grad:
                # causes gradient issues (probably due to numerical problems
                # when computing velocities from position changes)
                if requires_grad:
                    out_body_qd = wp.clone(state_out.body_qd)
                else:
                    out_body_qd = state_out.body_qd

                # update body velocities
                wp.launch(
                    kernel=update_body_velocities,
                    dim=model.body_count,
                    inputs=[state_out.body_q, body_q_init, model.body_com, dt],
                    outputs=[out_body_qd],
                    device=model.device,
                )

            if self.enable_restitution and contacts is not None:
                if model.particle_count:
                    wp.launch(
                        kernel=apply_particle_shape_restitution,
                        dim=contacts.soft_contact_max,
                        inputs=[
                            particle_qd,
                            self.particle_q_init,
                            self.particle_qd_init,
                            model.particle_radius,
                            model.particle_flags,
                            body_q,
                            body_q_init,
                            body_qd,
                            body_qd_init,
                            model.body_com,
                            model.shape_body,
                            model.particle_adhesion,
                            model.soft_contact_restitution,
                            contacts.soft_contact_count,
                            contacts.soft_contact_particle,
                            contacts.soft_contact_shape,
                            contacts.soft_contact_body_pos,
                            contacts.soft_contact_body_vel,
                            contacts.soft_contact_normal,
                            contacts.soft_contact_max,
                        ],
                        outputs=[state_out.particle_qd],
                        device=model.device,
                    )

                if model.body_count:
                    body_deltas.zero_()

                    wp.launch(
                        kernel=apply_rigid_restitution,
                        dim=contacts.rigid_contact_max,
                        inputs=[
                            state_out.body_q,
                            state_out.body_qd,
                            body_q_init,
                            body_qd_init,
                            model.body_com,
                            self.body_inv_mass_effective,
                            self.body_inv_inertia_effective,
                            model.body_world,
                            model.shape_body,
                            contacts.rigid_contact_count,
                            contacts.rigid_contact_normal,
                            contacts.rigid_contact_shape0,
                            contacts.rigid_contact_shape1,
                            model.shape_material_restitution,
                            contacts.rigid_contact_point0,
                            contacts.rigid_contact_point1,
                            contacts.rigid_contact_offset0,
                            contacts.rigid_contact_offset1,
                            contacts.rigid_contact_margin0,
                            contacts.rigid_contact_margin1,
                            rigid_contact_inv_weight_init,
                            model.gravity,
                            dt,
                        ],
                        outputs=[
                            body_deltas,
                        ],
                        device=model.device,
                    )

                    wp.launch(
                        kernel=apply_body_delta_velocities,
                        dim=model.body_count,
                        inputs=[
                            body_deltas,
                        ],
                        outputs=[state_out.body_qd],
                        device=model.device,
                    )

            if model.body_count:
                self.copy_kinematic_body_state(model, state_in, state_out)

    @override
    def update_contacts(self, contacts: Contacts, state: State | None = None) -> None:
        """Populate ``contacts.force`` from XPBD contact impulses accumulated during the last :meth:`step`.

        Both force [N] and torque [N·m] components are written.  The torque
        includes torsional and rolling friction contributions that cannot be
        reconstructed from the linear force alone.

        When ``rigid_contact_con_weighting`` is enabled, the raw per-contact
        impulse is scaled to reflect the ``1/N`` correction that
        ``apply_body_deltas`` applies.  For contacts between a dynamic and a
        kinematic body, ``N`` is the dynamic body's contact count.  For
        contacts between two dynamic bodies, the harmonic mean
        ``2/(N_a + N_b)`` is used so that the reported force is symmetric with
        respect to body ordering.  This is an approximation -- the solver
        applies ``1/N_a`` and ``1/N_b`` independently to each side, so no
        single scalar can exactly represent both.

        Args:
            contacts: :class:`Contacts` object whose :attr:`~Contacts.force` buffer will be written.
                Must have been created with ``"force"`` in its requested attributes and must
                match the :class:`Contacts` instance (same ``rigid_contact_max``) passed to
                the preceding :meth:`step`.
            state: Unused (accepted for API compatibility with :class:`SolverBase`).

        Raises:
            ValueError: If ``contacts.force`` is ``None`` (not requested), if no step has been run yet,
                or if the contacts capacity does not match the one used in the last :meth:`step`.
        """
        if contacts.force is None:
            raise ValueError(
                "contacts.force is not allocated. Call model.request_contact_attributes('force') "
                "before creating the Contacts object."
            )
        if not hasattr(self, "_contact_impulse") or self._contact_impulse is None:
            raise ValueError("No contact impulse data available. Call step() before update_contacts().")
        if contacts.rigid_contact_max != self._contact_impulse_capacity:
            raise ValueError(
                f"Contacts capacity mismatch: update_contacts() received rigid_contact_max="
                f"{contacts.rigid_contact_max}, but step() used {self._contact_impulse_capacity}. "
                f"Pass the same Contacts instance to both step() and update_contacts()."
            )

        contacts.force.zero_()

        wp.launch(
            kernel=convert_contact_impulse_to_force,
            dim=contacts.rigid_contact_max,
            inputs=[
                contacts.rigid_contact_count,
                self._contact_impulse,
                self._last_dt,
            ],
            outputs=[contacts.force],
            device=self.model.device,
        )
