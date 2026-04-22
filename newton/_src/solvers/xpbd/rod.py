# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Internal particle-frame rod projector for :class:`SolverXPBD`."""

from __future__ import annotations

import numpy as np
import warp as wp

from ...geometry import ParticleFlags
from ...sim import Model, State
from ..xpbd_rod.kernels_assembly import (
    _warp_assemble_jmjt_blocks_batched,
    _warp_compute_inv_inertia_world_batched,
)
from ..xpbd_rod.kernels_collision import (
    _warp_apply_accumulated_corrections,
    _warp_compute_corrections_parallel_batched,
    _warp_zero_float,
    _warp_zero_vec3,
)
from ..xpbd_rod.kernels_constraints import (
    _warp_build_rhs,
    _warp_compute_jacobians_batched,
    _warp_prepare_compliance_batched,
    _warp_update_constraints_batched_v2,
)
from ..xpbd_rod.kernels_integration import (
    _warp_integrate_rotations_batched,
    _warp_predict_rotations_batched,
)
from ..xpbd_rod.kernels_math import _inv_inertia_mul_vec, _warp_jacobian_index
from ..xpbd_rod.kernels_solvers import _warp_block_thomas_solve_batched


@wp.kernel
def _gather_vec3(
    src: wp.array(dtype=wp.vec3),
    indices: wp.array(dtype=wp.int32),
    dest: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    dest[tid] = src[indices[tid]]


@wp.kernel
def _gather_float(
    src: wp.array(dtype=wp.float32),
    indices: wp.array(dtype=wp.int32),
    dest: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    dest[tid] = src[indices[tid]]


@wp.kernel
def _scatter_positions_and_velocities(
    compact_positions: wp.array(dtype=wp.vec3),
    compact_indices: wp.array(dtype=wp.int32),
    x_orig: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    dt: float,
    v_max: float,
    full_positions: wp.array(dtype=wp.vec3),
    full_velocities: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    particle_index = compact_indices[tid]

    if (particle_flags[particle_index] & int(ParticleFlags.ACTIVE)) == 0:
        return

    x_new = compact_positions[tid]
    x0 = x_orig[particle_index]
    v_new = (x_new - x0) / dt

    v_new_mag = wp.length(v_new)
    if v_new_mag > v_max:
        v_new *= v_max / v_new_mag

    full_positions[particle_index] = x_new
    full_velocities[particle_index] = v_new


# ---------------------------------------------------------------------------
# Per-edge local XPBD solve (parallel Jacobi alternative to block-Thomas)
# ---------------------------------------------------------------------------


@wp.func
def _safe_inverse_3x3(A: wp.mat33) -> wp.mat33:
    if wp.abs(wp.determinant(A)) < 1.0e-20:
        return wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return wp.inverse(A)


@wp.kernel
def _solve_rod_edges_local_batched(
    constraint_values: wp.array[wp.float32],
    compliance: wp.array[wp.float32],
    lambda_sum: wp.array[wp.float32],
    jacobian_rot: wp.array[wp.float32],
    inv_masses: wp.array[wp.float32],
    inv_inertia: wp.array[wp.float32],
    rod_offsets: wp.array[wp.int32],
    edge_offsets: wp.array[wp.int32],
    edge_rod_id: wp.array[wp.int32],
    delta_lambda: wp.array[wp.float32],
):
    """Per-edge local XPBD solve for rod stretch and Darboux constraints.

    Each edge independently assembles its local diagonal 6x6 JMJT block
    and solves the coupled stretch-Darboux system via Schur complement
    (two 3x3 solves).  Inter-edge coupling present in the direct (Thomas)
    path is ignored, trading convergence rate for full parallelism.
    """
    global_edge = wp.tid()
    rod_id = edge_rod_id[global_edge]
    local_edge = global_edge - edge_offsets[rod_id]
    p0_idx = rod_offsets[rod_id] + local_edge
    p1_idx = p0_idx + 1

    w0 = inv_masses[p0_idx]
    w1 = inv_masses[p1_idx]
    w_sum = w0 + w1

    i = global_edge
    base = i * 6
    reg = 1.0e-6

    # Stretch rows (0-2), particle 0 (cols 0-2)
    jr0_s0 = wp.vec3(
        jacobian_rot[_warp_jacobian_index(i, 0, 0)],
        jacobian_rot[_warp_jacobian_index(i, 0, 1)],
        jacobian_rot[_warp_jacobian_index(i, 0, 2)],
    )
    jr0_s1 = wp.vec3(
        jacobian_rot[_warp_jacobian_index(i, 1, 0)],
        jacobian_rot[_warp_jacobian_index(i, 1, 1)],
        jacobian_rot[_warp_jacobian_index(i, 1, 2)],
    )
    jr0_s2 = wp.vec3(
        jacobian_rot[_warp_jacobian_index(i, 2, 0)],
        jacobian_rot[_warp_jacobian_index(i, 2, 1)],
        jacobian_rot[_warp_jacobian_index(i, 2, 2)],
    )
    # Stretch rows (0-2), particle 1 (cols 3-5)
    jr1_s0 = wp.vec3(
        jacobian_rot[_warp_jacobian_index(i, 0, 3)],
        jacobian_rot[_warp_jacobian_index(i, 0, 4)],
        jacobian_rot[_warp_jacobian_index(i, 0, 5)],
    )
    jr1_s1 = wp.vec3(
        jacobian_rot[_warp_jacobian_index(i, 1, 3)],
        jacobian_rot[_warp_jacobian_index(i, 1, 4)],
        jacobian_rot[_warp_jacobian_index(i, 1, 5)],
    )
    jr1_s2 = wp.vec3(
        jacobian_rot[_warp_jacobian_index(i, 2, 3)],
        jacobian_rot[_warp_jacobian_index(i, 2, 4)],
        jacobian_rot[_warp_jacobian_index(i, 2, 5)],
    )
    # Darboux rows (3-5), particle 0 (cols 0-2)
    jr0_d0 = wp.vec3(
        jacobian_rot[_warp_jacobian_index(i, 3, 0)],
        jacobian_rot[_warp_jacobian_index(i, 3, 1)],
        jacobian_rot[_warp_jacobian_index(i, 3, 2)],
    )
    jr0_d1 = wp.vec3(
        jacobian_rot[_warp_jacobian_index(i, 4, 0)],
        jacobian_rot[_warp_jacobian_index(i, 4, 1)],
        jacobian_rot[_warp_jacobian_index(i, 4, 2)],
    )
    jr0_d2 = wp.vec3(
        jacobian_rot[_warp_jacobian_index(i, 5, 0)],
        jacobian_rot[_warp_jacobian_index(i, 5, 1)],
        jacobian_rot[_warp_jacobian_index(i, 5, 2)],
    )
    # Darboux rows (3-5), particle 1 (cols 3-5)
    jr1_d0 = wp.vec3(
        jacobian_rot[_warp_jacobian_index(i, 3, 3)],
        jacobian_rot[_warp_jacobian_index(i, 3, 4)],
        jacobian_rot[_warp_jacobian_index(i, 3, 5)],
    )
    jr1_d1 = wp.vec3(
        jacobian_rot[_warp_jacobian_index(i, 4, 3)],
        jacobian_rot[_warp_jacobian_index(i, 4, 4)],
        jacobian_rot[_warp_jacobian_index(i, 4, 5)],
    )
    jr1_d2 = wp.vec3(
        jacobian_rot[_warp_jacobian_index(i, 5, 3)],
        jacobian_rot[_warp_jacobian_index(i, 5, 4)],
        jacobian_rot[_warp_jacobian_index(i, 5, 5)],
    )

    Ij0_s0 = _inv_inertia_mul_vec(inv_inertia, p0_idx, jr0_s0)
    Ij0_s1 = _inv_inertia_mul_vec(inv_inertia, p0_idx, jr0_s1)
    Ij0_s2 = _inv_inertia_mul_vec(inv_inertia, p0_idx, jr0_s2)
    Ij1_s0 = _inv_inertia_mul_vec(inv_inertia, p1_idx, jr1_s0)
    Ij1_s1 = _inv_inertia_mul_vec(inv_inertia, p1_idx, jr1_s1)
    Ij1_s2 = _inv_inertia_mul_vec(inv_inertia, p1_idx, jr1_s2)

    Ij0_d0 = _inv_inertia_mul_vec(inv_inertia, p0_idx, jr0_d0)
    Ij0_d1 = _inv_inertia_mul_vec(inv_inertia, p0_idx, jr0_d1)
    Ij0_d2 = _inv_inertia_mul_vec(inv_inertia, p0_idx, jr0_d2)
    Ij1_d0 = _inv_inertia_mul_vec(inv_inertia, p1_idx, jr1_d0)
    Ij1_d1 = _inv_inertia_mul_vec(inv_inertia, p1_idx, jr1_d1)
    Ij1_d2 = _inv_inertia_mul_vec(inv_inertia, p1_idx, jr1_d2)

    # Assemble the 6x6 JMJT as three 3x3 blocks:
    #   A_ss = stretch-stretch (+ (w0+w1)*I on diagonal)
    #   A_dd = Darboux-Darboux (rotation-only)
    #   A_sd = stretch-Darboux coupling (A_ds = A_sd^T)
    s00 = w_sum + wp.dot(jr0_s0, Ij0_s0) + wp.dot(jr1_s0, Ij1_s0) + compliance[base + 0] + reg
    s01 = wp.dot(jr0_s0, Ij0_s1) + wp.dot(jr1_s0, Ij1_s1)
    s02 = wp.dot(jr0_s0, Ij0_s2) + wp.dot(jr1_s0, Ij1_s2)
    s10 = wp.dot(jr0_s1, Ij0_s0) + wp.dot(jr1_s1, Ij1_s0)
    s11 = w_sum + wp.dot(jr0_s1, Ij0_s1) + wp.dot(jr1_s1, Ij1_s1) + compliance[base + 1] + reg
    s12 = wp.dot(jr0_s1, Ij0_s2) + wp.dot(jr1_s1, Ij1_s2)
    s20 = wp.dot(jr0_s2, Ij0_s0) + wp.dot(jr1_s2, Ij1_s0)
    s21 = wp.dot(jr0_s2, Ij0_s1) + wp.dot(jr1_s2, Ij1_s1)
    s22 = w_sum + wp.dot(jr0_s2, Ij0_s2) + wp.dot(jr1_s2, Ij1_s2) + compliance[base + 2] + reg

    d00 = wp.dot(jr0_d0, Ij0_d0) + wp.dot(jr1_d0, Ij1_d0) + compliance[base + 3] + reg
    d01 = wp.dot(jr0_d0, Ij0_d1) + wp.dot(jr1_d0, Ij1_d1)
    d02 = wp.dot(jr0_d0, Ij0_d2) + wp.dot(jr1_d0, Ij1_d2)
    d10 = wp.dot(jr0_d1, Ij0_d0) + wp.dot(jr1_d1, Ij1_d0)
    d11 = wp.dot(jr0_d1, Ij0_d1) + wp.dot(jr1_d1, Ij1_d1) + compliance[base + 4] + reg
    d12 = wp.dot(jr0_d1, Ij0_d2) + wp.dot(jr1_d1, Ij1_d2)
    d20 = wp.dot(jr0_d2, Ij0_d0) + wp.dot(jr1_d2, Ij1_d0)
    d21 = wp.dot(jr0_d2, Ij0_d1) + wp.dot(jr1_d2, Ij1_d1)
    d22 = wp.dot(jr0_d2, Ij0_d2) + wp.dot(jr1_d2, Ij1_d2) + compliance[base + 5] + reg

    A_sd = wp.mat33(
        wp.dot(jr0_s0, Ij0_d0) + wp.dot(jr1_s0, Ij1_d0),
        wp.dot(jr0_s0, Ij0_d1) + wp.dot(jr1_s0, Ij1_d1),
        wp.dot(jr0_s0, Ij0_d2) + wp.dot(jr1_s0, Ij1_d2),
        wp.dot(jr0_s1, Ij0_d0) + wp.dot(jr1_s1, Ij1_d0),
        wp.dot(jr0_s1, Ij0_d1) + wp.dot(jr1_s1, Ij1_d1),
        wp.dot(jr0_s1, Ij0_d2) + wp.dot(jr1_s1, Ij1_d2),
        wp.dot(jr0_s2, Ij0_d0) + wp.dot(jr1_s2, Ij1_d0),
        wp.dot(jr0_s2, Ij0_d1) + wp.dot(jr1_s2, Ij1_d1),
        wp.dot(jr0_s2, Ij0_d2) + wp.dot(jr1_s2, Ij1_d2),
    )

    # Solve the coupled 6x6 system via Schur complement:
    #   S = A_dd - A_ds * A_ss^{-1} * A_sd
    #   dl_d = S^{-1} * (rhs_d - A_ds * A_ss^{-1} * rhs_s)
    #   dl_s = A_ss^{-1} * (rhs_s - A_sd * dl_d)
    rhs_s = wp.vec3(
        -constraint_values[base + 0] - compliance[base + 0] * lambda_sum[base + 0],
        -constraint_values[base + 1] - compliance[base + 1] * lambda_sum[base + 1],
        -constraint_values[base + 2] - compliance[base + 2] * lambda_sum[base + 2],
    )
    rhs_d = wp.vec3(
        -constraint_values[base + 3] - compliance[base + 3] * lambda_sum[base + 3],
        -constraint_values[base + 4] - compliance[base + 4] * lambda_sum[base + 4],
        -constraint_values[base + 5] - compliance[base + 5] * lambda_sum[base + 5],
    )

    A_ss = wp.mat33(s00, s01, s02, s10, s11, s12, s20, s21, s22)
    A_dd = wp.mat33(d00, d01, d02, d10, d11, d12, d20, d21, d22)
    A_ss_inv = _safe_inverse_3x3(A_ss)
    A_ds = wp.transpose(A_sd)

    X = A_ss_inv * A_sd
    S = A_dd - A_ds * X

    y_s = A_ss_inv * rhs_s
    dl_d = _safe_inverse_3x3(S) * (rhs_d - A_ds * y_s)
    dl_s = y_s - X * dl_d

    delta_lambda[base + 0] = dl_s[0]
    delta_lambda[base + 1] = dl_s[1]
    delta_lambda[base + 2] = dl_s[2]
    delta_lambda[base + 3] = dl_d[0]
    delta_lambda[base + 4] = dl_d[1]
    delta_lambda[base + 5] = dl_d[2]


class _XPBDRodProjector:
    """Compact batched rod projector that reuses the standalone rod kernels.

    Supports two solve strategies selected via *solve_method*:

    * ``"direct"`` — Block-Thomas (tridiagonal) solve that couples all edges
      within a rod.  Better convergence per iteration but sequential per rod.
    * ``"local"`` — Per-edge parallel Jacobi solve that ignores inter-edge
      coupling.  Fully parallel across all edges but needs more iterations.
    """

    def __init__(self, model: Model, solve_method: str = "direct"):
        if solve_method not in ("direct", "local"):
            raise ValueError(f"rod solve_method must be 'direct' or 'local', got {solve_method!r}")
        self._solve_method = solve_method

        rod_count = model.get_custom_frequency_count("xpbd:rod")
        rod_particle_count = model.get_custom_frequency_count("xpbd:rod_particle")
        edge_count = model.get_custom_frequency_count("xpbd:rod_edge")

        if rod_count == 0 or rod_particle_count == 0 or edge_count == 0:
            raise ValueError("XPBD rod projector requires non-empty xpbd rod custom attributes.")

        self.device = model.device
        self.rod_count = rod_count
        self.total_particles = rod_particle_count
        self.total_edges = edge_count
        self.total_dofs = edge_count * 6

        xpbd = model.xpbd
        self.particle_indices = xpbd.particle_index
        self.quat_inv_masses = xpbd.quat_inv_mass
        self.rest_lengths = xpbd.rest_length
        self.rest_darboux = xpbd.rest_darboux
        self.bend_stiffness = xpbd.bend_stiffness
        self.young_modulus = xpbd.young_modulus
        self.torsion_modulus = xpbd.torsion_modulus

        particle_counts = np.asarray(xpbd.particle_count.numpy(), dtype=np.int32)
        edge_counts = np.asarray(xpbd.edge_count.numpy(), dtype=np.int32)
        particle_starts = np.asarray(xpbd.particle_start.numpy(), dtype=np.int32)
        edge_starts = np.asarray(xpbd.edge_start.numpy(), dtype=np.int32)

        particle_rod_id = np.repeat(np.arange(rod_count, dtype=np.int32), particle_counts)
        edge_rod_id = np.repeat(np.arange(rod_count, dtype=np.int32), edge_counts)

        rod_offsets = np.concatenate((particle_starts, np.array([self.total_particles], dtype=np.int32)))
        edge_offsets = np.concatenate((edge_starts, np.array([self.total_edges], dtype=np.int32)))

        self.rod_offsets = wp.array(rod_offsets, dtype=wp.int32, device=self.device)
        self.edge_offsets = wp.array(edge_offsets, dtype=wp.int32, device=self.device)
        self.particle_rod_id = wp.array(particle_rod_id, dtype=wp.int32, device=self.device)
        self.edge_rod_id = wp.array(edge_rod_id, dtype=wp.int32, device=self.device)

        self.positions = wp.zeros(self.total_particles, dtype=wp.vec3, device=self.device)
        self.orientations = wp.zeros(self.total_particles, dtype=wp.quat, device=self.device)
        self.predicted_orientations = wp.zeros(self.total_particles, dtype=wp.quat, device=self.device)
        self.prev_orientations = wp.zeros(self.total_particles, dtype=wp.quat, device=self.device)
        self.angular_velocities = wp.zeros(self.total_particles, dtype=wp.vec3, device=self.device)
        self.torques = wp.zeros(self.total_particles, dtype=wp.vec3, device=self.device)
        self.inv_masses = wp.zeros(self.total_particles, dtype=wp.float32, device=self.device)
        self.inv_inertia = wp.zeros(self.total_particles * 9, dtype=wp.float32, device=self.device)
        self.inv_inertia_local_diag = wp.array(
            [wp.vec3(1.0, 1.0, 1.0)] * rod_count,
            dtype=wp.vec3,
            device=self.device,
        )

        self.constraint_values = wp.zeros(self.total_dofs, dtype=wp.float32, device=self.device)
        self.compliance = wp.zeros(self.total_dofs, dtype=wp.float32, device=self.device)
        self.lambda_sum = wp.zeros(self.total_dofs, dtype=wp.float32, device=self.device)
        self.jacobian_pos = wp.zeros(self.total_edges * 36, dtype=wp.float32, device=self.device)
        self.jacobian_rot = wp.zeros(self.total_edges * 36, dtype=wp.float32, device=self.device)

        self.delta_lambda = wp.zeros(self.total_dofs, dtype=wp.float32, device=self.device)

        # Buffers only needed by the direct (Thomas) path
        if solve_method == "direct":
            self.rhs = wp.zeros(self.total_dofs, dtype=wp.float32, device=self.device)
            self.diag_blocks = wp.zeros(self.total_edges * 36, dtype=wp.float32, device=self.device)
            self.offdiag_blocks = wp.zeros(self.total_edges * 36, dtype=wp.float32, device=self.device)
            self.c_blocks = wp.zeros(self.total_edges * 36, dtype=wp.float32, device=self.device)
            self.d_prime = wp.zeros(self.total_edges * 6, dtype=wp.float32, device=self.device)

        self.pos_corrections = wp.zeros(self.total_particles, dtype=wp.vec3, device=self.device)
        self.rot_corrections = wp.zeros(self.total_particles, dtype=wp.vec3, device=self.device)
        self._delta_lambda_max = wp.zeros(1, dtype=wp.float32, device=self.device)
        self._correction_max = wp.zeros(1, dtype=wp.float32, device=self.device)

        wp.launch(
            kernel=_gather_float,
            dim=self.total_particles,
            inputs=[model.particle_inv_mass, self.particle_indices],
            outputs=[self.inv_masses],
            device=self.device,
        )

    @classmethod
    def from_model(cls, model: Model, solve_method: str = "direct") -> _XPBDRodProjector | None:
        if not hasattr(model, "xpbd"):
            return None
        try:
            if model.get_custom_frequency_count("xpbd:rod") == 0:
                return None
        except KeyError:
            return None
        return cls(model, solve_method=solve_method)

    def begin_step(self, state_in: State, dt: float, angular_damping: float) -> None:
        self.orientations.assign(state_in.xpbd.orientation)
        self.predicted_orientations.assign(self.orientations)
        self.prev_orientations.assign(self.orientations)
        self.angular_velocities.assign(state_in.xpbd.angular_velocity)
        self.torques.assign(state_in.xpbd.torque)
        self.lambda_sum.zero_()

        wp.launch(
            kernel=_warp_predict_rotations_batched,
            dim=self.total_particles,
            inputs=[
                self.orientations,
                self.angular_velocities,
                self.torques,
                self.quat_inv_masses,
                float(dt),
                float(angular_damping),
                self.predicted_orientations,
            ],
            device=self.device,
        )

        wp.launch(
            kernel=_warp_prepare_compliance_batched,
            dim=self.total_edges,
            inputs=[
                self.rest_lengths,
                self.bend_stiffness,
                self.edge_rod_id,
                self.young_modulus,
                self.torsion_modulus,
                float(dt),
                self.compliance,
            ],
            device=self.device,
        )

    def project(
        self,
        particle_q: wp.array,
        particle_qd: wp.array,
        particle_q_init: wp.array,
        particle_flags: wp.array,
        dt: float,
        particle_max_velocity: float,
    ) -> None:
        wp.launch(
            kernel=_gather_vec3,
            dim=self.total_particles,
            inputs=[particle_q, self.particle_indices],
            outputs=[self.positions],
            device=self.device,
        )

        wp.launch(
            kernel=_warp_update_constraints_batched_v2,
            dim=self.total_edges,
            inputs=[
                self.positions,
                self.predicted_orientations,
                self.rest_lengths,
                self.rest_darboux,
                self.rod_offsets,
                self.edge_offsets,
                self.edge_rod_id,
                self.constraint_values,
            ],
            device=self.device,
        )

        wp.launch(
            kernel=_warp_compute_jacobians_batched,
            dim=self.total_edges,
            inputs=[
                self.predicted_orientations,
                self.rest_lengths,
                self.rod_offsets,
                self.edge_offsets,
                self.edge_rod_id,
                self.jacobian_pos,
                self.jacobian_rot,
            ],
            device=self.device,
        )

        wp.launch(
            kernel=_warp_compute_inv_inertia_world_batched,
            dim=self.total_particles,
            inputs=[
                self.predicted_orientations,
                self.quat_inv_masses,
                self.inv_inertia_local_diag,
                self.particle_rod_id,
                self.inv_inertia,
            ],
            device=self.device,
        )

        # ---- Solve (method-specific) ----
        if self._solve_method == "direct":
            wp.launch(
                kernel=_warp_assemble_jmjt_blocks_batched,
                dim=self.total_edges,
                inputs=[
                    self.jacobian_pos,
                    self.jacobian_rot,
                    self.compliance,
                    self.inv_masses,
                    self.inv_inertia,
                    self.rod_offsets,
                    self.edge_offsets,
                    self.edge_rod_id,
                    self.diag_blocks,
                    self.offdiag_blocks,
                ],
                device=self.device,
            )
            wp.launch(
                kernel=_warp_build_rhs,
                dim=self.total_dofs,
                inputs=[self.constraint_values, self.compliance, self.lambda_sum, int(self.total_dofs), self.rhs],
                device=self.device,
            )
            wp.launch(
                kernel=_warp_block_thomas_solve_batched,
                dim=self.rod_count,
                inputs=[
                    self.diag_blocks,
                    self.offdiag_blocks,
                    self.rhs,
                    self.edge_offsets,
                    int(self.rod_count),
                    self.c_blocks,
                    self.d_prime,
                    self.delta_lambda,
                ],
                device=self.device,
            )
        else:
            wp.launch(
                kernel=_solve_rod_edges_local_batched,
                dim=self.total_edges,
                inputs=[
                    self.constraint_values,
                    self.compliance,
                    self.lambda_sum,
                    self.jacobian_rot,
                    self.inv_masses,
                    self.inv_inertia,
                    self.rod_offsets,
                    self.edge_offsets,
                    self.edge_rod_id,
                    self.delta_lambda,
                ],
                device=self.device,
            )

        # ---- Compute and apply corrections from delta_lambda ----
        wp.launch(kernel=_warp_zero_vec3, dim=self.total_particles, inputs=[self.pos_corrections], device=self.device)
        wp.launch(kernel=_warp_zero_vec3, dim=self.total_particles, inputs=[self.rot_corrections], device=self.device)
        wp.launch(kernel=_warp_zero_float, dim=1, inputs=[self._delta_lambda_max], device=self.device)
        wp.launch(kernel=_warp_zero_float, dim=1, inputs=[self._correction_max], device=self.device)

        wp.launch(
            kernel=_warp_compute_corrections_parallel_batched,
            dim=self.total_edges,
            inputs=[
                self.positions,
                self.inv_masses,
                self.quat_inv_masses,
                self.inv_inertia,
                self.jacobian_pos,
                self.jacobian_rot,
                self.delta_lambda,
                self.lambda_sum,
                self.rod_offsets,
                self.edge_offsets,
                self.edge_rod_id,
                self.pos_corrections,
                self.rot_corrections,
                self._delta_lambda_max,
                self._correction_max,
            ],
            device=self.device,
        )

        wp.launch(
            kernel=_warp_apply_accumulated_corrections,
            dim=self.total_particles,
            inputs=[
                self.positions,
                self.predicted_orientations,
                self.pos_corrections,
                self.rot_corrections,
                int(self.total_particles),
            ],
            device=self.device,
        )

        wp.launch(
            kernel=_scatter_positions_and_velocities,
            dim=self.total_particles,
            inputs=[
                self.positions,
                self.particle_indices,
                particle_q_init,
                particle_flags,
                float(dt),
                float(particle_max_velocity),
            ],
            outputs=[particle_q, particle_qd],
            device=self.device,
        )

    def finalize_step(self, state_out: State, dt: float) -> None:
        wp.launch(
            kernel=_warp_integrate_rotations_batched,
            dim=self.total_particles,
            inputs=[
                self.orientations,
                self.predicted_orientations,
                self.prev_orientations,
                self.angular_velocities,
                self.quat_inv_masses,
                float(dt),
            ],
            device=self.device,
        )
        state_out.xpbd.orientation.assign(self.orientations)
        state_out.xpbd.angular_velocity.assign(self.angular_velocities)
