# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""warp.fem-backed FEM solver for volumetric soft bodies."""

from __future__ import annotations

import warnings

import numpy as np
import warp as wp
import warp.fem as fem
import warp.optim.linear as wpla
import warp.sparse as wps

import newton

from ...core.types import override
from ...sim import Contacts, Control, State
from ..flags import SolverNotifyFlags
from ..solver import SolverBase
from .kernels import (
    add_external_force,
    accumulate_displacement_delta,
    build_dirichlet_projector_blocks,
    build_mass_blocks,
    build_newton_rhs,
    compute_gravity_force,
    refresh_lame_parameters,
    scatter_soft_contact_forces,
    project_particles_against_infinite_planes,
    project_particle_drag_constraint,
    sync_displacement_state,
    write_state_outputs,
)

__all__ = ["SolverFEM"]


@wp.func
def _cofactor_3x3(F: wp.mat33) -> wp.mat33:
    """Cofactor matrix of a 3x3 matrix; equal to J * F^{-T} = adj(F)^T."""
    c00 = F[1, 1] * F[2, 2] - F[1, 2] * F[2, 1]
    c01 = -(F[1, 0] * F[2, 2] - F[1, 2] * F[2, 0])
    c02 = F[1, 0] * F[2, 1] - F[1, 1] * F[2, 0]
    c10 = -(F[0, 1] * F[2, 2] - F[0, 2] * F[2, 1])
    c11 = F[0, 0] * F[2, 2] - F[0, 2] * F[2, 0]
    c12 = -(F[0, 0] * F[2, 1] - F[0, 1] * F[2, 0])
    c20 = F[0, 1] * F[1, 2] - F[0, 2] * F[1, 1]
    c21 = -(F[0, 0] * F[1, 2] - F[0, 2] * F[1, 0])
    c22 = F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]
    return wp.mat33(c00, c01, c02, c10, c11, c12, c20, c21, c22)


@fem.integrand
def _neohookean_internal_force_form(
    s: fem.Sample,
    v: fem.Field,
    u_cur: fem.Field,
    mu_e: wp.array[float],
    lambda_e: wp.array[float],
):
    """P : grad(v) integrand for the Stable Neo-Hookean PK1 stress."""
    mu = mu_e[s.element_index]
    lam = lambda_e[s.element_index]
    F = wp.identity(n=3, dtype=float) + fem.grad(u_cur, s)
    J = wp.determinant(F)
    cof = _cofactor_3x3(F)
    # Stable Neo-Hookean (Smith et al. 2018):
    # Psi(F) = 0.5 * mu * (||F||^2 - 3) + 0.5 * lam * (J - 1 - mu/lam)^2
    # dPsi/dF = mu * F + (lam * (J - 1) - mu) * cof(F)
    P = mu * F + (lam * (J - 1.0) - mu) * cof
    return wp.ddot(fem.grad(v, s), P)


@fem.integrand
def _neohookean_tangent_form(
    s: fem.Sample,
    u: fem.Field,
    v: fem.Field,
    u_cur: fem.Field,
    mu_e: wp.array[float],
    lambda_e: wp.array[float],
):
    """Gauss-Newton approximation of grad(v) : (d^2 Psi/dF^2 : grad(u))."""
    mu = mu_e[s.element_index]
    lam = lambda_e[s.element_index]
    F = wp.identity(n=3, dtype=float) + fem.grad(u_cur, s)
    cof = _cofactor_3x3(F)
    grad_u = fem.grad(u, s)
    grad_v = fem.grad(v, s)
    # Gauss-Newton drops the d^2 J / dF^2 term and keeps the SPD outer-product
    # contribution from the volumetric energy.
    return mu * wp.ddot(grad_u, grad_v) + lam * wp.ddot(cof, grad_u) * wp.ddot(cof, grad_v)


class SolverFEM(SolverBase):
    """Experimental warp.fem-backed implicit FEM solver for tet soft bodies.

    Drives Newton's particle state through a displacement-based
    backward-Euler Newton solve using 3D tet4 Stable Neo-Hookean elements
    (Smith et al. 2018). Element assembly and sparse matrix construction are
    delegated to :mod:`warp.fem`. Newton's :class:`~newton.Contacts` arrays
    are consumed as external soft-contact forces; contact tangents are not
    included in the Newton matrix.

    The solver assumes a single soft mesh whose particles span the entire
    ``model.particle_q`` array (the standard layout produced by
    :meth:`~newton.ModelBuilder.add_soft_mesh`). It is intended for demos and
    research prototypes rather than FEBio feature parity.

    Args:
        model: The Newton model. Must contain tetrahedral elements
            (``model.tet_count > 0``).
        iterations: Number of Newton-Raphson iterations per ``step()``.
            With ``iterations=1`` this collapses to a single semi-implicit
            backward-Euler step (the default).
        k_mu: Optional scalar override for the first Lamé parameter μ [Pa].
            Defaults to per-element values from ``model.tet_materials[:, 0]``.
        k_lambda: Optional scalar override for the second Lamé parameter λ [Pa].
            Defaults to per-element values from ``model.tet_materials[:, 1]``.
        k_damp: Mass-proportional velocity damping coefficient [1/s].
            Default ``0.0``.
        contact_ke: Soft-contact stiffness [N/m]. Defaults to
            ``model.soft_contact_ke``.
        contact_kd: Soft-contact damping [N·s/m]. Defaults to
            ``model.soft_contact_kd``.
        contact_mu: Soft-contact Coulomb friction. Defaults to
            ``model.soft_contact_mu``.
        cg_tol: Relative tolerance for the inner CG solve.
        cg_max_iters: Maximum CG iterations per Newton iteration. ``0``
            means equal to the system size.
        plane_contact_projection_iterations: Number of post-solve position
            projection iterations against infinite plane shapes. ``0`` keeps
            the historical FEM-only soft-contact path.
        plane_contact_projection_relaxation: Normal correction multiplier for
            plane contact projection. ``1.0`` applies the full correction.
        particle_drag_projection_iterations: Number of post-solve position
            projection iterations for an externally supplied picked soft-body
            surface point. ``0`` disables mouse-drag projection.
        particle_drag_projection_relaxation: Correction multiplier for the
            particle drag projection. ``1.0`` applies the full correction.
        fp64: Reserved for future double-precision support; currently
            ignored with a warning.
    """

    def __init__(
        self,
        model: newton.Model,
        *,
        iterations: int = 1,
        k_mu: float | None = None,
        k_lambda: float | None = None,
        k_damp: float = 0.0,
        contact_ke: float | None = None,
        contact_kd: float | None = None,
        contact_mu: float | None = None,
        cg_tol: float = 1.0e-5,
        cg_max_iters: int = 0,
        plane_contact_projection_iterations: int = 0,
        plane_contact_projection_relaxation: float = 1.0,
        particle_drag_projection_iterations: int = 0,
        particle_drag_projection_relaxation: float = 1.0,
        fp64: bool = False,
    ):
        super().__init__(model)

        if model.tet_count == 0 or model.tet_indices is None:
            raise ValueError(
                "SolverFEM requires a model with tetrahedral elements; "
                "use ModelBuilder.add_soft_mesh() or add_soft_grid() to add one."
            )

        if fp64:
            warnings.warn(
                "SolverFEM does not yet support double precision; falling back to fp32.",
                stacklevel=2,
            )

        self.iterations = max(1, int(iterations))
        self._k_damp = float(k_damp)
        self._cg_tol = float(cg_tol)
        self._cg_max_iters = int(cg_max_iters)
        self.plane_contact_projection_iterations = max(0, int(plane_contact_projection_iterations))
        self.plane_contact_projection_relaxation = float(plane_contact_projection_relaxation)
        if self.plane_contact_projection_relaxation < 0.0:
            raise ValueError("plane_contact_projection_relaxation must be non-negative.")
        self.particle_drag_projection_iterations = max(0, int(particle_drag_projection_iterations))
        self.particle_drag_projection_relaxation = float(particle_drag_projection_relaxation)
        if self.particle_drag_projection_relaxation < 0.0:
            raise ValueError("particle_drag_projection_relaxation must be non-negative.")
        self._particle_drag_indices = None
        self._particle_drag_weights = None
        self._particle_drag_target = None
        self._particle_drag_point = None

        self._mu_override = k_mu
        self._lambda_override = k_lambda

        self._contact_ke = float(contact_ke) if contact_ke is not None else float(model.soft_contact_ke)
        self._contact_kd = float(contact_kd) if contact_kd is not None else float(model.soft_contact_kd)
        self._contact_mu = float(contact_mu) if contact_mu is not None else float(model.soft_contact_mu)

        device = model.device
        self.particle_count = int(model.particle_count)
        self.tet_count = int(model.tet_count)
        self._has_dirichlet_constraints = self._check_has_dirichlet_constraints(model)

        # Snapshot rest positions from the model (these were written by
        # ModelBuilder.finalize() and currently match state.particle_q before
        # the first step).
        rest_np = model.particle_q.numpy()
        tet_np = model.tet_indices.numpy().reshape(-1, 4).astype(np.int32)

        with wp.ScopedDevice(device):
            self._rest_positions = wp.array(rest_np, dtype=wp.vec3)
            positions = wp.array(rest_np, dtype=wp.vec3)
            tets = wp.array(tet_np, dtype=wp.int32)

            self._geo = fem.Tetmesh(tet_vertex_indices=tets, positions=positions)
            self._domain = fem.Cells(geometry=self._geo)
            self._u_space = fem.make_polynomial_space(self._geo, degree=1, dtype=wp.vec3)
            self._u_field = self._u_space.make_field()
            self._u_test = fem.make_test(space=self._u_space, domain=self._domain)
            self._u_trial = fem.make_trial(space=self._u_space, domain=self._domain)

            self._u_n = wp.zeros(self.particle_count, dtype=wp.vec3)
            self._v_n = wp.zeros(self.particle_count, dtype=wp.vec3)
            self._delta_u = wp.zeros(self.particle_count, dtype=wp.vec3)
            self._f_ext = wp.zeros(self.particle_count, dtype=wp.vec3)
            self._rhs = wp.zeros(self.particle_count, dtype=wp.vec3)
            self._mu_e = wp.zeros(self.tet_count, dtype=float)
            self._lambda_e = wp.zeros(self.tet_count, dtype=float)

            mass_blocks = wp.zeros(self.particle_count, dtype=wp.mat33)
            wp.launch(
                build_mass_blocks,
                dim=self.particle_count,
                inputs=[model.particle_mass],
                outputs=[mass_blocks],
            )
            self._mass_bsr = wps.bsr_diag(mass_blocks)

            projector_blocks = wp.zeros(self.particle_count, dtype=wp.mat33)
            self._dirichlet_projector = wps.bsr_diag(projector_blocks)

            self._refresh_material_parameters()

        self._axpy_work = wps.bsr_axpy_work_arrays()

        # Cache an empty body_q/body_qd/body_com to feed the contact kernel
        # when the scene has no rigid bodies.
        self._empty_body_q = wp.empty(0, dtype=wp.transform, device=device)
        self._empty_body_qd = wp.empty(0, dtype=wp.spatial_vector, device=device)
        self._empty_body_com = wp.empty(0, dtype=wp.vec3, device=device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @override
    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        del control
        model = self.model
        device = model.device
        n = self.particle_count

        with wp.ScopedDevice(device):
            if state_in.particle_q is None or state_in.particle_qd is None:
                raise ValueError("SolverFEM requires particle positions and velocities in state_in.")

            wp.launch(
                sync_displacement_state,
                dim=n,
                inputs=[self._rest_positions, state_in.particle_q, state_in.particle_qd],
                outputs=[self._u_n, self._v_n, self._u_field.dof_values],
            )

            # External force = gravity + (optional) user-supplied particle_f
            # + soft-contact response.
            wp.launch(
                compute_gravity_force,
                dim=n,
                inputs=[model.particle_mass, model.gravity, model.particle_world],
                outputs=[self._f_ext],
            )

            if state_in.particle_f is not None:
                wp.launch(
                    add_external_force,
                    dim=n,
                    inputs=[state_in.particle_f],
                    outputs=[self._f_ext],
                )

            self._apply_soft_contacts(state_in, contacts)

            if self._has_dirichlet_constraints:
                wp.launch(
                    build_dirichlet_projector_blocks,
                    dim=n,
                    inputs=[model.particle_mass, model.particle_flags],
                    outputs=[self._dirichlet_projector.values],
                )

            dt_inv = 1.0 / dt
            mass_scale = dt_inv * dt_inv + self._k_damp * dt_inv

            # Displacement Newton iterations for backward Euler.
            for _ in range(self.iterations):
                f_int = fem.integrate(
                    _neohookean_internal_force_form,
                    fields={"v": self._u_test, "u_cur": self._u_field},
                    values={"mu_e": self._mu_e, "lambda_e": self._lambda_e},
                    output_dtype=wp.vec3,
                )

                K = fem.integrate(
                    _neohookean_tangent_form,
                    fields={"u": self._u_trial, "v": self._u_test, "u_cur": self._u_field},
                    values={"mu_e": self._mu_e, "lambda_e": self._lambda_e},
                    output_dtype=wp.float32,
                )

                # System matrix A = M/dt^2 + k_damp*M/dt + K(u).
                A = wps.bsr_copy(self._mass_bsr)
                wps.bsr_axpy(K, A, alpha=1.0, beta=mass_scale, work_arrays=self._axpy_work)

                wp.launch(
                    build_newton_rhs,
                    dim=n,
                    inputs=[
                        self._f_ext,
                        f_int,
                        model.particle_mass,
                        self._u_field.dof_values,
                        self._u_n,
                        self._v_n,
                        dt,
                        self._k_damp,
                    ],
                    outputs=[self._rhs],
                )

                if self._has_dirichlet_constraints:
                    fem.project_linear_system(
                        A,
                        self._rhs,
                        self._dirichlet_projector,
                        fixed_value=None,
                        normalize_projector=False,
                    )

                self._delta_u.zero_()
                wpla.cg(
                    A,
                    b=self._rhs,
                    x=self._delta_u,
                    M=wpla.preconditioner(A, "diag"),
                    tol=self._cg_tol,
                    maxiter=self._cg_max_iters,
                    check_every=0 if device.is_capturing else 10,
                )

                wp.launch(
                    accumulate_displacement_delta,
                    dim=n,
                    inputs=[self._delta_u, model.particle_mass, model.particle_flags],
                    outputs=[self._u_field.dof_values],
                )

            wp.launch(
                write_state_outputs,
                dim=n,
                inputs=[
                    self._rest_positions,
                    self._u_field.dof_values,
                    self._u_n,
                    model.particle_mass,
                    model.particle_flags,
                    dt,
                ],
                outputs=[state_out.particle_q, state_out.particle_qd],
            )

            self._apply_plane_contact_projection(state_in, state_out, dt)
            self._apply_particle_drag_projection(state_in, state_out, dt)

    @override
    def notify_model_changed(self, flags: int) -> None:
        if flags & SolverNotifyFlags.SHAPE_PROPERTIES:
            self._refresh_material_parameters()

    def refresh_material_parameters(self) -> None:
        """Re-read Lamé parameters from ``model.tet_materials``.

        Convenience hook for example UIs that mutate ``model.tet_materials``
        in place and need the solver to pick up the new values without
        recreating the solver.
        """
        self._refresh_material_parameters()

    def set_particle_drag_constraint(
        self,
        particle_indices: wp.array,
        particle_weights: wp.array,
        target_world: wp.array,
        picked_point_world: wp.array,
    ) -> None:
        """Connect an external soft-body mouse-drag target to the FEM solver.

        Args:
            particle_indices: Three particle indices defining the picked point,
                or ``[-1, -1, -1]`` when inactive.
            particle_weights: Barycentric weights for ``particle_indices``.
            target_world: Single ``wp.vec3`` mouse target in physics space.
            picked_point_world: Single ``wp.vec3`` updated with the constrained
                point after projection, for viewer feedback.
        """
        self._particle_drag_indices = particle_indices
        self._particle_drag_weights = particle_weights
        self._particle_drag_target = target_world
        self._particle_drag_point = picked_point_world

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _check_has_dirichlet_constraints(model: newton.Model) -> bool:
        particle_mass = model.particle_mass.numpy()
        particle_flags = model.particle_flags.numpy()
        active = int(newton.ParticleFlags.ACTIVE)
        return bool(np.any((particle_flags & active) == 0) or np.any(particle_mass == 0.0))

    def _refresh_material_parameters(self) -> None:
        with wp.ScopedDevice(self.model.device):
            wp.launch(
                refresh_lame_parameters,
                dim=self.tet_count,
                inputs=[
                    self.model.tet_materials,
                    int(self._mu_override is not None),
                    float(self._mu_override) if self._mu_override is not None else 0.0,
                    int(self._lambda_override is not None),
                    float(self._lambda_override) if self._lambda_override is not None else 0.0,
                ],
                outputs=[self._mu_e, self._lambda_e],
            )

    def _apply_soft_contacts(self, state_in: State, contacts: Contacts | None) -> None:
        if contacts is None:
            return
        if contacts.soft_contact_max <= 0:
            return
        if state_in.particle_q is None or state_in.particle_qd is None:
            return

        model = self.model
        body_q = state_in.body_q if state_in.body_q is not None else self._empty_body_q
        body_qd = state_in.body_qd if state_in.body_qd is not None else self._empty_body_qd
        body_com = model.body_com if model.body_com is not None else self._empty_body_com

        wp.launch(
            scatter_soft_contact_forces,
            dim=contacts.soft_contact_max,
            inputs=[
                contacts.soft_contact_count,
                contacts.soft_contact_particle,
                contacts.soft_contact_shape,
                contacts.soft_contact_body_pos,
                contacts.soft_contact_body_vel,
                contacts.soft_contact_normal,
                model.shape_body,
                body_q,
                body_qd,
                body_com,
                state_in.particle_q,
                state_in.particle_qd,
                model.particle_radius,
                self._contact_ke,
                self._contact_kd,
                self._contact_mu,
            ],
            outputs=[self._f_ext],
        )

    def _apply_plane_contact_projection(self, state_in: State, state_out: State, dt: float) -> None:
        if self.plane_contact_projection_iterations <= 0:
            return
        if self.model.shape_count <= 0:
            return
        if state_in.particle_q is None or state_out.particle_q is None or state_out.particle_qd is None:
            return

        model = self.model
        body_q = state_in.body_q if state_in.body_q is not None else self._empty_body_q

        for _ in range(self.plane_contact_projection_iterations):
            wp.launch(
                project_particles_against_infinite_planes,
                dim=self.particle_count,
                inputs=[
                    state_in.particle_q,
                    state_out.particle_q,
                    state_out.particle_qd,
                    model.particle_mass,
                    model.particle_flags,
                    model.particle_radius,
                    model.particle_world,
                    model.shape_transform,
                    model.shape_body,
                    model.shape_type,
                    model.shape_scale,
                    model.shape_flags,
                    model.shape_world,
                    body_q,
                    model.shape_count,
                    dt,
                    self.plane_contact_projection_relaxation,
                ],
            )

    def _apply_particle_drag_projection(self, state_in: State, state_out: State, dt: float) -> None:
        if self.particle_drag_projection_iterations <= 0:
            return
        if self._particle_drag_indices is None:
            return
        if state_in.particle_q is None or state_out.particle_q is None or state_out.particle_qd is None:
            return

        model = self.model
        for _ in range(self.particle_drag_projection_iterations):
            wp.launch(
                project_particle_drag_constraint,
                dim=1,
                inputs=[
                    state_in.particle_q,
                    state_out.particle_q,
                    state_out.particle_qd,
                    model.particle_mass,
                    model.particle_flags,
                    self._particle_drag_indices,
                    self._particle_drag_weights,
                    self._particle_drag_target,
                    self._particle_drag_point,
                    dt,
                    self.particle_drag_projection_relaxation,
                ],
            )
