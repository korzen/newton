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

from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.slicer.vtk_loader import load_vtk_unstructured_grid

MESH_PATH = Path(__file__).resolve().parent / "vsd_device" / "mesh3.1.vtk"
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


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.solver_type = args.solver
        self.sim_time = 0.0
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 8
        self.iterations = 8
        self.sim_dt = self.frame_dt / self.sim_substeps
        self._needs_graph_recapture = False

        if self.solver_type not in {"xpbd", "vbd", "fem"}:
            raise ValueError("The VSD device example only supports the XPBD, VBD, and FEM solvers.")

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
        self.mesh_scale = 0.05 * float(args.scale)
        ox, oy, oz = (float(v) for v in args.offset)
        self.mesh_pos = wp.vec3(0.0 + ox, 0.0 + oy, 0.45 + oz)

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

        # Color the mesh for VBD solver
        builder.color()

        self.model = builder.finalize()
        self.model.soft_contact_ke = 1.0e2
        self.model.soft_contact_kd = 0
        self.model.soft_contact_mu = 1.0

        self.solver = self._create_solver()

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.contacts = self.model.contacts()

        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(wp.vec3(1.8, -2.0, 1.1), -18.0, 132.0)

        self.capture()

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
                iterations=1,
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

            # swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

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
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

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
            "--offset",
            help="Translation vector (x y z) [m] added to the initial mesh position.",
            type=float,
            nargs=3,
            metavar=("X", "Y", "Z"),
            default=[0.0, 0.0, 0.0],
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
