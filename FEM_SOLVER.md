# SolverFEM Development Notes

This document records the current state of the experimental Warp FEM-backed
`newton.solvers.SolverFEM` implementation and the most useful next steps.

## What Was Developed

`SolverFEM` is now an experimental implicit tet4 soft-body FEM solver intended
for GPU tissue-deformation demos and research prototypes. It is not intended to
match FEBio feature coverage or numerical behavior.

The solver currently supports:

- A single Newton soft-body tet mesh whose particles span `model.particle_q`.
- Stable Neo-Hookean volumetric elasticity for tet4 elements.
- Per-element first and second Lame parameters read from
  `model.tet_materials[:, 0]` and `model.tet_materials[:, 1]`.
- Optional scalar `k_mu` and `k_lambda` constructor overrides that fill all
  elements with the same material value.
- Lumped nodal mass from `model.particle_mass`.
- Fixed or inactive particles through zero mass or disabled
  `ParticleFlags.ACTIVE`.
- External user particle forces, gravity, and existing Newton soft-contact
  force scattering.
- Optional post-solve position projection for active particles against
  infinite `GeoType.PLANE` shapes with `ShapeFlags.COLLIDE_PARTICLES`.
- Optional post-solve mouse-drag projection of a picked soft-body surface
  point, driven by the viewer's right-click particle picking state.
- Mass-proportional damping through `k_damp`.
- Sparse linear solves through Warp BSR matrices and CG with diagonal
  preconditioning.

## Time Integration

The previous velocity-style FEM loop was replaced with a displacement-based
backward-Euler Newton loop.

At the start of each `step()`:

- `u_n = state_in.particle_q - rest_positions`
- `v_n = state_in.particle_qd`
- The current nonlinear iterate `u` is initialized to `u_n`.

Each nonlinear iteration assembles internal force and tangent stiffness from
the current displacement:

```text
f_int = f_int(u)
K = K(u)
```

It then solves for a displacement increment:

```text
A * delta_u = rhs

A =
    M / dt^2
  + k_damp * M / dt
  + K(u)

rhs =
    f_ext
  - f_int(u)
  - M * (u - u_n - dt * v_n) / dt^2
  - k_damp * M * (u - u_n) / dt
```

After the CG solve, the increment is accumulated:

```text
u = u + delta_u
```

At the end of the step:

```text
state_out.particle_q = rest_positions + u
state_out.particle_qd = (u - u_n) / dt
```

Fixed or inactive particle velocities are written as zero.

## Fixed-DOF Handling

Fixed particles are detected when either:

- `model.particle_mass[i] == 0.0`, or
- `(model.particle_flags[i] & ParticleFlags.ACTIVE) == 0`.

The solver builds a 3x3 block-diagonal Dirichlet projector each step:

- Identity block for fixed or inactive particles.
- Zero block for active dynamic particles.

Each Newton linear system is projected with:

```python
fem.project_linear_system(
    A,
    rhs,
    projector,
    fixed_value=None,
    normalize_projector=False,
)
```

The accumulation and final writeback kernels also defensively zero fixed
increments and velocities.

## Material Refresh

The solver owns device arrays:

- `_mu_e`
- `_lambda_e`

These are populated from `model.tet_materials` unless scalar overrides were
provided in the constructor.

`SolverFEM.refresh_material_parameters()` re-reads material data in place. This
is used by the VSD device example after the GUI changes tet stiffness values.
The refresh is scoped to `model.device` so it works even if the current Warp
device differs from the model device.

## Contact Scope

Contacts are currently external-force-only:

- Existing soft contacts are scattered into `f_ext`.
- Contact stiffness/tangents are not added to the Newton matrix.

`SolverFEM` also has an opt-in floor projection path through
`plane_contact_projection_iterations`. When enabled, the solver first runs the
same implicit displacement Newton solve, then projects active dynamic particles
out of infinite plane shapes and recomputes particle velocities from the
corrected positions. This v1 projection is normal-only, respects particle/shape
world filtering, treats plane bodies as kinematic obstacles, and is limited to
infinite planes such as those created by `ModelBuilder.add_ground_plane()`.
Finite planes, arbitrary SDF shapes, particle-particle contacts, and friction
constraints are not projected.

For interactive manipulation, `SolverFEM` can also consume a viewer-provided
soft pick via `set_particle_drag_constraint()`. The OpenGL viewer raycasts the
rendered soft-body surface triangles on right-click, stores the selected
barycentric particle triple and mouse target, and FEM projects that weighted
point to the target after each implicit solve. This is XPBD-style positional
dragging: selected particles are moved according to inverse mass and their
velocities are recomputed from the corrected positions. Rigid body picking keeps
using the existing force path.

This keeps the implementation simple and demo-oriented, but means contact is
not fully implicit and can require smaller time steps or more substeps.

## Public Demo Surface

The public constructor remains backward-compatible:

- `iterations` now means nonlinear Newton iterations.
- `cg_tol` and `cg_max_iters` remain the inner linear solve controls.
- No new public solver class or dependency was added.

The VSD device example uses FEM mode through:

```bash
uv run -m newton.examples vsd_device --solver fem
```

FEM mode now defaults to `iterations=4` in that example, while the existing GUI
iteration control still updates `solver.iterations`.
It also enables `plane_contact_projection_iterations=1` for hard floor
non-penetration and `particle_drag_projection_iterations=1` for right-click
mouse dragging in that demo. General `SolverFEM` users remain on the historical
FEM-only contact-force behavior unless they opt in to projection features.

## Tests Added

`newton/tests/test_solver_fem.py` covers:

- Rest stability with no gravity or external force.
- Gravity deformation with fixed nodes preserved exactly.
- Multiple Newton iterations producing a different displacement than one
  iteration on a small nonlinear fixed grid.
- Material refresh changing deformation after `model.tet_materials` edits.

Verification run so far:

```bash
uv run python -m py_compile \
  newton/_src/solvers/fem/solver_fem.py \
  newton/_src/solvers/fem/kernels.py \
  newton/tests/test_solver_fem.py \
  newton/examples/slicer/example_vsd_device.py \
  newton/solvers.py

uv run --extra dev -m newton.tests -k fem

uv run --extra dev -m newton.examples vsd_device \
  --solver fem --viewer null --num-frames 2 --test --quiet --device cuda:0
```

The targeted FEM tests passed on CPU and CUDA. The short VSD FEM smoke run
passed on CUDA. A direct `ruff check` was attempted but `ruff` was not installed
in the active environment.

## Current Limitations

The current implementation deliberately does not include:

- FEBio XML import.
- Multiple materials beyond per-element scalar Lame parameters.
- Mooney-Rivlin, HGO fibers, viscoelasticity, plasticity, or anisotropy.
- Mixed displacement-pressure elements.
- Shells, beams, hexes, wedges, or higher-order elements.
- Contact tangents or fully implicit contact.
- Self-contact.
- Rigid body coupling through a monolithic system.
- Sparse direct solves, AMG, or robust nonlinear globalization.
- `fp64=True` support.
- Multiple independently indexed soft-body meshes in one `SolverFEM`.

## Potential Next Steps

High-value next steps:

1. Add Newton convergence diagnostics.
   Track residual norm, CG iteration count, and whether each nonlinear solve
   converged. Expose this in debug logs or an optional stats object.

2. Add line search or increment damping.
   The current solver always accepts the full `delta_u`. A simple backtracking
   line search would improve robustness for large time steps, strong gravity,
   or high stiffness contrast.

3. Add better nonlinear tangent options.
   The current tangent is a Gauss-Newton-style SPD approximation. A fuller
   material tangent, or a selectable tangent mode, would improve convergence
   for large deformation.

4. Improve material modeling for tissue demos.
   Add optional nearly incompressible presets, spatial material maps, and
   possibly anisotropic fiber directions. Keep these demo-oriented unless a
   separate FEBio-parity project is started.

5. Add contact tangent support.
   Include soft-contact normal stiffness in the linear system for more stable
   ground and tool contact at larger time steps.

6. Support multiple soft bodies.
   The current solver assumes one soft mesh spanning the particle array.
   Supporting multiple meshes requires clear ownership of rest geometry,
   element ranges, material ranges, and fixed/projector ranges.

7. Improve solver performance.
   Reuse sparsity patterns where possible, reduce allocation churn during FEM
   assembly/projection, tune CG tolerances for interactive demos, and profile
   CUDA behavior on the VSD mesh.

8. Broaden tests.
   Add multi-element material-contrast tests, damping tests, external force
   tests, soft-contact smoke tests, and longer GPU stability tests.

9. Add a documented VSD FEM acceptance script.
   Keep a short headless smoke path for CI-like checks and a longer manual GPU
   path for 100-300 frame interactive stability testing.

10. Decide whether FEM stays demo-only or becomes a general solver.
    If it should become a production-facing solver, define the API for meshes,
    materials, contacts, diagnostics, and solver failure handling before adding
    many features.
