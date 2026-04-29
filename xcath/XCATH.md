# XCATH Catheter Simulation

Interactive demo of a catheter (256-particle Cosserat elastic rod) being inserted
through an aortic vessel mesh. Uses Newton's XPBD elastic-rod solver with three
sample-local extensions:

- **Track-guided insertion** — non-tip rod particles slide along a fixed insertion
  axis, simulating the rigid catheter shaft being fed through a guide.
- **Vessel containment** — rod particles are projected to stay inside the closed
  vessel mesh. Two collision paths are available: a single-closest-point SDF
  query, or an AABB-based mesh-edge path that handles vertex/triangle and
  rod-segment/mesh-edge contacts.
- **Bendable tip** — a configurable number of edges at the rod's distal tip have
  a non-zero rest curvature, letting the user steer.

The vessel mesh is loaded from `DynamicAorta.usdc` and can be repositioned/
rotated at runtime to test alignment.

## Running

```bash
uv run xcath/xcath.py            # direct
```

The viewer accepts the same flags as other Newton examples (`--device`,
`--output-path` for USD recording, etc.).

## Keyboard controls

All keys are held-to-act except `G`, which is edge-triggered.

| Key                       | Action                                          |
|---------------------------|-------------------------------------------------|
| `PgUp`, `2`, `I`          | Insert (push catheter forward along track)      |
| `PgDn`, `1`, `K`          | Retract (pull catheter back along track)        |
| `,` (comma), `J`          | Rotate root counter-clockwise around shaft axis |
| `.` (period), `L`         | Rotate root clockwise around shaft axis         |
| `=` / `+`, NumPad `+`     | Increase tip bend angle                         |
| `-`, NumPad `-`           | Decrease tip bend angle                         |
| `G`                       | Toggle gravity on/off (also reflected in GUI)   |

Insertion is clamped to `>= 0`. Tip bend is clamped to `[-1.5, 1.5]` rad in code
(slider exposes `[-1.8, 1.8]`).

## ImGui panel

### Catheter pose

| Control            | Range          | Description                                                |
|--------------------|----------------|------------------------------------------------------------|
| Insertion          | 0 – 10 m       | Distance the root is advanced along the track axis.        |
| Tip Bend           | -1.8 – 1.8 rad | Total rest curvature applied across the bendable tip.      |
| Tip Bend Segments  | 1 – 30 edges   | Number of distal edges that share the tip bend curvature.  |
| Root Rotation      | -π – π rad     | Twist applied to the root particle around the shaft axis.  |

The tip bend is distributed evenly: each of the last *N* edges gets a rest
Darboux of `bend / N` around the local x-axis.

### Toggles

| Toggle                                | Effect                                                                                     |
|---------------------------------------|--------------------------------------------------------------------------------------------|
| Gravity                               | Adds `-9.81` m/s² along world z. Off by default.                                           |
| Track Sliding                         | Snaps non-tip particles toward the insertion axis with `track_stiffness`.                  |
| Mesh Collision                        | Enables vessel containment. If off, the rod can pass freely through the aorta.             |
| Use Mesh Edge Collision Path          | Switches to the AABB / per-triangle containment kernels. If off, single SDF query is used. |
| Use Smooth Collision Normals          | Uses area-weighted vertex normals (smoothed across triangles) instead of flat face normal. |
| Collision Before Rod Constraints      | Runs vessel containment *before* the XPBD solve. Rare; gives constraints a head start.     |
| Collision After Rod Constraints       | Runs vessel containment *after* the XPBD solve. Default and recommended.                   |

The two stage toggles are independent — both can be on (collision runs twice
per substep) or both off (no containment regardless of `Mesh Collision`).

### Material & damping

| Control          | Range        | Description                                                  |
|------------------|--------------|--------------------------------------------------------------|
| Bend Stiffness   | 0 – 1        | Per-edge bend XPBD stiffness multiplier (x and y components).|
| Twist Stiffness  | 0 – 1        | Per-edge twist XPBD stiffness multiplier (z component).      |
| Young Modulus    | Pa           | Material Young's modulus, used in compliance computation.    |
| Torsion Modulus  | Pa           | Material torsion modulus, used in compliance computation.    |
| Track Stiffness  | 0 – 1        | Blend factor for track snapping (0 = free, 1 = locked).      |
| Pos Damping      | 0 – `DAMPING_SLIDER_MAX` | Linear velocity damping applied during prediction. |
| Rot Damping      | 0 – `DAMPING_SLIDER_MAX` | Angular velocity damping applied during prediction.|

Lower stiffness and modulus make the rod "softer". Increase if the catheter
visibly stretches or twists during insertion.

### Solver

| Control              | Range   | Description                                                                |
|----------------------|---------|----------------------------------------------------------------------------|
| Substeps             | 1 – 32  | Solver substeps per render frame. More = more stable, slower.              |
| Collision Iterations | 1 – 16  | How many times vessel containment runs per substep (Gauss-Seidel-style).   |
| Collision Radius     | 0.001 – 0.05 m | Effective rod radius for containment, also drives `target_phi` and `max_dist`. |

A bigger `Collision Radius` keeps the rod farther from the wall but can cause
ringing if too aggressive relative to segment length.

### Mesh transform

Six sliders to translate/rotate the aorta mesh in world space:

| Control         | Range            | Description                                  |
|-----------------|------------------|----------------------------------------------|
| Mesh X / Y / Z  | -20 / -5 / -5 m  | World-space translation (x range is wider).  |
| Mesh Rot X/Y/Z  | -π – π rad       | Euler XYZ rotation around the mesh origin.   |

Changing any mesh slider rebuilds the BVH and collision normals on the GPU and
prints the current transform to the console (handy for hardcoding a good
alignment back into the constants).

## Status line

The bottom of the panel shows current sim time and a key reference. The viewer
title also shows FPS and frame timing.

## Tips

- Start with **Mesh Collision** on, **Use Mesh Edge Collision Path** off, and
  **Collision After Rod Constraints** on. That's the cheapest, most stable
  configuration.
- If the catheter visibly tunnels through the vessel, increase
  **Collision Iterations** before lowering **Substeps**.
- For very curvy regions, enable **Use Smooth Collision Normals** to avoid
  facet-aligned ringing in the contact direction.
- The track is captured **once** at startup from the initial rod layout. If you
  rotate the mesh, the track does not follow — re-run the demo with new
  `mesh_offset` / `mesh_rotation` constants if you need a track aligned to the
  new vessel pose.
- Gravity is off by default because it would immediately collapse the
  unsupported tip during insertion. Toggle on with `G` to test bending under
  load.
