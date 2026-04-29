"""Minimal XCATH catheter simulation using Newton's XPBD elastic rod solver.

Simulates catheter insertion through an aorta model with:
- BVH-based particle-mesh collision
- Track-guided insertion
- Bendable tip steering
- Root rotation control

Run: python xcath/xcath.py --viewer gl
"""

from __future__ import annotations

import math
import os

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.solvers
import newton.usd
from newton.solvers import xpbd_rod

try:
    from xcath.xcath_solver import XCathRodSolver, compute_signed_distances, compute_smooth_vertex_normals
except ImportError:
    from xcath_solver import XCathRodSolver, compute_signed_distances, compute_smooth_vertex_normals

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_POINTS = 128
SEGMENT_LENGTH = 0.05
PARTICLE_MASS = 1.0
PARTICLE_RADIUS = 0.02
MESH_SCALE = 0.01
SUBSTEPS = 8
COLLISION_ITERATIONS = 1
TIP_NUM_EDGES = 16
MESH_PRIM_PATH = "/root/A4009/A4007/Xueguan_rudong/Dynamic_vessels/Mesh"
LINEAR_DAMPING = 0.01
ANGULAR_DAMPING = 0.01
DAMPING_SLIDER_MAX = 0.2

# ---------------------------------------------------------------------------
# Warp kernels
# ---------------------------------------------------------------------------


@wp.kernel
def _set_root_position_kernel(
    positions: wp.array(dtype=wp.vec3),
    predicted: wp.array(dtype=wp.vec3),
    new_pos: wp.vec3,
):
    """Set root particle position on both position arrays."""
    tid = wp.tid()
    if tid != 0:
        return
    positions[0] = new_pos
    predicted[0] = new_pos


@wp.kernel
def _set_particle_radius_kernel(
    particle_radius: wp.array(dtype=wp.float32),
    start: int,
    count: int,
    radius: float,
):
    i = wp.tid()
    if i >= count:
        return
    particle_radius[start + i] = radius


@wp.kernel
def _update_tip_rest_darboux_kernel(
    rest_darboux: wp.array(dtype=wp.vec3),
    num_edges: int,
    tip_num_edges: int,
    bend_angle: float,
):
    """Set rest Darboux vector: tip edges get curvature, others stay straight."""
    e = wp.tid()
    if e >= num_edges:
        return
    tip_start = num_edges - tip_num_edges
    if e >= tip_start:
        per_edge = bend_angle / float(tip_num_edges)
        rest_darboux[e] = wp.vec3(per_edge, 0.0, 0.0)
    else:
        rest_darboux[e] = wp.vec3(0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Multiply two quaternions [x, y, z, w]."""
    return np.array(
        [
            a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
            a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
            a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
            a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
        ],
        dtype=np.float32,
    )


def _rotation_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    """Build a 3x3 rotation matrix from Euler angles (XYZ order, radians)."""
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    return np.array(
        [
            [cy * cz, -cy * sz, sy],
            [sx * sy * cz + cx * sz, -sx * sy * sz + cx * cz, -sx * cy],
            [-cx * sy * cz + sx * sz, cx * sy * sz + sx * cz, cx * cy],
        ],
        dtype=np.float32,
    )


def _transform_vertices(
    verts: np.ndarray,
    normals: np.ndarray | None,
    offset: np.ndarray,
    rot: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Apply rotation then translation to vertices (and rotation to normals)."""
    transformed = (verts @ rot.T) + offset
    transformed_normals = None
    if normals is not None:
        transformed_normals = (normals @ rot.T).astype(np.float32)
    return transformed.astype(np.float32), transformed_normals


# ---------------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------------


class Example:
    def __init__(self, viewer, args=None):
        self.viewer = viewer
        self.sim_time = 0.0
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = SUBSTEPS
        self.sim_dt = self.frame_dt / self.sim_substeps

        # Tuneable parameters
        self.bend_stiffness = 0.1
        self.twist_stiffness = 1.0
        self.young_modulus = 1000000.0
        self.torsion_modulus = 1_000_000_000.0
        self.linear_damping = LINEAR_DAMPING
        self.angular_damping = ANGULAR_DAMPING
        self.gravity_enabled = False
        self.track_enabled = True
        self.track_stiffness = 1.0
        self.collision_enabled = True
        self.collision_radius = PARTICLE_RADIUS
        self.collision_pre_constraints_enabled = True
        self.collision_post_constraints_enabled = False
        self.collision_iterations = COLLISION_ITERATIONS
        self.mesh_edge_collision_enabled = False
        self.smooth_collision_normals_enabled = False

        # Controls
        self.insertion = 0.0
        self.root_rotation = 0.0
        self.tip_bend_angle = 0.0
        self.tip_num_edges = TIP_NUM_EDGES
        self._g_pressed = False

        # --- Load aorta mesh ---
        from pxr import Usd  # noqa: PLC0415

        asset_dir = os.path.dirname(os.path.abspath(__file__))
        usd_path = os.path.join(asset_dir, "DynamicAorta.usdc")
        stage = Usd.Stage.Open(usd_path)
        prim = stage.GetPrimAtPath(MESH_PRIM_PATH)
        newton_mesh = newton.usd.get_mesh(prim, load_normals=True)

        # Scale and store raw (untransformed) mesh data
        self._raw_verts = newton_mesh.vertices.copy() * MESH_SCALE
        self._raw_normals = newton_mesh.normals.copy() if newton_mesh.normals is not None else None
        self._raw_indices = newton_mesh.indices.copy().astype(np.int32)

        self.device = wp.get_device()

        # Mesh transform (Euler XYZ radians + translation)
        self.mesh_offset = np.array([9.833, 0.111, 0.503], dtype=np.float32)
        self.mesh_rotation = np.array([-1.333, 1.369, 0.000], dtype=np.float32)

        # Build initial transformed mesh arrays
        self.aorta_verts_wp = None
        self.aorta_normals_wp = None
        self.collision_normals_wp = None
        self.aorta_indices_wp = wp.array(self._raw_indices, dtype=wp.int32, device=self.device)
        self.collision_mesh = None
        self._rebuild_aorta_mesh()

        # --- Build rod ---
        builder = newton.ModelBuilder()
        newton.solvers.SolverXPBDRod.register_custom_attributes(builder)

        positions = np.zeros((NUM_POINTS, 3), dtype=np.float32)
        for i in range(NUM_POINTS):
            positions[i, 0] = i * SEGMENT_LENGTH
            positions[i, 2] = 1.0

        self.track_start = positions[0].copy()
        self.track_end = positions[-1].copy()
        track_vec = self.track_end - self.track_start
        self.track_length = float(np.linalg.norm(track_vec))
        self.track_dir = track_vec / self.track_length

        xpbd_rod.add_elastic_rod(
            builder,
            positions=positions,
            radius=self.collision_radius,
            particle_mass=PARTICLE_MASS,
            bend_stiffness=self.bend_stiffness,
            twist_stiffness=self.twist_stiffness,
            young_modulus=self.young_modulus,
            torsion_modulus=self.torsion_modulus,
            lock_root=True,
            lock_root_rotation=True,
        )

        self.model = builder.finalize()
        # Override gravity (starts disabled)
        self.model.gravity = wp.array(
            [[0.0, 0.0, 0.0]], dtype=wp.vec3, device=self.device
        )

        self.solver = XCathRodSolver(
            model=self.model,
            linear_damping=self.linear_damping,
            angular_damping=self.angular_damping,
            solver_backend="block_thomas",
            floor_z=None,
            collision_mesh=self.collision_mesh,
            collision_mesh_normals=self.collision_normals_wp,
            track_start=self.track_start,
            track_dir=self.track_dir,
            track_length=self.track_length,
            tip_num_edges=self.tip_num_edges,
            particle_radius=self.collision_radius,
            segment_length=SEGMENT_LENGTH,
            track_stiffness=self.track_stiffness,
            track_enabled=self.track_enabled,
            collision_enabled=self.collision_enabled,
            collision_iterations=self.collision_iterations,
            collision_pre_constraints_enabled=self.collision_pre_constraints_enabled,
            collision_post_constraints_enabled=self.collision_post_constraints_enabled,
            mesh_edge_collision_enabled=self.mesh_edge_collision_enabled,
            smooth_collision_normals_enabled=self.smooth_collision_normals_enabled,
            sign_scale=-1.0,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        # Shorthand references
        self.ws = self.solver._rods[0]
        self.rod_particle_start = self.solver._rod_particle_starts[0]

        # Cache base root orientation
        self.base_orientation = self.ws.orientations_wp.numpy()[0].copy()

        # Rod tube mesher
        self.mesher = xpbd_rod.RodMesher(
            num_points=NUM_POINTS,
            radius=self.collision_radius,
            resolution=8,
            smoothing=3,
            device=self.device,
        )

        self.viewer.set_model(self.model)
        self.viewer.show_particles = False

    def _apply_collision_radius(self):
        """Apply collision radius to solver and visual radius buffers."""
        radius = max(0.001, float(self.collision_radius))
        self.collision_radius = radius
        self.solver.target_phi = -radius
        self.solver.max_dist = 2.0 * radius + SEGMENT_LENGTH
        self.solver.mesh_edge_collision_radius = radius
        self.solver.mesh_edge_collision_query_radius = radius
        self.solver.mesh_edge_collision_max_triangles = 128
        self.mesher.radius = radius
        if self.model.particle_radius is not None:
            wp.launch(
                _set_particle_radius_kernel,
                dim=NUM_POINTS,
                inputs=[self.model.particle_radius, self.rod_particle_start, NUM_POINTS, radius],
                device=self.device,
            )

    # ----- Mesh transform -----

    def _rebuild_aorta_mesh(self):
        """Recompute transformed aorta vertices and rebuild collision BVH."""
        rot = _rotation_matrix(*self.mesh_rotation)
        verts, normals = _transform_vertices(
            self._raw_verts, self._raw_normals, self.mesh_offset, rot
        )
        self.aorta_verts_wp = wp.array(verts, dtype=wp.vec3, device=self.device)
        self.aorta_normals_wp = (
            wp.array(normals, dtype=wp.vec3, device=self.device)
            if normals is not None
            else None
        )
        collision_normals = normals
        if collision_normals is None or collision_normals.shape[0] != verts.shape[0]:
            collision_normals = compute_smooth_vertex_normals(verts, self._raw_indices)
        self.collision_normals_wp = wp.array(collision_normals, dtype=wp.vec3, device=self.device)
        self.collision_mesh = wp.Mesh(
            points=wp.array(verts, dtype=wp.vec3, device=self.device),
            indices=wp.array(self._raw_indices, dtype=wp.int32, device=self.device),
        )
        if hasattr(self, "solver"):
            self.solver.set_collision_mesh(self.collision_mesh, self.collision_normals_wp)

    # ----- Input handling -----

    def _handle_input(self):
        v = self.viewer
        if not hasattr(v, "is_key_down"):
            return

        try:
            import pyglet  # noqa: PLC0415

            KEY = pyglet.window.key
        except Exception:
            return

        insert_speed = 0.5
        rotate_speed = 1.5
        bend_speed = 1.0
        dt = self.frame_dt

        # Insertion
        if v.is_key_down(KEY.PAGEUP) or v.is_key_down("2") or v.is_key_down("i"):
            self.insertion += insert_speed * dt
        if v.is_key_down(KEY.PAGEDOWN) or v.is_key_down("1") or v.is_key_down("k"):
            self.insertion -= insert_speed * dt
        self.insertion = max(0.0, self.insertion)

        # Root rotation around local Z
        if v.is_key_down(KEY.COMMA) or v.is_key_down("j"):
            self.root_rotation -= rotate_speed * dt
        if v.is_key_down(KEY.PERIOD) or v.is_key_down("l"):
            self.root_rotation += rotate_speed * dt

        # Tip bending
        if v.is_key_down(KEY.EQUAL) or v.is_key_down(KEY.NUM_ADD):
            self.tip_bend_angle += bend_speed * dt
        if v.is_key_down(KEY.MINUS) or v.is_key_down(KEY.NUM_SUBTRACT):
            self.tip_bend_angle -= bend_speed * dt
        self.tip_bend_angle = max(-1.5, min(1.5, self.tip_bend_angle))

        # Gravity toggle (edge-triggered)
        if v.is_key_down("g"):
            if not self._g_pressed:
                self.gravity_enabled = not self.gravity_enabled
                g = wp.vec3(0.0, 0.0, -9.81) if self.gravity_enabled else wp.vec3(0.0, 0.0, 0.0)
                self.ws.gravity = g
                self._g_pressed = True
        else:
            self._g_pressed = False

    # ----- Root & tip control -----

    def _apply_root_control(self):
        ws = self.ws

        # Root position along track
        root_pos = self.track_start + self.track_dir * self.insertion
        wp.launch(
            _set_root_position_kernel,
            dim=1,
            inputs=[ws.positions_wp, ws.predicted_positions_wp, wp.vec3(*root_pos.tolist())],
            device=self.device,
        )

        # Root rotation
        half = self.root_rotation * 0.5
        q_twist = np.array(
            [0.0, 0.0, math.sin(half), math.cos(half)], dtype=np.float32
        )
        q_new = _qmul(self.base_orientation, q_twist)
        q_new /= np.linalg.norm(q_new)
        self.solver.set_root_orientation(
            0, wp.quat(float(q_new[0]), float(q_new[1]), float(q_new[2]), float(q_new[3]))
        )

        # Tip bend rest Darboux
        wp.launch(
            _update_tip_rest_darboux_kernel,
            dim=ws.num_edges,
            inputs=[ws.rest_darboux_wp, ws.num_edges, self.tip_num_edges, self.tip_bend_angle],
            device=self.device,
        )

    # ----- Simulation step -----

    def step(self):
        self._handle_input()
        self.solver.track_enabled = self.track_enabled
        self.solver.track_stiffness = self.track_stiffness
        self.solver.linear_damping = max(0.0, min(DAMPING_SLIDER_MAX, float(self.linear_damping)))
        self.solver.angular_damping = max(0.0, min(DAMPING_SLIDER_MAX, float(self.angular_damping)))
        self.tip_num_edges = max(1, min(self.ws.num_edges, int(self.tip_num_edges)))
        self.solver.tip_num_edges = self.tip_num_edges
        self._apply_collision_radius()
        self.solver.collision_enabled = self.collision_enabled
        self.solver.collision_pre_constraints_enabled = self.collision_pre_constraints_enabled
        self.solver.collision_post_constraints_enabled = self.collision_post_constraints_enabled
        self.solver.mesh_edge_collision_enabled = self.mesh_edge_collision_enabled
        self.solver.smooth_collision_normals_enabled = self.smooth_collision_normals_enabled
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.solver.collision_iterations = max(1, int(self.collision_iterations))

        for _ in range(self.sim_substeps):
            self._apply_root_control()

            self.solver.step(
                self.state_0, self.state_1, self.control, self.contacts, self.sim_dt
            )

            self.state_0, self.state_1 = self.state_1, self.state_0

        self.sim_time += self.frame_dt

    # ----- Rendering -----

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)

        # Aorta mesh (gray)
        self.viewer.log_mesh(
            "/aorta",
            self.aorta_verts_wp,
            self.aorta_indices_wp,
            self.aorta_normals_wp,
        )
        self._set_mesh_color(self.viewer, "/aorta", 0.6, 0.6, 0.6)

        # Rod tube mesh (blue)
        ps = self.rod_particle_start
        self.mesher.update(self.state_0.particle_q[ps : ps + NUM_POINTS])
        self.viewer.log_mesh(
            "/catheter",
            self.mesher.vertices,
            self.mesher.indices,
            self.mesher.normals,
            self.mesher.uvs,
        )
        self._set_mesh_color(self.viewer, "/catheter", 0.3, 0.45, 0.85)

        self.viewer.end_frame()

    # ----- Mesh color -----

    @staticmethod
    def _set_mesh_color(viewer, name: str, r: float, g: float, b: float):
        """Patch a logged GL mesh so it sets its albedo color before each draw."""
        if not hasattr(viewer, "objects"):
            return
        mesh_obj = viewer.objects.get(name)
        if mesh_obj is None or hasattr(mesh_obj, "_color_patched"):
            return
        try:
            from newton._src.viewer.gl.opengl import RendererGL  # noqa: PLC0415

            gl_mod = RendererGL.gl
        except Exception:
            return

        original_render = mesh_obj.render

        def _colored_render(_gl=gl_mod, _r=r, _g=g, _b=b, _orig=original_render, _vao=mesh_obj.vao):
            _gl.glBindVertexArray(_vao)
            _gl.glVertexAttrib3f(7, _r, _g, _b)
            _gl.glBindVertexArray(0)
            _orig()

        mesh_obj.render = _colored_render
        mesh_obj._color_patched = True

    # ----- Stiffness update -----

    def _update_stiffness(self):
        ws = self.ws
        ws.young_modulus = self.young_modulus
        ws.torsion_modulus = self.torsion_modulus
        bs = np.full(
            (ws.num_edges, 3),
            [self.bend_stiffness, self.bend_stiffness, self.twist_stiffness],
            dtype=np.float32,
        )
        ws.bend_stiffness_wp.assign(
            wp.array(bs, dtype=wp.vec3, device=ws.device)
        )

    # ----- ImGui -----

    def gui(self, imgui):
        imgui.text("XCATH Catheter Simulation")
        imgui.separator()

        _, self.insertion = imgui.slider_float("Insertion", self.insertion, 0.0, 10.0)
        _, self.tip_bend_angle = imgui.slider_float(
            "Tip Bend", self.tip_bend_angle, -1.8, 1.8
        )
        _, self.tip_num_edges = imgui.slider_int(
            "Tip Bend Segments", self.tip_num_edges, 1, 30
        )
        _, self.root_rotation = imgui.slider_float(
            "Root Rotation ", self.root_rotation, -math.pi, math.pi
        )

        imgui.separator()

        changed_g, self.gravity_enabled = imgui.checkbox("Gravity", self.gravity_enabled)
        if changed_g:
            g = wp.vec3(0.0, 0.0, -9.81) if self.gravity_enabled else wp.vec3(0.0, 0.0, 0.0)
            self.ws.gravity = g

        _, self.track_enabled = imgui.checkbox("Track Sliding", self.track_enabled)
        _, self.collision_enabled = imgui.checkbox("Mesh Collision", self.collision_enabled)
        _, self.mesh_edge_collision_enabled = imgui.checkbox(
            "Use Mesh Edge Collision Path", self.mesh_edge_collision_enabled
        )
        _, self.smooth_collision_normals_enabled = imgui.checkbox(
            "Use Smooth Collision Normals", self.smooth_collision_normals_enabled
        )
        _, self.collision_pre_constraints_enabled = imgui.checkbox(
            "Collision Before Rod Constraints", self.collision_pre_constraints_enabled
        )
        _, self.collision_post_constraints_enabled = imgui.checkbox(
            "Collision After Rod Constraints", self.collision_post_constraints_enabled
        )

        imgui.separator()

        changed_b, self.bend_stiffness = imgui.slider_float(
            "Bend Stiffness", self.bend_stiffness, 0.0, 1.0
        )
        changed_t, self.twist_stiffness = imgui.slider_float(
            "Twist Stiffness", self.twist_stiffness, 0.0, 1.0
        )
        if changed_b or changed_t:
            self._update_stiffness()

        changed_E, self.young_modulus = imgui.input_float(
            "Young Modulus [Pa]", self.young_modulus, format="%.1f"
        )
        changed_G, self.torsion_modulus = imgui.input_float(
            "Torsion Modulus [Pa]", self.torsion_modulus, format="%.1f"
        )
        if changed_E or changed_G:
            self._update_stiffness()

        _, self.track_stiffness = imgui.slider_float(
            "Track Stiffness", self.track_stiffness, 0.0, 1.0
        )
        _, self.linear_damping = imgui.slider_float(
            "Pos Damping", self.linear_damping, 0.0, DAMPING_SLIDER_MAX
        )
        _, self.angular_damping = imgui.slider_float(
            "Rot Damping", self.angular_damping, 0.0, DAMPING_SLIDER_MAX
        )

        imgui.separator()

        _, self.sim_substeps = imgui.slider_int("Substeps", self.sim_substeps, 1, 32)
        _, self.collision_iterations = imgui.slider_int(
            "Collision Iterations", self.collision_iterations, 1, 16
        )
        _, self.collision_radius = imgui.slider_float(
            "Collision Radius", self.collision_radius, 0.001, 0.05
        )

        imgui.separator()
        imgui.text("Mesh Transform")
        mesh_changed = False
        c, self.mesh_offset[0] = imgui.slider_float("Mesh X", float(self.mesh_offset[0]), -20.0, 20.0)
        mesh_changed = mesh_changed or c
        c, self.mesh_offset[1] = imgui.slider_float("Mesh Y", float(self.mesh_offset[1]), -5.0, 5.0)
        mesh_changed = mesh_changed or c
        c, self.mesh_offset[2] = imgui.slider_float("Mesh Z", float(self.mesh_offset[2]), -5.0, 5.0)
        mesh_changed = mesh_changed or c
        c, self.mesh_rotation[0] = imgui.slider_float(
            "Mesh Rot X", float(self.mesh_rotation[0]), -math.pi, math.pi
        )
        mesh_changed = mesh_changed or c
        c, self.mesh_rotation[1] = imgui.slider_float(
            "Mesh Rot Y", float(self.mesh_rotation[1]), -math.pi, math.pi
        )
        mesh_changed = mesh_changed or c
        c, self.mesh_rotation[2] = imgui.slider_float(
            "Mesh Rot Z", float(self.mesh_rotation[2]), -math.pi, math.pi
        )
        mesh_changed = mesh_changed or c
        if mesh_changed:
            print(
                f"Mesh transform: offset=({self.mesh_offset[0]:.3f}, {self.mesh_offset[1]:.3f}, {self.mesh_offset[2]:.3f})"
                f"  rot=({self.mesh_rotation[0]:.3f}, {self.mesh_rotation[1]:.3f}, {self.mesh_rotation[2]:.3f})"
            )
            self._rebuild_aorta_mesh()

        imgui.separator()
        imgui.text(f"Sim time: {self.sim_time:.2f}s")
        imgui.text("Keys: PgUp/PgDn or I/K=insert, ,/. or J/L=rotate")
        imgui.text("      +/-=bend tip, G=gravity")

    # ----- Test -----

    def test_final(self):
        particle_q = self.state_0.particle_q.numpy()
        assert np.all(np.isfinite(particle_q)), "Particle positions must stay finite"
        if self.collision_enabled and self.collision_mesh is not None:
            ps = self.rod_particle_start
            signed_distance_wp, hit_wp = compute_signed_distances(
                self.state_0.particle_q[ps : ps + NUM_POINTS],
                self.collision_mesh.id,
                sign_scale=self.solver.sign_scale,
                max_dist=self.solver.max_dist,
                device=self.device,
            )
            signed_distance = signed_distance_wp.numpy()
            hit = hit_wp.numpy().astype(bool)
            inv_mass = self.ws.inv_masses_wp.numpy()
            active_hits = hit & (inv_mass > 0.0)
            assert np.all(
                signed_distance[active_hits] <= 1.0e-3
            ), "Contained catheter particles must not end outside the vessel"


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer=viewer, args=args)
    newton.examples.run(example, args)
