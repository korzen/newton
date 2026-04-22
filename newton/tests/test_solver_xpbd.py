# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the XPBD solver.

Includes tests for particle-particle friction using relative velocity correctly.
"""

import unittest

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.tests.unittest_utils import add_function_test, get_test_devices


def test_particle_particle_friction_uses_relative_velocity(test, device):
    """
    Test that particle-particle friction correctly uses relative velocity.

    This test verifies the fix for the bug where friction was computed using
    absolute velocity instead of relative velocity:
        WRONG: vt = v - n * vn        (uses absolute velocity v)
        RIGHT: vt = vrel - n * vn     (uses relative velocity vrel)

    Setup:
    - Two particles in contact (overlapping slightly)
    - Both particles moving with the same tangential velocity
    - With friction coefficient > 0

    Expected behavior:
    - Since relative tangential velocity is zero, friction should not
      affect their relative motion
    - Both particles should continue moving together at the same velocity
      (modulo normal contact forces)

    If the bug existed (using absolute velocity), the friction would
    incorrectly compute a non-zero tangential component and try to
    slow down both particles differently.
    """
    builder = newton.ModelBuilder(up_axis="Y")

    # Two particles that are slightly overlapping (in contact)
    # Positioned along X axis, both at y=0, z=0
    particle_radius = 0.5
    overlap = 0.1  # small overlap to ensure contact
    separation = 2.0 * particle_radius - overlap

    pos = [
        wp.vec3(0.0, 0.0, 0.0),
        wp.vec3(separation, 0.0, 0.0),
    ]

    # Both particles moving with the same tangential velocity (along Z axis)
    # The contact normal will be along X axis, so Z velocity is tangential
    tangential_velocity = 10.0
    vel = [
        wp.vec3(0.0, 0.0, tangential_velocity),
        wp.vec3(0.0, 0.0, tangential_velocity),
    ]

    mass = [1.0, 1.0]
    radius = [particle_radius, particle_radius]

    builder.add_particles(pos=pos, vel=vel, mass=mass, radius=radius)

    model = builder.finalize(device=device)

    # Disable gravity so we only see friction effects
    model.set_gravity((0.0, 0.0, 0.0))

    # Set particle-particle friction coefficient (XPBD particle-particle contact uses model.particle_mu)
    model.particle_mu = 1.0  # high friction
    model.particle_cohesion = 0.0

    # Use XPBD solver which uses the solve_particle_particle_contacts kernel
    solver = newton.solvers.SolverXPBD(
        model=model,
        iterations=20,
    )

    state0 = model.state()
    state1 = model.state()
    contacts = model.contacts()

    # Apply equal and opposite forces to keep the particles in sustained contact.
    # Without this, the initial overlap may be resolved in ~1 iteration and friction becomes hard to observe,
    # making the test flaky across devices/precision.
    press_force = 50.0
    assert state0.particle_f is not None
    state0.particle_f.assign(
        wp.array(
            [
                wp.vec3(wp.float32(press_force), wp.float32(0.0), wp.float32(0.0)),
                wp.vec3(wp.float32(-press_force), wp.float32(0.0), wp.float32(0.0)),
            ],
            dtype=wp.vec3,
            device=device,
        )
    )

    dt = 1.0 / 60.0
    num_steps = 60

    # Store initial relative velocity
    initial_vel = state0.particle_qd.numpy().copy()
    initial_relative_z_vel = initial_vel[0, 2] - initial_vel[1, 2]

    # Run simulation
    for _ in range(num_steps):
        model.collide(state0, contacts)
        control = model.control()
        solver.step(state0, state1, control, contacts, dt)
        state0, state1 = state1, state0

    # Get final velocities
    final_vel = state0.particle_qd.numpy()
    final_relative_z_vel = final_vel[0, 2] - final_vel[1, 2]

    # The key assertion: relative tangential velocity should remain near zero
    # since both particles started with the same tangential velocity
    test.assertAlmostEqual(
        initial_relative_z_vel,
        0.0,
        places=5,
        msg="Initial relative tangential velocity should be zero",
    )
    test.assertAlmostEqual(
        final_relative_z_vel,
        0.0,
        places=3,
        msg="Final relative tangential velocity should remain near zero "
        "(friction should not affect particles moving together)",
    )

    # Also verify both particles still have similar Z velocities
    # (they should move together, not be affected differently by friction)
    test.assertAlmostEqual(
        final_vel[0, 2],
        final_vel[1, 2],
        places=3,
        msg="Both particles should have the same tangential velocity after simulation",
    )


def test_particle_particle_friction_with_relative_motion(test, device):
    """
    Test that friction DOES affect particles with different tangential velocities.

    This is the complementary test - when particles have different tangential
    velocities, friction should work to equalize them.

    Notes on test design:
    - Particle-particle friction in XPBD is applied during constraint projection while particles are in contact.
      If particles are not kept in sustained contact, you may only get a single contact correction and the
      effect of friction can be near-zero and noisy.
    - To make this robust, we apply equal-and-opposite forces along the contact normal so the particles stay
      pressed together while sliding tangentially, and we compare against a mu=0 baseline.
    """
    # Keep this test to a single time step with guaranteed initial penetration.
    # XPBD's particle-particle friction term is limited by the *incremental* normal correction (penetration error),
    # so once the overlap is resolved to touching, friction can become effectively zero. A long multi-step
    # "relative velocity must decrease" assertion is therefore inherently flaky.

    particle_radius = 0.5
    overlap = 0.1
    separation = 2.0 * particle_radius - overlap

    dt = 1.0 / 30.0  # larger dt to make frictional slip correction clearly measurable

    def run(mu: float) -> float:
        builder = newton.ModelBuilder(up_axis="Y")

        pos = [
            wp.vec3(0.0, 0.0, 0.0),
            wp.vec3(separation, 0.0, 0.0),
        ]

        # Different tangential velocities along Z (tangent to the X-axis contact normal).
        vel = [
            wp.vec3(0.0, 0.0, 10.0),
            wp.vec3(0.0, 0.0, 0.0),
        ]

        mass = [1.0, 1.0]
        radius = [particle_radius, particle_radius]

        builder.add_particles(pos=pos, vel=vel, mass=mass, radius=radius)

        model = builder.finalize(device=device)
        model.set_gravity((0.0, 0.0, 0.0))
        model.particle_mu = mu
        model.particle_cohesion = 0.0

        solver = newton.solvers.SolverXPBD(model=model, iterations=30)

        state0 = model.state()
        state1 = model.state()
        contacts = model.contacts()

        # One step: measure tangential slip (relative z displacement).
        model.collide(state0, contacts)
        control = model.control()
        solver.step(state0, state1, control, contacts, dt)

        q1 = state1.particle_q.numpy()
        return float(abs(q1[0, 2] - q1[1, 2]))

    slip_no_friction = run(mu=0.0)
    slip_with_friction = run(mu=1.0)

    # With mu=0, slip should be close to v_rel * dt (~10 * dt).
    test.assertGreater(
        slip_no_friction,
        0.2,
        msg="With mu=0, relative tangential slip over one step should be significant",
    )
    test.assertLess(
        slip_with_friction,
        slip_no_friction * 0.95,
        msg="With mu>0, particle-particle friction should reduce tangential slip over one step vs mu=0 baseline",
    )


def test_particle_shape_restitution_correct_particle(test, device):
    """
    Regression test for the bug where apply_particle_shape_restitution wrote
    restitution velocity to particle_v_out[tid] (contact index) instead of
    particle_v_out[particle_index].

    Setup:
    - Particle 0 ("decoy"): high above the ground (y=10), zero velocity, no contact.
    - Particle 1 ("bouncer"): at the ground surface with downward velocity, will contact.
    - The first contact has tid=0 but contact_particle[0] = 1.
    - With the old bug, restitution dv was written to particle 0 (the decoy).
    - After fix, restitution dv is written to particle 1 (the bouncer).

    Assert: particle 1's y-velocity should be positive (bouncing up) and
    particle 0's y-velocity should remain near zero.
    """
    builder = newton.ModelBuilder(up_axis="Y")

    particle_radius = 0.1

    # Particle 0: decoy, far above the ground — should never contact
    builder.add_particle(pos=(0.0, 10.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0, radius=particle_radius)

    # Particle 1: at ground level with downward velocity — will contact
    builder.add_particle(pos=(0.0, particle_radius, 0.0), vel=(0.0, -5.0, 0.0), mass=1.0, radius=particle_radius)

    # Add a ground plane so particle 1 can bounce
    builder.add_ground_plane()

    model = builder.finalize(device=device)

    # Disable gravity so decoy particle stays at rest
    model.set_gravity((0.0, 0.0, 0.0))

    # Enable restitution
    model.soft_contact_restitution = 1.0

    solver = newton.solvers.SolverXPBD(
        model=model,
        iterations=10,
        enable_restitution=True,
    )

    state0 = model.state()
    state1 = model.state()

    dt = 1.0 / 60.0

    # Run a single step — enough for the contact + restitution pass
    contacts = model.contacts()
    model.collide(state0, contacts)
    control = model.control()
    solver.step(state0, state1, control, contacts, dt)

    vel = state1.particle_qd.numpy()

    # Particle 0 (decoy, no contact): y-velocity should be ~0
    test.assertAlmostEqual(
        float(vel[0, 1]),
        0.0,
        places=2,
        msg="Decoy particle (no contact) should have zero y-velocity; restitution was incorrectly applied to it",
    )

    # Particle 1 (bouncer): y-velocity should be positive (bouncing up)
    test.assertGreater(
        float(vel[1, 1]),
        0.0,
        msg="Bouncing particle should have positive y-velocity after restitution",
    )


def test_particle_shape_restitution_accounts_for_body_velocity(test, device):
    """
    Regression test for the bug where apply_particle_shape_restitution
    did not account for the rigid body velocity at the contact point when
    computing relative velocity for restitution (#1273).

    Setup:
    - A rigid box moving upward at 5 m/s.
    - A stationary particle sitting just above the top face of the box.
    - Restitution = 1.0, gravity disabled.

    Without the fix, the kernel computes relative velocity from the
    particle velocity alone (ignoring the approaching body), so the
    approaching normal velocity appears zero and no restitution impulse
    is applied — the particle stays nearly at rest.

    With the fix, the kernel correctly subtracts the body velocity at
    the contact point, detects the closing velocity, and applies a
    restitution impulse that launches the particle upward.
    """
    builder = newton.ModelBuilder(up_axis="Y")

    # Add a dynamic rigid box centered at origin
    body_id = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
    )
    builder.add_shape_box(body=body_id, hx=1.0, hy=0.5, hz=1.0)

    # Add a stationary particle just above the box's top face (y=0.5)
    particle_radius = 0.1
    builder.add_particle(
        pos=(0.0, 0.5 + particle_radius, 0.0),
        vel=(0.0, 0.0, 0.0),
        mass=1.0,
        radius=particle_radius,
    )

    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, 0.0))
    model.soft_contact_restitution = 1.0

    solver = newton.solvers.SolverXPBD(
        model=model,
        iterations=10,
        enable_restitution=True,
    )

    state0 = model.state()
    state1 = model.state()

    # Give the rigid body an upward velocity so it approaches the particle
    body_vel = np.array([[0.0, 5.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    state0.body_qd.assign(wp.array(body_vel, dtype=wp.spatial_vector, device=device))

    dt = 1.0 / 60.0
    contacts = model.contacts()
    model.collide(state0, contacts)
    control = model.control()
    solver.step(state0, state1, control, contacts, dt)

    vel = state1.particle_qd.numpy()

    # Without the fix, the position solver alone gives the particle ~5 m/s.
    # With the fix, restitution adds another ~5 m/s on top (elastic bounce
    # against a body moving at 5 m/s), yielding ~10 m/s total.
    test.assertGreater(
        float(vel[0, 1]),
        7.0,
        msg=f"Particle should receive restitution impulse from the moving body (expected ~10 m/s, got {float(vel[0, 1]):.2f})",
    )


def test_articulation_contact_drift(test, device):
    """
    Regression test for articulated bodies drifting laterally on the ground (#2030).

    When joints are solved before contacts in the XPBD iteration loop, joint
    corrections displace bodies laterally and contact friction can't fully
    counteract the displacement. Over many steps, the residual accumulates
    into visible sliding.

    Setup:
    - Load a quadruped URDF on its side on the ground plane.
    - Let it settle for 2 seconds, then simulate for 3 more seconds.
    - Check that the root body hasn't drifted laterally.
    """
    builder = newton.ModelBuilder()
    builder.default_joint_cfg.armature = 0.01
    builder.default_joint_cfg.target_ke = 2000.0
    builder.default_joint_cfg.target_kd = 1.0
    builder.default_shape_cfg.ke = 1.0e4
    builder.default_shape_cfg.kd = 1.0e2
    builder.default_shape_cfg.kf = 1.0e2
    builder.default_shape_cfg.mu = 1.0

    # Place the quadruped on its side (rotated 90 degrees around X axis)
    rot = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.PI * 0.5)
    builder.add_urdf(
        newton.examples.get_asset("quadruped.urdf"),
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.3), rot),
        floating=True,
        enable_self_collisions=False,
        ignore_inertial_definitions=True,
    )
    armature_inertia = wp.mat33(np.eye(3, dtype=np.float32)) * 0.01
    for i in range(builder.body_count):
        builder.body_inertia[i] = builder.body_inertia[i] + armature_inertia

    builder.joint_q[-12:] = [0.2, 0.4, -0.6, -0.2, -0.4, 0.6, -0.2, 0.4, -0.6, 0.2, -0.4, 0.6]
    builder.joint_target_pos[-12:] = builder.joint_q[-12:]
    builder.add_ground_plane()

    model = builder.finalize(device=device)
    solver = newton.solvers.SolverXPBD(model)

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = model.contacts()

    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    fps = 100
    frame_dt = 1.0 / fps
    sim_substeps = 10
    sim_dt = frame_dt / sim_substeps

    # Let the quadruped settle after drop (2 seconds)
    for _ in range(200):
        for _ in range(sim_substeps):
            state_0.clear_forces()
            model.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, sim_dt)
            state_0, state_1 = state_1, state_0

    body_q = state_0.body_q.numpy()
    initial_x = float(body_q[0][0])
    initial_y = float(body_q[0][1])

    # Simulate for 3 more seconds
    for _ in range(300):
        for _ in range(sim_substeps):
            state_0.clear_forces()
            model.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, sim_dt)
            state_0, state_1 = state_1, state_0

    body_q = state_0.body_q.numpy()
    final_x = float(body_q[0][0])
    final_y = float(body_q[0][1])

    drift_x = abs(final_x - initial_x)
    drift_y = abs(final_y - initial_y)
    drift_xy = float(np.hypot(drift_x, drift_y))

    # The root body should not drift more than 1 cm laterally over 3 seconds
    # (Z is up, so X and Y are the lateral axes)
    # Without the fix, Y drifts ~5.9 mm/s → ~1.8 cm over 3 seconds.
    max_drift = 0.01
    test.assertLess(
        drift_xy,
        max_drift,
        msg=(
            f"Root body drifted {drift_xy:.4f} m laterally over 3 seconds "
            f"(dx={drift_x:.4f}, dy={drift_y:.4f}, max allowed: {max_drift})"
        ),
    )


def _test_xpbd_contact_force_sphere_on_plane(test, device, radius, density):
    """A sphere resting on a ground plane must report contact force equal to its weight."""
    mass = density * (4.0 / 3.0) * np.pi * radius**3
    gravity = 9.81

    builder = newton.ModelBuilder()
    builder.default_shape_cfg.density = density
    builder.add_ground_plane()
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, radius), wp.quat_identity()))
    builder.add_shape_sphere(body=body, radius=radius)
    model = builder.finalize(device=device)
    model.request_contact_attributes("force")

    solver = newton.solvers.SolverXPBD(model, iterations=32)
    state_in = model.state()
    state_out = model.state()
    control = model.control()
    contacts = model.contacts()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)

    dt = 1.0 / 60.0
    num_substeps = 8
    sub_dt = dt / num_substeps
    settle_steps = 120
    avg_steps = 60

    for _ in range(settle_steps):
        for _ in range(num_substeps):
            state_in.clear_forces()
            model.collide(state_in, contacts)
            solver.step(state_in, state_out, control, contacts, sub_dt)
            state_in, state_out = state_out, state_in

    force_acc = np.zeros(3)
    for _ in range(avg_steps):
        for _ in range(num_substeps):
            state_in.clear_forces()
            model.collide(state_in, contacts)
            solver.step(state_in, state_out, control, contacts, sub_dt)
            state_in, state_out = state_out, state_in
        solver.update_contacts(contacts, state_in)
        ncontacts = int(contacts.rigid_contact_count.numpy()[0])
        if ncontacts > 0:
            f = contacts.force.numpy()[:ncontacts, :3]
            force_acc += np.sum(f, axis=0)

    avg_force = force_acc / avg_steps
    expected_fz = mass * gravity
    np.testing.assert_allclose(avg_force[2], -expected_fz, rtol=0.05, err_msg="Vertical contact force should match -mg")
    np.testing.assert_allclose(avg_force[0], 0.0, atol=0.5, err_msg="Horizontal X force should be ~0")
    np.testing.assert_allclose(avg_force[1], 0.0, atol=0.5, err_msg="Horizontal Y force should be ~0")


def test_xpbd_contact_force_sphere_on_plane(test, device):
    _test_xpbd_contact_force_sphere_on_plane(test, device, radius=0.25, density=1000.0)


def test_xpbd_contact_force_heavy_sphere_on_plane(test, device):
    _test_xpbd_contact_force_sphere_on_plane(test, device, radius=0.5, density=2000.0)


def test_xpbd_contact_force_box_on_plane(test, device):
    """A box on a ground plane has multiple contact points; total force must equal mg, not N*mg.

    This specifically tests the fix for the N*contact force inflation bug when
    ``rigid_contact_con_weighting`` is enabled (the default).  A box generates 4
    bottom-face contacts, so without the fix the summed force would be ~4x the
    true weight.
    """
    hx, hy, hz = 0.5, 0.5, 0.5
    density = 1000.0
    volume = (2.0 * hx) * (2.0 * hy) * (2.0 * hz)
    mass = density * volume
    gravity = 9.81

    builder = newton.ModelBuilder()
    builder.default_shape_cfg.density = density
    builder.add_ground_plane()
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, hz), wp.quat_identity()))
    builder.add_shape_box(body=body, hx=hx, hy=hy, hz=hz)
    model = builder.finalize(device=device)
    model.request_contact_attributes("force")

    solver = newton.solvers.SolverXPBD(model, iterations=32, rigid_contact_con_weighting=True)
    state_in = model.state()
    state_out = model.state()
    control = model.control()
    contacts = model.contacts()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)

    dt = 1.0 / 60.0
    num_substeps = 8
    sub_dt = dt / num_substeps
    settle_steps = 120
    avg_steps = 60

    for _ in range(settle_steps):
        for _ in range(num_substeps):
            state_in.clear_forces()
            model.collide(state_in, contacts)
            solver.step(state_in, state_out, control, contacts, sub_dt)
            state_in, state_out = state_out, state_in

    force_acc = np.zeros(3)
    for _ in range(avg_steps):
        for _ in range(num_substeps):
            state_in.clear_forces()
            model.collide(state_in, contacts)
            solver.step(state_in, state_out, control, contacts, sub_dt)
            state_in, state_out = state_out, state_in
        solver.update_contacts(contacts, state_in)
        ncontacts = int(contacts.rigid_contact_count.numpy()[0])
        test.assertGreater(ncontacts, 1, "Box should generate multiple contact points")
        if ncontacts > 0:
            f = contacts.force.numpy()[:ncontacts, :3]
            force_acc += np.sum(f, axis=0)

    avg_force = force_acc / avg_steps
    expected_fz = mass * gravity
    np.testing.assert_allclose(
        avg_force[2],
        -expected_fz,
        rtol=0.10,
        err_msg="Total vertical contact force over multiple contacts should match -mg, not N*mg",
    )
    np.testing.assert_allclose(avg_force[0], 0.0, atol=1.0, err_msg="Horizontal X force should be ~0")
    np.testing.assert_allclose(avg_force[1], 0.0, atol=1.0, err_msg="Horizontal Y force should be ~0")


def test_xpbd_contact_force_mini_pyramid(test, device):
    """Two-layer pyramid: ground reaction forces must reflect stacked weight.

    Layout (Z-up):
        - Ground plane at z=0 (shape 0, body -1)
        - Two bottom cubes (bodies 0, 1) side-by-side on the ground
        - One top cube (body 2) centered on top of both

    All cubes have the same mass m.  At steady state the ground pushes up on
    each bottom cube with 1.5*mg (own weight + half the top cube).

    Only ground-contact forces are checked here.  Inter-body forces between
    dynamic bodies are under-reported by the per-body contact weighting
    scheme (``rigid_contact_con_weighting``) because a body's weight mixes
    contacts from different pairs.
    """
    h = 0.5
    density = 1000.0
    volume = (2.0 * h) ** 3
    mass = density * volume
    gravity = 9.81
    mg = mass * gravity

    builder = newton.ModelBuilder()
    builder.default_shape_cfg.density = density
    builder.add_ground_plane()

    b0 = builder.add_body(xform=wp.transform(wp.vec3(-h, 0.0, h), wp.quat_identity()))
    builder.add_shape_box(body=b0, hx=h, hy=h, hz=h)
    b1 = builder.add_body(xform=wp.transform(wp.vec3(h, 0.0, h), wp.quat_identity()))
    builder.add_shape_box(body=b1, hx=h, hy=h, hz=h)
    b2 = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 3.0 * h), wp.quat_identity()))
    builder.add_shape_box(body=b2, hx=h, hy=h, hz=h)

    model = builder.finalize(device=device)
    model.request_contact_attributes("force")

    solver = newton.solvers.SolverXPBD(model, iterations=32, rigid_contact_con_weighting=True)
    state_in = model.state()
    state_out = model.state()
    control = model.control()
    contacts = model.contacts()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)

    dt = 1.0 / 60.0
    num_substeps = 8
    sub_dt = dt / num_substeps
    settle_steps = 200
    avg_steps = 60

    for _ in range(settle_steps):
        for _ in range(num_substeps):
            state_in.clear_forces()
            model.collide(state_in, contacts)
            solver.step(state_in, state_out, control, contacts, sub_dt)
            state_in, state_out = state_out, state_in

    shape_body_np = model.shape_body.numpy()
    ground_shape = 0

    ground_fz_on_b0 = 0.0
    ground_fz_on_b1 = 0.0

    for _ in range(avg_steps):
        for _ in range(num_substeps):
            state_in.clear_forces()
            model.collide(state_in, contacts)
            solver.step(state_in, state_out, control, contacts, sub_dt)
            state_in, state_out = state_out, state_in
        solver.update_contacts(contacts, state_in)

        nc = int(contacts.rigid_contact_count.numpy()[0])
        if nc == 0:
            continue
        forces = contacts.force.numpy()[:nc, :3]
        s0 = contacts.rigid_contact_shape0.numpy()[:nc]
        s1 = contacts.rigid_contact_shape1.numpy()[:nc]
        for ci in range(nc):
            if s0[ci] != ground_shape:
                continue
            fz = forces[ci, 2]
            body_b = shape_body_np[s1[ci]] if s1[ci] >= 0 else -1
            if body_b == b0:
                ground_fz_on_b0 += -fz
            elif body_b == b1:
                ground_fz_on_b1 += -fz

    ground_fz_on_b0 /= avg_steps
    ground_fz_on_b1 /= avg_steps

    np.testing.assert_allclose(
        ground_fz_on_b0,
        1.5 * mg,
        rtol=0.15,
        err_msg=f"Ground reaction on bottom cube 0 should be ~1.5*mg={1.5 * mg:.0f}, got {ground_fz_on_b0:.0f}",
    )
    np.testing.assert_allclose(
        ground_fz_on_b1,
        1.5 * mg,
        rtol=0.15,
        err_msg=f"Ground reaction on bottom cube 1 should be ~1.5*mg={1.5 * mg:.0f}, got {ground_fz_on_b1:.0f}",
    )


def test_xpbd_contact_force_zero_when_no_contact(test, device):
    """A sphere in free-fall (no ground) should produce zero contact force."""
    radius = 0.25

    builder = newton.ModelBuilder()
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 5.0), wp.quat_identity()))
    builder.add_shape_sphere(body=body, radius=radius)
    model = builder.finalize(device=device)
    model.request_contact_attributes("force")

    solver = newton.solvers.SolverXPBD(model, iterations=2)
    state_in = model.state()
    state_out = model.state()
    control = model.control()
    contacts = model.contacts()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)

    dt = 1.0 / 60.0
    state_in.clear_forces()
    model.collide(state_in, contacts)
    solver.step(state_in, state_out, control, contacts, dt)
    solver.update_contacts(contacts, state_out)

    ncontacts = int(contacts.rigid_contact_count.numpy()[0])
    if ncontacts > 0:
        forces = contacts.force.numpy()[:ncontacts]
        np.testing.assert_allclose(forces, 0.0, atol=1e-6, err_msg="No contact force expected in free-fall")


def test_xpbd_contact_force_zero_when_not_touching(test, device):
    """A sphere near a ground plane with a large gap: contact pair exists but force is zero."""
    radius = 0.25
    gap = 1.0
    # Place sphere so it's within the gap (contact pair generated) but not penetrating.
    # Ground is at z=0, sphere center at z = radius + 0.5*gap (well above surface).
    z = radius + 0.5 * gap

    builder = newton.ModelBuilder()
    builder.default_shape_cfg.gap = gap
    builder.add_ground_plane()
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, z), wp.quat_identity()))
    builder.add_shape_sphere(body=body, radius=radius)
    model = builder.finalize(device=device)
    model.set_gravity(wp.vec3(0.0, 0.0, 0.0))
    model.request_contact_attributes("force")

    solver = newton.solvers.SolverXPBD(model, iterations=2)
    state_in = model.state()
    state_out = model.state()
    control = model.control()
    contacts = model.contacts()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)

    state_in.clear_forces()
    model.collide(state_in, contacts)

    ncontacts = int(contacts.rigid_contact_count.numpy()[0])
    test.assertGreater(ncontacts, 0, "Gap should cause a contact pair to be generated")

    solver.step(state_in, state_out, control, contacts, 1.0 / 60.0)
    solver.update_contacts(contacts, state_out)

    forces = contacts.force.numpy()[:ncontacts, :3]
    np.testing.assert_allclose(
        forces,
        0.0,
        atol=1e-6,
        err_msg="Contact pair within gap but not touching should report zero force",
    )


def test_xpbd_update_contacts_requires_force_attribute(test, device):
    """update_contacts should raise ValueError when contacts.force is not allocated."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.25), wp.quat_identity()))
    builder.add_shape_sphere(body=body, radius=0.25)
    model = builder.finalize(device=device)

    solver = newton.solvers.SolverXPBD(model, iterations=2)
    state_in = model.state()
    state_out = model.state()
    control = model.control()
    contacts = model.contacts()

    state_in.clear_forces()
    model.collide(state_in, contacts)
    solver.step(state_in, state_out, control, contacts, 1.0 / 60.0)

    test.assertIsNone(contacts.force)
    with test.assertRaises(ValueError):
        solver.update_contacts(contacts)


def _make_particle_frame_rod_model(
    device,
    *,
    num_points=12,
    spacing=0.05,
    height=1.0,
    particle_mass=0.05,
    radius=0.02,
    bend_stiffness=0.1,
    twist_stiffness=0.1,
    young_modulus=1.0e4,
    torsion_modulus=1.0e4,
    gravity=(0.0, 0.0, -9.81),
    add_ground=False,
):
    builder = newton.ModelBuilder(up_axis="Z")
    newton.solvers.SolverXPBD.register_custom_attributes(builder)

    positions = np.zeros((num_points, 3), dtype=np.float32)
    positions[:, 0] = np.arange(num_points, dtype=np.float32) * spacing
    positions[:, 2] = height

    newton.solvers.xpbd_rod.add_elastic_rod(
        builder,
        positions=positions,
        radius=radius,
        particle_mass=particle_mass,
        bend_stiffness=bend_stiffness,
        twist_stiffness=twist_stiffness,
        young_modulus=young_modulus,
        torsion_modulus=torsion_modulus,
    )

    if add_ground:
        builder.add_ground_plane()

    model = builder.finalize(device=device)
    if gravity != (0.0, 0.0, -9.81):
        model.set_gravity(gravity)
    return model, positions


def _step_particle_frame_rod(model, *, iterations=4, num_steps=200, dt=0.001, rod_solve_method="direct"):
    solver = newton.solvers.SolverXPBD(model=model, iterations=iterations, rod_solve_method=rod_solve_method)
    state0 = model.state()
    state1 = model.state()
    control = model.control()
    contacts = model.contacts()

    for _ in range(num_steps):
        model.collide(state0, contacts)
        solver.step(state0, state1, control, contacts, dt)
        state0, state1 = state1, state0

    return solver, state0


def _quat_alignment(q0: np.ndarray, q1: np.ndarray) -> float:
    return float(abs(np.dot(q0, q1)))


def test_xpbd_particle_frame_rod_builder_dual_path(test, device):
    """Builder helper populates both legacy and base-XPBD rod data when both schemas are registered."""
    builder = newton.ModelBuilder(up_axis="Z")
    newton.solvers.SolverXPBD.register_custom_attributes(builder)
    newton.solvers.SolverXPBDRod.register_custom_attributes(builder)

    positions = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.1, 0.0, 1.0],
            [0.2, 0.0, 1.0],
            [0.3, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    newton.solvers.xpbd_rod.add_elastic_rod(builder, positions=positions)
    model = builder.finalize(device=device)

    test.assertTrue(hasattr(model, "xpbd"))
    test.assertTrue(hasattr(model, "xpbd_rod"))
    test.assertEqual(model.get_custom_frequency_count("xpbd:rod"), 1)
    test.assertEqual(model.get_custom_frequency_count("xpbd:rod_particle"), positions.shape[0])
    test.assertEqual(model.get_custom_frequency_count("xpbd:rod_edge"), positions.shape[0] - 1)
    test.assertEqual(model.xpbd_rod["rod_num_points"], [positions.shape[0]])
    test.assertEqual(len(model.xpbd_rod["orientations"]), positions.shape[0])
    test.assertEqual(len(model.xpbd_rod["rest_lengths"]), positions.shape[0] - 1)
    np.testing.assert_allclose(
        model.xpbd.orientation.numpy()[0],
        np.asarray(model.xpbd_rod["orientations"][0], dtype=np.float32),
        atol=1.0e-6,
    )


def test_xpbd_particle_frame_rod_under_gravity(test, device):
    """Particle-frame XPBD rod sags under gravity while keeping the root fixed."""
    model, initial_positions = _make_particle_frame_rod_model(device, num_points=64)
    _solver, state = _step_particle_frame_rod(model, num_steps=400)

    positions = state.particle_q.numpy()
    orientations = state.xpbd.orientation.numpy()
    test.assertFalse(np.any(np.isnan(positions)))
    test.assertFalse(np.any(np.isnan(orientations)))
    test.assertLess(positions[-1, 2], initial_positions[-1, 2] - 0.05)
    test.assertLess(np.linalg.norm(positions[0] - initial_positions[0]), 1.0e-5)


def test_xpbd_particle_frame_rod_zero_gravity_rest(test, device):
    """Particle-frame XPBD rod stays close to its initial rest shape without gravity."""
    model, initial_positions = _make_particle_frame_rod_model(device, gravity=(0.0, 0.0, 0.0))
    initial_orientations = model.xpbd.orientation.numpy().copy()
    _solver, state = _step_particle_frame_rod(model, num_steps=120)

    positions = state.particle_q.numpy()
    orientations = state.xpbd.orientation.numpy()
    test.assertLess(float(np.max(np.abs(positions - initial_positions))), 1.0e-3)
    test.assertGreater(_quat_alignment(orientations[0], initial_orientations[0]), 0.9999)


def test_xpbd_particle_frame_rod_multi_rod_independence(test, device):
    """Multiple particle-frame rods keep their separation and solve independently."""
    builder = newton.ModelBuilder(up_axis="Z")
    newton.solvers.SolverXPBD.register_custom_attributes(builder)

    for rod_idx in range(2):
        positions = np.zeros((10, 3), dtype=np.float32)
        positions[:, 0] = np.arange(10, dtype=np.float32) * 0.05
        positions[:, 1] = rod_idx * 0.5
        positions[:, 2] = 1.0
        newton.solvers.xpbd_rod.add_elastic_rod(
            builder,
            positions=positions,
            particle_mass=0.05,
            bend_stiffness=0.1,
            twist_stiffness=0.1,
            young_modulus=1.0e4,
            torsion_modulus=1.0e4,
        )

    model = builder.finalize(device=device)
    _solver, state = _step_particle_frame_rod(model, num_steps=200)

    positions = state.particle_q.numpy()
    test.assertEqual(positions.shape[0], 20)
    test.assertLess(abs(float(np.mean(positions[:10, 1])) - 0.0), 1.0e-4)
    test.assertLess(abs(float(np.mean(positions[10:, 1])) - 0.5), 1.0e-4)


def test_xpbd_particle_frame_rod_stiffness_response(test, device):
    """Bending response changes when Young's modulus and bend stiffness change."""
    soft_model, _ = _make_particle_frame_rod_model(
        device,
        num_points=64,
        young_modulus=1.0e2,
        torsion_modulus=1.0e2,
        bend_stiffness=0.0,
        twist_stiffness=0.0,
    )
    stiff_model, _ = _make_particle_frame_rod_model(
        device,
        num_points=64,
        young_modulus=1.0e8,
        torsion_modulus=1.0e8,
        bend_stiffness=1.0,
        twist_stiffness=1.0,
    )

    _soft_solver, soft_state = _step_particle_frame_rod(soft_model, num_steps=240)
    _stiff_solver, stiff_state = _step_particle_frame_rod(stiff_model, num_steps=240)

    soft_tip_z = float(soft_state.particle_q.numpy()[-1, 2])
    stiff_tip_z = float(stiff_state.particle_q.numpy()[-1, 2])
    test.assertGreater(
        stiff_tip_z,
        soft_tip_z + 0.03,
        msg=f"Expected stiffer rod to sag less, got soft_tip_z={soft_tip_z:.4f}, stiff_tip_z={stiff_tip_z:.4f}",
    )


def test_xpbd_particle_frame_rod_root_lock(test, device):
    """Root position and material-frame orientation stay locked when requested."""
    model, initial_positions = _make_particle_frame_rod_model(device)
    initial_root_orientation = model.xpbd.orientation.numpy()[0].copy()
    _solver, state = _step_particle_frame_rod(model, num_steps=300)

    positions = state.particle_q.numpy()
    orientations = state.xpbd.orientation.numpy()
    test.assertLess(np.linalg.norm(positions[0] - initial_positions[0]), 1.0e-5)
    test.assertGreater(_quat_alignment(orientations[0], initial_root_orientation), 0.9999)


def test_xpbd_particle_frame_rod_ground_collision(test, device):
    """Particle-frame XPBD rod collides through the existing particle contact path."""
    radius = 0.03
    model, initial_positions = _make_particle_frame_rod_model(
        device,
        num_points=32,
        height=0.05,
        radius=radius,
        bend_stiffness=0.0,
        twist_stiffness=0.0,
        young_modulus=1.0e2,
        torsion_modulus=1.0e2,
        add_ground=True,
    )
    _solver, state = _step_particle_frame_rod(model, num_steps=300)

    positions = state.particle_q.numpy()
    test.assertLess(positions[-1, 2], initial_positions[-1, 2] - 0.005)
    test.assertGreaterEqual(float(np.min(positions[:, 2])), radius - 5.0e-3)


# ---------------------------------------------------------------------------
# Local (per-edge parallel Jacobi) rod solve method tests
# ---------------------------------------------------------------------------


def test_xpbd_particle_frame_rod_local_under_gravity(test, device):
    """Local rod solve: rod sags under gravity while keeping the root fixed.

    The per-edge Jacobi solver ignores inter-edge tridiagonal coupling, so
    it converges much slower than the direct Thomas path (~100-200x less sag
    for the same iteration count).  This test verifies the solver is stable
    and produces non-zero downward displacement, not that it matches the
    direct method's convergence.
    """
    model, initial_positions = _make_particle_frame_rod_model(device)
    _solver, state = _step_particle_frame_rod(model, num_steps=600, iterations=32, rod_solve_method="local")

    positions = state.particle_q.numpy()
    orientations = state.xpbd.orientation.numpy()
    test.assertFalse(np.any(np.isnan(positions)))
    test.assertFalse(np.any(np.isnan(orientations)))
    # The tip must have moved downward (any amount) and the root stays locked.
    test.assertLess(positions[-1, 2], initial_positions[-1, 2] - 1.0e-5)
    test.assertLess(np.linalg.norm(positions[0] - initial_positions[0]), 1.0e-5)


def test_xpbd_particle_frame_rod_local_zero_gravity_rest(test, device):
    """Local rod solve: rod stays close to rest shape without gravity."""
    model, initial_positions = _make_particle_frame_rod_model(device, gravity=(0.0, 0.0, 0.0))
    initial_orientations = model.xpbd.orientation.numpy().copy()
    _solver, state = _step_particle_frame_rod(model, num_steps=120, iterations=32, rod_solve_method="local")

    positions = state.particle_q.numpy()
    orientations = state.xpbd.orientation.numpy()
    test.assertLess(float(np.max(np.abs(positions - initial_positions))), 1.0e-3)
    test.assertGreater(_quat_alignment(orientations[0], initial_orientations[0]), 0.9999)


def test_xpbd_particle_frame_rod_local_multi_rod_independence(test, device):
    """Local rod solve: multiple rods keep their separation."""
    builder = newton.ModelBuilder(up_axis="Z")
    newton.solvers.SolverXPBD.register_custom_attributes(builder)

    for rod_idx in range(2):
        positions = np.zeros((10, 3), dtype=np.float32)
        positions[:, 0] = np.arange(10, dtype=np.float32) * 0.05
        positions[:, 1] = rod_idx * 0.5
        positions[:, 2] = 1.0
        newton.solvers.xpbd_rod.add_elastic_rod(
            builder,
            positions=positions,
            particle_mass=0.05,
            bend_stiffness=0.1,
            twist_stiffness=0.1,
            young_modulus=1.0e4,
            torsion_modulus=1.0e4,
        )

    model = builder.finalize(device=device)
    _solver, state = _step_particle_frame_rod(model, num_steps=200, iterations=32, rod_solve_method="local")

    positions = state.particle_q.numpy()
    test.assertEqual(positions.shape[0], 20)
    test.assertLess(abs(float(np.mean(positions[:10, 1])) - 0.0), 1.0e-4)
    test.assertLess(abs(float(np.mean(positions[10:, 1])) - 0.5), 1.0e-4)


def test_xpbd_particle_frame_rod_local_root_lock(test, device):
    """Local rod solve: root position and orientation stay locked."""
    model, initial_positions = _make_particle_frame_rod_model(device)
    initial_root_orientation = model.xpbd.orientation.numpy()[0].copy()
    _solver, state = _step_particle_frame_rod(model, num_steps=300, iterations=32, rod_solve_method="local")

    positions = state.particle_q.numpy()
    orientations = state.xpbd.orientation.numpy()
    test.assertLess(np.linalg.norm(positions[0] - initial_positions[0]), 1.0e-5)
    test.assertGreater(_quat_alignment(orientations[0], initial_root_orientation), 0.9999)


def test_xpbd_particle_frame_rod_local_ground_collision(test, device):
    """Local rod solve: rod collides through existing particle contact path."""
    radius = 0.03
    model, initial_positions = _make_particle_frame_rod_model(
        device,
        height=0.35,
        radius=radius,
        add_ground=True,
    )
    _solver, state = _step_particle_frame_rod(model, num_steps=600, iterations=32, rod_solve_method="local")

    positions = state.particle_q.numpy()
    test.assertLess(positions[-1, 2], initial_positions[-1, 2] - 0.02)
    test.assertGreaterEqual(float(np.min(positions[:, 2])), radius - 5.0e-3)


devices = get_test_devices(mode="basic")


class TestSolverXPBD(unittest.TestCase):
    pass


add_function_test(
    TestSolverXPBD,
    "test_particle_particle_friction_uses_relative_velocity",
    test_particle_particle_friction_uses_relative_velocity,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverXPBD,
    "test_particle_particle_friction_with_relative_motion",
    test_particle_particle_friction_with_relative_motion,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverXPBD,
    "test_particle_shape_restitution_correct_particle",
    test_particle_shape_restitution_correct_particle,
    devices=devices,
    check_output=False,
)


add_function_test(
    TestSolverXPBD,
    "test_particle_shape_restitution_accounts_for_body_velocity",
    test_particle_shape_restitution_accounts_for_body_velocity,
    devices=devices,
    check_output=False,
)


add_function_test(
    TestSolverXPBD,
    "test_articulation_contact_drift",
    test_articulation_contact_drift,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverXPBD,
    "test_xpbd_contact_force_sphere_on_plane",
    test_xpbd_contact_force_sphere_on_plane,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverXPBD,
    "test_xpbd_contact_force_heavy_sphere_on_plane",
    test_xpbd_contact_force_heavy_sphere_on_plane,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverXPBD,
    "test_xpbd_contact_force_box_on_plane",
    test_xpbd_contact_force_box_on_plane,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverXPBD,
    "test_xpbd_contact_force_mini_pyramid",
    test_xpbd_contact_force_mini_pyramid,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverXPBD,
    "test_xpbd_contact_force_zero_when_no_contact",
    test_xpbd_contact_force_zero_when_no_contact,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverXPBD,
    "test_xpbd_contact_force_zero_when_not_touching",
    test_xpbd_contact_force_zero_when_not_touching,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverXPBD,
    "test_xpbd_update_contacts_requires_force_attribute",
    test_xpbd_update_contacts_requires_force_attribute,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverXPBD,
    "test_xpbd_particle_frame_rod_builder_dual_path",
    test_xpbd_particle_frame_rod_builder_dual_path,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverXPBD,
    "test_xpbd_particle_frame_rod_under_gravity",
    test_xpbd_particle_frame_rod_under_gravity,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverXPBD,
    "test_xpbd_particle_frame_rod_zero_gravity_rest",
    test_xpbd_particle_frame_rod_zero_gravity_rest,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverXPBD,
    "test_xpbd_particle_frame_rod_multi_rod_independence",
    test_xpbd_particle_frame_rod_multi_rod_independence,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverXPBD,
    "test_xpbd_particle_frame_rod_stiffness_response",
    test_xpbd_particle_frame_rod_stiffness_response,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverXPBD,
    "test_xpbd_particle_frame_rod_root_lock",
    test_xpbd_particle_frame_rod_root_lock,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverXPBD,
    "test_xpbd_particle_frame_rod_ground_collision",
    test_xpbd_particle_frame_rod_ground_collision,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverXPBD,
    "test_xpbd_particle_frame_rod_local_under_gravity",
    test_xpbd_particle_frame_rod_local_under_gravity,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverXPBD,
    "test_xpbd_particle_frame_rod_local_zero_gravity_rest",
    test_xpbd_particle_frame_rod_local_zero_gravity_rest,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverXPBD,
    "test_xpbd_particle_frame_rod_local_multi_rod_independence",
    test_xpbd_particle_frame_rod_local_multi_rod_independence,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverXPBD,
    "test_xpbd_particle_frame_rod_local_root_lock",
    test_xpbd_particle_frame_rod_local_root_lock,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverXPBD,
    "test_xpbd_particle_frame_rod_local_ground_collision",
    test_xpbd_particle_frame_rod_local_ground_collision,
    devices=devices,
    check_output=False,
)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
