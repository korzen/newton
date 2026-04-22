# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Builder helper for adding Cosserat elastic rods to a Newton model."""

from __future__ import annotations

import numpy as np
import warp as wp

from ...sim import ModelBuilder
from ...utils.cable import create_parallel_transport_rod_quaternions


_XPBD_ROD_FREQUENCIES = frozenset({"xpbd:rod", "xpbd:rod_particle", "xpbd:rod_edge"})


def _has_xpbd_particle_frame_schema(builder: ModelBuilder) -> bool:
    return _XPBD_ROD_FREQUENCIES.issubset(builder.get_custom_frequency_keys())


def _quaternions_to_numpy(quaternions: list[wp.quat]) -> np.ndarray:
    return np.asarray([[q[0], q[1], q[2], q[3]] for q in quaternions], dtype=np.float32)


def _compute_rest_darboux(orientations: np.ndarray) -> np.ndarray:
    rest_darboux = np.zeros((orientations.shape[0] - 1, 3), dtype=np.float32)
    for i in range(rest_darboux.shape[0]):
        q0 = wp.quat(*orientations[i])
        q1 = wp.quat(*orientations[i + 1])
        q_rel = wp.quat_inverse(q0) * q1
        rest_darboux[i, 0] = q_rel[0]
        rest_darboux[i, 1] = q_rel[1]
        rest_darboux[i, 2] = q_rel[2]
    return rest_darboux


def add_elastic_rod(
    builder: ModelBuilder,
    positions: np.ndarray,
    radius: float = 0.01,
    particle_mass: float = 0.1,
    bend_stiffness: float = 1.0,
    twist_stiffness: float = 1.0,
    young_modulus: float = 1.0e6,
    torsion_modulus: float = 1.0e6,
    lock_root: bool = True,
    lock_root_rotation: bool = True,
) -> list[int]:
    """Add a Cosserat elastic rod to the model.

    Adds particles and stores rod metadata for the standalone
    :class:`~newton.solvers.SolverXPBDRod` path and/or the particle-frame
    :class:`~newton.solvers.SolverXPBD` path, depending on which custom
    attributes were registered on the builder.

    Args:
        builder: Model builder to add the rod to.
        positions: Initial positions as ``(N, 3)`` array [m].
        radius: Rod cross-section radius [m].
        particle_mass: Mass of each particle [kg].
        bend_stiffness: Bending stiffness coefficient.
        twist_stiffness: Twist stiffness coefficient.
        young_modulus: Young's modulus [Pa].
        torsion_modulus: Torsion modulus [Pa].
        lock_root: Whether the first particle is position-locked.
        lock_root_rotation: Whether the first particle is rotation-locked.

    Returns:
        List of particle indices.
    """
    positions = np.asarray(positions, dtype=np.float32)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must be (N, 3)")

    num_points = positions.shape[0]
    if num_points < 2:
        raise ValueError("Rod requires at least 2 points")

    has_legacy_schema = hasattr(builder, "_xpbd_rod_data")
    has_xpbd_schema = _has_xpbd_particle_frame_schema(builder)
    if not has_legacy_schema and not has_xpbd_schema:
        raise ValueError(
            "Elastic rods require registered custom attributes. "
            "Call SolverXPBD.register_custom_attributes(builder) or "
            "SolverXPBDRod.register_custom_attributes(builder) before add_elastic_rod()."
        )

    num_edges = num_points - 1

    # Compute rest lengths from positions
    rest_lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1).astype(np.float32)
    if np.any(rest_lengths <= 1.0e-8):
        raise ValueError("positions must not contain duplicate consecutive points")

    points_wp = [wp.vec3(float(p[0]), float(p[1]), float(p[2])) for p in positions]
    orientations = _quaternions_to_numpy(create_parallel_transport_rod_quaternions(points_wp))
    rest_darboux = _compute_rest_darboux(orientations)

    # Compute edge bend/twist stiffness vectors
    bend_stiffness_vec = np.zeros((num_edges, 3), dtype=np.float32)
    bend_stiffness_vec[:, 0] = bend_stiffness
    bend_stiffness_vec[:, 1] = bend_stiffness
    bend_stiffness_vec[:, 2] = twist_stiffness

    # Inverse masses
    inv_mass = 0.0 if particle_mass == 0.0 else 1.0 / particle_mass
    inv_masses = np.full(num_points, inv_mass, dtype=np.float32)
    if lock_root:
        inv_masses[0] = 0.0

    # Quaternion inverse masses (rotation lock)
    quat_inv_masses = np.ones(num_points, dtype=np.float32)
    if lock_root_rotation:
        quat_inv_masses[0] = 0.0

    # Add particles
    particle_indices = []
    for i in range(num_points):
        mass = 0.0 if inv_masses[i] == 0.0 else particle_mass
        idx = builder.add_particle(
            pos=tuple(positions[i]),
            vel=(0.0, 0.0, 0.0),
            mass=mass,
            radius=radius,
        )
        particle_indices.append(idx)

    if has_legacy_schema:
        ns = builder._xpbd_rod_data
        ns["rod_num_points"].append(num_points)
        ns["rod_particle_start"].append(particle_indices[0])
        ns["rod_young_modulus"].append(young_modulus)
        ns["rod_torsion_modulus"].append(torsion_modulus)

        ns["orientations"].extend(orientations.tolist())
        ns["quat_inv_masses"].extend(quat_inv_masses.tolist())
        ns["rest_lengths"].extend(rest_lengths.tolist())
        ns["rest_darboux"].extend(rest_darboux.tolist())
        ns["bend_stiffness"].extend(bend_stiffness_vec.tolist())

    if has_xpbd_schema:
        particle_rows = builder.add_custom_values_batch(
            [
                {
                    "xpbd:particle_index": particle_indices[i],
                    "xpbd:quat_inv_mass": float(quat_inv_masses[i]),
                    "xpbd:orientation": wp.quat(*orientations[i]),
                    "xpbd:angular_velocity": wp.vec3(0.0, 0.0, 0.0),
                    "xpbd:torque": wp.vec3(0.0, 0.0, 0.0),
                }
                for i in range(num_points)
            ]
        )
        edge_rows = builder.add_custom_values_batch(
            [
                {
                    "xpbd:rest_length": float(rest_lengths[i]),
                    "xpbd:rest_darboux": wp.vec3(*rest_darboux[i]),
                    "xpbd:bend_stiffness": wp.vec3(*bend_stiffness_vec[i]),
                }
                for i in range(num_edges)
            ]
        )
        builder.add_custom_values(
            **{
                "xpbd:particle_start": particle_rows[0]["xpbd:particle_index"],
                "xpbd:particle_count": num_points,
                "xpbd:edge_start": edge_rows[0]["xpbd:rest_length"],
                "xpbd:edge_count": num_edges,
                "xpbd:young_modulus": float(young_modulus),
                "xpbd:torsion_modulus": float(torsion_modulus),
            }
        )

    return particle_indices
