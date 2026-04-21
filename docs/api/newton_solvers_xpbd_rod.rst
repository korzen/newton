.. SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

newton.solvers.xpbd_rod
=======================

XPBD Rod solver module.

This module provides :func:`add_elastic_rod` for setting up Cosserat elastic
rod assets. Use :class:`~newton.solvers.SolverXPBDRod` as the canonical public
solver import path.

.. py:module:: newton.solvers.xpbd_rod
.. currentmodule:: newton.solvers.xpbd_rod

.. rubric:: Classes

.. autoclass:: BatchedRodMesher

.. autoclass:: RodMesher


.. rubric:: Functions

.. autofunction:: add_elastic_rod

.. autofunction:: compute_director_lines_kernel

