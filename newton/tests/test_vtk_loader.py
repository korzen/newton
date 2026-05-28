# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import tempfile
import unittest
from pathlib import Path

import numpy as np

from newton.examples.slicer.vtk_loader import load_vtk_polygon_data, load_vtk_polydata


class TestVtkLoader(unittest.TestCase):
    def _write_polydata(self, text: str) -> Path:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        path = Path(tmpdir.name) / "surface.vtk"
        path.write_text(text, encoding="utf-8")
        return path

    def test_load_polydata_polygons_and_normals(self):
        path = self._write_polydata(
            """# vtk DataFile Version 4.2
polydata fixture
ASCII
DATASET POLYDATA
POINTS 5 float
0 0 0
1 0 0
1 1 0
0 1 0
0 0 1
POLYGONS 2 9
3 0 1 4
4 0 1 2 3
TRIANGLE_STRIPS 1 5
4 0 1 3 4
POINT_DATA 5
NORMALS Normals float
0 0 1
0 0 1
0 0 1
0 0 1
0 1 0
CELL_DATA 3
SCALARS labels int 1
LOOKUP_TABLE default
10 20 30
"""
        )

        polydata = load_vtk_polydata(path)

        self.assertEqual(polydata.header, "polydata fixture")
        self.assertEqual(polydata.num_points, 5)
        self.assertEqual(polydata.num_polygons, 2)
        self.assertEqual(polydata.num_triangle_strips, 1)
        np.testing.assert_array_equal(polydata.polygon_sizes(), np.array([3, 4], dtype=np.int32))
        np.testing.assert_array_equal(
            polydata.triangle_indices(),
            np.array(
                [
                    [0, 1, 4],
                    [0, 1, 2],
                    [0, 2, 3],
                    [0, 1, 3],
                    [3, 1, 4],
                ],
                dtype=np.int32,
            ),
        )
        self.assertIn("Normals", polydata.point_data)
        np.testing.assert_allclose(polydata.point_data["Normals"][4], np.array([0.0, 1.0, 0.0], dtype=np.float32))
        np.testing.assert_allclose(polydata.cell_data["labels"], np.array([10.0, 20.0, 30.0], dtype=np.float32))

        mesh = polydata.to_mesh()
        self.assertEqual(mesh.vertices.shape, (5, 3))
        self.assertEqual(mesh.indices.shape, (15,))
        self.assertIsNotNone(mesh.normals)
        self.assertEqual(mesh.normals.shape, (5, 3))

    def test_load_polygon_data_alias(self):
        path = self._write_polydata(
            """# vtk DataFile Version 4.2
triangles only
ASCII
DATASET POLYDATA
POINTS 3 float
0 0 0 1 0 0 0 1 0
POLYGONS 1 4
3 0 1 2
"""
        )

        polydata = load_vtk_polygon_data(path)

        self.assertEqual(polydata.num_polygons, 1)
        np.testing.assert_array_equal(polydata.triangle_indices(), np.array([[0, 1, 2]], dtype=np.int32))


if __name__ == "__main__":
    unittest.main(verbosity=2)
