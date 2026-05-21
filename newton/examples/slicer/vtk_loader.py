# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Loader for legacy VTK ``UNSTRUCTURED_GRID`` files.

The slicer example family ships several volumetric/anatomy meshes authored in
the legacy VTK 3.0 ASCII format (see
https://docs.vtk.org/en/latest/vtk_file_formats/vtk_legacy_file_format.html).
This module provides a single-pass, dependency-free parser for that format
plus a few convenience extractors for the cell types we actually need
downstream (triangles, tetrahedra, hexahedra, and boundary-triangle
extraction for hex meshes).

Only the subset of the legacy format we encounter in this repository is
supported: ASCII ``UNSTRUCTURED_GRID`` datasets with ``POINTS``, ``CELLS``,
``CELL_TYPES``, and ``POINT_DATA``/``CELL_DATA`` containing ``SCALARS``,
``VECTORS``, ``NORMALS``, or ``COLOR_SCALARS`` attribute blocks. Binary
encodings, ``FIELD`` data, structured datasets, and other variants raise
:class:`ValueError`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import newton

__all__ = [
    "VTK_HEXAHEDRON",
    "VTK_LINE",
    "VTK_PIXEL",
    "VTK_POLY_LINE",
    "VTK_POLY_VERTEX",
    "VTK_PYRAMID",
    "VTK_QUAD",
    "VTK_TETRA",
    "VTK_TRIANGLE",
    "VTK_TRIANGLE_STRIP",
    "VTK_VERTEX",
    "VTK_VOXEL",
    "VTK_WEDGE",
    "VtkUnstructuredGrid",
    "load_vtk_unstructured_grid",
]

# Cell type ids per the VTK legacy spec.
VTK_VERTEX = 1
VTK_POLY_VERTEX = 2
VTK_LINE = 3
VTK_POLY_LINE = 4
VTK_TRIANGLE = 5
VTK_TRIANGLE_STRIP = 6
VTK_PIXEL = 8
VTK_QUAD = 9
VTK_TETRA = 10
VTK_VOXEL = 11
VTK_HEXAHEDRON = 12
VTK_WEDGE = 13
VTK_PYRAMID = 14

# Canonical VTK hex face winding (outward normals when the hex itself is
# right-handed). Each tuple is one quad face given as four vertex offsets
# into the hex's 8-vertex connectivity. Output triangles split each quad as
# (a, b, c) + (a, c, d) — preserves orientation.
_HEX_FACES = (
    (0, 3, 2, 1),  # -Z
    (4, 5, 6, 7),  # +Z
    (0, 1, 5, 4),  # -Y
    (1, 2, 6, 5),  # +X
    (2, 3, 7, 6),  # +Y
    (3, 0, 4, 7),  # -X
)

# Tetrahedron face winding (outward normals for a positive-orientation tet).
_TET_FACES = (
    (0, 2, 1),
    (0, 1, 3),
    (1, 2, 3),
    (2, 0, 3),
)

# Wedge (linear prism) face winding. Bottom and top are triangles; the
# remaining three faces are quads connecting them.
_WEDGE_FACES = (
    (0, 2, 1),
    (3, 4, 5),
    (0, 1, 4, 3),
    (1, 2, 5, 4),
    (2, 0, 3, 5),
)

# Pyramid face winding: base quad + 4 triangular sides.
_PYRAMID_FACES = (
    (0, 3, 2, 1),
    (0, 1, 4),
    (1, 2, 4),
    (2, 3, 4),
    (3, 0, 4),
)


@dataclass
class VtkUnstructuredGrid:
    """Parsed contents of a legacy ASCII VTK ``UNSTRUCTURED_GRID`` file.

    Attributes:
        points: Vertex positions, shape ``(num_points, 3)``, float32.
        cells: Tuple of per-cell vertex-index arrays. Length is the cell
            count; each entry is a 1-D int32 array of the vertex indices
            of that cell.
        cell_types: Per-cell type id (one of the ``VTK_*`` constants),
            shape ``(num_cells,)``, int32.
        point_data: Mapping from attribute name to point-attribute array.
            Shapes are ``(num_points,)`` for scalars and
            ``(num_points, c)`` for vectors / multi-component scalars.
        cell_data: Same as ``point_data`` but defined per cell.
        header: The free-form header line from the file (line 2).
    """

    points: np.ndarray
    cells: tuple[np.ndarray, ...]
    cell_types: np.ndarray
    point_data: dict[str, np.ndarray] = field(default_factory=dict)
    cell_data: dict[str, np.ndarray] = field(default_factory=dict)
    header: str = ""

    @property
    def num_points(self) -> int:
        return int(self.points.shape[0])

    @property
    def num_cells(self) -> int:
        return int(self.cell_types.shape[0])

    def cells_of_type(self, cell_type: int) -> np.ndarray:
        """Return the contiguous (num_matching, k) connectivity array for
        cells of the given VTK type, where ``k`` is the fixed vertex count
        for that type (e.g. 3 for triangles, 4 for tets, 8 for hexes).

        Raises:
            ValueError: If no cell of that type is present, or if the
                matching cells have inconsistent vertex counts.
        """
        mask = self.cell_types == cell_type
        if not mask.any():
            raise ValueError(f"No cells of VTK type {cell_type} found.")

        sizes = {int(self.cells[i].shape[0]) for i in np.flatnonzero(mask)}
        if len(sizes) != 1:
            raise ValueError(
                f"Cells of VTK type {cell_type} have inconsistent vertex counts: {sorted(sizes)}."
            )
        k = sizes.pop()
        out = np.empty((int(mask.sum()), k), dtype=np.int32)
        for row, cell_idx in enumerate(np.flatnonzero(mask)):
            out[row] = self.cells[int(cell_idx)]
        return out

    def triangle_indices(self) -> np.ndarray:
        """Return a ``(num_triangles, 3)`` int32 array of triangle indices.

        Includes both native ``VTK_TRIANGLE`` cells and the boundary
        triangles extracted from any volumetric cells (tetrahedra,
        hexahedra) present in the dataset. When the mesh is a pure surface
        mesh this is just the triangle connectivity.
        """
        parts: list[np.ndarray] = []

        if np.any(self.cell_types == VTK_TRIANGLE):
            parts.append(self.cells_of_type(VTK_TRIANGLE))

        volumetric_faces = (
            (VTK_TETRA, _TET_FACES),
            (VTK_HEXAHEDRON, _HEX_FACES),
            (VTK_WEDGE, _WEDGE_FACES),
            (VTK_PYRAMID, _PYRAMID_FACES),
        )
        if any(np.any(self.cell_types == ct) for ct, _ in volumetric_faces):
            parts.append(self._volumetric_boundary_triangles(volumetric_faces))

        if not parts:
            return np.empty((0, 3), dtype=np.int32)
        return np.concatenate(parts, axis=0)

    def to_tet_mesh(self) -> newton.TetMesh:
        """Build a :class:`newton.TetMesh` from the dataset's ``VTK_TETRA`` cells.

        Raises:
            ValueError: If the dataset contains no tetrahedral cells.
        """
        tets = self.cells_of_type(VTK_TETRA)
        return newton.TetMesh(
            vertices=self.points,
            tet_indices=tets.reshape(-1).astype(np.int32),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _volumetric_boundary_triangles(
        self,
        face_tables: tuple[tuple[int, tuple[tuple[int, ...], ...]], ...],
    ) -> np.ndarray:
        """Fan-triangulate the boundary faces of all volumetric cells.

        Iterates every cell of every type in ``face_tables`` and tallies each
        face by its sorted-vertex key so that faces shared by neighbouring
        cells cancel out. The remaining (single-count) faces form the
        watertight boundary, which is then fan-triangulated to produce a
        ``(num_triangles, 3)`` index array.
        """
        face_counts: dict[tuple[int, ...], int] = {}
        face_oriented: dict[tuple[int, ...], tuple[int, ...]] = {}

        for cell_type, face_table in face_tables:
            for cell_idx in np.flatnonzero(self.cell_types == cell_type):
                verts = self.cells[int(cell_idx)]
                for face_offsets in face_table:
                    oriented = tuple(int(verts[o]) for o in face_offsets)
                    key = tuple(sorted(oriented))
                    face_counts[key] = face_counts.get(key, 0) + 1
                    face_oriented[key] = oriented

        triangles: list[tuple[int, int, int]] = []
        for key, count in face_counts.items():
            if count != 1:
                continue
            oriented = face_oriented[key]
            for i in range(1, len(oriented) - 1):
                triangles.append((oriented[0], oriented[i], oriented[i + 1]))

        if not triangles:
            return np.empty((0, 3), dtype=np.int32)
        return np.asarray(triangles, dtype=np.int32)


# ----------------------------------------------------------------------
# Top-level parser
# ----------------------------------------------------------------------


def load_vtk_unstructured_grid(path: str | Path) -> VtkUnstructuredGrid:
    """Parse a legacy ASCII VTK ``UNSTRUCTURED_GRID`` file.

    Args:
        path: Filesystem path to the ``.vtk`` file.

    Returns:
        A populated :class:`VtkUnstructuredGrid`.

    Raises:
        ValueError: If the file is not a legacy ASCII ``UNSTRUCTURED_GRID``
            or if a recognized section is malformed.
    """
    path = Path(path)
    tokens = path.read_text(encoding="utf-8").replace(",", " ").split()

    cursor = _expect_keyword(tokens, 0, "#")
    # Format identifier "# vtk DataFile Version X.Y".
    while cursor < len(tokens) and tokens[cursor] != "ASCII" and tokens[cursor] != "BINARY":
        cursor += 1
    if cursor >= len(tokens):
        raise ValueError(f"VTK file '{path}' does not declare ASCII or BINARY format.")
    if tokens[cursor].upper() == "BINARY":
        raise ValueError(f"Only legacy ASCII VTK files are supported, got BINARY in '{path}'.")
    cursor += 1

    dataset_idx = _find_keyword(tokens, "DATASET", cursor)
    if dataset_idx + 1 >= len(tokens) or tokens[dataset_idx + 1].upper() != "UNSTRUCTURED_GRID":
        raise ValueError(f"Expected DATASET UNSTRUCTURED_GRID in '{path}'.")
    cursor = dataset_idx + 2

    points, cursor = _read_points(tokens, cursor, path)
    cells, cursor = _read_cells(tokens, cursor, path)
    cell_types, cursor = _read_cell_types(tokens, cursor, path, expected=len(cells))

    point_data: dict[str, np.ndarray] = {}
    cell_data: dict[str, np.ndarray] = {}

    while cursor < len(tokens):
        token = tokens[cursor].upper()
        if token == "POINT_DATA":
            cursor = _read_data_section(
                tokens, cursor, count=int(tokens[cursor + 1]), dest=point_data
            )
        elif token == "CELL_DATA":
            cursor = _read_data_section(
                tokens, cursor, count=int(tokens[cursor + 1]), dest=cell_data
            )
        else:
            # Tolerate unknown trailing tokens — some writers append metadata
            # we don't model.
            cursor += 1

    # The header is the line following "# vtk DataFile Version X.Y", but we
    # only kept tokens. Reconstruct from the raw file head.
    header_line = ""
    try:
        with path.open(encoding="utf-8") as fh:
            for _ in range(1):
                fh.readline()
            header_line = fh.readline().strip()
    except OSError:
        pass

    return VtkUnstructuredGrid(
        points=points,
        cells=cells,
        cell_types=cell_types,
        point_data=point_data,
        cell_data=cell_data,
        header=header_line,
    )


# ----------------------------------------------------------------------
# Internals
# ----------------------------------------------------------------------


def _expect_keyword(tokens: list[str], idx: int, keyword: str) -> int:
    if idx >= len(tokens) or tokens[idx] != keyword:
        raise ValueError(f"Expected '{keyword}' at token {idx}, got {tokens[idx] if idx < len(tokens) else '<EOF>'}.")
    return idx + 1


def _find_keyword(tokens: list[str], keyword: str, start: int) -> int:
    keyword = keyword.upper()
    for i in range(start, len(tokens)):
        if tokens[i].upper() == keyword:
            return i
    raise ValueError(f"VTK file is missing the '{keyword}' section.")


def _read_points(tokens: list[str], cursor: int, path: Path) -> tuple[np.ndarray, int]:
    cursor = _find_keyword(tokens, "POINTS", cursor)
    count = int(tokens[cursor + 1])
    # tokens[cursor + 2] is dataType (e.g. "float"); we always parse as float32.
    start = cursor + 3
    end = start + count * 3
    if end > len(tokens):
        raise ValueError(f"POINTS section in '{path}' ended before {count} points were read.")
    points = np.asarray(tokens[start:end], dtype=np.float32).reshape(-1, 3)
    return points, end


def _read_cells(
    tokens: list[str], cursor: int, path: Path
) -> tuple[tuple[np.ndarray, ...], int]:
    cursor = _find_keyword(tokens, "CELLS", cursor)
    cell_count = int(tokens[cursor + 1])
    list_size = int(tokens[cursor + 2])
    start = cursor + 3
    end = start + list_size
    if end > len(tokens):
        raise ValueError(f"CELLS section in '{path}' ended before all cell data was read.")

    cells: list[np.ndarray] = []
    pos = start
    for cell_id in range(cell_count):
        size = int(tokens[pos])
        pos += 1
        cell_end = pos + size
        if cell_end > end:
            raise ValueError(f"CELLS section in '{path}' ended inside cell {cell_id}.")
        cells.append(np.asarray(tokens[pos:cell_end], dtype=np.int32))
        pos = cell_end
    if pos != end:
        raise ValueError(
            f"CELLS list size mismatch in '{path}': declared {list_size}, consumed {pos - start}."
        )

    return tuple(cells), end


def _read_cell_types(
    tokens: list[str], cursor: int, path: Path, expected: int
) -> tuple[np.ndarray, int]:
    cursor = _find_keyword(tokens, "CELL_TYPES", cursor)
    count = int(tokens[cursor + 1])
    if count != expected:
        raise ValueError(
            f"CELL_TYPES count {count} does not match CELLS count {expected} in '{path}'."
        )
    start = cursor + 2
    end = start + count
    if end > len(tokens):
        raise ValueError(f"CELL_TYPES section in '{path}' ended before {count} types were read.")
    cell_types = np.asarray(tokens[start:end], dtype=np.int32)
    return cell_types, end


# Top-level attribute keywords that close the current POINT_DATA / CELL_DATA
# section (or otherwise mark a new region of the file).
_SECTION_TERMINATORS = frozenset({"POINT_DATA", "CELL_DATA", "DATASET", "POINTS", "CELLS", "CELL_TYPES"})


def _read_data_section(
    tokens: list[str], cursor: int, count: int, dest: dict[str, np.ndarray]
) -> int:
    """Parse a POINT_DATA / CELL_DATA block of *count* tuples into ``dest``."""
    # Skip "POINT_DATA n" or "CELL_DATA n" header.
    cursor += 2

    while cursor < len(tokens):
        keyword = tokens[cursor].upper()

        if keyword in _SECTION_TERMINATORS and cursor > 0:
            return cursor

        if keyword == "SCALARS":
            cursor = _read_scalars(tokens, cursor, count, dest)
        elif keyword in ("VECTORS", "NORMALS"):
            cursor = _read_vectors(tokens, cursor, count, dest)
        elif keyword == "TENSORS":
            cursor = _read_tensors(tokens, cursor, count, dest)
        elif keyword == "COLOR_SCALARS":
            cursor = _read_color_scalars(tokens, cursor, count, dest)
        elif keyword == "LOOKUP_TABLE":
            # Standalone lookup table definition: LOOKUP_TABLE name size, then
            # size * 4 floats. We parse and discard.
            cursor = _skip_lookup_table_def(tokens, cursor)
        elif keyword in ("TENSORS6", "FIELD", "TEXTURE_COORDINATES"):
            raise ValueError(
                f"Unsupported VTK attribute '{tokens[cursor]}' at token {cursor}."
            )
        else:
            # Unknown token — assume the data section ended.
            return cursor

    return cursor


def _read_scalars(
    tokens: list[str], cursor: int, count: int, dest: dict[str, np.ndarray]
) -> int:
    # SCALARS name dataType [numComp]
    name = tokens[cursor + 1]
    # dataType at cursor + 2 is informational; we always cast to float32.
    cursor += 3
    num_comp = 1
    if cursor < len(tokens) and tokens[cursor].isdigit():
        num_comp = int(tokens[cursor])
        cursor += 1
    # LOOKUP_TABLE tableName (no size).
    if cursor >= len(tokens) or tokens[cursor].upper() != "LOOKUP_TABLE":
        raise ValueError(f"Expected LOOKUP_TABLE after SCALARS '{name}'.")
    cursor += 2  # skip "LOOKUP_TABLE" and the table name.

    end = cursor + count * num_comp
    if end > len(tokens):
        raise ValueError(f"SCALARS '{name}' ended before all values were read.")
    values = np.asarray(tokens[cursor:end], dtype=np.float32)
    if num_comp > 1:
        values = values.reshape(count, num_comp)
    dest[name] = values
    return end


def _read_vectors(
    tokens: list[str], cursor: int, count: int, dest: dict[str, np.ndarray]
) -> int:
    # VECTORS name dataType, then count * 3 floats.
    name = tokens[cursor + 1]
    cursor += 3
    end = cursor + count * 3
    if end > len(tokens):
        raise ValueError(f"VECTORS '{name}' ended before all values were read.")
    values = np.asarray(tokens[cursor:end], dtype=np.float32).reshape(count, 3)
    dest[name] = values
    return end


def _read_tensors(
    tokens: list[str], cursor: int, count: int, dest: dict[str, np.ndarray]
) -> int:
    # TENSORS name dataType, then count * 9 floats (row-major 3x3 per tuple).
    name = tokens[cursor + 1]
    cursor += 3
    end = cursor + count * 9
    if end > len(tokens):
        raise ValueError(f"TENSORS '{name}' ended before all values were read.")
    values = np.asarray(tokens[cursor:end], dtype=np.float32).reshape(count, 3, 3)
    dest[name] = values
    return end


def _read_color_scalars(
    tokens: list[str], cursor: int, count: int, dest: dict[str, np.ndarray]
) -> int:
    # COLOR_SCALARS name nValues, then count * nValues floats.
    name = tokens[cursor + 1]
    n_values = int(tokens[cursor + 2])
    cursor += 3
    end = cursor + count * n_values
    if end > len(tokens):
        raise ValueError(f"COLOR_SCALARS '{name}' ended before all values were read.")
    values = np.asarray(tokens[cursor:end], dtype=np.float32).reshape(count, n_values)
    dest[name] = values
    return end


def _skip_lookup_table_def(tokens: list[str], cursor: int) -> int:
    # LOOKUP_TABLE tableName size, then size * 4 floats (RGBA).
    size = int(tokens[cursor + 2])
    return cursor + 3 + size * 4
