"""What a hole IS: the lattice it sits in, its cells, and its identity.

WHY THIS MODULE EXISTS. Atlas had ten representations of a hole and no module
that owned the concept. Two lattice recoveries with two refusal protocols (one
raising, one returning a silent ``(0, 0)``); component labelling written twice;
island ids assigned in `path_hole_repair` while the anchors they are joined
against were computed in `planar_hole_patch`, reconciled by a comment reading
"both modules label the same lattice, so anchors coincide". Every engine
re-derived all of it from a mesh that had already lost its `hole_mask` at the
proxy-primitive seam.

Six defects fixed in one session lived in that gap — a vertical mirror in patch
materialization, a lattice bail that returned silently, an equal-depth tolerance
that did not scale, a single-pass bridge, a distance limit that was a no-op, and
island rejections reported without ids. None was visible from the interface that
produced it, because no interface produced it.

So identity is decided ONCE, here:

    ids are 1..N, ordered by (size, top-left cell)
    the anchor is min(cells) — the top-left cell in (row, col) order

Both orderings are the ones `path_hole_repair` already used; they are pinned by
tests because renumbering islands would renumber every report an artist reads.

Host-agnostic: numpy only, no ComfyUI, no torch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - guarded import
        raise RuntimeError(
            "atlas_camera.core.hole_field requires numpy. Install with: "
            "pip install -e .[vision]") from exc
    return np


class LatticeError(ValueError):
    """The mesh is no longer a structured relief lattice.

    Subclasses ValueError deliberately: `planar_hole_patch` raised bare
    ValueErrors here for three distinct refusals and callers catch that type.
    Naming the failure does not get to break them.
    """


# --------------------------------------------------------------------------
# Components
# --------------------------------------------------------------------------

_NEIGHBOURS_4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_NEIGHBOURS_8 = _NEIGHBOURS_4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def components(mask: Any, *, connectivity: int = 4) -> list[set[tuple[int, int]]]:
    """Connected cell components of a boolean cell mask.

    4-connectivity is the default because that is what the hole engines have
    always used: a diagonal touch is not a shared edge, and a patch fitted
    across one would span two unrelated surfaces.
    """
    np = _require_numpy()
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    offsets = _NEIGHBOURS_4 if connectivity == 4 else _NEIGHBOURS_8
    remaining = {tuple(int(v) for v in rc) for rc in np.argwhere(mask)}
    out: list[set[tuple[int, int]]] = []
    while remaining:
        seed = remaining.pop()
        component = {seed}
        stack = [seed]
        while stack:
            r, c = stack.pop()
            for dr, dc in offsets:
                neighbour = (r + dr, c + dc)
                if neighbour in remaining:
                    remaining.remove(neighbour)
                    component.add(neighbour)
                    stack.append(neighbour)
        out.append(component)
    return out


# --------------------------------------------------------------------------
# Island identity
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Island:
    """One hole. `anchor` is the deterministic top-left cell."""

    id: int
    cells: frozenset[tuple[int, int]]
    anchor: tuple[int, int]

    @property
    def size(self) -> int:
        return len(self.cells)


class HoleField:
    """Every hole in one cell mask, labelled once.

    Pass this object to the engines rather than letting each label the mask
    again — that is the whole point. Two labellings of the same mask agree only
    as long as nobody changes a sort key.
    """

    __slots__ = ("islands", "id_map")

    def __init__(self, islands: list[Island], id_map: Any) -> None:
        self.islands = islands
        self.id_map = id_map

    @classmethod
    def from_cell_mask(cls, mask: Any, *, connectivity: int = 4) -> HoleField:
        np = _require_numpy()
        mask = np.asarray(mask, dtype=bool)
        found = components(mask, connectivity=connectivity)
        # (size, top-left cell) — the order path_hole_repair assigned before
        # this module existed. Island numbers appear in artist-facing reports.
        found.sort(key=lambda cells: (len(cells), min(cells)))
        id_map = np.zeros(mask.shape, dtype=np.int32)
        islands: list[Island] = []
        for island_id, cells in enumerate(found, start=1):
            anchor = min(cells)
            islands.append(Island(id=island_id, cells=frozenset(cells),
                                  anchor=anchor))
            for row, col in cells:
                id_map[row, col] = island_id
        return cls(islands, id_map)

    def id_at(self, cell: tuple[int, int]) -> int:
        """Island id covering `cell`, or 0 — out-of-bounds included.

        This is the lookup that joins a fit rejection back to the island it
        rejected. It returns 0 rather than raising because a rejection can name
        a cell outside the caller's candidate set.
        """
        row, col = int(cell[0]), int(cell[1])
        if row < 0 or col < 0:
            return 0
        if row >= self.id_map.shape[0] or col >= self.id_map.shape[1]:
            return 0
        return int(self.id_map[row, col])

    def island(self, island_id: int) -> Island | None:
        for island in self.islands:
            if island.id == int(island_id):
                return island
        return None

    def as_component_list(self) -> list[set[tuple[int, int]]]:
        """Cells per island in id order — the shape `build_island_candidates`
        already returns as `components`."""
        return [set(island.cells) for island in self.islands]

    def component_by_id(self) -> dict[int, set[tuple[int, int]]]:
        return {island.id: set(island.cells) for island in self.islands}


# --------------------------------------------------------------------------
# Lattice recovery
# --------------------------------------------------------------------------

def _mode_step(values: Any) -> int:
    np = _require_numpy()
    unique = np.unique(np.asarray(values, dtype=np.int64))
    diffs = np.diff(unique)
    diffs = diffs[diffs > 0]
    if not len(diffs):
        return 0
    counts = np.bincount(diffs)
    return int(np.argmax(counts[1:]) + 1)


def _axis_lattice(length: int, step: int) -> Any:
    np = _require_numpy()
    values = np.arange(0, int(length), int(step), dtype=np.int64)
    if not len(values) or values[-1] != int(length) - 1:
        values = np.append(values, int(length) - 1)
    return values


def recover_lattice(mesh: Any, width: int, height: int) -> dict[str, Any]:
    """Recover the pre-compaction relief grid and locate every existing face.

    Works in PIXEL space and builds the FULL expected lattice from the image
    size and the recovered step, so a row torn away entirely still occupies its
    index. (The UV-space reconstruction in `mesh_repair` derives its rows from
    the occupied lines alone and cannot represent that case; the two are not
    interchangeable, which is why this one is the public entry point.)
    """
    np = _require_numpy()
    vertices = np.asarray(mesh.vertices, dtype=np.float64).reshape(-1, 3)
    faces = np.asarray(mesh.faces, dtype=np.int64).reshape(-1, 3)
    uvs = np.asarray(mesh.uvs, dtype=np.float64).reshape(-1, 2)
    if len(vertices) < 3 or len(faces) < 1 or len(uvs) != len(vertices):
        raise LatticeError(
            "relief mesh is empty or has mismatched vertex/UV arrays")

    px = np.rint(uvs[:, 0] * max(width - 1, 1)).astype(np.int64)
    py = np.rint((1.0 - uvs[:, 1]) * max(height - 1, 1)).astype(np.int64)
    step_x = _mode_step(px)
    step_y = _mode_step(py)
    candidates = [v for v in (step_x, step_y) if v > 0]
    if not candidates:
        raise LatticeError("could not recover a regular UV lattice")
    step = max(candidates, key=candidates.count)
    rows = _axis_lattice(height, step)
    cols = _axis_lattice(width, step)

    row_lookup = {int(v): i for i, v in enumerate(rows)}
    col_lookup = {int(v): i for i, v in enumerate(cols)}
    index_grid = np.full((len(rows), len(cols)), -1, dtype=np.int64)
    grid_coords = np.full((len(vertices), 2), -1, dtype=np.int64)
    mapped = 0
    for vertex_index, (x, y) in enumerate(zip(px, py)):
        r = row_lookup.get(int(y))
        c = col_lookup.get(int(x))
        if r is None or c is None:
            continue
        if index_grid[r, c] < 0:
            index_grid[r, c] = vertex_index
        grid_coords[vertex_index] = (r, c)
        mapped += 1
    if mapped < max(3, int(0.95 * len(vertices))):
        raise LatticeError(
            "mesh UVs are no longer a structured relief lattice; "
            "run Atlas Planar Hole Patch before retopology"
        )

    face_cells = np.full((len(faces), 2), -1, dtype=np.int64)
    coverage = np.zeros((len(rows) - 1, len(cols) - 1), dtype=np.int16)
    for face_index, face in enumerate(faces):
        coords = grid_coords[face]
        if (coords < 0).any():
            continue
        if int(coords[:, 0].max() - coords[:, 0].min()) > 1:
            raise LatticeError(
                "mesh contains non-grid faces; run Atlas Planar Hole Patch "
                "before retopology"
            )
        if int(coords[:, 1].max() - coords[:, 1].min()) > 1:
            raise LatticeError(
                "mesh contains non-grid faces; run Atlas Planar Hole Patch "
                "before retopology"
            )
        r = int(coords[:, 0].min())
        c = int(coords[:, 1].min())
        if r < coverage.shape[0] and c < coverage.shape[1]:
            face_cells[face_index] = (r, c)
            coverage[r, c] += 1

    return {
        "vertices": vertices,
        "faces": faces,
        "uvs": uvs,
        "rows": rows,
        "cols": cols,
        "index_grid": index_grid,
        "grid_coords": grid_coords,
        "face_cells": face_cells,
        "coverage": coverage,
    }
