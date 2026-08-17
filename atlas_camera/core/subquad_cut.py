"""Cut a torn quad at the depth cliff instead of deleting the whole cell.

`build_relief_mesh` decides tearing per TRIANGLE on an axis-aligned lattice and
then simply does not emit the triangle. That decision is correct — the two sides
of a depth cliff must never be joined — but its granularity is a whole grid cell,
so the surviving silhouette is a staircase whose amplitude is one cell, and the
cell's worth of surface on BOTH sides of the cliff is thrown away with it.
Measured on a synthetic diagonal cliff at `relief_grid=128` on a 1024 px plate
(step = 8 px): the boundary sits a mean of 5.67 px from the true cliff, which is
*larger* than the step/2 quantization bound precisely because the whole cell goes.

This module keeps the tear and recovers the surface. Inside a torn cell it:

1. classifies each of the four corners as the NEAR or the FAR sheet (log-depth
   against the cell's own midpoint);
2. locates, for each cell edge whose two corners disagree, the actual cliff
   position by scanning the FULL-RESOLUTION depth along that edge — not by
   interpolating the two lattice samples, which for step-shaped data always
   lands mid-edge no matter where the cliff really is;
3. emits the near polygon and the far polygon separately, each running up to the
   crossing points, **sharing no vertex with the other**.

Both sheets therefore reach the true cliff, the silhouette lands at plate
precision rather than lattice precision, and the tear is preserved: the crossing
points spawn two vertices at the same pixel carrying different depths — the
layered-depth underlap `AtlasRefineOcclusionSeams` uses, and the opposite of the
"long near-to-far curtain" DESIGN_RULES forbids.

Deliberately NOT a marching-squares case table. Walking the quad's CCW corner
cycle and appending corners and crossings in order yields each polygon already
correctly wound, which is the same construction with none of the 16-case
bookkeeping. The one configuration it cannot express is the diagonal saddle (two
near corners that are not adjacent in the cycle): the cell contains two separate
cliffs and either resolution is a guess, so it is left torn exactly as today.
"""

from __future__ import annotations

from typing import Any, Callable


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - exercised only without numpy
        raise RuntimeError(
            "subquad_cut requires numpy. Install with: pip install -e .[vision]"
        ) from exc
    return np


#: The quad's corners in CCW order as seen from the camera, matching the
#: `tri_a = (i00, i10, i01)` / `tri_b = (i10, i11, i01)` winding
#: `build_relief_mesh` emits. Order is (row offset, col offset).
_CYCLE = ((0, 0), (1, 0), (1, 1), (0, 1))


def _cliff_crossing(
    depth_full: Any,
    valid_full: Any,
    r_a: int, c_a: int,
    r_b: int, c_b: int,
    np: Any,
) -> tuple[float, float, float] | None:
    """Where along the lattice edge (a → b) does the depth actually jump?

    Returns ``(t, depth_near_side, depth_far_side)`` with ``t`` in (0, 1) measured
    from a, or None when the full-res samples carry no usable jump.

    Scanning matters. Interpolating the two lattice depths to their midpoint —
    the textbook marching-squares crossing — puts a step function's crossing at
    t = 0.5 regardless of where the step is, which merely halves the error. The
    plate knows the answer to the pixel.
    """
    n = max(abs(r_b - r_a), abs(c_b - c_a))
    if n < 1:
        return None
    rr = np.linspace(r_a, r_b, n + 1).round().astype(np.int64)
    cc = np.linspace(c_a, c_b, n + 1).round().astype(np.int64)
    line = np.asarray(depth_full[rr, cc], dtype=np.float64)
    good = np.asarray(valid_full[rr, cc], dtype=bool) & np.isfinite(line) & (line > 1e-6)
    if good.sum() < 2:
        return None
    logs = np.where(good, np.log(np.maximum(line, 1e-6)), np.nan)
    jump = np.abs(np.diff(logs))
    if not np.isfinite(jump).any():
        return None
    k = int(np.nanargmax(jump))
    # The cliff lies between sample k and k+1; place the cut on the boundary
    # between those two pixels.
    t = (k + 0.5) / n
    return float(min(max(t, 1e-3), 1.0 - 1e-3)), float(line[k]), float(line[k + 1])


def cut_torn_quads(
    *,
    grid_depth: Any,
    grid_valid: Any,
    rows: Any,
    cols: Any,
    torn: Any,
    depth_full: Any,
    valid_full: Any,
    lattice_index: Any,
    unproject: Callable[[float, float, float], Any],
    max_cut_cells: int = 20000,
) -> dict[str, Any]:
    """Recover both sheets inside torn cells that straddle a single depth cliff.

    ``torn`` is the (nr-1, nc-1) boolean of cells the tear tests rejected.
    ``lattice_index`` is the (nr, nc) vertex index map. ``unproject(u_px, v_px,
    depth)`` must be the SAME back-projection `build_relief_mesh` uses for its
    lattice vertices — passing it in rather than re-deriving it keeps one copy of
    the ray math, so a cut vertex can never drift from the lattice it joins.

    Returns ``{"positions", "uv_pixels", "faces", "stats"}``: new vertices in
    world space, their fractional source pixels (for UVs), and faces indexed into
    ``lattice_index`` values first, then ``base + k`` for the k-th new vertex.
    """
    np = _require_numpy()

    depth_grid = np.asarray(grid_depth, dtype=np.float64)
    valid_grid = np.asarray(grid_valid, dtype=bool)
    torn = np.asarray(torn, dtype=bool)
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    base = int(np.asarray(lattice_index).size)

    # Only cells that are torn, whose four corners are ALL valid, are cliff
    # candidates. A cell with an invalid corner is torn because there is no data
    # there — genuinely nothing to recover, and inventing a boundary inside it
    # would be the "backdrop is not a hole rim" mistake.
    cand = torn.copy()
    cand &= valid_grid[:-1, :-1] & valid_grid[1:, :-1]
    cand &= valid_grid[1:, 1:] & valid_grid[:-1, 1:]
    cand &= np.isfinite(depth_grid[:-1, :-1]) & np.isfinite(depth_grid[1:, :-1])
    cand &= np.isfinite(depth_grid[1:, 1:]) & np.isfinite(depth_grid[:-1, 1:])

    cell_rows, cell_cols = np.nonzero(cand)
    n_candidates = int(cell_rows.size)
    budget = max(0, int(max_cut_cells))
    truncated = n_candidates > budget
    if truncated:
        cell_rows = cell_rows[:budget]
        cell_cols = cell_cols[:budget]

    positions: list[Any] = []
    uv_pixels: list[tuple[float, float]] = []
    faces: list[tuple[int, int, int]] = []
    n_saddles = 0
    n_cut = 0

    for cell_r, cell_c in zip(cell_rows.tolist(), cell_cols.tolist()):
        corner_d = [float(depth_grid[cell_r + dr, cell_c + dc]) for dr, dc in _CYCLE]
        logs = np.log(np.maximum(np.asarray(corner_d), 1e-6))
        mid = 0.5 * (float(logs.min()) + float(logs.max()))
        near = [bool(value < mid) for value in logs]
        if all(near) or not any(near):
            continue  # torn on edge length or normal bend, not on a depth cliff
        if sum(near) == 2 and near[0] == near[2]:
            n_saddles += 1  # diagonal saddle: two cliffs in one cell, left torn
            continue

        corner_px = [
            (float(cols[cell_c + dc]), float(rows[cell_r + dr])) for dr, dc in _CYCLE
        ]
        corner_idx = [
            int(lattice_index[cell_r + dr, cell_c + dc]) for dr, dc in _CYCLE
        ]

        crossings: list[tuple[float, float] | None] = []
        crossing_depths: list[tuple[float, float] | None] = []
        usable = True
        for k in range(4):
            nxt = (k + 1) % 4
            if near[k] == near[nxt]:
                crossings.append(None)
                crossing_depths.append(None)
                continue
            dr_a, dc_a = _CYCLE[k]
            dr_b, dc_b = _CYCLE[nxt]
            found = _cliff_crossing(
                depth_full, valid_full,
                int(rows[cell_r + dr_a]), int(cols[cell_c + dc_a]),
                int(rows[cell_r + dr_b]), int(cols[cell_c + dc_b]),
                np,
            )
            if found is None:
                usable = False
                break
            t, d_before, d_after = found
            ax, ay = corner_px[k]
            bx, by = corner_px[nxt]
            crossings.append((ax + t * (bx - ax), ay + t * (by - ay)))
            # d_before belongs to corner k's side, d_after to corner nxt's.
            crossing_depths.append(
                (d_before, d_after) if near[k] else (d_after, d_before)
            )
        if not usable:
            continue

        def _emit(want_near: bool) -> bool:
            """Append one sheet's polygon; True when it produced triangles."""
            loop: list[int] = []
            for k in range(4):
                if near[k] == want_near:
                    loop.append(corner_idx[k])
                cross = crossings[k]
                if cross is None:
                    continue
                near_depth, far_depth = crossing_depths[k]
                depth_here = near_depth if want_near else far_depth
                loop.append(base + len(positions))
                positions.append(unproject(cross[0], cross[1], depth_here))
                uv_pixels.append(cross)
            if len(loop) < 3:
                return False
            for j in range(1, len(loop) - 1):
                faces.append((loop[0], loop[j], loop[j + 1]))
            return True

        produced_near = _emit(True)
        produced_far = _emit(False)
        if produced_near or produced_far:
            n_cut += 1

    stats = {
        "n_candidate_cells": n_candidates,
        "n_cut_cells": int(n_cut),
        "n_saddle_cells": int(n_saddles),
        "n_new_vertices": int(len(positions)),
        "n_new_faces": int(len(faces)),
        "budget_truncated": bool(truncated),
        "max_cut_cells": budget,
    }
    if not positions:
        return {
            "positions": np.zeros((0, 3), dtype=np.float64),
            "uv_pixels": np.zeros((0, 2), dtype=np.float64),
            "faces": np.zeros((0, 3), dtype=np.int64),
            "stats": stats,
        }
    return {
        "positions": np.asarray(positions, dtype=np.float64).reshape(-1, 3),
        "uv_pixels": np.asarray(uv_pixels, dtype=np.float64).reshape(-1, 2),
        "faces": np.asarray(faces, dtype=np.int64).reshape(-1, 3),
        "stats": stats,
    }
