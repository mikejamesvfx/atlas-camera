"""Mask-authoritative local surface reconstruction for structured relief meshes.

Unlike a boundary-loop filler, this pass may cut faces from an intact mesh.
The image mask selects cells, a configurable collar manufactures a clean local
rim, and forward camera depth is harmonically interpolated across that rim.
The reconstructed vertices stay on their original camera rays, preserving the
relief mesh's projective UV contract.  Numpy is the only runtime dependency.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from atlas_camera.core.hole_field import components, recover_lattice


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Masked surface reconstruction requires numpy. "
            "Install with: pip install -e .[vision]"
        ) from exc
    return np


@dataclass(frozen=True, slots=True)
class MaskedSurfaceReconstructConfig:
    """Safety and solve controls for :func:`reconstruct_masked_surface`."""

    rim_cells: int = 1
    max_components: int = 64
    max_hole_fraction: float = 0.05
    enclosed_only: bool = True
    smooth_iterations: int = 128


def _dilate_cells(mask: Any, radius: int) -> Any:
    np = _require_numpy()
    source = np.asarray(mask, dtype=bool)
    result = source.copy()
    for _ in range(max(0, int(radius))):
        grown = result.copy()
        grown[1:, :] |= result[:-1, :]
        grown[:-1, :] |= result[1:, :]
        grown[:, 1:] |= result[:, :-1]
        grown[:, :-1] |= result[:, 1:]
        result = grown
    return result


def _corner_is_boundary(
    corner: tuple[int, int],
    component: set[tuple[int, int]],
    cell_shape: tuple[int, int],
) -> bool:
    row, col = corner
    adjacent = (
        (row - 1, col - 1),
        (row - 1, col),
        (row, col - 1),
        (row, col),
    )
    return any(
        (rr, cc) not in component
        for rr, cc in adjacent
        if 0 <= rr < cell_shape[0] and 0 <= cc < cell_shape[1]
    )


def _harmonic_depths(
    corners: set[tuple[int, int]],
    fixed: dict[tuple[int, int], float],
    iterations: int,
) -> dict[tuple[int, int], float]:
    """Jacobi-solve scalar forward depth with the manufactured rim fixed."""
    np = _require_numpy()
    initial = float(np.mean(list(fixed.values())))
    values = {corner: fixed.get(corner, initial) for corner in corners}
    unknown = sorted(corners.difference(fixed))
    for _ in range(max(1, int(iterations))):
        updated = dict(values)
        max_delta = 0.0
        for row, col in unknown:
            neighbours = [
                values[neighbour]
                for neighbour in (
                    (row - 1, col),
                    (row + 1, col),
                    (row, col - 1),
                    (row, col + 1),
                )
                if neighbour in values
            ]
            if neighbours:
                value = float(sum(neighbours) / len(neighbours))
                updated[(row, col)] = value
                max_delta = max(max_delta, abs(value - values[(row, col)]))
        values = updated
        if max_delta < 1e-7:
            break
    return values


def reconstruct_masked_surface(
    mesh: Any,
    hole_mask: Any,
    *,
    view_matrix: Any,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    image_width: int,
    image_height: int,
    config: MaskedSurfaceReconstructConfig | None = None,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    """Cut and reconstruct masked relief cells without Blender.

    Returns ``(mesh, remaining_mask, created_region, report)``.  The input
    mesh is never mutated.  The mask is authoritative: selected cells are
    eligible even when they still contain two intact triangles, which is what
    lets this pass manufacture a boundary for a topology-boundary-less gap.
    """
    np = _require_numpy()
    from atlas_camera.core.relief_mesh import ReliefMesh

    cfg = config or MaskedSurfaceReconstructConfig()
    width = int(image_width)
    height = int(image_height)
    selected_mask = np.asarray(hole_mask, dtype=bool)
    if selected_mask.shape != (height, width):
        raise ValueError(
            f"hole mask shape {selected_mask.shape} does not match "
            f"camera image {(height, width)}"
        )
    if float(fx) <= 0.0 or float(fy) <= 0.0:
        raise ValueError("fx and fy must be positive")

    lattice = recover_lattice(mesh, width, height)
    vertices = lattice["vertices"]
    faces = lattice["faces"]
    uvs = lattice["uvs"]
    rows = lattice["rows"]
    cols = lattice["cols"]
    index_grid = lattice["index_grid"]
    face_cells = lattice["face_cells"]
    cell_shape = (len(rows) - 1, len(cols) - 1)
    row_centers = ((rows[:-1] + rows[1:]) // 2).astype(np.int64)
    col_centers = ((cols[:-1] + cols[1:]) // 2).astype(np.int64)
    selected_cells = selected_mask[np.ix_(row_centers, col_centers)]
    grown_cells = _dilate_cells(selected_cells, cfg.rim_cells)
    found = components(grown_cells)
    found.sort(key=lambda cells: (len(cells), min(cells)))

    vm = np.asarray(view_matrix, dtype=np.float64).reshape(4, 4)
    c2w = np.linalg.inv(vm)
    rotation = c2w[:3, :3]
    camera = c2w[:3, 3]
    new_vertices = vertices.tolist()
    new_uvs = uvs.tolist()
    new_faces: list[list[int]] = []
    remove_face = np.zeros(len(faces), dtype=bool)
    remaining_mask = selected_mask.copy()
    created_region = np.zeros((height, width), dtype=bool)
    records: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    accepted = 0
    attempted = 0
    support_min = float("inf")
    support_max = float("-inf")
    total_cells = max(int(grown_cells.size), 1)

    risk_raw = getattr(mesh, "edge_risk", None)
    if risk_raw is None:
        edge_risk = None
    else:
        edge_risk = np.asarray(risk_raw, dtype=np.float64).reshape(-1).tolist()
        if len(edge_risk) != len(vertices):
            edge_risk = None

    for component_index, component in enumerate(found, start=1):
        anchor = [int(v) for v in min(component)]
        touches_frame = any(
            row == 0
            or col == 0
            or row == cell_shape[0] - 1
            or col == cell_shape[1] - 1
            for row, col in component
        )
        fraction = len(component) / total_cells
        reason = None
        if cfg.enclosed_only and touches_frame:
            reason = "touches_frame"
        elif fraction > float(cfg.max_hole_fraction):
            reason = "exceeds_max_hole_fraction"
        elif attempted >= max(0, int(cfg.max_components)):
            reason = "component_budget_exceeded"
        if reason is not None:
            rejected.append({
                "component": component_index,
                "anchor_cell": anchor,
                "cells": len(component),
                "reason": reason,
            })
            continue
        attempted += 1

        corners = {
            corner
            for row, col in component
            for corner in (
                (row, col),
                (row + 1, col),
                (row, col + 1),
                (row + 1, col + 1),
            )
        }
        rim_corners = {
            corner
            for corner in corners
            if _corner_is_boundary(corner, component, cell_shape)
        }
        missing_rim = [corner for corner in sorted(rim_corners)
                       if int(index_grid[corner]) < 0]
        if missing_rim:
            rejected.append({
                "component": component_index,
                "anchor_cell": anchor,
                "cells": len(component),
                "reason": "incomplete_support_rim",
                "missing_rim_vertices": len(missing_rim),
            })
            continue

        fixed: dict[tuple[int, int], float] = {}
        for corner in rim_corners:
            index = int(index_grid[corner])
            homogeneous = np.append(vertices[index], 1.0)
            depth = -float((vm @ homogeneous)[2])
            if not np.isfinite(depth) or depth <= 1e-6:
                fixed = {}
                break
            fixed[corner] = depth
        if not fixed:
            rejected.append({
                "component": component_index,
                "anchor_cell": anchor,
                "cells": len(component),
                "reason": "invalid_support_depth",
            })
            continue

        depths = _harmonic_depths(corners, fixed, cfg.smooth_iterations)
        corner_indices: dict[tuple[int, int], int] = {}
        vertex_checkpoint = len(new_vertices)
        uv_checkpoint = len(new_uvs)
        risk_checkpoint = len(edge_risk) if edge_risk is not None else 0
        failed = False
        for row, col in sorted(corners):
            if (row, col) in rim_corners:
                corner_indices[(row, col)] = int(index_grid[row, col])
                continue
            depth = float(depths[(row, col)])
            if not np.isfinite(depth) or depth <= 1e-6:
                failed = True
                break
            x = float(cols[col])
            y = float(rows[row])
            ray_camera = np.asarray(
                ((x - cx) / fx, -(y - cy) / fy, -1.0), dtype=np.float64
            )
            point = camera + depth * (rotation @ ray_camera)
            corner_indices[(row, col)] = len(new_vertices)
            new_vertices.append(point.tolist())
            new_uvs.append([
                x / max(width - 1, 1),
                1.0 - y / max(height - 1, 1),
            ])
            if edge_risk is not None:
                edge_risk.append(0.0)
        if failed:
            del new_vertices[vertex_checkpoint:]
            del new_uvs[uv_checkpoint:]
            if edge_risk is not None:
                del edge_risk[risk_checkpoint:]
            rejected.append({
                "component": component_index,
                "anchor_cell": anchor,
                "cells": len(component),
                "reason": "depth_solve_failed",
            })
            continue

        for row, col in sorted(component):
            i00 = corner_indices[(row, col)]
            i10 = corner_indices[(row + 1, col)]
            i01 = corner_indices[(row, col + 1)]
            i11 = corner_indices[(row + 1, col + 1)]
            new_faces.append([i00, i10, i01])
            new_faces.append([i10, i11, i01])
            y0, y1 = int(rows[row]), int(rows[row + 1])
            x0, x1 = int(cols[col]), int(cols[col + 1])
            remaining_mask[y0:y1 + 1, x0:x1 + 1] = False
            created_region[y0:y1 + 1, x0:x1 + 1] = True
        for face_index, (row, col) in enumerate(face_cells):
            if (int(row), int(col)) in component:
                remove_face[face_index] = True

        local_min = float(min(fixed.values()))
        local_max = float(max(fixed.values()))
        support_min = min(support_min, local_min)
        support_max = max(support_max, local_max)
        accepted += 1
        records.append({
            "component": component_index,
            "anchor_cell": anchor,
            "cells": len(component),
            "rim_vertices": len(rim_corners),
            "vertices_added": len(new_vertices) - vertex_checkpoint,
            "faces_added": 2 * len(component),
            "support_depth_min": local_min,
            "support_depth_max": local_max,
        })

    kept_faces = faces[~remove_face]
    if new_faces:
        all_faces = np.concatenate(
            (kept_faces, np.asarray(new_faces, dtype=np.int64).reshape(-1, 3)),
            axis=0,
        )
    else:
        all_faces = kept_faces.copy()

    report = {
        "components_found": len(found),
        "components_attempted": attempted,
        "components_reconstructed": accepted,
        "components_rejected": len(rejected),
        "rim_cells": max(0, int(cfg.rim_cells)),
        "vertices_added": len(new_vertices) - len(vertices),
        "faces_removed": int(remove_face.sum()),
        "faces_added": len(new_faces),
        "support_depth_min": support_min if accepted else None,
        "support_depth_max": support_max if accepted else None,
        "component_records": records + rejected,
    }
    stats = copy.deepcopy(getattr(mesh, "stats", None) or {})
    stats["masked_surface_reconstruct"] = copy.deepcopy(report)
    stats["n_vertices"] = len(new_vertices)
    stats["n_faces"] = len(all_faces)
    rebuilt = ReliefMesh(
        vertices=np.asarray(new_vertices, dtype=np.float32),
        faces=np.asarray(all_faces, dtype=np.int32),
        uvs=np.asarray(new_uvs, dtype=np.float32),
        stats=stats,
        hole_mask=remaining_mask,
        filled_mask=getattr(mesh, "filled_mask", None),
        edge_risk=(np.asarray(edge_risk, dtype=np.float32)
                   if edge_risk is not None else None),
    )
    return rebuilt, remaining_mask, created_region, report

