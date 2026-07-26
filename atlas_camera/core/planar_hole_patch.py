"""Normal-guided planar patches for enclosed relief-mesh holes.

The relief mesh is a compacted image-space lattice.  This module recovers that
lattice from projective UVs, finds selected missing quads, fits a local plane
from the valid boundary normals/positions, and replaces each accepted hole with
camera-ray/plane intersections.  Perimeter vertices are reused where possible,
so the patch is part of the relief mesh rather than a disconnected primitive.

The fit is deliberately conservative: open frame/sky gaps, large components,
mixed-normal boundaries, and high-residual planes are rejected.  Those cases
need a separate clean-plate or hidden-geometry layer, not a rubber sheet.
"""

from __future__ import annotations

import copy
import heapq
import math
from dataclasses import dataclass
from typing import Any


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Planar hole patching requires numpy. "
            "Install with: pip install -e .[vision]"
        ) from exc
    return np


@dataclass(frozen=True, slots=True)
class PlanarHolePatchConfig:
    """Controls for conservative local plane fitting and patch generation."""

    ring_cells: int = 2
    max_components: int = 64
    normal_tolerance_deg: float = 25.0
    max_plane_error_m: float = 0.15
    max_hole_fraction: float = 0.05
    enclosed_only: bool = True
    min_support_vertices: int = 8
    min_normal_support_fraction: float = 0.30


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


def _recover_lattice(mesh: Any, width: int, height: int) -> dict[str, Any]:
    """Recover the pre-compaction relief grid and locate every existing face."""
    np = _require_numpy()
    vertices = np.asarray(mesh.vertices, dtype=np.float64).reshape(-1, 3)
    faces = np.asarray(mesh.faces, dtype=np.int64).reshape(-1, 3)
    uvs = np.asarray(mesh.uvs, dtype=np.float64).reshape(-1, 2)
    if len(vertices) < 3 or len(faces) < 1 or len(uvs) != len(vertices):
        raise ValueError("relief mesh is empty or has mismatched vertex/UV arrays")

    px = np.rint(uvs[:, 0] * max(width - 1, 1)).astype(np.int64)
    py = np.rint((1.0 - uvs[:, 1]) * max(height - 1, 1)).astype(np.int64)
    step_x = _mode_step(px)
    step_y = _mode_step(py)
    candidates = [v for v in (step_x, step_y) if v > 0]
    if not candidates:
        raise ValueError("could not recover a regular UV lattice")
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
        raise ValueError(
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
            raise ValueError(
                "mesh contains non-grid faces; run Atlas Planar Hole Patch "
                "before retopology"
            )
        if int(coords[:, 1].max() - coords[:, 1].min()) > 1:
            raise ValueError(
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
        "face_cells": face_cells,
        "coverage": coverage,
    }


def _components(mask: Any) -> list[set[tuple[int, int]]]:
    np = _require_numpy()
    remaining = {tuple(int(v) for v in rc) for rc in np.argwhere(mask)}
    out: list[set[tuple[int, int]]] = []
    while remaining:
        seed = remaining.pop()
        component = {seed}
        stack = [seed]
        while stack:
            r, c = stack.pop()
            for neighbor in ((r - 1, c), (r + 1, c),
                             (r, c - 1), (r, c + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)
        out.append(component)
    return out


def _vertex_normals(vertices: Any, faces: Any) -> Any:
    np = _require_numpy()
    normals = np.zeros_like(vertices, dtype=np.float64)
    tri = vertices[faces]
    face_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    lengths = np.linalg.norm(face_normals, axis=1)
    valid = lengths > 1e-10
    face_normals[valid] /= lengths[valid, None]
    face_normals[~valid] = 0.0
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1e-10
    normals[valid] /= lengths[valid, None]
    return normals


def _ring_cells(
    component: set[tuple[int, int]],
    shape: tuple[int, int],
    rings: int,
) -> Any:
    np = _require_numpy()
    source = np.zeros(shape, dtype=bool)
    for r, c in component:
        source[r, c] = True
    grown = source.copy()
    for _ in range(max(1, int(rings))):
        previous = grown.copy()
        grown[1:, :] |= previous[:-1, :]
        grown[:-1, :] |= previous[1:, :]
        grown[:, 1:] |= previous[:, :-1]
        grown[:, :-1] |= previous[:, 1:]
        grown[1:, 1:] |= previous[:-1, :-1]
        grown[1:, :-1] |= previous[:-1, 1:]
        grown[:-1, 1:] |= previous[1:, :-1]
        grown[:-1, :-1] |= previous[1:, 1:]
    return grown & ~source


def _support_indices(
    component: set[tuple[int, int]],
    index_grid: Any,
    ring_cells: int,
) -> Any:
    np = _require_numpy()
    ring = _ring_cells(component, (index_grid.shape[0] - 1,
                                   index_grid.shape[1] - 1), ring_cells)
    cells = np.argwhere(ring)
    indices: set[int] = set()
    for r, c in cells:
        for rr, cc in ((r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)):
            index = int(index_grid[rr, cc])
            if index >= 0:
                indices.add(index)
    return np.asarray(sorted(indices), dtype=np.int64)


def _fit_plane(
    points: Any,
    normals: Any,
    config: PlanarHolePatchConfig,
) -> tuple[
    tuple[Any, float, dict[str, Any]] | None,
    dict[str, Any],
]:
    np = _require_numpy()
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1e-8
    points = points[valid]
    normals = normals[valid] / lengths[valid, None]
    diagnostic: dict[str, Any] = {
        "support_vertices": int(len(points)),
    }
    if len(points) < config.min_support_vertices:
        return None, {
            **diagnostic,
            "reason": "insufficient valid normal support",
            "required_support_vertices": int(config.min_support_vertices),
        }

    cosine = math.cos(math.radians(float(config.normal_tolerance_deg)))
    # Do not materialize an N×N agreement matrix: a 4K relief grid can place
    # tens of thousands of vertices in a boundary ring.  A deterministic
    # 64-seed medoid approximation keeps this O(64N) with the same cluster
    # gate for ordinary hole sizes.
    if len(normals) <= 64:
        seed_indices = np.arange(len(normals), dtype=np.int64)
    else:
        seed_indices = np.linspace(
            0, len(normals) - 1, 64, dtype=np.int64)
    best_count = -1
    seed_index = int(seed_indices[0])
    selected = None
    for candidate in seed_indices:
        candidate_selected = np.abs(normals @ normals[candidate]) >= cosine
        count = int(candidate_selected.sum())
        if count > best_count:
            best_count = count
            seed_index = int(candidate)
            selected = candidate_selected
    assert selected is not None
    support_fraction = float(selected.mean())
    diagnostic["normal_support_fraction"] = support_fraction
    diagnostic["required_normal_support_fraction"] = float(
        config.min_normal_support_fraction)
    if (int(selected.sum()) < config.min_support_vertices
            or support_fraction < config.min_normal_support_fraction):
        return None, {
            **diagnostic,
            "reason": "normal consensus below threshold",
            "normal_inliers": int(selected.sum()),
            "required_support_vertices": int(config.min_support_vertices),
        }

    selected_normals = normals[selected].copy()
    seed = normals[seed_index]
    flip = (selected_normals @ seed) < 0.0
    selected_normals[flip] *= -1.0
    normal = selected_normals.mean(axis=0)
    normal /= max(float(np.linalg.norm(normal)), 1e-12)
    selected_points = points[selected]
    offsets = selected_points @ normal
    tolerance = max(float(config.max_plane_error_m), 1e-6)

    # Position the averaged-normal plane on the densest local offset band.
    # This is not a second agreement gate: it only chooses which side supplies
    # the plane position when the same orientation exists at multiple depths.
    order = np.argsort(offsets)
    sorted_offsets = offsets[order]
    best_lo = best_hi = 0
    hi = 0
    for lo in range(len(sorted_offsets)):
        hi = max(hi, lo)
        while (hi + 1 < len(sorted_offsets)
               and sorted_offsets[hi + 1] - sorted_offsets[lo]
               <= 2.0 * tolerance):
            hi += 1
        if hi - lo > best_hi - best_lo:
            best_lo, best_hi = lo, hi
    inlier_order = order[best_lo:best_hi + 1]
    if len(inlier_order) < config.min_support_vertices:
        return None, {
            **diagnostic,
            "reason": "insufficient plane-position support",
            "plane_inliers": int(len(inlier_order)),
            "required_support_vertices": int(config.min_support_vertices),
        }

    # Average the normals that agree in both angle and local plane position.
    final_normals = selected_normals[inlier_order]
    normal = final_normals.mean(axis=0)
    normal /= max(float(np.linalg.norm(normal)), 1e-12)
    inlier_points = selected_points[inlier_order]
    inlier_offsets = inlier_points @ normal
    offset = float(np.median(inlier_offsets))
    residuals = np.abs(inlier_offsets - offset)
    residual_p95 = float(np.percentile(residuals, 95))
    diagnostic["plane_error_p95_m"] = residual_p95
    diagnostic["max_plane_error_m"] = tolerance
    diagnostic["plane_normal_world"] = [float(v) for v in normal]
    diagnostic["plane_offset_m"] = offset
    diagnostic["plane_support_fraction"] = float(
        len(inlier_points) / max(len(selected_points), 1))
    if residual_p95 > tolerance:
        return None, {
            **diagnostic,
            "reason": "plane residual exceeds tolerance",
            "plane_inliers": int(len(inlier_points)),
        }
    fit_info = {
        **diagnostic,
        "plane_inliers": int(len(inlier_points)),
    }
    return (normal, offset, fit_info), fit_info


def _corner_is_boundary(
    corner: tuple[int, int],
    component: set[tuple[int, int]],
    cell_shape: tuple[int, int],
) -> bool:
    r, c = corner
    adjacent = (
        (r - 1, c - 1), (r - 1, c),
        (r, c - 1), (r, c),
    )
    for rr, cc in adjacent:
        if 0 <= rr < cell_shape[0] and 0 <= cc < cell_shape[1]:
            if (rr, cc) not in component:
                return True
    return False


def patch_planar_holes(
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
    config: PlanarHolePatchConfig | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    """Return ``(patched_mesh, remaining_hole_mask, report)``.

    Accepted components replace any surviving half-quad faces in the selected
    cells, share existing perimeter vertices, and add planar interior vertices
    with exact projective UVs.  The input mesh is never mutated.
    """
    np = _require_numpy()
    from atlas_camera.core.relief_mesh import ReliefMesh

    cfg = config or PlanarHolePatchConfig()
    width = int(image_width)
    height = int(image_height)
    selected_mask = np.asarray(hole_mask, dtype=bool)
    if selected_mask.shape != (height, width):
        raise ValueError(
            f"hole mask shape {selected_mask.shape} does not match "
            f"camera image {(height, width)}"
        )

    lattice = _recover_lattice(mesh, width, height)
    vertices = lattice["vertices"]
    faces = lattice["faces"]
    uvs = lattice["uvs"]
    rows = lattice["rows"]
    cols = lattice["cols"]
    index_grid = lattice["index_grid"]
    coverage = lattice["coverage"]
    face_cells = lattice["face_cells"]
    row_centers = ((rows[:-1] + rows[1:]) // 2).astype(np.int64)
    col_centers = ((cols[:-1] + cols[1:]) // 2).astype(np.int64)
    selected_cells = selected_mask[np.ix_(row_centers, col_centers)]
    candidates = selected_cells & (coverage < 2)
    components = _components(candidates)
    vertex_normals = _vertex_normals(vertices, faces)

    vm = np.asarray(view_matrix, dtype=np.float64).reshape(4, 4)
    c2w = np.linalg.inv(vm)
    rotation = c2w[:3, :3]
    camera = c2w[:3, 3]
    new_vertices = vertices.tolist()
    new_uvs = uvs.tolist()
    new_faces: list[list[int]] = []
    remove_face = np.zeros(len(faces), dtype=bool)
    remaining_mask = selected_mask.copy()
    edge_risk_raw = getattr(mesh, "edge_risk", None)
    if edge_risk_raw is None:
        edge_risk = None
    else:
        edge_risk = np.asarray(edge_risk_raw, dtype=np.float64).reshape(-1).tolist()
        if len(edge_risk) != len(vertices):
            edge_risk = None

    filled: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    cell_count = max(int(candidates.size), 1)

    # Reject obviously unsafe regions before applying the fit budget.  The old
    # largest-first loop let a handful of frame/sky components consume the
    # entire budget, so the small enclosed tears the node is meant to repair
    # were never attempted.
    eligible: list[set[tuple[int, int]]] = []
    for component in components:
        fraction = len(component) / cell_count
        touches_frame = any(
            r == 0 or c == 0
            or r == candidates.shape[0] - 1
            or c == candidates.shape[1] - 1
            for r, c in component
        )
        if cfg.enclosed_only and touches_frame:
            rejected.append({
                "cells": len(component),
                "reason": "touches image frame",
            })
        elif fraction > float(cfg.max_hole_fraction):
            rejected.append({
                "cells": len(component),
                "reason": "component exceeds max_hole_fraction",
                "hole_fraction": float(fraction),
                "max_hole_fraction": float(cfg.max_hole_fraction),
            })
        else:
            eligible.append(component)

    # Select only the k smallest eligible islands.  A bounded heap keeps this
    # O(n log k) when max_components is much smaller than the number of holes,
    # rather than sorting every component O(n log n).  The top-left cell breaks
    # equal-size ties so processing remains deterministic across runs.
    fit_budget = max(0, int(cfg.max_components))
    ordered_components = heapq.nsmallest(
        fit_budget,
        eligible,
        key=lambda component: (len(component), min(component)),
    )
    selected_component_ids = {id(component) for component in ordered_components}
    budget_skipped = len(eligible) - len(ordered_components)
    for component in eligible:
        if id(component) not in selected_component_ids:
            rejected.append({
                "cells": len(component),
                "reason": "component budget exceeded",
            })

    attempted_components = 0
    for component in ordered_components:
        attempted_components += 1

        support = _support_indices(component, index_grid, int(cfg.ring_cells))
        fit, fit_diagnostic = _fit_plane(
            vertices[support], vertex_normals[support], cfg)
        if fit is None:
            rejected.append({
                "cells": len(component),
                **fit_diagnostic,
            })
            continue
        normal, plane_offset, fit_info = fit

        corners = {
            corner
            for r, c in component
            for corner in ((r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1))
        }
        vertex_checkpoint = len(new_vertices)
        uv_checkpoint = len(new_uvs)
        risk_checkpoint = len(edge_risk) if edge_risk is not None else 0
        corner_indices: dict[tuple[int, int], int] = {}
        failed_intersection = False
        for r, c in sorted(corners):
            existing = int(index_grid[r, c])
            if (existing >= 0
                    and _corner_is_boundary((r, c), component, candidates.shape)):
                corner_indices[(r, c)] = existing
                continue
            x = float(cols[c])
            y = float(rows[r])
            ray_camera = np.asarray(
                ((x - cx) / fx, -(y - cy) / fy, -1.0), dtype=np.float64)
            ray_world = rotation @ ray_camera
            denominator = float(normal @ ray_world)
            if abs(denominator) < 1e-9:
                failed_intersection = True
                break
            distance = (plane_offset - float(normal @ camera)) / denominator
            if not np.isfinite(distance) or distance <= 1e-6:
                failed_intersection = True
                break
            point = camera + distance * ray_world
            index = len(new_vertices)
            new_vertices.append(point.tolist())
            new_uvs.append([
                x / max(width - 1, 1),
                1.0 - y / max(height - 1, 1),
            ])
            if edge_risk is not None:
                edge_risk.append(0.0)
            corner_indices[(r, c)] = index
        if failed_intersection:
            del new_vertices[vertex_checkpoint:]
            del new_uvs[uv_checkpoint:]
            if edge_risk is not None:
                del edge_risk[risk_checkpoint:]
            rejected.append({
                "cells": len(component),
                "reason": "plane is behind or parallel to camera rays",
            })
            continue

        component_face_start = len(new_faces)
        for r, c in sorted(component):
            i00 = corner_indices[(r, c)]
            i10 = corner_indices[(r + 1, c)]
            i01 = corner_indices[(r, c + 1)]
            i11 = corner_indices[(r + 1, c + 1)]
            new_faces.append([i00, i10, i01])
            new_faces.append([i10, i11, i01])
            y0, y1 = int(rows[r]), int(rows[r + 1])
            x0, x1 = int(cols[c]), int(cols[c + 1])
            remaining_mask[y0:y1 + 1, x0:x1 + 1] = False
        for face_index, (r, c) in enumerate(face_cells):
            if (int(r), int(c)) in component:
                remove_face[face_index] = True
        if edge_risk is not None:
            patch_indices = {
                index for face in new_faces[component_face_start:] for index in face
            }
            for index in patch_indices:
                edge_risk[index] = 0.0
        filled.append({
            "cells": len(component),
            "faces_added": 2 * len(component),
            **fit_info,
        })

    kept_faces = faces[~remove_face]
    all_faces = (
        np.concatenate(
            [kept_faces, np.asarray(new_faces, dtype=np.int64).reshape(-1, 3)],
            axis=0,
        )
        if new_faces else kept_faces.copy()
    )
    stats = copy.deepcopy(getattr(mesh, "stats", None) or {})
    report = {
        "components_found": int(len(components)),
        "components_eligible": int(len(eligible)),
        "components_attempted": int(attempted_components),
        "components_budget_skipped": int(budget_skipped),
        "components_filled": int(len(filled)),
        "components_rejected": int(len(rejected)),
        "vertices_added": int(len(new_vertices) - len(vertices)),
        "faces_removed": int(remove_face.sum()),
        "faces_added": int(len(new_faces)),
        "filled": filled,
        "rejected": rejected,
    }
    stats["planar_hole_patch"] = report
    stats["n_vertices"] = int(len(new_vertices))
    stats["n_faces"] = int(len(all_faces))
    patched = ReliefMesh(
        vertices=np.asarray(new_vertices, dtype=np.float32),
        faces=np.asarray(all_faces, dtype=np.int32),
        uvs=np.asarray(new_uvs, dtype=np.float32),
        stats=stats,
        hole_mask=remaining_mask,
        filled_mask=getattr(mesh, "filled_mask", None),
        edge_risk=(np.asarray(edge_risk, dtype=np.float32)
                   if edge_risk is not None else None),
    )
    return patched, remaining_mask, report
