"""Depth-safe occlusion seam strips for structured relief meshes.

The relief builder deliberately tears triangles across depth discontinuities.
On a decimated lattice that correct topological decision leaves a staircase
silhouette.  This module does not reconnect the two depths.  It extends each
boundary sheet independently into the selected hole, smooths the new outer
contour in image space, and zippers the old and new contours with a narrow
strip.  Near and far strips may overlap in projection but never share faces or
vertices, which is the layered-depth-image underlap used by this pass.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Occlusion seam refinement requires numpy. "
            "Install with: pip install -e .[vision]"
        ) from exc
    return np


@dataclass(frozen=True, slots=True)
class OcclusionSeamConfig:
    """Controls for :func:`refine_occlusion_seams`."""

    seam_width_cells: float = 2.0
    smooth_iterations: int = 8
    smooth_strength: float = 0.35
    max_chains: int = 256
    max_layer_depth_rel: float = 0.08
    min_chain_edges: int = 2
    global_direction: str = "away_from_camera"


def _pixel_positions(uvs: Any, width: int, height: int) -> Any:
    np = _require_numpy()
    uv = np.asarray(uvs, dtype=np.float64)
    return np.stack(
        [uv[:, 0] * max(width - 1, 1),
         (1.0 - uv[:, 1]) * max(height - 1, 1)],
        axis=1,
    )


def _lattice_step(pixel_positions: Any) -> float:
    np = _require_numpy()
    candidates: list[float] = []
    for axis in (0, 1):
        values = np.unique(np.rint(pixel_positions[:, axis]).astype(np.int64))
        diffs = np.diff(values)
        diffs = diffs[diffs > 0]
        if len(diffs):
            candidates.append(float(np.median(diffs)))
    return float(np.median(candidates)) if candidates else 1.0


def _directed_boundary_edges(faces: Any) -> list[tuple[int, int]]:
    """Boundary half-edges in the mesh's existing face winding."""
    np = _require_numpy()
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    directed = [
        (int(a), int(b))
        for tri in f
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0]))
    ]
    counts: dict[tuple[int, int], int] = {}
    for a, b in directed:
        key = (min(a, b), max(a, b))
        counts[key] = counts.get(key, 0) + 1
    return [
        (a, b) for a, b in directed
        if counts[(min(a, b), max(a, b))] == 1
    ]


def _sample_mask(mask: Any, point: Any) -> bool:
    row = int(round(float(point[1])))
    col = int(round(float(point[0])))
    if row < 0 or col < 0 or row >= mask.shape[0] or col >= mask.shape[1]:
        return False
    return bool(mask[row, col])


def _edge_touches_mask(mask: Any, p0: Any, p1: Any, step: float) -> bool:
    np = _require_numpy()
    midpoint = 0.5 * (p0 + p1)
    tangent = p1 - p0
    length = float(np.linalg.norm(tangent))
    if length <= 1e-9:
        return False
    normal = np.asarray((-tangent[1], tangent[0]), dtype=np.float64) / length
    radius = max(1.0, 0.8 * float(step))
    return any(_sample_mask(mask, midpoint + sign * radius * normal)
               for sign in (-1.0, 0.0, 1.0))


def _ordered_chains(edges: list[tuple[int, int]]) -> list[list[int]]:
    """Decompose selected directed half-edges into maximal ordered chains."""
    outgoing: dict[int, list[int]] = {}
    incoming: dict[int, int] = {}
    for a, b in edges:
        outgoing.setdefault(a, []).append(b)
        incoming[b] = incoming.get(b, 0) + 1
    unused = set(edges)
    chains: list[list[int]] = []

    def walk(start: tuple[int, int]) -> list[int]:
        a, b = start
        chain = [a, b]
        unused.remove(start)
        while True:
            candidates = [n for n in outgoing.get(chain[-1], [])
                          if (chain[-1], n) in unused]
            if len(candidates) != 1:
                break
            edge = (chain[-1], candidates[0])
            unused.remove(edge)
            chain.append(edge[1])
            if chain[-1] == chain[0]:
                break
        return chain

    starts = [edge for edge in edges if incoming.get(edge[0], 0) != 1]
    for edge in starts:
        if edge in unused:
            chains.append(walk(edge))
    while unused:
        chains.append(walk(next(iter(unused))))
    return chains


def _smooth_polyline(points: Any, iterations: int, strength: float,
                     closed: bool) -> Any:
    np = _require_numpy()
    values = np.asarray(points, dtype=np.float64).copy()
    if len(values) < 3 or iterations <= 0 or strength <= 0.0:
        return values
    lam = min(max(float(strength), 0.0), 0.95)
    mu = -min(0.53, 1.06 * lam)
    for _ in range(int(iterations)):
        for step in (lam, mu):
            prior = values.copy()
            if closed:
                midpoint = 0.5 * (
                    np.roll(prior, 1, axis=0) + np.roll(prior, -1, axis=0))
                values = prior + step * (midpoint - prior)
            else:
                midpoint = 0.5 * (prior[:-2] + prior[2:])
                values[1:-1] = prior[1:-1] + step * (
                    midpoint - prior[1:-1])
    return values


def _chain_hole_side(points: Any, mask: Any, step: float,
                     closed: bool) -> tuple[float, Any] | None:
    np = _require_numpy()
    n = len(points)
    segment_count = n if closed else n - 1
    if segment_count < 1:
        return None
    left_score = 0
    right_score = 0
    segment_normals: list[Any] = []
    probe = max(1.0, 0.8 * float(step))
    for index in range(segment_count):
        p0 = points[index]
        p1 = points[(index + 1) % n]
        tangent = p1 - p0
        length = float(np.linalg.norm(tangent))
        if length <= 1e-9:
            segment_normals.append(np.zeros(2, dtype=np.float64))
            continue
        normal = np.asarray((-tangent[1], tangent[0]), dtype=np.float64) / length
        segment_normals.append(normal)
        midpoint = 0.5 * (p0 + p1)
        left_score += int(_sample_mask(mask, midpoint + probe * normal))
        right_score += int(_sample_mask(mask, midpoint - probe * normal))
    if max(left_score, right_score) == 0:
        return None
    sign = 1.0 if left_score >= right_score else -1.0
    vertex_normals = np.zeros((n, 2), dtype=np.float64)
    if closed:
        for index in range(n):
            vertex_normals[index] = (
                segment_normals[index - 1] + segment_normals[index])
    else:
        vertex_normals[0] = segment_normals[0]
        vertex_normals[-1] = segment_normals[-1]
        for index in range(1, n - 1):
            vertex_normals[index] = (
                segment_normals[index - 1] + segment_normals[index])
    lengths = np.linalg.norm(vertex_normals, axis=1)
    valid = lengths > 1e-9
    vertex_normals[valid] /= lengths[valid, None]
    return sign, vertex_normals


def refine_occlusion_seams(
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
    config: OcclusionSeamConfig | None = None,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    """Append depth-safe underlap strips along selected interior boundaries.

    Returns ``(refined_mesh, remaining_mask, created_region, report)``.  Old
    vertices, faces and UVs are byte-for-byte preserved.  Each added strip is
    derived from one existing boundary chain only; the per-edge relative-depth
    gate makes a foreground/background curtain impossible.
    """
    np = _require_numpy()
    from atlas_camera.core.mesh_voxel import render_depth_grid
    from atlas_camera.core.relief_mesh import ReliefMesh

    started = time.perf_counter()
    cfg = config or OcclusionSeamConfig()
    direction = str(cfg.global_direction).strip().lower()
    if direction not in {"away_from_camera", "screen_normal_receding"}:
        raise ValueError(
            "global_direction must be 'away_from_camera' or "
            "'screen_normal_receding'"
        )
    width = int(image_width)
    height = int(image_height)
    selected = np.asarray(hole_mask, dtype=bool)
    if selected.shape != (height, width):
        raise ValueError(
            f"hole mask shape {selected.shape} does not match "
            f"camera image {(height, width)}"
        )
    if min(width, height) <= 1 or float(fx) <= 0.0 or float(fy) <= 0.0:
        raise ValueError("valid image dimensions and positive fx/fy are required")

    vertices = np.asarray(mesh.vertices, dtype=np.float64).reshape(-1, 3)
    faces = np.asarray(mesh.faces, dtype=np.int64).reshape(-1, 3)
    uvs = np.asarray(mesh.uvs, dtype=np.float64).reshape(-1, 2)
    if len(vertices) != len(uvs):
        raise ValueError("mesh vertex and UV counts differ")
    pixels = _pixel_positions(uvs, width, height)
    step = _lattice_step(pixels)

    vm = np.asarray(view_matrix, dtype=np.float64).reshape(4, 4)
    homogeneous = np.concatenate(
        [vertices, np.ones((len(vertices), 1), dtype=np.float64)], axis=1)
    forward = -(homogeneous @ vm.T)[:, 2]
    boundary = _directed_boundary_edges(faces)

    def on_frame(index: int) -> bool:
        x, y = pixels[index]
        tolerance = max(0.75, 0.25 * step)
        return (x <= tolerance or x >= width - 1 - tolerance
                or y <= tolerance or y >= height - 1 - tolerance)

    eligible_edges: list[tuple[int, int]] = []
    rejected_cross_depth = 0
    rejected_frame_edges = 0
    for a, b in boundary:
        if on_frame(a) or on_frame(b):
            rejected_frame_edges += 1
            continue
        if not _edge_touches_mask(selected, pixels[a], pixels[b], step):
            continue
        depth_min = max(min(float(forward[a]), float(forward[b])), 1e-6)
        depth_rel = abs(float(forward[a] - forward[b])) / depth_min
        if (not np.isfinite(depth_rel)
                or depth_rel > float(cfg.max_layer_depth_rel)):
            rejected_cross_depth += 1
            continue
        eligible_edges.append((a, b))

    found_chains = _ordered_chains(eligible_edges)
    found_chains.sort(key=lambda chain: (-len(chain), chain[0]))
    budget = max(0, int(cfg.max_chains))
    selected_chains = found_chains[:budget]
    budget_skipped = max(0, len(found_chains) - len(selected_chains))

    c2w = np.linalg.inv(vm)
    rotation = c2w[:3, :3]
    camera = c2w[:3, 3]
    camera_forward = rotation @ np.asarray(
        (0.0, 0.0, -1.0), dtype=np.float64)
    new_vertices = vertices.tolist()
    new_uvs = uvs.tolist()
    added_faces: list[list[int]] = []
    records: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    width_px = max(0.0, float(cfg.seam_width_cells)) * step

    for chain_number, raw_chain in enumerate(selected_chains, start=1):
        closed = len(raw_chain) > 2 and raw_chain[-1] == raw_chain[0]
        chain = raw_chain[:-1] if closed else raw_chain
        edge_count = len(chain) if closed else len(chain) - 1
        if edge_count < max(1, int(cfg.min_chain_edges)) or width_px <= 0.0:
            rejected.append({
                "chain": chain_number,
                "edges": edge_count,
                "reason": "chain_too_short_or_width_zero",
            })
            continue
        points = pixels[np.asarray(chain, dtype=np.int64)]
        side = _chain_hole_side(points, selected, step, closed)
        if side is None:
            rejected.append({
                "chain": chain_number,
                "edges": edge_count,
                "reason": "no_selected_hole_side",
            })
            continue
        sign, normals = side
        unsmoothed = points + sign * width_px * normals
        target = _smooth_polyline(
            unsmoothed,
            int(cfg.smooth_iterations),
            float(cfg.smooth_strength),
            closed,
        )
        # Smoothing is allowed to round the staircase but not to drift farther
        # than one requested seam width from its directly-offset contour.
        displacement = target - unsmoothed
        distance = np.linalg.norm(displacement, axis=1)
        over = distance > width_px
        if over.any():
            displacement[over] *= (width_px / distance[over])[:, None]
            target = unsmoothed + displacement
        if ((target[:, 0] <= 0.5).any()
                or (target[:, 0] >= width - 1.5).any()
                or (target[:, 1] <= 0.5).any()
                or (target[:, 1] >= height - 1.5).any()):
            rejected.append({
                "chain": chain_number,
                "edges": edge_count,
                "reason": "extension_touches_frame",
            })
            continue
        # Validate the swept strip, not only its outer contour.  Relief holes
        # are commonly one raster pixel wide while a useful underlap is two or
        # more lattice cells wide, so requiring the target itself to remain in
        # the mask would reject every valid strip.  The source edges have
        # already been selected individually; this additional check ensures
        # the offset still sweeps across the requested region after smoothing.
        swept_hits = 0
        swept_samples = 0
        for source_point, target_point in zip(points, target):
            for fraction in (0.25, 0.5, 0.75, 1.0):
                swept_samples += 1
                swept_hits += int(_sample_mask(
                    selected,
                    source_point + fraction * (target_point - source_point),
                ))
        if swept_hits == 0:
            rejected.append({
                "chain": chain_number,
                "edges": edge_count,
                "reason": "extension_misses_selected_mask",
            })
            continue

        outer_indices: list[int] = []
        valid_depth = True
        for vertex_index, point in zip(chain, target):
            source_depth = float(forward[vertex_index])
            if not np.isfinite(source_depth) or source_depth <= 1e-6:
                valid_depth = False
                break
            extension_px = float(np.linalg.norm(point - pixels[vertex_index]))
            extension_world = (
                source_depth * extension_px
                / max(min(float(fx), float(fy)), 1e-6)
            )
            if direction == "away_from_camera":
                # One global direction for every chain: the camera optical
                # forward vector.  There is no camera-space X/Y displacement,
                # so horizontal boundaries cannot grow world-Y shelves.
                world = vertices[vertex_index] + (
                    extension_world * camera_forward)
                camera_point = vm @ np.append(world, 1.0)
                depth = -float(camera_point[2])
                x = float(fx) * float(camera_point[0]) / depth + float(cx)
                y = float(cy) - float(fy) * float(camera_point[1]) / depth
            else:
                # Presentation-oriented alternative: expand in the chosen
                # image-space normal while receding one local cell per cell.
                x, y = float(point[0]), float(point[1])
                depth = source_depth + extension_world
                ray_camera = np.asarray(
                    ((x - cx) / fx, -(y - cy) / fy, -1.0),
                    dtype=np.float64,
                )
                world = camera + depth * (rotation @ ray_camera)
            outer_indices.append(len(new_vertices))
            new_vertices.append(world.tolist())
            new_uvs.append([
                x / max(width - 1, 1),
                1.0 - y / max(height - 1, 1),
            ])
        if not valid_depth:
            del new_vertices[-len(outer_indices):]
            del new_uvs[-len(outer_indices):]
            rejected.append({
                "chain": chain_number,
                "edges": edge_count,
                "reason": "invalid_boundary_depth",
            })
            continue

        face_start = len(added_faces)
        for index in range(edge_count):
            next_index = (index + 1) % len(chain)
            a, b = int(chain[index]), int(chain[next_index])
            oa, ob = outer_indices[index], outer_indices[next_index]
            # Existing boundary traverses a→b. The first new face traverses
            # b→a, preserving the manifold half-edge winding.
            added_faces.append([b, a, oa])
            added_faces.append([b, oa, ob])
        records.append({
            "chain": chain_number,
            "edges": edge_count,
            "faces_added": len(added_faces) - face_start,
            "vertices_added": len(outer_indices),
            "median_depth_m": float(np.median(forward[chain])),
            "selected_sweep_fraction": float(swept_hits / swept_samples),
            "contour_length_before_px": float(np.linalg.norm(
                points - np.roll(points, -1, axis=0), axis=1
            ).sum() if closed else np.linalg.norm(np.diff(points, axis=0), axis=1).sum()),
            "contour_length_after_px": float(np.linalg.norm(
                target - np.roll(target, -1, axis=0), axis=1
            ).sum() if closed else np.linalg.norm(np.diff(target, axis=0), axis=1).sum()),
        })

    if added_faces:
        added_array = np.asarray(added_faces, dtype=np.int64).reshape(-1, 3)
        all_faces = np.concatenate([faces, added_array], axis=0)
    else:
        added_array = np.empty((0, 3), dtype=np.int64)
        all_faces = faces.copy()
    all_vertices = np.asarray(new_vertices, dtype=np.float32)
    all_uvs = np.asarray(new_uvs, dtype=np.float32)

    if len(added_array):
        added_depth = render_depth_grid(
            all_vertices,
            added_array,
            vm,
            float(fx),
            float(fy),
            float(cx),
            float(cy),
            width,
            height,
        )
        added_coverage = np.isfinite(added_depth)
    else:
        added_coverage = np.zeros((height, width), dtype=bool)
    created_region = selected & added_coverage
    remaining = selected & ~added_coverage

    edge_risk_raw = getattr(mesh, "edge_risk", None)
    if edge_risk_raw is not None and len(np.asarray(edge_risk_raw)) == len(vertices):
        edge_risk = np.concatenate([
            np.asarray(edge_risk_raw, dtype=np.float32),
            np.zeros(len(all_vertices) - len(vertices), dtype=np.float32),
        ])
    else:
        edge_risk = None

    report = {
        "boundary_edges_found": len(boundary),
        "eligible_edges": len(eligible_edges),
        "chains_found": len(found_chains),
        "chains_attempted": len(selected_chains),
        "chains_refined": len(records),
        "chains_rejected": len(rejected),
        "chains_budget_skipped": budget_skipped,
        "cross_depth_edges_rejected": rejected_cross_depth,
        "frame_edges_rejected": rejected_frame_edges,
        "vertices_added": len(all_vertices) - len(vertices),
        "faces_added": len(added_faces),
        "camera_mask_pixels_covered": int(created_region.sum()),
        "remaining_mask_pixels": int(remaining.sum()),
        "seam_width_cells": float(cfg.seam_width_cells),
        "global_direction": direction,
        "lattice_step_px": float(step),
        "elapsed_ms": float((time.perf_counter() - started) * 1000.0),
        "refined": records,
        "rejected": rejected,
    }
    stats = copy.deepcopy(getattr(mesh, "stats", None) or {})
    stats["occlusion_seam_refine"] = copy.deepcopy(report)
    stats["n_vertices"] = len(all_vertices)
    stats["n_faces"] = len(all_faces)
    refined = ReliefMesh(
        vertices=all_vertices,
        faces=np.asarray(all_faces, dtype=np.int32),
        uvs=all_uvs,
        stats=stats,
        hole_mask=remaining,
        filled_mask=getattr(mesh, "filled_mask", None),
        edge_risk=edge_risk,
    )
    return refined, remaining, created_region, report
