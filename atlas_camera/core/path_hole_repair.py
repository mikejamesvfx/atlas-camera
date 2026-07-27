"""Camera-path-guided selection of planar relief-hole candidates.

The missing cells themselves have no surface to ray-cast from a moved camera.
Instead, this module fits the same normal-guided candidate planes used by the
planar patcher, gives every connected source-space island a stable integer ID,
and rasterises those candidate faces from a selected camera-path frame.  A
painted moved-camera mask can therefore select IDs and recover the exact
original-image hole cells without an ambiguous screen-space unprojection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from atlas_camera.core.camera_path import sample_camera_path
from atlas_camera.core.planar_hole_patch import (
    PlanarHolePatchConfig,
    _components,
    _recover_lattice,
    patch_planar_holes,
)


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Path-guided hole repair requires numpy. "
            "Install with: pip install -e .[vision]"
        ) from exc
    return np


@dataclass(frozen=True, slots=True)
class PathHoleRepairConfig:
    """Controls for the candidate fit, view, and automatic island selection."""

    frame_offset_from_end: int = 0
    lens_scale_override: float = 0.0
    resolution: int = 768
    selection_mode: str = "all_visible"
    max_selected_islands: int = 0
    min_visible_pixels: int = 8
    paint_overlap_fraction: float = 0.02
    ring_cells: int = 2
    max_components: int = 1024
    normal_tolerance_deg: float = 30.0
    max_plane_error_m: float = 0.45
    max_hole_fraction: float = 0.04
    enclosed_only: bool = False
    min_normal_support_fraction: float = 0.20


def _fit_long_edge(width: int, height: int, long_edge: int) -> tuple[int, int]:
    scale = float(long_edge) / max(width, height, 1)
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def _project_vertices(
    vertices: Any,
    view_matrix: Any,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> tuple[Any, Any]:
    np = _require_numpy()
    points = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    vm = np.asarray(view_matrix, dtype=np.float64).reshape(4, 4)
    homogeneous = np.concatenate(
        (points, np.ones((len(points), 1), dtype=np.float64)), axis=1)
    camera_points = (vm @ homogeneous.T).T[:, :3]
    depth = -camera_points[:, 2]
    safe = np.maximum(depth, 1e-12)
    projected = np.stack(
        (
            fx * camera_points[:, 0] / safe + cx,
            -fy * camera_points[:, 1] / safe + cy,
        ),
        axis=1,
    )
    projected[depth <= 1e-6] = np.nan
    return projected, depth


def _rasterize_triangles(
    projected: Any,
    depth: Any,
    faces: Any,
    face_ids: Any,
    z_buffer: Any,
    id_map: Any | None = None,
) -> None:
    """Small dependency-free triangle rasteriser with a conventional z test."""
    np = _require_numpy()
    height, width = z_buffer.shape
    triangles = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    ids = np.asarray(face_ids, dtype=np.int32).reshape(-1)
    for face, island_id in zip(triangles, ids):
        xy = projected[face]
        z = depth[face]
        if (not np.isfinite(xy).all() or not np.isfinite(z).all()
                or (z <= 1e-6).any()):
            continue
        x0 = max(0, int(math.floor(float(xy[:, 0].min()))))
        x1 = min(width - 1, int(math.ceil(float(xy[:, 0].max()))))
        y0 = max(0, int(math.floor(float(xy[:, 1].min()))))
        y1 = min(height - 1, int(math.ceil(float(xy[:, 1].max()))))
        if x1 < x0 or y1 < y0:
            continue
        ax, ay = xy[0]
        bx, by = xy[1]
        cx_, cy_ = xy[2]
        denominator = (by - cy_) * (ax - cx_) + (cx_ - bx) * (ay - cy_)
        if abs(float(denominator)) < 1e-12:
            continue
        xs = np.arange(x0, x1 + 1, dtype=np.float64)
        ys = np.arange(y0, y1 + 1, dtype=np.float64)
        xx, yy = np.meshgrid(xs, ys)
        wa = ((by - cy_) * (xx - cx_) + (cx_ - bx) * (yy - cy_)) / denominator
        wb = ((cy_ - ay) * (xx - cx_) + (ax - cx_) * (yy - cy_)) / denominator
        wc = 1.0 - wa - wb
        inside = (wa >= -1e-7) & (wb >= -1e-7) & (wc >= -1e-7)
        if not inside.any():
            continue
        zz = wa * z[0] + wb * z[1] + wc * z[2]
        view = z_buffer[y0:y1 + 1, x0:x1 + 1]
        nearer = inside & (zz < view)
        if not nearer.any():
            continue
        view[nearer] = zz[nearer]
        if id_map is not None and int(island_id) > 0:
            id_view = id_map[y0:y1 + 1, x0:x1 + 1]
            id_view[nearer] = int(island_id)


def build_path_hole_repair(
    mesh: Any,
    hole_mask: Any,
    *,
    source_camera: Any,
    camera_path: Any,
    paint_mask: Any | None = None,
    config: PathHoleRepairConfig | None = None,
) -> dict[str, Any]:
    """Return moved-view IDs plus an exact source-space repair mask.

    ``frame_offset_from_end=0`` selects the last path frame.  ``lens_scale`` is
    a focal multiplier, matching the Camera Path playback slider: values below
    one widen the view.  Plane candidates deliberately bypass only the final
    edge-stretch acceptance gate; the returned mask still feeds a normal
    ``AtlasPlanarHolePatch`` whose own gate decides whether geometry is built.
    """
    np = _require_numpy()
    cfg = config or PathHoleRepairConfig()
    sampled = sample_camera_path(camera_path)
    src_intr = source_camera.intrinsics
    src_width = int(src_intr.image_width or 0)
    src_height = int(src_intr.image_height or 0)
    selected_mask = np.asarray(hole_mask, dtype=bool)
    if selected_mask.shape != (src_height, src_width):
        raise ValueError(
            f"hole mask shape {selected_mask.shape} does not match "
            f"source camera image {(src_height, src_width)}"
        )
    if not sampled:
        return {
            "repair_mask": np.zeros_like(selected_mask),
            "view_id_map": np.zeros((1, 1), dtype=np.int32),
            "frame_index": -1,
            "lens_scale": 1.0,
            "selected_ids": [],
            "visible_ids": [],
            "report": "camera path has no sampled frames — bake/author a path first",
        }

    frame_index = max(
        0, min(len(sampled) - 1,
               len(sampled) - 1 - max(0, int(cfg.frame_offset_from_end))))
    view_extrinsics = sampled[frame_index]
    path_lens_scale = max(0.05, float(getattr(camera_path, "lens_scale", 1.0)))
    lens_scale = (
        max(0.05, float(cfg.lens_scale_override))
        if float(cfg.lens_scale_override) > 0.0 else path_lens_scale
    )
    out_width, out_height = _fit_long_edge(
        src_width, src_height, max(64, int(cfg.resolution)))
    sx = out_width / max(src_width, 1)
    sy = out_height / max(src_height, 1)
    fx = float(src_intr.fx_px or 1.0) * sx * lens_scale
    fy = float(src_intr.fy_px or src_intr.fx_px or 1.0) * sy * lens_scale
    cx = float(src_intr.cx_px if src_intr.cx_px is not None
               else src_width / 2.0) * sx
    cy = float(src_intr.cy_px if src_intr.cy_px is not None
               else src_height / 2.0) * sy

    lattice = _recover_lattice(mesh, src_width, src_height)
    rows = lattice["rows"]
    cols = lattice["cols"]
    coverage = lattice["coverage"]
    row_centers = ((rows[:-1] + rows[1:]) // 2).astype(np.int64)
    col_centers = ((cols[:-1] + cols[1:]) // 2).astype(np.int64)
    selected_cells = selected_mask[np.ix_(row_centers, col_centers)]
    candidate_cells = selected_cells & (coverage < 2)
    components = _components(candidate_cells)
    components.sort(key=lambda item: (len(item), min(item)))
    cell_to_id = np.zeros(candidate_cells.shape, dtype=np.int32)
    component_by_id: dict[int, set[tuple[int, int]]] = {}
    for island_id, component in enumerate(components, start=1):
        component_by_id[island_id] = component
        for row, col in component:
            cell_to_id[row, col] = island_id

    fit_cfg = PlanarHolePatchConfig(
        ring_cells=int(cfg.ring_cells),
        max_components=int(cfg.max_components),
        normal_tolerance_deg=float(cfg.normal_tolerance_deg),
        max_plane_error_m=float(cfg.max_plane_error_m),
        max_hole_fraction=float(cfg.max_hole_fraction),
        enclosed_only=bool(cfg.enclosed_only),
        min_normal_support_fraction=float(cfg.min_normal_support_fraction),
        # Preview candidates that passed the plane fit even when the ordinary
        # repair rejected their camera-ray stretch. The downstream patch node
        # remains the geometry acceptance gate.
        max_patch_edge_factor=1.0e9,
    )
    candidate_mesh, _remaining, fit_report = patch_planar_holes(
        mesh,
        selected_mask,
        view_matrix=source_camera.extrinsics.camera_view_matrix,
        fx=float(src_intr.fx_px or 1.0),
        fy=float(src_intr.fy_px or src_intr.fx_px or 1.0),
        cx=float(src_intr.cx_px if src_intr.cx_px is not None
                 else src_width / 2.0),
        cy=float(src_intr.cy_px if src_intr.cy_px is not None
                 else src_height / 2.0),
        image_width=src_width,
        image_height=src_height,
        config=fit_cfg,
    )

    faces_added = int(fit_report.get("faces_added", 0))
    candidate_faces = (
        np.asarray(candidate_mesh.faces, dtype=np.int64).reshape(-1, 3)[-faces_added:]
        if faces_added else np.zeros((0, 3), dtype=np.int64)
    )
    face_ids = np.zeros(len(candidate_faces), dtype=np.int32)
    if len(candidate_faces):
        uvs = np.asarray(candidate_mesh.uvs, dtype=np.float64).reshape(-1, 2)
        centroids = uvs[candidate_faces].mean(axis=1)
        px = centroids[:, 0] * max(src_width - 1, 1)
        py = (1.0 - centroids[:, 1]) * max(src_height - 1, 1)
        rr = np.clip(np.searchsorted(rows, py, side="right") - 1,
                     0, len(rows) - 2)
        cc = np.clip(np.searchsorted(cols, px, side="right") - 1,
                     0, len(cols) - 2)
        face_ids = cell_to_id[rr, cc]

    z_buffer = np.full((out_height, out_width), np.inf, dtype=np.float64)
    id_map = np.zeros((out_height, out_width), dtype=np.int32)
    base_xy, base_z = _project_vertices(
        mesh.vertices, view_extrinsics.camera_view_matrix, fx, fy, cx, cy)
    _rasterize_triangles(
        base_xy, base_z, mesh.faces,
        np.zeros(len(mesh.faces), dtype=np.int32), z_buffer)
    if len(candidate_faces):
        candidate_xy, candidate_z = _project_vertices(
            candidate_mesh.vertices, view_extrinsics.camera_view_matrix,
            fx, fy, cx, cy)
        # The support mesh and candidate plane may meet at numerically equal
        # depths. A tiny relative bias lets the candidate own its hole without
        # pulling it through genuinely nearer geometry.
        z_buffer *= 1.0 + 1.0e-6
        _rasterize_triangles(
            candidate_xy, candidate_z, candidate_faces, face_ids,
            z_buffer, id_map)

    visible_counts = {
        int(island_id): int(count)
        for island_id, count in zip(*np.unique(id_map[id_map > 0],
                                               return_counts=True))
        if int(count) >= max(1, int(cfg.min_visible_pixels))
    }
    visible_ids = sorted(
        visible_counts,
        key=lambda island_id: (
            len(component_by_id.get(island_id, ())), island_id))

    mode = str(cfg.selection_mode or "all_visible")
    selected_ids: list[int]
    if mode == "paint_overlap":
        if paint_mask is None:
            selected_ids = []
        else:
            painted = np.asarray(paint_mask, dtype=bool)
            if painted.shape != id_map.shape:
                y_idx = np.minimum(
                    (np.arange(out_height) * painted.shape[0] / out_height).astype(int),
                    painted.shape[0] - 1)
                x_idx = np.minimum(
                    (np.arange(out_width) * painted.shape[1] / out_width).astype(int),
                    painted.shape[1] - 1)
                painted = painted[np.ix_(y_idx, x_idx)]
            selected_ids = []
            for island_id in visible_ids:
                island_pixels = id_map == island_id
                overlap = int((island_pixels & painted).sum())
                fraction = overlap / max(int(island_pixels.sum()), 1)
                if overlap > 0 and fraction >= float(cfg.paint_overlap_fraction):
                    selected_ids.append(island_id)
    elif mode == "largest_visible":
        selected_ids = list(reversed(visible_ids))
    elif mode == "smallest_visible":
        selected_ids = visible_ids
    else:
        selected_ids = visible_ids

    limit = max(0, int(cfg.max_selected_islands))
    if limit:
        selected_ids = selected_ids[:limit]
    repair_mask = np.zeros_like(selected_mask)
    for island_id in selected_ids:
        for row, col in component_by_id.get(island_id, ()):
            y0, y1 = int(rows[row]), int(rows[row + 1])
            x0, x1 = int(cols[col]), int(cols[col + 1])
            repair_mask[y0:y1 + 1, x0:x1 + 1] = True
    repair_mask &= selected_mask

    report_lines = [
        f"path frame {frame_index}/{len(sampled) - 1} "
        f"(offset {max(0, int(cfg.frame_offset_from_end))} from end)",
        f"view lens {lens_scale:.3f}x focal "
        f"({'path' if float(cfg.lens_scale_override) <= 0.0 else 'override'}; "
        "below 1.0 = wider)",
        f"{len(components)} source island(s), "
        f"{fit_report.get('components_filled', 0)} candidate plane(s), "
        f"{len(visible_ids)} visible, {len(selected_ids)} selected",
        f"selection={mode}; source mask contains {int(repair_mask.sum())} pixels",
    ]
    rejected = fit_report.get("rejected") or []
    if rejected:
        reasons: dict[str, int] = {}
        for item in rejected:
            reason = str(item.get("reason", "unknown"))
            reasons[reason] = reasons.get(reason, 0) + 1
        report_lines.append(
            "candidate-fit rejections: "
            + ", ".join(f"{count} {reason}"
                        for reason, count in sorted(reasons.items())))
    return {
        "repair_mask": repair_mask,
        "view_id_map": id_map,
        "frame_index": frame_index,
        "lens_scale": lens_scale,
        "path_lens_scale": path_lens_scale,
        "selected_ids": selected_ids,
        "visible_ids": visible_ids,
        "visible_counts": visible_counts,
        "report": "\n".join(report_lines),
    }
