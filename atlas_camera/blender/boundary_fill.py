"""Scoped, topology-preserving hole fill through headless Blender.

This is deliberately *not* a remesher.  A repair may append faces across a
known interior boundary loop, but it may not move vertices, replace original
faces, or cap the source-image perimeter.  That makes it a safe counterpart to
the experimental organic voxel path: a missing boundary is reported rather
than concealed with an invented, scene-scale shell.

The module owns the Blender adapter only.  Core mesh topology remains Blender
agnostic and coordinate conversion happens in :mod:`atlas_camera.blender.exchange`.
"""
from __future__ import annotations

from collections import Counter
import tempfile
from pathlib import Path
from typing import Any

from atlas_camera.blender.exchange import read_result, write_exchange
from atlas_camera.blender.runner import run_recipe
from atlas_camera.core.mesh_repair import _perimeter_loops, boundary_edges, walk_loops


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "atlas_camera.blender requires numpy. Install with: "
            "pip install -e .[vision]") from exc
    return np


def _points_inside_polygon(points: Any, polygon: Any) -> Any:
    """Even-odd point-in-polygon test, vectorised over ``points``.

    UV loops are small while a selected mask component can contain many pixels,
    so keeping the loop scalar and the pixels vectorised is both simple and
    avoids adding an image/geometry dependency to Atlas core.
    """
    np = _require_numpy()
    p = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    q = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    inside = np.zeros(len(p), dtype=bool)
    if len(q) < 3 or not len(p):
        return inside
    x, y = p[:, 0], p[:, 1]
    j = len(q) - 1
    for i in range(len(q)):
        xi, yi = q[i]
        xj, yj = q[j]
        crosses = ((yi > y) != (yj > y))
        # The tiny denominator only handles horizontal segments.  For all
        # other edges it is several orders below source-image UV precision.
        at_x = (xj - xi) * (y - yi) / ((yj - yi) + 1e-20) + xi
        inside ^= crosses & (x < at_x)
        j = i
    return inside


def select_masked_interior_loops(
    faces: Any,
    uvs: Any,
    hole_mask: Any,
    *,
    image_width: int,
    image_height: int,
    max_hole_edges: int = 256,
) -> tuple[list[list[int]], dict[str, int]]:
    """Return only real interior loops overlapped by ``hole_mask``.

    ``hole_mask`` is the source-space ``remaining_holes`` output of
    :class:`AtlasPlanarHolePatch`; a non-zero pixel alone is never evidence
    that Blender may create geometry.  It must overlap an actual boundary loop
    in the relief mesh.  Conversely, a boundary loop at the UV frame is the
    plate perimeter, not a repair candidate.

    A mask with no matching loop is intentional information: it means the
    discontinuity has no boundary edge for a topology fill to bridge (for
    example an overlap, bad depth ordering, or a filled-but-visually-dark
    region).  The caller surfaces that state in its report instead of applying
    a global remesh or a synthetic "snowglobe" cap.
    """
    np = _require_numpy()
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    uv = np.asarray(uvs, dtype=np.float64).reshape(-1, 2)
    mask = np.asarray(hole_mask, dtype=np.float64)
    if mask.ndim != 2:
        raise ValueError(f"hole_mask must be 2D after normalization, got {mask.shape}")
    if image_width <= 1 or image_height <= 1:
        raise ValueError("image_width and image_height must both be greater than 1")
    if len(f) == 0 or len(uv) == 0:
        return [], {"boundary_loops": 0, "perimeter_loops": 0,
                    "candidate_loops": 0, "too_large_loops": 0,
                    "masked_pixels": int((mask > 0.5).sum()),
                    "matched_loops": 0}

    loops = walk_loops(boundary_edges(f), faces=f)
    perimeter = _perimeter_loops(loops, uv, image_width, image_height)
    yy, xx = np.nonzero(mask > 0.5)
    report = {
        "boundary_loops": len(loops),
        "perimeter_loops": len(perimeter),
        "candidate_loops": 0,
        "too_large_loops": 0,
        "masked_pixels": int(len(xx)),
        "matched_loops": 0,
    }
    if not len(xx):
        return [], report

    # Comfy mask pixels are top-left origin; relief UVs are bottom-left origin.
    mask_points = np.column_stack((xx / float(image_width - 1),
                                   1.0 - yy / float(image_height - 1)))
    chosen: list[list[int]] = []
    edge_limit = max(3, int(max_hole_edges))
    for k, loop in enumerate(loops):
        if k in perimeter:
            continue
        if len(loop) > edge_limit:
            report["too_large_loops"] += 1
            continue
        idx = np.asarray(loop, dtype=np.int64)
        if (idx < 0).any() or (idx >= len(uv)).any():
            continue
        report["candidate_loops"] += 1
        polygon = uv[idx]
        lo, hi = polygon.min(axis=0), polygon.max(axis=0)
        in_box = ((mask_points[:, 0] >= lo[0]) & (mask_points[:, 0] <= hi[0])
                  & (mask_points[:, 1] >= lo[1]) & (mask_points[:, 1] <= hi[1]))
        if in_box.any() and _points_inside_polygon(mask_points[in_box], polygon).any():
            chosen.append([int(v) for v in loop])

    report["matched_loops"] = len(chosen)
    return chosen, report


def _normalise_loops(loops: list[list[int]], vertex_count: int) -> list[list[int]]:
    out: list[list[int]] = []
    for loop in loops:
        clean = [int(v) for v in loop]
        if len(clean) < 3 or len(set(clean)) != len(clean):
            raise ValueError("selected boundary loops must be simple polygons with 3+ vertices")
        if min(clean) < 0 or max(clean) >= vertex_count:
            raise ValueError("selected boundary loop contains an out-of-range vertex index")
        out.append(clean)
    return out


def _face_counter(faces: Any) -> Counter[tuple[int, int, int]]:
    np = _require_numpy()
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    return Counter(tuple(sorted(map(int, tri))) for tri in f)


def _validate_result(original_vertices: Any, original_faces: Any,
                     returned_vertices: Any, returned_faces: Any) -> None:
    """Enforce the non-destructive contract after a Blender round trip."""
    np = _require_numpy()
    before_v = np.asarray(original_vertices, dtype=np.float64).reshape(-1, 3)
    after_v = np.asarray(returned_vertices, dtype=np.float64).reshape(-1, 3)
    if len(after_v) != len(before_v):
        raise RuntimeError(
            "Blender boundary fill changed the vertex count "
            f"({len(before_v)} -> {len(after_v)}); refusing a topology rewrite")
    # Blender mesh coordinates are float32.  This admits that storage round-off
    # but catches any geometric operation, even on kilometre-scale scenes.
    tolerance = max(1.0, float(np.abs(before_v).max(initial=0.0))) * 1e-6
    moved = np.linalg.norm(after_v - before_v, axis=1) > tolerance
    if moved.any():
        raise RuntimeError(
            "Blender boundary fill altered "
            f"{int(moved.sum())} existing vertex/vertices; refusing a topology rewrite")
    missing = _face_counter(original_faces) - _face_counter(returned_faces)
    if missing:
        raise RuntimeError(
            "Blender boundary fill removed or rewrote existing face(s); "
            "refusing a topology rewrite")


def fill_selected_boundary_loops(
    vertices: Any,
    faces: Any,
    selected_loops: list[list[int]],
    *,
    backend: str = "native",
    blender_path: str = "",
    timeout_s: int = 600,
    exchange_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Fill already-scoped loops in Blender and prove it preserved the mesh.

    ``backend='native'`` is Atlas's clean-room BMesh implementation.
    ``backend='fill_mesh_addon'`` invokes the user's installed Fill Mesh add-on
    operator on exactly the same selected loops for an A/B comparison; no add-on
    source is bundled, copied, or required for the native path.
    """
    np = _require_numpy()
    if backend not in {"native", "fill_mesh_addon"}:
        raise ValueError("backend must be 'native' or 'fill_mesh_addon'")
    v = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    loops = _normalise_loops(selected_loops, len(v))
    if not loops:
        return {"vertices": v.copy(), "faces": f.copy(),
                "report": {"backend": backend, "faces_created": 0,
                           "selected_loops": 0, "skipped": "no matched boundary loops"}}

    ex = Path(exchange_dir) if exchange_dir else Path(
        tempfile.mkdtemp(prefix="atlas_blender_boundary_fill_"))
    write_exchange(
        ex, patch_vertices=v, patch_faces=f, target_vertices=v, target_faces=f,
        camera_position=[0.0, 0.0, 0.0],
        params={"backend": backend, "selected_loops": loops},
    )
    report = run_recipe("boundary_fill.py", ex, blender_path=blender_path,
                        timeout_s=int(timeout_s))
    got = read_result(ex)
    _validate_result(v, f, got["vertices"], got["faces"])
    # Preserve exact Atlas coordinates rather than serialising float32 Blender
    # coordinates back into the solve.  Validation above proves that this did
    # not mask a geometric operation.
    out_faces = np.asarray(got["faces"], dtype=np.int64).reshape(-1, 3)
    report = dict(report)
    report.setdefault("backend", backend)
    report.setdefault("selected_loops", len(loops))
    report.setdefault("faces_created", int(len(out_faces) - len(f)))
    report["existing_vertices_preserved"] = int(len(v))
    report["existing_faces_preserved"] = int(len(f))
    return {"vertices": v.copy(), "faces": out_faces, "report": report,
            "exchange_dir": str(ex)}
