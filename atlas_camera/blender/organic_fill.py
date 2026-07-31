"""Atlas-side driver: close in numpy, place in Blender, gate, stitch.

The division of labour, both halves measured:

    numpy voxel_remesh   closes every interior tear, ~3.4s          (Phase 0)
    Blender shrinkwrap   places it EXACTLY on measured surface,     (Phase 2)
                         p95 0.0000m against a calibrated 0.0 floor

THE GATE MEASURES MOVEMENT, NOT FINAL POSITION. The plan called for a rim-drift
gate on distance-to-measured-surface. After calibration that number is zero BY
CONSTRUCTION — NEAREST_SURFACEPOINT projects onto the surface, so the gate would
pass everything and catch nothing. What it cannot see is a vertex slid ALONG the
surface to the wrong place.

So the gate is how far shrinkwrap had to DRAG the fill. A closure that landed
near the truth needs a small correction; one that needs hauling several edge
lengths was badly placed, and "it ended up on the surface" does not redeem it.
That distance is real evidence and the recipe already reports it per vertex.

Rejection is a THIRD outcome, not an error: a raise kills a long graph, a silent
accept ships bad geometry. The layer is left untouched and the reason is stated
with its numbers, which is what the repo's gate doctrine requires.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from atlas_camera.blender.exchange import read_result, write_exchange
from atlas_camera.blender.runner import run_recipe


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "atlas_camera.blender requires numpy. Install with: "
            "pip install -e .[vision]") from exc
    return np


def median_edge_length(vertices: Any, faces: Any) -> float:
    """Scene-scale ruler. Every tolerance here is expressed in these units so a
    setting tuned on a 10 m interior still means something on a 2 km vista."""
    np = _require_numpy()
    v = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if not len(f):
        return 0.0
    return float(np.median(np.linalg.norm(v[f[:, 0]] - v[f[:, 1]], axis=1)))


def shrinkwrap_patch(patch_vertices: Any, patch_faces: Any,
                     target_vertices: Any, target_faces: Any, *,
                     blender_path: str = "", limit_scale: float = 4.0,
                     wrap_method: str = "NEAREST_SURFACEPOINT",
                     timeout_s: int = 600,
                     exchange_dir: str | Path | None = None) -> dict[str, Any]:
    """Run the shrinkwrap recipe; return result arrays plus the recipe report."""
    ex = Path(exchange_dir) if exchange_dir else Path(
        tempfile.mkdtemp(prefix="atlas_blender_"))
    med = median_edge_length(target_vertices, target_faces)
    write_exchange(ex, patch_vertices=patch_vertices, patch_faces=patch_faces,
                   target_vertices=target_vertices, target_faces=target_faces,
                   camera_position=[0.0, 0.0, 0.0],
                   params={"shrinkwrap_limit_m": float(limit_scale) * med,
                           "wrap_method": wrap_method})
    report = run_recipe("shrinkwrap.py", ex, blender_path=blender_path,
                        timeout_s=timeout_s)
    out = read_result(ex)
    out["report"] = report
    out["exchange_dir"] = str(ex)
    out["median_edge_m"] = med
    return out


def gate_movement(moved_median_m: float, moved_max_m: float,
                  median_edge_m: float, *, max_move_scale: float = 3.0
                  ) -> tuple[bool, str]:
    """Accept/reject on how far shrinkwrap had to drag the fill.

    Returns ``(accepted, reason)`` — the reason is always populated, including
    on acceptance, because a silent pass tells a reader nothing about how close
    the call was.
    """
    if median_edge_m <= 0:
        return False, ("cannot gate: the target mesh has no measurable edge "
                       "length, so there is no scale to judge movement against")
    ratio = moved_median_m / median_edge_m
    limit = float(max_move_scale)
    if ratio > limit:
        return False, (
            f"REJECTED — shrinkwrap had to move the fill a median "
            f"{moved_median_m:.4f}m ({ratio:.1f}x the {median_edge_m:.4f}m "
            f"median edge, limit {limit:.1f}x). It now sits ON the measured "
            f"surface, but a correction that large means the closure was badly "
            f"placed and may have slid somewhere wrong. Layer left unchanged.")
    return True, (
        f"accepted — shrinkwrap moved the fill a median {moved_median_m:.4f}m "
        f"({ratio:.2f}x median edge, limit {limit:.1f}x); max {moved_max_m:.4f}m")


def weld_to_anchor(new_vertices: Any, anchor_vertices: Any, *,
                   tolerance_m: float) -> dict[str, Any]:
    """Map returned rim vertices onto preserved anchor vertices by proximity.

    Reports the UNWELDED count rather than swallowing it: an unwelded rim vertex
    is a residual seam, and `boundary_edges` will read it as a fresh tear on the
    next pass. Silence there would look like the repair had worked.
    """
    np = _require_numpy()
    new_v = np.asarray(new_vertices, dtype=np.float64).reshape(-1, 3)
    anchor = np.asarray(anchor_vertices, dtype=np.float64).reshape(-1, 3)
    if not len(anchor) or not len(new_v):
        return {"pairs": np.zeros((0, 2), dtype=np.int64),
                "unwelded": int(len(anchor)), "welded": 0}
    pairs = []
    # Brute force: the rim is a few hundred vertices, so this stays cheap and
    # avoids adding a spatial index (Atlas has no scipy).
    for i, a in enumerate(anchor):
        d = np.linalg.norm(new_v - a, axis=1)
        j = int(np.argmin(d))
        if d[j] <= tolerance_m:
            pairs.append((j, i))
    arr = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    return {"pairs": arr, "welded": len(arr),
            "unwelded": int(len(anchor) - len(arr))}
