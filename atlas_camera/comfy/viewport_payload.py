"""The viewport wire protocol — the payload `atlas_blockout.js` consumes.

Split out of `node_helpers.py` in phase 1 of
`docs/dev/node_helpers_layering_plan.md`. `_extract_blockout_camera` alone was
231 lines — 57% of everything genuinely ComfyUI-coupled in that module — and it
grows every time the viewport gains a feature (the occlusion depth-packing block
was added 2026-07-20). It is a protocol, not a helper, and it deserves a file
that says so.

Everything here is the SERIALISATION boundary between a solved scene and the
browser: what the frontend receives, in the shape it expects. Two conventions
are load-bearing and easy to break from a distance:

- ``fx``/``fy``/``cx``/``cy``/``image_width``/``image_height`` describe how the
  PHOTO was shot and are read by `makeProjectionMaterial` to sample the plate.
  ``render_fy``/``render_image_height`` are separate keys read ONLY by
  `applyRecoveredCamera` for the viewing camera, so a ShotCam can change the
  render format without corrupting how the photo projects onto geometry.
- ``primary_depth_b64`` is bit-packed R/G/B = high/mid/low bytes of one 24-bit
  millimetre integer. `atlas_blockout.js` unpacks it as
  ``z_mm = R*65536 + G*256 + B``; the two are a contract, and the texture must
  be sampled NEAREST because interpolating the bytes yields garbage distances.
  The transport map is capped at a 2048px long edge (still above the default
  viewport render) and carries its own width/height; shipping a full 7K RAW
  map as high-entropy RGB otherwise produces a ~34MB data URI for one toggle.

This module stays in `comfy/` deliberately — it is adapter code by definition,
not host-agnostic math.
"""

from __future__ import annotations

import base64
import io
import logging
import math
from typing import Any

from atlas_camera.comfy.node_helpers import (
    _depth_map_for_solve,
    _horizon_y_from_solve,
    _image_tensor_to_preview_b64,
    _require_numpy,
    _require_pil,
)

_PRIMARY_DEPTH_MAX_EDGE = 2048


def _decimate_metric_depth_for_viewport(metric_depth, max_edge: int = _PRIMARY_DEPTH_MAX_EDGE):
    """Nearest-sample a metric depth map for browser transport.

    Packed RGB depth cannot use ordinary image interpolation: interpolating
    the three bytes independently corrupts the integer, while averaging metric
    depth across a discontinuity invents a false surface.  A deterministic
    nearest grid preserves exact millimetre samples and sharp jumps.
    """
    np = _require_numpy()
    arr = np.asarray(metric_depth)
    if arr.ndim != 2:
        raise ValueError("metric depth must be a 2D array")
    h, w = arr.shape
    if max(h, w) <= int(max_edge):
        return arr
    scale = float(max_edge) / float(max(h, w))
    out_h = max(1, int(round(h * scale)))
    out_w = max(1, int(round(w * scale)))
    ys = np.rint(np.linspace(0, h - 1, out_h)).astype(np.int64)
    xs = np.rint(np.linspace(0, w - 1, out_w)).astype(np.int64)
    return arr[ys[:, None], xs[None, :]]

def _fit_long_edge(width: int, height: int, long_edge: int, multiple: int = 8) -> tuple[int, int]:
    """Scale (width, height) so its longest side is ``long_edge``, rounded to ``multiple``."""
    width = max(1, int(width))
    height = max(1, int(height))
    scale = long_edge / float(max(width, height))
    def _round(v: float) -> int:
        return max(multiple, int(round(v / multiple)) * multiple)
    return _round(width * scale), _round(height * scale)
def _plate_ref_to_dict(plate_ref) -> dict[str, Any] | None:
    if plate_ref is None:
        return None
    if hasattr(plate_ref, "to_dict"):
        return plate_ref.to_dict()
    if isinstance(plate_ref, dict):
        return dict(plate_ref)
    return None
def _output_profile_to_dict(output_profile) -> dict[str, Any] | None:
    if output_profile is None:
        return None
    if hasattr(output_profile, "to_dict"):
        return output_profile.to_dict()
    if isinstance(output_profile, dict):
        return dict(output_profile)
    return None


def _serialize_projection_sources(solve) -> list[dict[str, Any]]:
    """Serialize multi-angle patch sources attached to a solve for viewport projection materials."""
    from atlas_camera.core.proxy_geometry import serialize_proxy_geometry
    from atlas_camera.core.schema import AtlasProjectionScene

    sources = getattr(solve, "projection_sources", None) or []
    projection_sources = []
    for src in sources:
        s_intr = src.camera.intrinsics
        s_extr = src.camera.extrinsics
        s_fx = s_intr.fx_px or 0.0
        s_fy = s_intr.fy_px or s_fx
        s_cx = s_intr.cx_px if s_intr.cx_px is not None else (s_intr.image_width or 1) / 2.0
        s_cy = s_intr.cy_px if s_intr.cy_px is not None else (s_intr.image_height or 1) / 2.0
        projection_sources.append({
            "name": src.name,
            "view_matrix": [list(row) for row in s_extr.camera_view_matrix],
            "camera_position": list(s_extr.camera_position),
            "fx": s_fx, "fy": s_fy, "cx": s_cx, "cy": s_cy,
            "image_width": s_intr.image_width,
            "image_height": s_intr.image_height,
            "image_b64": src.image_b64 or "",
            "mask_b64": getattr(src, "mask_b64", None) or "",
            "normal_map_b64": getattr(src, "normal_map_b64", None) or "",
            "plate_ref": _plate_ref_to_dict(getattr(src, "plate_ref", None)),
            "priority": float(src.priority),
            "azimuth_deg": float(src.azimuth_deg),
            "elevation_deg": float(src.elevation_deg),
            "projection_mode": (src.metadata or {}).get("projection_mode"),
            "near_m": (src.metadata or {}).get("near_m"),
            "far_m": (src.metadata or {}).get("far_m"),
            "band_geometry": (src.metadata or {}).get("band_geometry"),
            "hidden_mask_b64": (src.metadata or {}).get("hidden_mask_b64") or "",
            "hidden_backend": (src.metadata or {}).get("hidden_backend") or "",
            "proxy_geometry": serialize_proxy_geometry(
                AtlasProjectionScene(proxy_geometry=list(src.proxy_geometry)),
            ),
        })
    return projection_sources


def _pack_primary_depth(solve, primary_depth, intr, extr, fx: float, fy: float, cx: float, cy: float) -> tuple[str, int, int]:
    """Pack 24-bit metric depth PNG in millimetres for viewport occlusion culling."""
    if primary_depth is None:
        return "", 0, 0

    from atlas_camera.core.relief_mesh import estimate_ground_scale

    np = _require_numpy()
    PILImage = _require_pil()
    try:
        p_map = _depth_map_for_solve(primary_depth, intr.image_width, intr.image_height)
        p_scale, _ = estimate_ground_scale(
            p_map, view_matrix=extr.camera_view_matrix,
            fx=fx, fy=fy, cx=cx, cy=cy,
            horizon_y=_horizon_y_from_solve(solve))
        primary_metric_map = np.asarray(p_map, dtype=np.float64) * float(p_scale)
        primary_metric_map = _decimate_metric_depth_for_viewport(primary_metric_map)
        h, w = primary_metric_map.shape

        depth_mm = np.clip(primary_metric_map * 1000.0, 0, 0xFFFFFF).astype(np.uint32)
        rgb_depth = np.zeros(depth_mm.shape + (3,), dtype=np.uint8)
        rgb_depth[..., 0] = (depth_mm >> 16) & 0xFF
        rgb_depth[..., 1] = (depth_mm >> 8) & 0xFF
        rgb_depth[..., 2] = depth_mm & 0xFF

        buf = io.BytesIO()
        PILImage.fromarray(rgb_depth, mode="RGB").save(buf, format="PNG", optimize=True)
        depth_b64 = ("data:image/png;base64,"
                     + base64.b64encode(buf.getvalue()).decode("ascii"))
        return depth_b64, w, h
    except (ValueError, TypeError, AttributeError, IndexError, ArithmeticError) as exc:
        import logging
        logging.warning("primary_depth could not be packed for the occlusion "
                        "cull; it will stay disabled: %s", exc)
        return "", 0, 0


def _extract_blockout_camera(
solve, source_image, target_width: int, target_height: int,
                              preview_expand: float = 1.0, shot_intrinsics=None,
                              output_profile=None, solve_fingerprint: str = "", primary_depth=None) -> dict[str, Any]:
    """Serialize the recovered camera into a dict the browser extension can consume.

    `shot_intrinsics` (optional, from AtlasShotCam via intrinsics_from_shot_cam)
    conforms the RENDER/VIEWING camera to a project-level shot format. It must
    stay entirely separate from `fx`/`fy`/`cx`/`cy` below: those are also read
    by the frontend's makeProjectionMaterial() for the PRIMARY source's own
    texture-sampling (applyCamera(data) and setProxies(data) — which builds
    the primary's projection material — both consume this SAME dict), so
    overwriting them would corrupt how the actual photo gets projected onto
    geometry. Only `render_fy`/`render_image_height` (read solely by
    applyRecoveredCamera for the live orbit camera's FOV) carry the
    shot-conformed values; `target_width`/`target_height` are always already
    independent of `image_width`/`image_height` (routinely resized via
    resolution/_fit_long_edge regardless of shot_cam), so they're set
    directly from the shot format by the caller with no separate key needed.
    """
    cam = solve.camera
    intr = cam.intrinsics
    extr = cam.extrinsics
    fx = intr.fx_px or 0.0
    fy = intr.fy_px or fx
    cx = intr.cx_px if intr.cx_px is not None else intr.image_width / 2.0
    cy = intr.cy_px if intr.cy_px is not None else intr.image_height / 2.0
    if shot_intrinsics is not None:
        render_fy = shot_intrinsics.fy_px or shot_intrinsics.fx_px or fy
        render_image_height = shot_intrinsics.image_height
    else:
        render_fy = fy
        render_image_height = intr.image_height
    # view_matrix is the Atlas camera_view_matrix (4×4, row-major)
    vm = [list(row) for row in extr.camera_view_matrix]

    try:
        from atlas_camera.core.camera_math import ground_lookat_pivot
        orbit_pivot = [float(v) for v in ground_lookat_pivot(extr)]
    except Exception:
        orbit_pivot = [0.0, 0.0, 0.0]

    # Encode source image as JPEG base64 so the browser can use it as background
    source_b64 = _image_tensor_to_preview_b64(source_image, quality=85)
    source_plate = getattr(solve, "source_plate", None)
    if not source_b64 and source_plate is not None:
        source_b64 = source_plate.preview_b64 or ""

    # Derived projection proxies (ground/walls/boxes/cylinders/backdrop) for the
    # viewport to build; empty list when nothing was derived. preview_expand>1
    # dilates them outward from the camera for wider orbit coverage — display
    # only, never mutates the primitives stored on the solve.
    from atlas_camera.core.proxy_geometry import serialize_proxy_geometry
    proxy_geometry = serialize_proxy_geometry(
        solve.projection_scene,
        preview_expand=preview_expand,
        preview_pivot=extr.camera_position,
    )

    # Multi-angle patch sources (AtlasAddPatchView)
    projection_sources = _serialize_projection_sources(solve)

    # Vanishing points + horizon (2D image-space diagnostics — meaningful only
    # against the flat source photo, not the 3D scene) for the viewport's
    # layered VP/horizon/ground overlay.
    vanishing_points = [
        {
            "position_px": list(vp.position_px),
            "direction_label": vp.direction_label,
            "confidence": float(vp.confidence),
        }
        for vp in (solve.vanishing_points or [])
    ]
    horizon_line = None
    if solve.horizon_line is not None:
        horizon_line = {
            "endpoints_px": [list(p) for p in solve.horizon_line.endpoints_px]
                            if solve.horizon_line.endpoints_px else None,
            "line_coefficients": list(solve.horizon_line.line_coefficients),
            "confidence": float(solve.horizon_line.confidence),
        }

    # Solved latent-camera metadata (lens, distance, provenance) for the HUD.
    fov_h_deg = None
    if fx > 0 and intr.image_width:
        fov_h_deg = math.degrees(2.0 * math.atan(intr.image_width / (2.0 * fx)))
    scene_depth_m = None
    for prim in solve.projection_scene.proxy_geometry:
        if prim.name == "projection_backdrop":
            scene_depth_m = (prim.metadata or {}).get("distance_m")
            break
    from atlas_camera.core.scene_health import scale_health
    sh = scale_health(solve)
    camera_meta = {
        "confidence": float(getattr(solve, "confidence", 0.0) or 0.0),
        "source_method": getattr(solve, "source_method", None),
        "scale_source": (solve.debug_metadata or {}).get("scale_source"),
        "focal_mm": intr.focal_length_mm,
        "sensor_mm": intr.sensor_width_mm,
        "fov_h_deg": fov_h_deg,
        "camera_height_m": float(extr.camera_position[1]) if extr.camera_position else None,
        "scene_depth_m": scene_depth_m,
        "scale_health": {"status": sh.status,
                         "safe_to_export": sh.safe_to_export,
                         "detail": sh.detail},
    }

    primary_depth_b64, primary_depth_width, primary_depth_height = _pack_primary_depth(
        solve, primary_depth, intr, extr, fx, fy, cx, cy
    )


    return {
        "view_matrix": vm,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "camera_position": list(extr.camera_position),
        "image_width": intr.image_width,
        "image_height": intr.image_height,
        "target_width": target_width,
        "target_height": target_height,
        "shot_cam": shot_intrinsics is not None,
        "render_fy": render_fy,
        "render_image_height": render_image_height,
        # ground_lookat_pivot: the EXACT pivot orbit_camera uses to construct
        # patch cameras backend-side. The 📐 Extract Angle button computes its
        # orbit delta about this (not the viewport's own geometry-centroid
        # orbit pivot, which differs) so extracted angles round-trip exactly
        # through AtlasAddPatchView/AtlasOcclusionMask's orbit_camera call.
        "orbit_pivot": orbit_pivot,
        # Identity of this solve+image — 📐 Extract Angle echoes it back so a
        # stale extraction (different photo) can never drive the patch branch.
        "solve_fingerprint": solve_fingerprint,
        "focal_mm": intr.focal_length_mm,
        "sensor_mm": intr.sensor_width_mm,
        "source_image_b64": source_b64,
        "source_plate": _plate_ref_to_dict(source_plate),
        "output_profile": _output_profile_to_dict(
            output_profile if output_profile is not None else getattr(solve, "output_profile", None)
        ),
        "proxy_geometry": proxy_geometry,
        "projection_sources": projection_sources,
        "vanishing_points": vanishing_points,
        "horizon_line": horizon_line,
        "camera_meta": camera_meta,
        "primary_depth_b64": primary_depth_b64,
        "primary_depth_width": primary_depth_width,
        "primary_depth_height": primary_depth_height,
    }

__all__ = [
    "_fit_long_edge",
    "_plate_ref_to_dict",
    "_output_profile_to_dict",
    "_decimate_metric_depth_for_viewport",
    "_extract_blockout_camera",
]
