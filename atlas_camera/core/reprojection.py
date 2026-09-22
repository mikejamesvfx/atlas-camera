"""Where a camera's pixels land in the PLATE, assuming infinity.

MOVED HERE 2026-09-22, verbatim, from ``dynamic.occlusion_fill``. The body was
already nothing but ``core`` calls (``depth_geometry.back_project_normals`` and
``patch_registration._project``), so the only thing keeping it in ``dynamic/``
was where it was first needed. That location had become a layering wall:
``core.ghost_pixels`` classifies a target view using this mapping's ``inside``
field and explicitly may not import ``dynamic`` (the layering runs one way), so
the per-frame conditioning sequence could not live in ``core`` while its sky
test lived in ``dynamic``. ``dynamic.occlusion_fill`` re-exports the name, which
its own ``not_disocclusion_mask`` / ``survey_hole_rois`` still call.

THE MAPPING IS ROTATION-ONLY BY CONSTRUCTION. Every pixel is back-projected at
``_INFINITY_M``, where the translation between the two cameras cancels. That is
exactly the property the sky test needs — two views differing by a rotation do
not see the sky at the same pixels, and sky has no parallax to find — and it is
exactly the property optical flow must NOT have. This function is the sky test.
It is never a motion field; see ``core.conditioning`` for flow.

Host-agnostic: numpy only, no Pillow, no torch, no ComfyUI.
"""
from __future__ import annotations

from typing import Any

#: Stand-in for "infinitely far". Large enough that the inter-camera
#: translation is negligible against it, small enough to stay exact in float64.
INFINITY_M = 1.0e6


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - guarded import
        raise RuntimeError(
            "atlas_camera.core.reprojection requires numpy. Install with: "
            "pip install -e .[vision]") from exc
    return np


def reproject_at_infinity(plate, *, view, fx, fy, cx, cy, width, height):
    """Where each pixel of a camera lands in the PLATE, assuming infinity.

    Returns ``{"inside", "u", "v", "plate_width", "plate_height"}`` — boolean
    plus float plate-raster coordinates (the raster ``plate`` was surveyed at).
    Split out of ``not_disocclusion_mask`` because the coordinates are useful in
    their own right: sky is at infinity, so sampling the plate through this
    mapping IS the correct sky for the moved camera, which is how a dropped
    (not disocclusion) pixel gets real content instead of a sentinel.

    ``plate`` is a dict carrying ``mask`` (for its raster), ``view``, ``fx``,
    ``fy``, ``cx``, ``cy`` — the camera the photograph was taken through.
    """
    np = _require_numpy()
    from atlas_camera.core.depth_geometry import back_project_normals
    from atlas_camera.core.patch_registration import _project

    p_h, p_w = np.asarray(plate["mask"]).shape[:2]
    height, width = int(height), int(width)
    depth = np.full((height, width), INFINITY_M, dtype=np.float64)
    bp = back_project_normals(depth, view_matrix=view,
                              fx=fx, fy=fy, cx=cx, cy=cy)
    u, v, fwd = _project(bp.pts_world.reshape(-1, 3),
                         {"view_matrix": plate["view"], "fx": plate["fx"],
                          "fy": plate["fy"], "cx": plate["cx"],
                          "cy": plate["cy"]})
    inside = (np.isfinite(u) & np.isfinite(v) & (fwd > 1e-6)
              & (u >= 0) & (u < p_w) & (v >= 0) & (v < p_h))
    return {"inside": inside.reshape(height, width),
            "u": np.nan_to_num(u).reshape(height, width),
            "v": np.nan_to_num(v).reshape(height, width),
            "plate_width": p_w, "plate_height": p_h}
