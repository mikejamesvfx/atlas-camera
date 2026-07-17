"""lensfun geometry correction for RAW imports ([raw-lens] extra).

Separate optional extra from [raw]: lensfunpy wheels can lag new Python /
Windows releases, and decode/metadata must keep working without it. Every
lookup miss is a STATUS the pipeline reports, never an exception — Fuji X
bodies especially rely on in-body corrections and have thin lensfun coverage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _require_lensfunpy():
    try:
        import lensfunpy
    except ImportError as exc:
        raise RuntimeError(
            "Lens undistortion requires lensfunpy. "
            "Install with: pip install -e .[raw-lens]") from exc
    return lensfunpy


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "Lens undistortion requires opencv-python for the remap. "
            "Install with: pip install -e .[raw]") from exc
    return cv2


@dataclass(slots=True)
class UndistortResult:
    status: str                     # "applied"|"no_profile_camera"|"no_profile_lens"|"no_lens_metadata"
    cam_name: str | None = None
    lens_name: str | None = None
    coords: Any | None = None       # HxWx2 float32 remap grid (None unless "applied")
    distortion: dict[str, float] = field(default_factory=dict)


def build_undistort_map(meta, width: int, height: int) -> UndistortResult:
    """EXIF make/model/lens -> lensfun profile -> remap grid.

    The correction maps to the rectilinear ideal at the same nominal focal and
    a centered principal point, so the EXIF focal stays valid for intrinsics
    (nominal-vs-calibrated residual is ~1-2%; the report notes it).
    """
    lensfunpy = _require_lensfunpy()

    if not meta.camera_model:
        return UndistortResult("no_lens_metadata")
    db = lensfunpy.Database()
    cams = db.find_cameras(meta.camera_make or "", meta.camera_model,
                           loose_search=True)
    if not cams:
        return UndistortResult("no_profile_camera")
    cam = cams[0]
    if not meta.lens_model:
        return UndistortResult("no_lens_metadata", cam_name=str(cam.model))
    lenses = db.find_lenses(cam, meta.lens_make or "", meta.lens_model,
                            loose_search=True)
    if not lenses:
        return UndistortResult("no_profile_lens", cam_name=str(cam.model))
    lens = lenses[0]

    focal = meta.focal_length_mm or getattr(lens, "min_focal", None) or 35.0
    aperture = meta.aperture or 8.0
    mod = lensfunpy.Modifier(lens, cam.crop_factor, width, height)
    mod.initialize(float(focal), float(aperture), 1000.0,
                   flags=lensfunpy.ModifyFlags.DISTORTION)
    coords = mod.apply_geometry_distortion()
    if coords is None:
        # Lens found but carries no distortion calibration at this focal.
        return UndistortResult("no_profile_lens", cam_name=str(cam.model),
                               lens_name=str(lens.model))

    distortion = {"lensfun_crop_factor": float(cam.crop_factor),
                  "lensfun_focal_mm": float(focal)}
    return UndistortResult("applied", cam_name=str(cam.model),
                           lens_name=str(lens.model), coords=coords,
                           distortion=distortion)


def apply_undistort(arr, coords):
    """Remap one HxWx3 float32 array through the shared lensfun grid."""
    cv2 = _require_cv2()
    return cv2.remap(arr, coords, None, cv2.INTER_LANCZOS4,
                     borderMode=cv2.BORDER_REPLICATE)
