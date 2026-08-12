"""Crop-adjusted camera intrinsics for Dynamic Plates.

When a region of a solved still is extracted (a dynamic-plate ROI), the crop is
NOT an unrelated image: it shares the solved camera's pose and focal length,
and only the principal point shifts. For an unscaled crop::

    fx' = fx            cx' = cx - crop_x
    fy' = fy            cy' = cy - crop_y

If the crop is later resized for model inference the intrinsics scale by the
resize factors — never re-derived from FOV guesses::

    fx'' = fx' * sx     cx'' = cx' * sx      (sx = out_w / crop_w)
    fy'' = fy' * sy     cy'' = cy' * sy      (sy = out_h / crop_h)

`CropTransform` persists enough to invert the whole mapping exactly
(full-image pixel <-> resized-crop pixel).

The inverse operation (a plate that GREW) already exists as
`depth_outpaint.widen_intrinsics`; this module is the crop-side counterpart
and follows the same conventions: deepcopy, principal-point ladder resolved
before arithmetic, sensor height kept consistent with the new aspect.

Host-agnostic: stdlib only.
"""
from __future__ import annotations

import copy

from dataclasses import dataclass
from typing import Any

from atlas_camera.core.camera_spec import CameraSpec
from atlas_camera.core.schema import _json_ready


@dataclass(slots=True)
class RegionROI:
    """Axis-aligned pixel rectangle in image coordinates (origin top-left)."""

    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        self.x = int(self.x)
        self.y = int(self.y)
        self.width = int(self.width)
        self.height = int(self.height)
        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                f"RegionROI needs positive extents, got {self.width}x{self.height}")

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "RegionROI | None":
        if not data:
            return None
        return cls(x=int(data.get("x", 0)), y=int(data.get("y", 0)),
                   width=int(data["width"]), height=int(data["height"]))

    def clamped(self, image_width: int, image_height: int) -> "RegionROI":
        """The intersection of this ROI with the image rectangle."""
        x0 = max(0, self.x)
        y0 = max(0, self.y)
        x1 = min(int(image_width), self.x + self.width)
        y1 = min(int(image_height), self.y + self.height)
        if x1 <= x0 or y1 <= y0:
            raise ValueError(
                f"ROI {self.to_dict()} lies outside a "
                f"{image_width}x{image_height} image")
        return RegionROI(x=x0, y=y0, width=x1 - x0, height=y1 - y0)

    def expanded(self, *, pad_px: int = 0, pad_frac: float = 0.0,
                 image_width: int, image_height: int) -> "RegionROI":
        """Overscan: grow each side by ``pad_px`` plus ``pad_frac`` of the
        larger ROI extent, then clamp to the image."""
        pad = int(round(int(pad_px) + float(pad_frac) * max(self.width, self.height)))
        grown = RegionROI(x=self.x - pad, y=self.y - pad,
                          width=self.width + 2 * pad,
                          height=self.height + 2 * pad)
        return grown.clamped(image_width, image_height)


def _resolved_centre(intrinsics: Any) -> tuple[float, float]:
    """cx/cy via the CameraSpec fallback ladder (cx_px -> principal -> centre)."""
    spec = CameraSpec.from_intrinsics(intrinsics)
    return spec.cx, spec.cy


def crop_intrinsics(intrinsics: Any, roi: RegionROI) -> Any:
    """The camera for an unscaled crop of the original plate.

    Focal length is untouched (a crop does not change the lens); the principal
    point shifts by the crop origin because the same optical centre now sits
    closer to the new top-left corner.
    """
    w = int(getattr(intrinsics, "image_width", 0) or 0)
    h = int(getattr(intrinsics, "image_height", 0) or 0)
    if roi.x < 0 or roi.y < 0 or roi.x + roi.width > w or roi.y + roi.height > h:
        raise ValueError(
            f"ROI {roi.to_dict()} does not lie within the {w}x{h} plate")
    cx, cy = _resolved_centre(intrinsics)
    out = copy.deepcopy(intrinsics)
    out.image_width = roi.width
    out.image_height = roi.height
    out.cx_px = cx - roi.x
    out.cy_px = cy - roi.y
    out.principal_point_px = (out.cx_px, out.cy_px)
    sw = getattr(out, "sensor_width_mm", None)
    if sw and out.image_width:
        # Sensor width now covers only the crop's share of the frame; keep the
        # mm-side story consistent so focal_length_mm -> fx_px still agrees.
        out.sensor_width_mm = float(sw) * (roi.width / float(w))
        out.sensor_height_mm = float(out.sensor_width_mm) * (
            out.image_height / float(out.image_width))
    return out


def scale_intrinsics(intrinsics: Any, out_width: int, out_height: int) -> Any:
    """The camera for a resized raster (e.g. crop downscaled for inference)."""
    w = int(getattr(intrinsics, "image_width", 0) or 0)
    h = int(getattr(intrinsics, "image_height", 0) or 0)
    if w <= 0 or h <= 0:
        raise ValueError("scale_intrinsics needs source image_width/height")
    out_width = int(out_width)
    out_height = int(out_height)
    if out_width <= 0 or out_height <= 0:
        raise ValueError("scale_intrinsics needs positive output extents")
    sx = out_width / float(w)
    sy = out_height / float(h)
    cx, cy = _resolved_centre(intrinsics)
    out = copy.deepcopy(intrinsics)
    out.image_width = out_width
    out.image_height = out_height
    if getattr(out, "fx_px", None) is not None:
        out.fx_px = float(out.fx_px) * sx
    if getattr(out, "fy_px", None) is not None:
        out.fy_px = float(out.fy_px) * sy
    out.cx_px = cx * sx
    out.cy_px = cy * sy
    out.principal_point_px = (out.cx_px, out.cy_px)
    return out


@dataclass(slots=True)
class CropTransform:
    """Exactly invertible full-image <-> resized-crop pixel mapping."""

    source_width: int
    source_height: int
    roi: RegionROI
    output_width: int
    output_height: int

    def _scales(self) -> tuple[float, float]:
        return (self.output_width / float(self.roi.width),
                self.output_height / float(self.roi.height))

    def full_to_crop(self, px: float, py: float) -> tuple[float, float]:
        sx, sy = self._scales()
        return ((float(px) - self.roi.x) * sx, (float(py) - self.roi.y) * sy)

    def crop_to_full(self, px: float, py: float) -> tuple[float, float]:
        sx, sy = self._scales()
        return (float(px) / sx + self.roi.x, float(py) / sy + self.roi.y)

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "CropTransform | None":
        if not data:
            return None
        roi = RegionROI.from_dict(data.get("roi"))
        if roi is None:
            return None
        return cls(source_width=int(data["source_width"]),
                   source_height=int(data["source_height"]),
                   roi=roi,
                   output_width=int(data["output_width"]),
                   output_height=int(data["output_height"]))
