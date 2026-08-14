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

Host-agnostic: stdlib only for the intrinsics math. ``hole_rois`` and
``composite_crops`` additionally need numpy and gate on it at call time.
"""
from __future__ import annotations

import copy

from dataclasses import dataclass, field
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

    @property
    def area_px(self) -> int:
        return self.width * self.height

    def snapped(self, snap: int, *, image_width: int,
                image_height: int) -> "RegionROI":
        """Grow to a multiple of ``snap`` on both axes, staying in the image.

        Diffusion models raster on a fixed grid — LTX-2.5 wants /32, and /64
        once an adapter declares ``reference_downscale_factor = 2.0``. Growing
        (never shrinking) keeps every hole pixel inside the crop; the ROI is
        then slid back into the frame rather than clamped, because clamping
        would break the grid the snap exists to hit. When the image itself is
        smaller than one snapped step the largest fitting multiple is used, and
        if even that is zero the extent is left unsnapped (the caller's raster
        is then the image, which no crop can improve).
        """
        snap = int(snap)
        if snap <= 1:
            return self.clamped(image_width, image_height)
        iw, ih = int(image_width), int(image_height)

        def _fit(extent: int, limit: int) -> int:
            grown = -(-int(extent) // snap) * snap  # ceil to a multiple
            if grown <= limit:
                return grown
            fitted = (limit // snap) * snap
            return fitted if fitted > 0 else limit

        w = _fit(self.width, iw)
        h = _fit(self.height, ih)
        # Keep the ROI centred on the same content, then slide (not clamp) it
        # fully inside the frame so the snapped extents survive.
        x = self.x - (w - self.width) // 2
        y = self.y - (h - self.height) // 2
        x = max(0, min(x, iw - w))
        y = max(0, min(y, ih - h))
        return RegionROI(x=x, y=y, width=w, height=h)


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


# ---------------------------------------------------------------------------
# Hole clustering (numpy-gated)

def _require_numpy():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "hole_rois/composite_crops require numpy. Install with:\n"
            "    pip install -e .[vision]") from exc
    return np


@dataclass(slots=True)
class HoleROISet:
    """The ROIs a hole-crop fill will actually generate, plus what it dropped.

    ``dropped`` is not diagnostic garnish: a pipeline that silently ignores
    small scattered holes reads as "covered everything" when it did not, and
    those holes usually want deterministic edge-extend rather than diffusion.
    """

    rois: list[RegionROI] = field(default_factory=list)
    dropped: list[dict[str, Any]] = field(default_factory=list)
    image_width: int = 0
    image_height: int = 0
    hole_area_px: int = 0
    component_count: int = 0

    @property
    def roi_area_px(self) -> int:
        return sum(r.area_px for r in self.rois)

    @property
    def image_area_px(self) -> int:
        return int(self.image_width) * int(self.image_height)

    @property
    def coverage_frac(self) -> float:
        """Generated pixels / full-frame pixels — the G1 compute-saved number.

        A value near 1.0 means the crops cover most of the frame and the
        hole-crop premise does not hold for this plate; say so rather than
        pretending the crop bought anything.
        """
        total = self.image_area_px
        return (self.roi_area_px / float(total)) if total else 0.0

    @property
    def dropped_area_px(self) -> int:
        return sum(int(d.get("area_px", 0)) for d in self.dropped)

    def to_dict(self) -> dict[str, Any]:
        return {"rois": [r.to_dict() for r in self.rois],
                "dropped": list(self.dropped),
                "image_width": int(self.image_width),
                "image_height": int(self.image_height),
                "hole_area_px": int(self.hole_area_px),
                "component_count": int(self.component_count),
                "roi_area_px": self.roi_area_px,
                "coverage_frac": self.coverage_frac,
                "dropped_area_px": self.dropped_area_px}


def _as_bool_mask(np, mask) -> Any:
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[..., 0]
    if m.dtype == np.bool_:
        return m
    if m.dtype == np.uint8:
        return m > 127
    return m > 0.5


def _component_boxes(np, mask) -> list[tuple[int, int, int, int, int]]:
    """Connected components (4-connectivity) as ``(x0, y0, x1, y1, area)``.

    Row-run union-find rather than per-pixel labelling: an 8K plate is 36M
    pixels but only a few thousand runs, and this keeps the module on
    stdlib+numpy (no scipy/cv2 dependency divergence between hosts).
    """
    height, width = mask.shape[:2]
    flat = mask.reshape(-1).astype(np.int8)
    padded = np.zeros((height, width + 2), dtype=np.int8)
    padded[:, 1:-1] = flat.reshape(height, width)
    diff = np.diff(padded, axis=1)
    starts_r, starts_c = np.nonzero(diff == 1)
    ends_r, ends_c = np.nonzero(diff == -1)
    if starts_r.size == 0:
        return []
    # (row, col_start, col_end_inclusive) — nonzero scans row-major, so runs
    # arrive sorted by row then column, which the sweep below relies on.
    runs = list(zip(starts_r.tolist(), starts_c.tolist(),
                    (ends_c - 1).tolist()))

    parent = list(range(len(runs)))

    def find(i: int) -> int:
        root = i
        while parent[root] != root:
            root = parent[root]
        while parent[i] != root:
            parent[i], i = root, parent[i]
        return root

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    row_start: dict[int, int] = {}
    for index, (row, _c0, _c1) in enumerate(runs):
        row_start.setdefault(row, index)

    for row in sorted(row_start):
        if row + 1 not in row_start:
            continue
        i = row_start[row]
        j = row_start[row + 1]
        while i < len(runs) and runs[i][0] == row and \
                j < len(runs) and runs[j][0] == row + 1:
            a0, a1 = runs[i][1], runs[i][2]
            b0, b1 = runs[j][1], runs[j][2]
            if a0 <= b1 and b0 <= a1:
                union(i, j)
            if a1 < b1:
                i += 1
            else:
                j += 1

    boxes: dict[int, list[int]] = {}
    for index, (row, c0, c1) in enumerate(runs):
        root = find(index)
        box = boxes.get(root)
        if box is None:
            boxes[root] = [c0, row, c1, row, c1 - c0 + 1]
        else:
            box[0] = min(box[0], c0)
            box[1] = min(box[1], row)
            box[2] = max(box[2], c1)
            box[3] = max(box[3], row)
            box[4] += c1 - c0 + 1
    return [tuple(b) for b in boxes.values()]


def rois_from_world_regions(regions, view, *, fx: float, fy: float,
                            cx: float, cy: float, image_width: int,
                            image_height: int, pad_frac: float = 0.05,
                            pad_px: int = 0, snap: int = 64) -> HoleROISet:
    """Artist-drawn world-space regions -> crop ROIs under ``view``.

    The artist marks the tears worth repairing in the 3D viewport, so the
    region arrives as WORLD corners rather than a screen rectangle. That is
    the useful form: the marker stays pinned to the same surface as the camera
    moves, so one selection frames the same content in every frame of the
    shot, and the ROI is simply its screen bounding box under whichever view
    is being repaired.

    Regions entirely behind the camera, or landing outside the frame, are
    recorded in ``dropped`` rather than skipped silently.
    """
    np = _require_numpy()
    view_matrix = np.asarray(view, dtype=np.float64).reshape(4, 4)
    result = HoleROISet(image_width=int(image_width),
                        image_height=int(image_height))
    for index, region in enumerate(regions or []):
        label = str((region or {}).get("label") or f"region {index + 1}")
        pts = [p for p in ((region or {}).get("points_world") or [])
               if p is not None and len(p) >= 3]
        if len(pts) < 3:
            result.dropped.append(
                {"x": 0, "y": 0, "width": 0, "height": 0, "area_px": 0,
                 "reason": f"{label}: needs at least 3 world corners"})
            continue
        world = np.asarray([[float(p[0]), float(p[1]), float(p[2]), 1.0]
                            for p in pts], dtype=np.float64)
        cam = world @ view_matrix.T
        forward = -cam[:, 2]
        ahead = forward > 1e-6
        if not bool(ahead.any()):
            result.dropped.append(
                {"x": 0, "y": 0, "width": 0, "height": 0, "area_px": 0,
                 "reason": f"{label}: entirely behind the camera in this view"})
            continue
        cam = cam[ahead]
        forward = forward[ahead]
        px = float(cx) + float(fx) * (cam[:, 0] / forward)
        py = float(cy) - float(fy) * (cam[:, 1] / forward)
        x0, x1 = int(np.floor(px.min())), int(np.ceil(px.max()))
        y0, y1 = int(np.floor(py.min())), int(np.ceil(py.max()))
        raw = RegionROI(x=x0, y=y0, width=max(1, x1 - x0),
                        height=max(1, y1 - y0))
        try:
            grown = raw.expanded(pad_px=pad_px, pad_frac=pad_frac,
                                 image_width=image_width,
                                 image_height=image_height)
        except ValueError:
            result.dropped.append(
                {**raw.to_dict(), "area_px": raw.area_px,
                 "reason": f"{label}: projects outside the frame in this view"})
            continue
        result.rois.append(grown.snapped(snap, image_width=image_width,
                                         image_height=image_height))
        result.component_count += 1
    return result


def hole_rois(masks, *, pad_frac: float = 0.15, pad_px: int = 0,
              min_area_px: int = 1024, snap: int = 64,
              max_rois: int = 4) -> HoleROISet:
    """Cluster disocclusion holes into crop ROIs.

    ``masks`` is one 2D mask or a sequence of them (uint8 / bool / float);
    they are UNIONED ACROSS TIME before clustering, so one crop serves the
    entire camera move and the generated content is temporally consistent by
    construction rather than by prompt luck.

    Each connected component becomes a `RegionROI`, padded by ``pad_frac`` of
    its larger extent for real context around the hole, clamped to the image
    and snapped to the model grid. The largest ``max_rois`` by area survive —
    every other component is recorded in ``HoleROISet.dropped`` with a reason.
    """
    np = _require_numpy()
    if masks is None:
        raise ValueError("hole_rois needs at least one mask")
    seq = masks if isinstance(masks, (list, tuple)) else None
    if seq is None:
        arr = np.asarray(masks)
        seq = list(arr) if arr.ndim == 3 and arr.shape[-1] not in (1, 3, 4) \
            else [arr]
    if not len(seq):
        raise ValueError("hole_rois needs at least one mask")

    union_mask = None
    for mask in seq:
        m = _as_bool_mask(np, mask)
        if union_mask is None:
            union_mask = m.copy()
        elif m.shape != union_mask.shape:
            raise ValueError(
                f"hole masks disagree on raster: {m.shape} vs "
                f"{union_mask.shape}")
        else:
            union_mask |= m
    height, width = union_mask.shape[:2]
    result = HoleROISet(image_width=int(width), image_height=int(height),
                        hole_area_px=int(union_mask.sum()))
    if not bool(union_mask.any()):
        return result

    components = _component_boxes(np, union_mask)
    result.component_count = len(components)
    components.sort(key=lambda b: b[4], reverse=True)

    kept = 0
    for x0, y0, x1, y1, area in components:
        tight = RegionROI(x=x0, y=y0, width=x1 - x0 + 1, height=y1 - y0 + 1)
        if area < int(min_area_px):
            result.dropped.append(
                {**tight.to_dict(), "area_px": int(area),
                 "reason": f"hole area {area}px < min_area_px {min_area_px}"})
            continue
        if max_rois > 0 and kept >= int(max_rois):
            result.dropped.append(
                {**tight.to_dict(), "area_px": int(area),
                 "reason": f"beyond max_rois {max_rois} (ranked by area)"})
            continue
        roi = tight.expanded(pad_px=pad_px, pad_frac=pad_frac,
                             image_width=width, image_height=height)
        result.rois.append(roi.snapped(snap, image_width=width,
                                       image_height=height))
        kept += 1
    return result


# ---------------------------------------------------------------------------
# Composite back (numpy-gated)

def _feather_alpha(np, mask, feather_px: int):
    """Alpha 1 over the whole mask, ramping down over ``feather_px`` OUTSIDE.

    The feather grows outward, never inward. Ramping inward leaves alpha near
    zero along the hole's own boundary — and what shows through there is the
    guide's inpaint sentinel, so an inward feather paints a green rim around
    every filled hole (seen on the first DSC_2289 composite). Outside the hole
    the base is real photographed pixels, which is what a blend should mix
    with.
    """
    alpha = mask.astype(np.float32)
    if feather_px <= 0:
        return alpha
    grown = mask.copy()
    step = 1.0 / (int(feather_px) + 1)
    for i in range(int(feather_px)):
        wider = grown.copy()
        wider[1:, :] |= grown[:-1, :]
        wider[:-1, :] |= grown[1:, :]
        wider[:, 1:] |= grown[:, :-1]
        wider[:, :-1] |= grown[:, 1:]
        # Pixels gained at this step sit i+1 pixels outside the mask.
        ring = wider & ~grown
        alpha[ring] = 1.0 - (i + 1) * step
        grown = wider
    return alpha


def match_reference_colour(crop, reference, mask, *, min_samples: int = 64):
    """Fit per-channel gain+offset on the UNMASKED pixels and apply it.

    A diffusion round trip re-renders the whole crop, not only the masked
    pixels: the VAE encode/decode plus the model's own tonal prior shift
    colour globally (measured on DSC_2289: a visible green/magenta cast and a
    saturation drop across the entire returned crop). Compositing only the
    masked pixels therefore lands a mismatched patch in a plate that was
    otherwise untouched.

    The unmasked region is the SAME content in both images, so it is a paired
    sample that pins the shift without any assumption about the fill: solve
    ``crop*g + o ~= reference`` per channel there, then apply that transfer to
    the whole crop, hole included. Returns the corrected crop (a copy); with
    too few paired samples the crop is returned unchanged rather than
    corrected from noise.
    """
    np = _require_numpy()
    src = np.asarray(crop, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    if src.shape != ref.shape:
        raise ValueError(
            f"colour match needs matching rasters, got {src.shape} vs "
            f"{ref.shape}")
    keep = ~_as_bool_mask(np, mask)
    if int(keep.sum()) < int(min_samples):
        return np.array(crop, copy=True)
    out = src.copy()
    channels = src.shape[-1] if src.ndim == 3 else 1
    for c in range(channels):
        s = src[..., c][keep] if src.ndim == 3 else src[keep]
        r = ref[..., c][keep] if ref.ndim == 3 else ref[keep]
        s_std = float(s.std())
        if s_std < 1e-6:
            gain, offset = 1.0, float(r.mean() - s.mean())
        else:
            gain = float(r.std() / s_std)
            offset = float(r.mean() - gain * s.mean())
        if src.ndim == 3:
            out[..., c] = src[..., c] * gain + offset
        else:
            out = src * gain + offset
    source = np.asarray(crop)
    if np.issubdtype(source.dtype, np.integer):
        info = np.iinfo(source.dtype)
        out = np.clip(np.rint(out), info.min, info.max)
    return out.astype(source.dtype)


def neutralize_fill_cast(crop, mask, *, reference=None, band_px: int = 32,
                         min_samples: int = 64):
    """Remove a colour CAST from the filled pixels, using the ring around them.

    An inpaint IC-LoRA leaves its own tint in what it invents — measured on
    DSC_2289, the fill ran +5.8 green-excess against a context sitting at
    -0.6, and it read as a green haze over every repaired band. That is a
    chroma error, not a brightness error: the fill's luminance is legitimately
    its own (it may be road where the ring is car), so only the colour BALANCE
    is transferred, per channel relative to each region's own luminance. The
    reference is an annulus ``band_px`` wide just outside the mask — near
    enough to share the local lighting, and made of real photographed pixels.

    ``reference`` is where that annulus is READ FROM and should be the plate
    render, not the generated crop: a diffusion round trip re-tints the whole
    crop, so sampling the ring from the crop measures the cast against itself
    and cancels almost nothing (it recovered 0.5 of a 5.8 error on DSC_2289;
    against the plate it recovered all of it). Defaults to the crop only so a
    caller with nothing better still gets the intra-crop correction.

    Returns a corrected copy; unchanged when either region is too small to
    measure.
    """
    np = _require_numpy()
    src = np.asarray(crop, dtype=np.float64)
    if src.ndim != 3 or src.shape[-1] < 3:
        raise ValueError("neutralize_fill_cast needs an HxWx3 crop")
    ref = src if reference is None else np.asarray(reference, dtype=np.float64)
    if ref.shape[:2] != src.shape[:2]:
        raise ValueError(
            f"reference raster {ref.shape[:2]} != crop {src.shape[:2]}")
    hole = _as_bool_mask(np, mask)
    grown = hole.copy()
    for _ in range(max(1, int(band_px))):
        wider = grown.copy()
        wider[1:, :] |= grown[:-1, :]
        wider[:-1, :] |= grown[1:, :]
        wider[:, 1:] |= grown[:, :-1]
        wider[:, :-1] |= grown[:, 1:]
        grown = wider
    ring = grown & ~hole
    if int(hole.sum()) < int(min_samples) or int(ring.sum()) < int(min_samples):
        return np.array(crop, copy=True)

    fill = src[hole][..., :3]
    ctx = ref[ring][..., :3]
    fill_lum = float(fill.mean())
    ctx_lum = float(ctx.mean())
    out = src.copy()
    for c in range(3):
        # each region's channel offset from its OWN luminance = its colour
        # balance; transfer only the difference between the two balances.
        shift = (float(ctx[:, c].mean()) - ctx_lum) - \
            (float(fill[:, c].mean()) - fill_lum)
        out[..., c][hole] = src[..., c][hole] + shift
    source = np.asarray(crop)
    if np.issubdtype(source.dtype, np.integer):
        info = np.iinfo(source.dtype)
        out = np.clip(np.rint(out), info.min, info.max)
    return out.astype(source.dtype)


def membrane_blend(fill, reference, mask, *, max_cg_iters: int = 600,
                   tol: float = 1e-4):
    """Erase the seam between a generated fill and the plate around it.

    Solves the Poisson-editing offset membrane: the mismatch
    ``reference - fill`` is sampled on the hole's RIM and extended
    harmonically (Laplace) into the hole, then added to the fill. The
    correction is smooth by construction, so the fill's own texture — its
    gradients — passes through untouched while the boundary matches the plate
    exactly.

    This exists because nothing generation-side moved the seam: across four
    arms (masked denoise both polarities, clean-plate LoRA, dev transformer)
    the rim gradient ratio stayed 2.0-8.5x the plate's own statistics, and
    cv2.seamlessClone's bbox/centre placement silently missed the target.
    The membrane took 2.22 -> 1.06 and 2.04 -> 1.03 (plate-native ~1.0) with
    fill gradient energy preserved, ~4 s for a 563k-px hole.

    Uses scipy's sparse direct solve when available; otherwise a pure-numpy
    conjugate-gradient on the same 5-point stencil (slower, no new
    dependency). Returns a corrected copy in the input dtype.
    """
    np = _require_numpy()
    src = np.asarray(fill, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    if src.shape != ref.shape or src.ndim != 3:
        raise ValueError(
            f"membrane_blend needs matching HxWx3 rasters, got "
            f"{src.shape} vs {ref.shape}")
    hole = _as_bool_mask(np, mask)
    if hole.shape != src.shape[:2]:
        raise ValueError(
            f"mask {hole.shape} does not match raster {src.shape[:2]}")
    if not bool(hole.any()):
        return np.array(fill, copy=True)

    height, width = hole.shape
    index = np.full((height, width), -1, dtype=np.int64)
    ys, xs = np.nonzero(hole)
    count = len(ys)
    index[ys, xs] = np.arange(count)

    # 5-point Laplacian over hole pixels; rim pixels are Dirichlet boundary
    # carrying the INTERFACE STEP — reference just outside minus fill just
    # inside. (Using ref-fill AT the outside pixel instead reads ~0 whenever
    # the fill already matches the plate outside the hole — it does, after
    # colour correction — and corrects nothing.) Both sides are 3x3
    # box-smoothed first: the raw one-pixel difference carries the plate's
    # own texture into the boundary condition and the membrane then bakes
    # that noise into the hole. Off-image neighbours simply drop out
    # (natural boundary), so holes touching the frame edge are fine.
    def _box3_masked(img, valid):
        """3x3 mean over VALID pixels only — the reference may hold the
        inpaint sentinel inside the hole, and an unmasked blur would smear it
        into the rim samples."""
        v = valid.astype(np.float64)
        acc = img * v[..., None]
        norm = v.copy()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1),
                       (1, 1), (1, -1), (-1, 1), (-1, -1)):
            shifted = np.roll(np.roll(img * v[..., None], dy, axis=0),
                              dx, axis=1)
            weight = np.roll(np.roll(v, dy, axis=0), dx, axis=1)
            # roll wraps; zero the wrapped rows/cols out of the average
            if dy == 1:
                shifted[0] = 0
                weight[0] = 0
            elif dy == -1:
                shifted[-1] = 0
                weight[-1] = 0
            if dx == 1:
                shifted[:, 0] = 0
                weight[:, 0] = 0
            elif dx == -1:
                shifted[:, -1] = 0
                weight[:, -1] = 0
            acc += shifted
            norm += weight
        return acc / np.maximum(norm, 1e-9)[..., None]

    ref_smooth = _box3_masked(ref, ~hole)      # plate side, sentinel excluded
    src_smooth = _box3_masked(src, hole)       # fill side only
    neighbours = []
    rhs = np.zeros((count, 3), dtype=np.float64)
    degree = np.zeros(count, dtype=np.float64)
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        yy = ys + dy
        xx = xs + dx
        inside = (yy >= 0) & (yy < height) & (xx >= 0) & (xx < width)
        degree += inside
        j = np.full(count, -1, dtype=np.int64)
        j[inside] = index[yy[inside], xx[inside]]
        boundary = inside & (j < 0)
        rhs[boundary] += (ref_smooth[yy[boundary], xx[boundary]] -
                          src_smooth[ys[boundary], xs[boundary]])
        neighbours.append(j)

    correction = np.zeros_like(src)
    solved = False
    try:
        from scipy import sparse
        from scipy.sparse.linalg import spsolve

        rows, cols, vals = [np.arange(count)], [np.arange(count)], [degree]
        for j in neighbours:
            have = j >= 0
            rows.append(np.arange(count)[have])
            cols.append(j[have])
            vals.append(np.full(int(have.sum()), -1.0))
        matrix = sparse.csr_matrix(
            (np.concatenate(vals),
             (np.concatenate(rows), np.concatenate(cols))),
            shape=(count, count))
        for channel in range(3):
            correction[ys, xs, channel] = spsolve(matrix, rhs[:, channel])
        solved = True
    except ImportError:
        pass
    if not solved:
        # Conjugate gradient on the same SPD system, matrix-free.
        def apply_laplacian(vec):
            out = degree * vec
            for j in neighbours:
                have = j >= 0
                out[have] -= vec[j[have]]
            return out

        for channel in range(3):
            b = rhs[:, channel]
            x = np.zeros(count)
            r = b - apply_laplacian(x)
            p = r.copy()
            rs = float(r @ r)
            b_norm = max(float(np.linalg.norm(b)), 1e-12)
            for _ in range(int(max_cg_iters)):
                ap = apply_laplacian(p)
                alpha = rs / max(float(p @ ap), 1e-30)
                x += alpha * p
                r -= alpha * ap
                rs_new = float(r @ r)
                if np.sqrt(rs_new) / b_norm < tol:
                    break
                p = r + (rs_new / rs) * p
                rs = rs_new
            correction[ys, xs, channel] = x

    out = src + correction
    source = np.asarray(fill)
    if np.issubdtype(source.dtype, np.integer):
        info = np.iinfo(source.dtype)
        out = np.clip(np.rint(out), info.min, info.max)
    return out.astype(source.dtype)


def composite_crops(base, crops, rois, *, masks=None, feather_px: int = 0):
    """Paste generated crops back into the full plate through their ROIs.

    ``base`` is the full-resolution image (HxWx3, uint8 or float); ``crops``
    are same-dtype rasters matching each ROI's extents exactly — a crop is
    pasted, never resized, because the ROI raster IS the generation raster in
    the hole-crop pipeline. ``masks`` (optional, one per ROI) restricts the
    paste to the pixels the generator was allowed to invent; ``feather_px``
    ramps the paste in from the mask edge so the seam is not a hard step.

    Returns a new array; ``base`` is untouched.
    """
    np = _require_numpy()
    out = np.array(base, copy=True)
    height, width = out.shape[:2]
    crops = list(crops)
    rois = list(rois)
    if len(crops) != len(rois):
        raise ValueError(
            f"composite_crops got {len(crops)} crops for {len(rois)} ROIs")
    if masks is not None:
        masks = list(masks)
        if len(masks) != len(rois):
            raise ValueError(
                f"composite_crops got {len(masks)} masks for {len(rois)} ROIs")

    for index, (crop, roi) in enumerate(zip(crops, rois)):
        patch = np.asarray(crop)
        if roi.x < 0 or roi.y < 0 or roi.x + roi.width > width or \
                roi.y + roi.height > height:
            raise ValueError(
                f"ROI {roi.to_dict()} does not lie within the "
                f"{width}x{height} plate")
        if patch.shape[:2] != (roi.height, roi.width):
            raise ValueError(
                f"crop {index} is {patch.shape[1]}x{patch.shape[0]} but its "
                f"ROI is {roi.width}x{roi.height}; resize before compositing")
        window = (slice(roi.y, roi.y + roi.height),
                  slice(roi.x, roi.x + roi.width))
        if masks is None:
            alpha = np.ones((roi.height, roi.width), dtype=np.float32)
        else:
            alpha = _feather_alpha(np, _as_bool_mask(np, masks[index]),
                                   feather_px)
        if alpha.shape != (roi.height, roi.width):
            raise ValueError(
                f"mask {index} is {alpha.shape[1]}x{alpha.shape[0]} but its "
                f"ROI is {roi.width}x{roi.height}")
        if out.ndim == 3:
            alpha = alpha[..., None]
        blended = out[window].astype(np.float32) * (1.0 - alpha) + \
            patch.astype(np.float32) * alpha
        if np.issubdtype(out.dtype, np.integer):
            info = np.iinfo(out.dtype)
            blended = np.clip(np.rint(blended), info.min, info.max)
        out[window] = blended.astype(out.dtype)
    return out
