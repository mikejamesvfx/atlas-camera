"""Metric scale from a human face — the reference that survives a crop.

Atlas's existing reference-object path (``solver.metric_height_from_reference``)
recovers camera height from an object **standing on the ground**: the base ray
hits the ground plane, the top sits a known height above it, and the apparent
size pins absolute scale. That is the strongest single-view anchor available,
and when the subject's feet are visible it should be preferred over anything
here.

It fails the moment the feet are not visible — a half-body portrait, a seated
subject, a figure behind a wall or a parked car, a tight crop. Those are common
in exactly the plates matte painters get handed. A face is still visible in all
of them, and a face has a known size.

The geometry
------------
A face does not touch the ground, so its rays cannot be intersected with the
ground plane directly. What a known-size feature DOES give is the metric
position of that feature relative to the camera. One further assumption — the
subject is an upright adult standing on the same ground plane the solve is
using — converts that into camera height:

    camera_height = anchor_height_above_ground - anchor_world_Y

where ``anchor_world_Y`` is the measured feature's height relative to the camera
(negative below it) and ``anchor_height_above_ground`` comes from stature. With
the camera at the origin the ground sits at ``Y = -camera_height``, so the
identity is just "how far above the ground the anchor is, minus how far above
the camera it is".

Accuracy, stated honestly
-------------------------
This is a WEAKER anchor than a ground-based reference, for two compounding
reasons, and it is labelled tier-1.5 rather than tier-1 for that reason:

  1. The anthropometric constant has population spread (see ``FACE_METRICS``).
  2. Stature itself is assumed, and it enters the anchor height directly.

A note on interpupillary distance: it is often claimed to be a tighter anchor
than stature. Measured as a coefficient of variation it is not — IPD is about
63 mm with an SD near 3.5 mm (~5.6%), against adult stature at roughly 4% within
sex and ~6% across a mixed population. They are comparable. The real reason to
reach for a face is not precision, it is APPLICABILITY: it is measurable when
the feet are not in frame at all, which is the case a ground reference simply
cannot serve.

Biometrics
----------
This module measures a distance between two image points. It does not identify
anyone, and deliberately computes no face embedding, descriptor or landmark set
— nothing that could persist into a solve JSON as a biometric identifier. The
caller supplies a pixel extent; what comes back is a scale factor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# ``_ray_world`` is private by convention but this is the same layer (core) and
# the same convention it encodes; re-deriving the pixel->world ray here would be
# a silent second copy of the projection convention, which is exactly the class
# of duplication this codebase treats as a defect.
from atlas_camera.core.solver import _ray_world, _require_numpy

__all__ = [
    "FACE_METRICS",
    "FaceScaleResult",
    "DEFAULT_STATURE_M",
    "EYE_HEIGHT_RATIO",
    "camera_height_from_face",
    "face_metric_choices",
]

#: Adult anthropometry. ``size_m`` is the population mean, ``sd_m`` its standard
#: deviation — carried so the caller can report an honest uncertainty band
#: instead of a bare number. ``axis`` says how the two marked points relate:
#: "vertical" segments are solved against world Y (like a ground reference),
#: "horizontal" ones by the angle subtended between their rays.
#:
#: ``anchor_ratio`` is the height of the segment's ANCHOR point above the ground
#: as a fraction of stature. For a vertical segment the anchor is its BOTTOM
#: (chin); for a horizontal one it is the segment's midpoint (eye line, ear
#: line). This is the number that converts a measured feature position into a
#: camera height.
#:
#: APPEND-ONLY: these keys are combo values on AtlasFaceScaleReference and
#: serialize into saved workflows.
FACE_METRICS: dict[str, dict[str, Any]] = {
    "head_chin_to_crown": {
        "size_m": 0.235, "sd_m": 0.011, "axis": "vertical", "anchor_ratio": 0.862,
        "label": "head height, chin to crown",
        "note": "Mark the WHOLE head including hair. Anchor is the chin.",
    },
    "face_chin_to_hairline": {
        "size_m": 0.185, "sd_m": 0.010, "axis": "vertical", "anchor_ratio": 0.862,
        "label": "face height, chin to hairline",
        "note": "Excludes hair — what a face segmenter usually returns. Anchor is the chin.",
    },
    "head_width": {
        "size_m": 0.152, "sd_m": 0.006, "axis": "horizontal", "anchor_ratio": 0.930,
        "label": "head breadth, ear to ear",
        "note": "Widest horizontal extent of the skull. Anchor is the segment midpoint.",
    },
    "interpupillary": {
        "size_m": 0.063, "sd_m": 0.0035, "axis": "horizontal", "anchor_ratio": 0.936,
        "label": "interpupillary distance, pupil to pupil",
        "note": "Mark the two pupil centres. Needs a frontal face; foreshortens badly in profile.",
    },
}

#: Assumed adult stature when the caller does not supply one, in metres.
DEFAULT_STATURE_M = 1.70

#: Eye height as a fraction of stature — the standing-adult constant behind the
#: horizontal metrics' anchor ratios. Exposed because it is the assumption most
#: worth overriding for a non-average subject.
EYE_HEIGHT_RATIO = 0.936


def face_metric_choices() -> list[str]:
    """Combo values for the face-metric widget, in registration order."""
    return list(FACE_METRICS)


@dataclass(slots=True)
class FaceScaleResult:
    """Outcome of one face measurement.

    ``camera_height`` is None when the geometry is degenerate or the inputs are
    unusable; ``reason`` then says why. ``camera_height_sd`` propagates the
    anthropometric spread only — it is NOT a total error bar, because the
    stature assumption and the pixel marking both contribute and neither has a
    defensible prior here.
    """

    camera_height: float | None
    camera_height_sd: float | None = None
    distance_m: float | None = None
    anchor_world: tuple[float, float, float] | None = None
    anchor_height_above_ground: float | None = None
    metric: str = ""
    real_size_m: float | None = None
    pixel_extent: float | None = None
    consistency: float = 0.0
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_height_m": self.camera_height,
            "camera_height_sd_m": self.camera_height_sd,
            "distance_m": self.distance_m,
            "anchor_world": list(self.anchor_world) if self.anchor_world else None,
            "anchor_height_above_ground_m": self.anchor_height_above_ground,
            "metric": self.metric,
            "real_size_m": self.real_size_m,
            "pixel_extent_px": self.pixel_extent,
            "consistency": self.consistency,
            "reason": self.reason,
        }


def _vertical_anchor(
    bottom_px: tuple[float, float],
    top_px: tuple[float, float],
    real_size_m: float,
    *,
    cam_to_world: Any,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> tuple[Any, float, float]:
    """Metric position of a vertical segment's BOTTOM point.

    Same two-ray least squares as ``solver.metric_height_from_reference``:
    ``alpha*r_b - beta*r_t = [0, -H, 0]`` fixes both depths, because the segment
    is known to be vertical and ``H`` long.

    It deliberately does NOT reuse that function, and the difference is the whole
    reason this helper exists. That one guards ``r_b[1] >= 0`` — "reference base
    is above the horizon" — because a ground object's base MUST look downward to
    meet the ground plane. A chin has no such obligation: point a camera at
    someone from below waist height and their chin is legitimately above the
    horizon while the geometry stays perfectly well posed. Importing that guard
    silently refused every low-camera face until the round-trip tests caught it.
    """
    np = _require_numpy()
    r_b = _ray_world(bottom_px[0], bottom_px[1], fx, fy, cx, cy, cam_to_world)
    r_t = _ray_world(top_px[0], top_px[1], fx, fy, cx, cy, cam_to_world)

    A = np.column_stack([r_b, -r_t])
    b = np.array([0.0, -float(real_size_m), 0.0])
    coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)
    alpha, beta = float(coeffs[0]), float(coeffs[1])
    residual = float(np.linalg.norm(A @ coeffs - b))

    if alpha <= 0 or beta <= 0:
        raise ValueError(
            "degenerate face geometry — the marked points do not describe a "
            "vertical segment in front of the camera")

    consistency = float(max(0.0, 1.0 - residual / max(float(real_size_m), 1e-3)))
    return alpha * r_b, float(alpha), consistency


def _horizontal_anchor(
    p0: tuple[float, float],
    p1: tuple[float, float],
    real_size_m: float,
    *,
    cam_to_world: Any,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> tuple[Any, float, float]:
    """Metric position of a horizontal segment's midpoint.

    Two rays subtend an angle theta; a segment of length L bridging them sits at
    ``d = (L/2) / tan(theta/2)`` along the bisector. Exact for a segment
    perpendicular to the bisector, which is the frontal-face case these metrics
    assume — a profile view foreshortens the segment and reads as farther away.
    """
    np = _require_numpy()
    r0 = _ray_world(p0[0], p0[1], fx, fy, cx, cy, cam_to_world)
    r1 = _ray_world(p1[0], p1[1], fx, fy, cx, cy, cam_to_world)

    cos_theta = float(np.clip(np.dot(r0, r1), -1.0, 1.0))
    theta = float(np.arccos(cos_theta))
    if theta < 1e-7:
        raise ValueError("marked points are coincident — no angle subtended")

    distance = (real_size_m / 2.0) / np.tan(theta / 2.0)
    bisector = r0 + r1
    norm = float(np.linalg.norm(bisector))
    if norm < 1e-9:
        raise ValueError("degenerate ray pair (points are diametrically opposed)")
    bisector = bisector / norm
    return bisector * distance, float(distance), theta


def camera_height_from_face(
    point_a: tuple[float, float],
    point_b: tuple[float, float],
    *,
    metric: str = "head_chin_to_crown",
    rotation: Any,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    stature_m: float = DEFAULT_STATURE_M,
    size_override_m: float | None = None,
) -> FaceScaleResult:
    """Metric camera height from one marked facial feature of known size.

    ``point_a``/``point_b`` are the feature's two endpoints in pixels. For a
    vertical metric order does not matter (the pair is sorted into top/bottom by
    image y); for a horizontal one they are the left/right extremes.

    ``rotation`` is the WORLD->CAM 3x3 — the same convention
    ``solver.apply_reference_scale`` passes, taken from the 4x4
    ``camera_view_matrix``'s rotation block rather than the transpose-ambiguous
    bare 3x3 (CLAUDE.md's view-matrix rule).

    Assumes an upright adult standing on the solve's ground plane. That
    assumption is the dominant error term and is recorded on the result rather
    than hidden.
    """
    np = _require_numpy()

    spec = FACE_METRICS.get(metric)
    if spec is None:
        return FaceScaleResult(
            camera_height=None, metric=metric,
            reason=f"unknown face metric '{metric}' (known: {', '.join(FACE_METRICS)})")

    real_size = float(size_override_m) if size_override_m else float(spec["size_m"])
    if real_size <= 0:
        return FaceScaleResult(camera_height=None, metric=metric,
                               reason="real size must be positive")
    if fx <= 0 or fy <= 0:
        return FaceScaleResult(camera_height=None, metric=metric,
                               reason="solve has no usable focal length")
    if stature_m <= 0:
        return FaceScaleResult(camera_height=None, metric=metric,
                               reason="stature must be positive")

    rotation = np.asarray(rotation, dtype=np.float64)
    cam_to_world = rotation.T
    anchor_height = float(spec["anchor_ratio"]) * float(stature_m)

    if spec["axis"] == "vertical":
        # Sort by image y: smaller y is higher in the world (image origin is
        # top-left), so the CROWN is point_top and the anchor is the chin.
        top_px, bottom_px = sorted((point_a, point_b), key=lambda p: p[1])
        pixel_extent = abs(float(bottom_px[1]) - float(top_px[1]))
        if pixel_extent < 1e-6:
            return FaceScaleResult(camera_height=None, metric=metric,
                                   reason="marked points have no vertical extent")

        try:
            anchor_world, _alpha, consistency = _vertical_anchor(
                bottom_px, top_px, real_size,
                cam_to_world=cam_to_world, fx=fx, fy=fy, cx=cx, cy=cy)
        except ValueError as exc:
            return FaceScaleResult(
                camera_height=None, metric=metric, real_size_m=real_size,
                pixel_extent=pixel_extent, reason=str(exc))
        distance = float(np.linalg.norm(anchor_world))
    else:
        left_px, right_px = sorted((point_a, point_b), key=lambda p: p[0])
        pixel_extent = float(
            np.hypot(right_px[0] - left_px[0], right_px[1] - left_px[1]))
        if pixel_extent < 1e-6:
            return FaceScaleResult(camera_height=None, metric=metric,
                                   reason="marked points are coincident")
        try:
            anchor_world, distance, _theta = _horizontal_anchor(
                left_px, right_px, real_size,
                cam_to_world=cam_to_world, fx=fx, fy=fy, cx=cx, cy=cy)
        except ValueError as exc:
            return FaceScaleResult(camera_height=None, metric=metric,
                                   real_size_m=real_size, pixel_extent=pixel_extent,
                                   reason=str(exc))
        # A horizontal metric has no internal redundancy to check, unlike the
        # vertical solve's least-squares residual. Report the honest value
        # rather than inventing agreement.
        consistency = 0.6

    camera_height = anchor_height - float(anchor_world[1])
    if not np.isfinite(camera_height) or camera_height <= 0:
        return FaceScaleResult(
            camera_height=None, metric=metric, real_size_m=real_size,
            distance_m=distance, pixel_extent=pixel_extent,
            anchor_world=tuple(float(v) for v in anchor_world),
            anchor_height_above_ground=anchor_height,
            reason="implied camera height is non-positive — the subject is likely "
                   "not standing on the solve's ground plane, or the marked "
                   "extent does not match the chosen metric")

    # Distance scales linearly with the assumed real size, and the anchor's
    # world Y with it, so the anthropometric SD propagates proportionally.
    rel_sd = float(spec["sd_m"]) / float(spec["size_m"])
    height_sd = abs(float(anchor_world[1])) * rel_sd

    return FaceScaleResult(
        camera_height=float(camera_height),
        camera_height_sd=float(height_sd),
        distance_m=float(distance),
        anchor_world=tuple(float(v) for v in anchor_world),
        anchor_height_above_ground=anchor_height,
        metric=metric,
        real_size_m=real_size,
        pixel_extent=float(pixel_extent),
        consistency=consistency,
    )
