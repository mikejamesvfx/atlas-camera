"""Which pixels does a fitted plane actually explain?

WHY THIS MODULE EXISTS. ``plane_extraction.extract_planes_ransac`` computes a
per-plane inlier mask (``inl_local``, :mod:`plane_extraction` line ~248) and
keeps only its COUNT. So a serialized ``ransac_planes.json`` describes eight
planes and cannot say which pixels belong to any of them, and a plane rebuilt
from that record is a bare quad. Project the plate through it and every plane
receives everything behind it — which is precisely the smear seen when orbiting
the roundtripped DSC_2552 scene: the photograph pasted across a handful of flat
rectangles instead of cropped to the objects those rectangles stand for.

A ``ProjectionSource`` already has the mechanism to fix this. ``mask_b64`` is
honoured as a hard CUT by the viewport (``atlas_blockout.js``: a hand-authored
mask is a cut, not a soft band edge) and decoded by the headless renderer, and
``gather_scene_meshes`` already walks a source's own ``proxy_geometry``. What
was missing was the mask itself.

THE LOAD-BEARING PROPERTY IS EXCLUSIVITY. Two planes that both claim a pixel
put the same photograph on two surfaces at different depths, and an orbit shows
that as a doubled, sliding ghost. Assignment here is therefore nearest-wins with
an explicit unassigned label, never "closest plane anyway".

Host-agnostic: numpy only. No torch, no ComfyUI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

#: Mirrors ``plane_extraction``'s own inlier tolerance — ``max(0.15, 0.02 *
#: median_depth)``. Reusing the rule that FITTED the plane rather than inventing
#: a second one is the whole reason these masks agree with the extraction that
#: produced the records.
MIN_TOLERANCE_M = 0.15
TOLERANCE_DEPTH_FRACTION = 0.02

#: Label for a pixel no plane explains. Sky and unmodelled clutter land here,
#: and they must: a backdrop that absorbs everything left over is the same
#: smear wearing a different name.
UNASSIGNED = -1


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - guarded import
        raise RuntimeError(
            "atlas_camera.core.plane_masks requires numpy. Install with: "
            "pip install -e .[vision]") from exc
    return np


@dataclass(frozen=True)
class PlaneFrame:
    """A fitted plane's basis, centre and rectangle, recovered from a record."""

    name: str
    u: Any
    v: Any
    normal: Any
    centre: Any
    width: float
    height: float

    def tolerance(self, camera_position: Any) -> float:
        np = _require_numpy()
        depth = float(np.linalg.norm(
            np.asarray(self.centre, dtype=np.float64)
            - np.asarray(camera_position, dtype=np.float64)))
        return max(MIN_TOLERANCE_M, TOLERANCE_DEPTH_FRACTION * depth)


def plane_frames_from_primitives(primitives: Sequence[Any]) -> list[PlaneFrame]:
    """Recover plane frames from ``AtlasProxyPrimitive``-shaped records.

    ``depth_geometry.plane_transform`` writes the local axes as COLUMNS —
    ``(u, v, n, c)``. Reading them as rows yields a perfectly plausible
    orthonormal basis that is silently transposed, so the extent test then
    crops along the wrong axes and the masks look almost right.
    """
    np = _require_numpy()
    frames: list[PlaneFrame] = []
    for primitive in primitives or []:
        if getattr(primitive, "primitive_type", None) != "plane":
            continue
        matrix = np.asarray(getattr(primitive, "transform_matrix", None),
                            dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError(
                f"plane {getattr(primitive, 'name', '?')!r} has no 4x4 transform")
        dims = tuple(float(d) for d in (getattr(primitive, "dimensions", None) or ()))
        if len(dims) < 2:
            raise ValueError(
                f"plane {getattr(primitive, 'name', '?')!r} has no (width, height)")
        frames.append(PlaneFrame(
            name=str(getattr(primitive, "name", f"plane_{len(frames)}")),
            u=matrix[:3, 0], v=matrix[:3, 1], normal=matrix[:3, 2],
            centre=matrix[:3, 3], width=dims[0], height=dims[1],
        ))
    return frames


#: Two planes closer than this in normal, AND within the offset threshold, are
#: the same surface found twice. Mirrors
#: ``plane_extraction.PlaneRansacConfig.duplicate_angle_deg`` — the extractor
#: applies the rule at FIT time, this applies it at CONSUME time, because a
#: released side-car was written before the extractor learned the rule and its
#: transforms are bound to an append-only ledger that cannot be re-minted.
DUPLICATE_ANGLE_DEG = 20.0

#: The offset band for calling two planes the same surface. DELIBERATELY
#: looser than the inlier tolerance above: assignment asks "is this pixel ON
#: that plane", dedup asks "is this plane THAT plane", and the second question
#: tolerates a worse fit. Same rule the extractor applies at fit time,
#: ``max(0.3, 0.05 * depth)``.
DUPLICATE_MIN_OFFSET_M = 0.3
DUPLICATE_OFFSET_DEPTH_FRACTION = 0.05


def _plane_offset(np: Any, frame: PlaneFrame) -> float:
    return float(np.asarray(frame.centre, dtype=np.float64)
                 @ np.asarray(frame.normal, dtype=np.float64))


def find_duplicate_planes(frames: Sequence[PlaneFrame], *, camera_position: Any,
                          angle_deg: float = DUPLICATE_ANGLE_DEG) -> list[dict]:
    """Planes that are another plane found twice, in emission order.

    The offset test is the real guard: a stepped facade is two walls at the
    SAME orientation, and only the distance between them along their shared
    normal separates a genuine second surface from a re-fit of the first. The
    tolerance is the plane's own depth-scaled one, so it matches the band the
    masks are assigned with.

    The EARLIER plane wins. Extraction emits in descending inlier count, so the
    first of a pair is the better-supported fit — on DSC_2552 that is a 172,882
    inlier ground against its 6,000 inlier shadow.
    """
    np = _require_numpy()
    cam = np.asarray(camera_position, dtype=np.float64).reshape(3)
    cos_limit = np.cos(np.radians(float(angle_deg)))
    duplicates: list[dict] = []
    for index, frame in enumerate(frames):
        for earlier in frames[:index]:
            if any(d["name"] == earlier.name for d in duplicates):
                continue  # never fold a plane into one already folded away
            dot = abs(float(np.asarray(frame.normal, dtype=np.float64)
                            @ np.asarray(earlier.normal, dtype=np.float64)))
            offset = abs(_plane_offset(np, frame) - _plane_offset(np, earlier))
            depth = float(np.linalg.norm(
                np.asarray(earlier.centre, dtype=np.float64) - cam))
            offset_limit = max(DUPLICATE_MIN_OFFSET_M,
                               DUPLICATE_OFFSET_DEPTH_FRACTION * depth)
            if dot > cos_limit and offset < offset_limit:
                duplicates.append({
                    "name": frame.name,
                    "duplicate_of": earlier.name,
                    "angle_deg": float(np.degrees(np.arccos(min(1.0, dot)))),
                    "offset_m": offset,
                })
                break
    return duplicates


def assign_points_to_planes(points_world: Any, frames: Sequence[PlaneFrame], *,
                            camera_position: Any, valid: Any = None,
                            tolerance_m: float | None = None) -> tuple[Any, dict]:
    """Assign each world point to at most one plane. Returns ``(labels, report)``.

    ``labels`` is an integer array of plane indices, :data:`UNASSIGNED` where no
    plane explains the point. A point qualifies for a plane only if it is both
    within the plane's depth tolerance AND inside the plane's rectangle — the
    extent test is what stops a wall claiming everything coplanar with it clear
    across the frame, which is the "crop to the object" half of the problem.

    Ties go to the plane NEARER THE CAMERA, because that is what occludes.
    Breaking a tie by list order would let extraction order decide occlusion.
    """
    np = _require_numpy()
    pts = np.asarray(points_world, dtype=np.float64)
    flat = pts.reshape(-1, 3)
    n_pts = flat.shape[0]

    finite = np.isfinite(flat).all(axis=1)
    if valid is not None:
        finite &= np.asarray(valid, dtype=bool).reshape(-1)

    labels = np.full(n_pts, UNASSIGNED, dtype=np.int64)
    best = np.full(n_pts, np.inf, dtype=np.float64)
    contested = np.zeros(n_pts, dtype=np.int64)

    cam = np.asarray(camera_position, dtype=np.float64).reshape(3)
    order = sorted(range(len(frames)),
                   key=lambda i: float(np.linalg.norm(
                       np.asarray(frames[i].centre, dtype=np.float64) - cam)))

    for index in order:                       # nearest plane first, so an exact
        frame = frames[index]                 # tie is already held by the nearer
        tol = frame.tolerance(cam) if tolerance_m is None else float(tolerance_m)
        rel = flat - np.asarray(frame.centre, dtype=np.float64)
        dist = np.abs(rel @ np.asarray(frame.normal, dtype=np.float64))
        a = np.abs(rel @ np.asarray(frame.u, dtype=np.float64))
        b = np.abs(rel @ np.asarray(frame.v, dtype=np.float64))
        eligible = (finite & (dist <= tol)
                    & (a <= frame.width / 2.0) & (b <= frame.height / 2.0))
        contested += eligible.astype(np.int64)
        take = eligible & (dist < best)
        labels[take] = index
        best[take] = dist[take]

    n_invalid = int((~finite).sum())
    report = {
        "planes": len(frames),
        "assigned_px": int((labels != UNASSIGNED).sum()),
        "unassigned_px": int((labels == UNASSIGNED).sum()) - n_invalid,
        "invalid_px": n_invalid,
        # How often more than one plane was in range for the same pixel. Not an
        # error — but a scene where most pixels are contested has planes that
        # are duplicates of each other, and the number says so.
        "contested_px": int((contested > 1).sum()),
    }
    return labels.reshape(pts.shape[:-1]), report


def plane_pixel_masks(points_world: Any, frames: Sequence[PlaneFrame], *,
                      camera_position: Any, valid: Any = None,
                      tolerance_m: float | None = None) -> tuple[list, dict]:
    """One boolean mask per plane, in the raster of ``points_world``.

    ``points_world`` is ``(H, W, 3)`` — the plate's own raster, back-projected
    through the recovered camera. The masks partition it: no pixel appears in
    two, which is what keeps the plate off two surfaces at once.
    """
    np = _require_numpy()
    pts = np.asarray(points_world, dtype=np.float64)
    if pts.ndim != 3 or pts.shape[-1] != 3:
        raise ValueError(f"expected an HxWx3 world-point raster, got {pts.shape}")

    labels, report = assign_points_to_planes(
        pts, frames, camera_position=camera_position, valid=valid,
        tolerance_m=tolerance_m)

    masks = [labels == index for index in range(len(frames))]
    total = float(labels.size) or 1.0
    report["raster"] = {"height": int(pts.shape[0]), "width": int(pts.shape[1])}
    report["assigned_fraction"] = report["assigned_px"] / total
    # Fraction of the frame is misleading when half of it has no depth at all
    # (DSC_2552: 699,620 of 1,570,816 pixels invalid, so 43.6% "assigned" is
    # really 78.6% of everything that could have been). Report both.
    measurable = float(report["assigned_px"] + report["unassigned_px"]) or 1.0
    report["assigned_fraction_of_valid"] = report["assigned_px"] / measurable
    report["per_plane_px"] = {frames[i].name: int(m.sum())
                              for i, m in enumerate(masks)}
    # A plane that explains nothing is a finding, not a crash: it was fitted
    # against a different depth map than the one being assigned, or its record
    # and the depth disagree about scale.
    report["empty_planes"] = [frames[i].name for i, m in enumerate(masks)
                              if not bool(m.any())]
    return masks, report
