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


def _resolve_modifier(meta, width: int, height: int):
    """EXIF -> (status, cam, lens, modifier). The modifier is initialised for
    ``width x height`` and can be asked for the correction over ANY region of
    that frame, including regions outside it — see :func:`extend_undistort_map`.

    Returns ``(status, cam, lens, mod)``; ``mod`` is None unless status is
    ``"applied"``. The three name-carrying misses keep the names they found so
    the caller's report can say how far the lookup got.
    """
    lensfunpy = _require_lensfunpy()

    if not meta.camera_model:
        return "no_lens_metadata", None, None, None
    db = lensfunpy.Database()
    cams = db.find_cameras(meta.camera_make or "", meta.camera_model,
                           loose_search=True)
    if not cams:
        return "no_profile_camera", None, None, None
    cam = cams[0]
    if not meta.lens_model:
        return "no_lens_metadata", cam, None, None
    lenses = db.find_lenses(cam, meta.lens_make or "", meta.lens_model,
                            loose_search=True)
    if not lenses:
        return "no_profile_lens", cam, None, None
    lens = lenses[0]

    focal = meta.focal_length_mm or getattr(lens, "min_focal", None) or 35.0
    aperture = meta.aperture or 8.0
    mod = lensfunpy.Modifier(lens, cam.crop_factor, width, height)
    mod.initialize(float(focal), float(aperture), 1000.0,
                   flags=lensfunpy.ModifyFlags.DISTORTION)
    return "applied", cam, lens, mod


def build_undistort_map(meta, width: int, height: int) -> UndistortResult:
    """EXIF make/model/lens -> lensfun profile -> remap grid.

    The correction maps to the rectilinear ideal at the same nominal focal and
    a centered principal point, so the EXIF focal stays valid for intrinsics
    (nominal-vs-calibrated residual is ~1-2%; the report notes it).
    """
    status, cam, lens, mod = _resolve_modifier(meta, width, height)
    if status != "applied":
        return UndistortResult(status,
                               cam_name=str(cam.model) if cam else None)
    coords = mod.apply_geometry_distortion()
    if coords is None:
        # Lens found but carries no distortion calibration at this focal.
        return UndistortResult("no_profile_lens", cam_name=str(cam.model),
                               lens_name=str(lens.model))

    focal = meta.focal_length_mm or getattr(lens, "min_focal", None) or 35.0
    distortion = {"lensfun_crop_factor": float(cam.crop_factor),
                  "lensfun_focal_mm": float(focal)}
    return UndistortResult("applied", cam_name=str(cam.model),
                           lens_name=str(lens.model), coords=coords,
                           distortion=distortion)


def extend_undistort_map(meta, width: int, height: int, coords,
                         overscan: tuple[int, int, int, int]):
    """The plate's own undistort grid, continued beyond the plate.

    ``overscan`` is ``(left, top, right, bottom)`` in undistorted pixels. The
    result is an ``(height+top+bottom, width+left+right, 2)`` grid over the
    RENDER frame: for each render pixel, where to sample in the distorted
    plate (plate pixel coordinates, so values outside ``[0, width-1]`` x
    ``[0, height-1]`` mean the lens never saw that pixel).

    Two things this deliberately does NOT do. It does not build a modifier for
    the bigger frame — lensfun normalises the radius by the frame it was given,
    so a modifier for ``width+2·left`` describes a different sensor and a
    different correction. It evaluates the SAME modifier over a region that
    starts at ``(-left, -top)``, which lensfun supports. And it does not trust
    that evaluation for the plate interior: lensfun's inverse-model iteration
    lands within ~0.5 px of itself between calls with different region origins
    (measured on an X-H2 / XF16-55 plate, 2026-09-11), so the interior is
    spliced from ``coords`` — the grid the plate was actually remapped with —
    and only the overscan band comes from the extended evaluation. A redistort
    map built on this grid therefore inverts exactly what was applied, and
    continues with the lens's own model where the plate ran out.
    """
    np = _require_numpy()
    left, top, right, bottom = (int(v) for v in overscan)
    if min(left, top, right, bottom) < 0:
        raise ValueError(f"overscan must be non-negative, got {overscan}")
    status, _cam, _lens, mod = _resolve_modifier(meta, width, height)
    if status != "applied":
        raise RuntimeError(
            f"cannot extend an undistort grid without a lens profile ({status})")
    ext = mod.apply_geometry_distortion(float(-left), float(-top),
                                        width + left + right,
                                        height + top + bottom)
    if ext is None:
        raise RuntimeError("lens profile carries no distortion at this focal")
    ext = np.asarray(ext, dtype=np.float32).copy()
    ext[top:top + height, left:left + width] = np.asarray(coords, dtype=np.float32)
    return ext


def measure_excursion(meta, width: int, height: int, *,
                      probe: float = 0.08, max_probe: float = 0.5
                      ) -> dict[str, float]:
    """How far, per edge, the rectilinear frame must extend beyond the plate
    for every distorted plate pixel to have a source. In undistorted pixels.

    Measured, not reasoned: the lens's own correction is evaluated over a
    probe band around the plate and the band is grown until the set of
    render pixels that sample inside the plate no longer touches the probe's
    boundary. A linear extrapolation from the edge gradient was tried first and
    is not good enough — the X-H2 / XF16-55 at 16 mm samples x=133.8 at the
    plate's left edge, but needs 219 undistorted pixels of overscan to reach
    distorted x=0, because the correction keeps steepening outside the frame.

    Returns ``{"left", "top", "right", "bottom"}`` as floats (pixels); a
    caller rounds them up into ``render.overscan``. All zero means the plate
    already covers itself (pincushion, or no correction at this focal).
    """
    np = _require_numpy()
    status, _cam, _lens, mod = _resolve_modifier(meta, width, height)
    if status != "applied":
        raise RuntimeError(
            f"cannot measure excursion without a lens profile ({status})")
    frac = float(probe)
    while True:
        L = T = R = B = int(round(frac * max(width, height)))
        ext = mod.apply_geometry_distortion(float(-L), float(-T),
                                            width + L + R, height + T + B)
        if ext is None:
            raise RuntimeError("lens profile carries no distortion at this focal")
        ext = np.asarray(ext)
        inside = ((ext[..., 0] >= 0) & (ext[..., 0] <= width - 1) &
                  (ext[..., 1] >= 0) & (ext[..., 1] <= height - 1))
        # Walk outward from the plate edge and stop at the first empty row or
        # column. Every polynomial lens model folds back at some radius, and
        # past the fold the probe samples the plate AGAIN — a ghost that would
        # otherwise read as "still needs more overscan" forever. Only the band
        # contiguous with the plate is the lens's real reach.
        cols = inside.any(axis=0)
        rows = inside.any(axis=1)

        def _reach(flags, start, step, limit):
            n = 0
            i = start
            while 0 <= i < flags.size and flags[i] and n < limit:
                n += 1
                i += step
            return n

        exc = {"left": float(_reach(cols, L - 1, -1, L)),
               "top": float(_reach(rows, T - 1, -1, T)),
               "right": float(_reach(cols, L + width, 1, R)),
               "bottom": float(_reach(rows, T + height, 1, B))}
        touches = (exc["left"] >= L or exc["right"] >= R or
                   exc["top"] >= T or exc["bottom"] >= B)
        if not touches:
            return exc
        frac *= 2.0
        if frac > max_probe:
            raise RuntimeError(
                f"excursion exceeds {max_probe:.0%} of the plate; refusing to "
                "guess — this is not a lens this pipeline should overscan for")


def _require_numpy():
    try:
        import numpy as np
        return np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Undistort grids require numpy. Install with: pip install -e .[raw]"
        ) from exc


def apply_undistort(arr, coords):
    """Remap one HxWx3 float32 array through the shared lensfun grid."""
    cv2 = _require_cv2()
    return cv2.remap(arr, coords, None, cv2.INTER_LANCZOS4,
                     borderMode=cv2.BORDER_REPLICATE)
