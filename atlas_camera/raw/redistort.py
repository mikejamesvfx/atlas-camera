"""Inverse of the RAW undistort: a Nuke-ready ST map for putting renders back.

Undistorting a plate is a ONE-WAY DOOR unless the inverse ships with it. Atlas
undistorts on RAW import so the solve, the depth and every derived mesh live in
a rectilinear space — which is correct for geometry and wrong for delivery. The
comp has to land back on the ORIGINAL distorted plate, so the render must be
re-distorted, and that needs the inverse mapping as a float ST map.

DIRECTIONS, because they are easy to invert by accident:

``undistort.build_undistort_map`` returns ``coords`` (H, W, 2): for each pixel of
the UNDISTORTED output, where to sample in the DISTORTED original. That is what
``cv2.remap`` consumes to straighten the plate.

Nuke's STMap runs the other way. For each pixel of its OUTPUT it looks up a UV in
its INPUT. To re-distort a rectilinear render, the map lives in DISTORTED space
and holds, per distorted pixel, the UV of the matching UNDISTORTED pixel — the
inverse of ``coords``.

The inverse is computed by fixed-point iteration rather than scatter. ``coords``
is smooth and near-identity, so ``u <- u + (p - coords(u))`` converges in a few
steps to sub-pixel accuracy, and it produces no scatter holes to fill in — a
gap-filled scatter would quietly invent geometry at the frame edge, which is
precisely where distortion is largest.

CONVENTIONS
- ST maps are normalized [0, 1].
- Nuke's origin is BOTTOM-LEFT; image arrays are top-left. The V channel is
  flipped on write. Same class of detail as the glTF exporter's UV flip.
- Channel layout is (u, v, 0) so the file reads as RGB in any viewer.
- Pixels whose source falls outside the undistorted frame are marked in an
  optional alpha; leaving them unflagged is how a comp silently stretches an
  edge pixel across a corner.
"""

from __future__ import annotations

from typing import Any


def _require_numpy():
    try:
        import numpy as np
        return np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Redistort maps require numpy. Install with: pip install -e .[vision]"
        ) from exc


def invert_remap(coords: Any, *, iterations: int = 12,
                 tolerance: float = 1e-3) -> Any:
    """Invert an (H, W, 2) remap grid by fixed-point iteration.

    ``coords[y, x] = (sx, sy)`` means output pixel (x, y) samples input (sx, sy).
    Returns ``(inv, residual)`` where ``inv[y, x] = (ux, uy)`` is the output
    pixel that lands on input pixel (x, y) — i.e. the inverse mapping — and
    ``residual`` is the PER-PIXEL error magnitude, not a scalar.

    Per-pixel matters. A real lens correction samples from an INSET region (a
    5178-wide plate had coords spanning x in [25.9, 3849]), so distorted pixels
    beyond that inset have NO solution inside the frame and can never converge.
    Reducing to a scalar max makes those unsolvable-by-definition pixels
    masquerade as a broken solver — measured live 2026-08-15, where a perfectly
    good inversion reported a 104 px "residual" that was entirely corner pixels
    with no answer. Callers judge convergence over the solvable set.
    """
    np = _require_numpy()
    coords = np.asarray(coords, dtype=np.float32)
    if coords.ndim != 3 or coords.shape[2] != 2:
        raise ValueError(f"coords must be (H, W, 2), got {coords.shape}")
    h, w = coords.shape[:2]

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    target = np.stack([xx, yy], axis=-1)          # the distorted pixel we want
    guess = target.copy()                          # near-identity start

    def sample(grid, pts):
        """Bilinear sample of an (H, W, 2) grid at float (…, 2) positions."""
        gx = np.clip(pts[..., 0], 0, w - 1)
        gy = np.clip(pts[..., 1], 0, h - 1)
        x0 = np.floor(gx).astype(np.int32)
        y0 = np.floor(gy).astype(np.int32)
        x1 = np.minimum(x0 + 1, w - 1)
        y1 = np.minimum(y0 + 1, h - 1)
        fx = (gx - x0)[..., None]
        fy = (gy - y0)[..., None]
        return ((grid[y0, x0] * (1 - fx) + grid[y0, x1] * fx) * (1 - fy) +
                (grid[y1, x0] * (1 - fx) + grid[y1, x1] * fx) * fy)

    err = np.zeros_like(guess)
    for _ in range(int(iterations)):
        err = target - sample(coords, guess)
        guess = guess + err
        # Bound the walk: an unsolvable pixel would otherwise wander far out of
        # frame and drag the guess for its neighbours through the bilinear tap.
        np.clip(guess[..., 0], -w, 2 * w, out=guess[..., 0])
        np.clip(guess[..., 1], -h, 2 * h, out=guess[..., 1])
        if float(np.abs(err).max()) < tolerance:
            break
    residual = np.hypot(err[..., 0], err[..., 1])
    return guess, residual


def build_redistort_stmap(coords: Any, *, with_alpha: bool = True,
                          iterations: int = 12) -> tuple[Any, dict[str, Any]]:
    """``coords`` -> a Nuke STMap that RE-DISTORTS a rectilinear render.

    Returns ``(stmap, info)``. ``stmap`` is (H, W, 3) or (H, W, 4) float32 with
    channels (u, v, 0[, a]), normalized [0, 1], V flipped for Nuke's bottom-left
    origin. ``info`` carries the inversion residual in pixels and the fraction of
    pixels whose source lies outside the frame.

    Wire it in Nuke as::

        Read(render, rectilinear) -> STMap(uv = Read(this_exr)) -> distorted
    """
    np = _require_numpy()
    inv, residual_map = invert_remap(coords, iterations=iterations)
    h, w = inv.shape[:2]

    outside = ((inv[..., 0] < 0) | (inv[..., 0] > w - 1) |
               (inv[..., 1] < 0) | (inv[..., 1] > h - 1))
    # Convergence is only meaningful where a solution EXISTS.
    solvable = ~outside
    residual = (float(residual_map[solvable].max()) if solvable.any()
                else float("inf"))
    residual_p999 = (float(np.percentile(residual_map[solvable], 99.9))
                     if solvable.any() else float("inf"))

    u = inv[..., 0] / max(w - 1, 1)
    v = inv[..., 1] / max(h - 1, 1)
    v = 1.0 - v                                   # top-left array -> bottom-left Nuke

    chans = [u, v, np.zeros_like(u)]
    if with_alpha:
        chans.append((~outside).astype(np.float32))
    stmap = np.stack(chans, axis=-1).astype(np.float32)

    info = {
        "inversion_residual_px": residual,
        "inversion_residual_p999_px": residual_p999,
        "converged": bool(residual < 1e-2),
        "outside_fraction": float(outside.mean()),
        "width": int(w), "height": int(h),
        "origin": "bottom-left (Nuke)",
        "channels": "u,v,0" + (",alpha" if with_alpha else ""),
    }
    return stmap, info


def redistort_stmap_for_raw(path: str, *, half_size: bool = False
                            ) -> tuple[Any, dict[str, Any]]:
    """Build the ST map straight from a RAW file's own lens profile.

    Rebuilds the lensfun modifier from the file's EXIF rather than requiring the
    remap grid to be threaded through the pipeline, so this can run standalone
    against any RAW the importer would undistort.

    Raises RuntimeError when no lensfun profile matches — that is the SAME
    condition under which the importer skips undistortion, and in that case no
    redistort map is needed because the plate was never straightened.
    """
    from atlas_camera.raw.metadata import read_raw_metadata
    from atlas_camera.raw.undistort import build_undistort_map

    meta = read_raw_metadata(path)
    res = getattr(meta, "raw_width", None), getattr(meta, "raw_height", None)
    if not all(res):
        from atlas_camera.raw.pipeline import decode_raw
        linear, _ = decode_raw(path, half_size=half_size)
        h, w = linear.shape[:2]
    else:
        w, h = int(res[0]), int(res[1])
        if half_size:
            w, h = w // 2, h // 2

    und = build_undistort_map(meta, w, h)
    if und.coords is None:
        raise RuntimeError(
            f"No lensfun profile matched ({und.status}) — the plate is not "
            "undistorted, so no redistort map is required.")
    stmap, info = build_redistort_stmap(und.coords)
    info.update(status=und.status, camera=und.cam_name, lens=und.lens_name)
    return stmap, info


def redistort_stmap_for_import(result: Any) -> tuple[Any, dict[str, Any]]:
    """Build the ST map for an already-decoded :class:`RawImportResult`.

    Uses the result's OWN decoded dimensions so the map lines up with the plate
    that was actually written, and re-reads the lens profile from
    ``source_path``. Returns ``(None, info)`` when the import did not undistort —
    then the plate was never straightened and no inverse is owed.
    """
    from atlas_camera.raw.metadata import read_raw_metadata
    from atlas_camera.raw.undistort import build_undistort_map

    if not getattr(result, "undistort_applied", False):
        return None, {"status": getattr(result, "undistort_status", "unknown"),
                      "reason": "plate was not undistorted; no redistort needed"}
    src = getattr(result, "source_path", None)
    if not src:
        return None, {"status": "no_source_path",
                      "reason": "cannot rebuild the lens profile without the RAW"}
    meta = read_raw_metadata(str(src))
    und = build_undistort_map(meta, int(result.width), int(result.height))
    if und.coords is None:
        return None, {"status": und.status,
                      "reason": "lens profile no longer resolves"}
    stmap, info = build_redistort_stmap(und.coords)
    info.update(status=und.status, camera=und.cam_name, lens=und.lens_name,
                source_raw=str(src))
    return stmap, info


def write_stmap_exr(stmap: Any, path: str) -> str:
    """Write a float32 ST map EXR (no colour transform — this is DATA, not pixels).

    An ST map must never pass through a colour pipeline: the channels are
    coordinates, so any transfer curve corrupts them. Written as raw linear
    float with no colourspace metadata attached.

    FULL float32, deliberately — these files are large (~200 MB at 5K) and half
    float looks like the obvious saving. It is not: half carries a 10-bit
    mantissa, so a UV in [0, 1] resolves to ~0.0005, which on a 5178 px plate is
    a **2.5 px** positional error. That is far coarser than the 0.004 px the
    inversion achieves and would waste the accuracy the map exists to carry.
    """
    np = _require_numpy()
    arr = np.asarray(stmap, dtype=np.float32)
    try:
        import OpenImageIO as oiio
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Writing ST map EXRs requires OpenImageIO. Install with: "
            "pip install -e .[oiio]") from exc

    h, w, c = arr.shape
    spec = oiio.ImageSpec(w, h, c, "float")
    spec.attribute("compression", "zip")
    spec.attribute("atlas:content", "redistort_stmap")
    spec.attribute("atlas:origin", "bottom-left")
    out = oiio.ImageOutput.create(path)
    if out is None:  # pragma: no cover
        raise RuntimeError(f"OpenImageIO cannot write {path}")
    out.open(path, spec)
    out.write_image(arr)
    out.close()
    return path
