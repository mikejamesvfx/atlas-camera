"""Volume -> LAYERED RAYS: march a distance field along the camera's own rays.

The adapter that lets a volumetric amodal predictor feed Atlas's existing
hidden-geometry consumer. ``core/hidden_geometry.py`` eats a per-pixel
front-to-back depth stack (H, W, L); a predictor like VolFill emits a 256^3
truncated distance field. Marching that field along the RECOVERED camera's rays
produces exactly the stack, so ``select_hidden_surface`` and everything
calibrated around it applies unchanged.

WHY THIS BEATS MARCHING CUBES HERE
----------------------------------
An UNSIGNED field has no interior, so its level set at ``t > 0`` is a closed
shell offset ``+/-t`` either side of the true surface — a front wall and a back
wall about ``2t`` apart. Measured on a real prediction: 78.3% of rays crossed the
0.5 level exactly twice and 98.6% crossed an EVEN number of times, with a band
thickness of exactly 2.0 voxels.

Meshing that shell directly (marching cubes) therefore yields:
  * double the geometry,
  * no consistent orientation — half the triangles face away, which reads as
    "every other triangle has flipped normals",
  * and a systematic ~1 voxel bias, because samples sit half a wall off the
    true surface.

Marching the ray instead makes the pairing explicit: consecutive crossings are
the entry and exit of one wall, and their MIDPOINT is the surface. One sample per
real surface, ordered front-to-back, correctly placed, single-sided by
construction.

CONVENTIONS
- Rays are built in the FIELD's own camera frame (OpenCV: x right, y down,
  +z forward), because that is the frame a MoGe-conditioned volume lives in.
  The parameter IS forward depth z, so the output is directly comparable with a
  depth map — which is what ``register_layers_to_depth`` expects.
- Output ``(H, W, L)`` float32, front-to-back, 0 where a ray has no Nth surface.
  That is ``hidden_geometry``'s existing "0 == invalid" convention.
"""

from __future__ import annotations

from typing import Any


def _require_numpy():
    try:
        import numpy as np
        return np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Volume ray-marching requires numpy. Install with: pip install -e .[vision]"
        ) from exc


def march_layers(
    field: Any,
    bbox_min: Any,
    extent: Any,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    threshold: float = 0.5,
    max_layers: int = 6,
    step_voxels: float = 0.5,
    z_chunk: int = 96,
) -> tuple[Any, dict[str, Any]]:
    """March ``field`` along camera rays -> ``(H, W, L)`` layered forward depth.

    ``field`` is (R, R, R) with array axes **(z, y, x)** and values in distance
    units (a TUDF). ``bbox_min``/``extent`` place it in the camera frame, in
    metres. ``fx``/``fy``/``cx``/``cy`` are for the OUTPUT raster
    (``width`` x ``height``), which need not match the source plate — the volume
    is 256^3, so rendering rays at much higher resolution buys nothing.

    ``step_voxels`` sets the march step as a fraction of a voxel. It must stay
    below 1.0 or a wall (2 voxels thick) can be stepped over entirely.

    Returns ``(layers, stats)``. Pairs of crossings are collapsed to their
    midpoint, so each layer is a real surface rather than a shell wall.
    """
    np = _require_numpy()
    field = np.asarray(field, dtype=np.float32)
    if field.ndim != 3:
        raise ValueError(f"field must be (R, R, R), got {field.shape}")
    bbox_min = np.asarray(bbox_min, dtype=np.float64)
    extent = np.asarray(extent, dtype=np.float64)
    res = np.array(field.shape[::-1], dtype=np.float64)      # (x, y, z) counts
    voxel = extent / res
    if step_voxels >= 1.0:
        raise ValueError(
            "step_voxels must be < 1.0: the isosurface shell is ~2 voxels thick, "
            "and a coarser step can skip a wall and lose a surface entirely.")

    # Ray directions in the field's camera frame (OpenCV), parameterised by z so
    # the marcher's parameter IS forward depth.
    uu, vv = np.meshgrid(np.arange(width, dtype=np.float64),
                         np.arange(height, dtype=np.float64))
    dx = (uu - cx) / fx
    dy = (vv - cy) / fy

    # March only across the slab the volume actually occupies.
    z_lo = max(float(bbox_min[2]), 1e-6)
    z_hi = float(bbox_min[2] + extent[2])
    step = float(np.min(voxel)) * float(step_voxels)
    n_steps = int(np.ceil((z_hi - z_lo) / step)) + 1
    if n_steps < 2:
        raise ValueError("volume slab is degenerate along z")

    H, W = height, width
    layers = np.zeros((H, W, int(max_layers)), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.int16)
    # Carried across chunks so a wall spanning a chunk edge is not lost.
    prev_val = np.full((H, W), np.nan, dtype=np.float32)
    prev_z = np.zeros((H, W), dtype=np.float64)
    open_entry = np.zeros((H, W), dtype=np.float64)   # z where the wall was entered
    is_open = np.zeros((H, W), dtype=bool)
    n_cross = np.zeros((H, W), dtype=np.int32)

    lo_xyz = bbox_min
    inv_voxel = 1.0 / voxel

    for start in range(0, n_steps, int(z_chunk)):
        zs = z_lo + step * np.arange(start, min(start + z_chunk, n_steps),
                                     dtype=np.float64)
        if zs.size == 0:
            break
        # (K, H, W) sample positions
        z = zs[:, None, None]
        px = dx[None] * z
        py = dy[None] * z
        pz = np.broadcast_to(z, px.shape)

        gx = (px - lo_xyz[0]) * inv_voxel[0]
        gy = (py - lo_xyz[1]) * inv_voxel[1]
        gz = (pz - lo_xyz[2]) * inv_voxel[2]
        inside = ((gx >= 0) & (gx < field.shape[2] - 1) &
                  (gy >= 0) & (gy < field.shape[1] - 1) &
                  (gz >= 0) & (gz < field.shape[0] - 1))
        ix = np.clip(gx, 0, field.shape[2] - 1).astype(np.int32)
        iy = np.clip(gy, 0, field.shape[1] - 1).astype(np.int32)
        iz = np.clip(gz, 0, field.shape[0] - 1).astype(np.int32)
        vals = field[iz, iy, ix]                       # nearest sample
        # Outside the box is "far from any surface", never a crossing.
        vals = np.where(inside, vals, np.float32(np.inf))

        # inf - inf is a legitimate outcome outside the box; the result is
        # masked by `entering`/`leaving` either way, so silence the warning
        # rather than let shipped code emit noise.
        for k in range(vals.shape[0]):
            cur = vals[k]
            cz = zs[k]
            valid = np.isfinite(prev_val) & np.isfinite(cur)
            below_prev = valid & (prev_val <= threshold)
            below_cur = valid & (cur <= threshold)

            entering = below_cur & ~below_prev & ~is_open
            if entering.any():
                # Linear interpolation to the crossing z.
                with np.errstate(invalid="ignore"):
                    denom = prev_val - cur
                    frac = np.where(np.abs(denom) > 1e-12,
                                    (prev_val - threshold) / np.where(
                                        np.abs(denom) > 1e-12, denom, 1.0), 0.0)
                open_entry[entering] = (prev_z + frac * (cz - prev_z))[entering]
                is_open[entering] = True
                n_cross[entering] += 1

            leaving = below_prev & ~below_cur & is_open
            if leaving.any():
                with np.errstate(invalid="ignore"):
                    denom = prev_val - cur
                    frac = np.where(np.abs(denom) > 1e-12,
                                    (prev_val - threshold) / np.where(
                                        np.abs(denom) > 1e-12, denom, 1.0), 0.0)
                exit_z = (prev_z + frac * (cz - prev_z))
                # THE PAIRING: entry and exit are the two walls of one shell, so
                # the surface is the midpoint. This is what removes the +/-t bias
                # that meshing the shell directly leaves behind.
                mid = 0.5 * (open_entry + exit_z)
                room = leaving & (count < max_layers)
                if room.any():
                    yy, xx = np.nonzero(room)
                    layers[yy, xx, count[yy, xx]] = mid[yy, xx].astype(np.float32)
                    count[yy, xx] += 1
                is_open[leaving] = False
                n_cross[leaving] += 1

            prev_val = cur
            prev_z = np.full((H, W), cz, dtype=np.float64)

    stats = {
        "resolution": [int(width), int(height), int(max_layers)],
        "z_range_m": [z_lo, z_hi],
        "step_m": step,
        "steps": int(n_steps),
        "threshold": float(threshold),
        "rays_with_surface": int((count > 0).sum()),
        "mean_layers_per_hit": float(count[count > 0].mean()) if (count > 0).any() else 0.0,
        "max_layers_reached": int((count >= max_layers).sum()),
        # Even crossing counts are the double-wall signature; a high ODD share
        # means walls are being clipped by the slab edge or the step is too coarse.
        "odd_crossing_fraction": float(
            ((n_cross % 2) == 1).sum() / max((n_cross > 0).sum(), 1)),
    }
    return layers, stats
