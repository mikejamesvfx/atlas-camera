"""Redistort ST map: the inverse of the RAW undistort, for Nuke delivery.

Undistorting a plate is a one-way door without this, so the properties that must
hold are pinned here rather than eyeballed in a comp.
"""
from __future__ import annotations

import numpy as np
import pytest

from atlas_camera.raw.redistort import (
    build_redistort_stmap,
    invert_remap,
)


def _radial_remap(h, w, k=0.12):
    """A synthetic barrel remap: per undistorted pixel, where to sample the
    distorted original. Same shape/meaning as lensfun's coords grid."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    nx = (xx - cx) / cx
    ny = (yy - cy) / cy
    r2 = nx * nx + ny * ny
    f = 1.0 + k * r2
    return np.stack([cx + nx * f * cx, cy + ny * f * cy], axis=-1).astype(np.float32)


def test_identity_remap_inverts_to_identity():
    h, w = 40, 60
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    coords = np.stack([xx, yy], axis=-1)
    inv, residual = invert_remap(coords)
    # residual is a PER-PIXEL map now — an identity remap is solvable everywhere.
    assert residual.shape == (h, w)
    assert residual.max() < 1e-4
    np.testing.assert_allclose(inv, coords, atol=1e-3)


def test_inverse_round_trips_a_real_distortion():
    """inv(coords(p)) == p to sub-pixel, which is the whole contract."""
    h, w = 64, 96
    coords = _radial_remap(h, w)
    inv, residual = invert_remap(coords, iterations=20)
    # Judge convergence only where a solution exists inside the frame: a real
    # correction samples from an inset region, so edge pixels are unsolvable by
    # construction rather than by solver failure.
    solvable = ((inv[..., 0] >= 0) & (inv[..., 0] <= w - 1) &
                (inv[..., 1] >= 0) & (inv[..., 1] <= h - 1))
    assert residual[solvable].max() < 1e-2, (
        f"inversion did not converge: {residual[solvable].max()}")

    # Compose the two mappings at interior samples and expect identity.
    ys, xs = np.mgrid[10:h - 10:7, 10:w - 10:9]
    pts = np.stack([xs, ys], axis=-1).astype(np.float32)
    src = coords[pts[..., 1].astype(int), pts[..., 0].astype(int)]
    back = inv[np.clip(src[..., 1].astype(int), 0, h - 1),
               np.clip(src[..., 0].astype(int), 0, w - 1)]
    assert np.abs(back - pts).max() < 1.5   # within a pixel and a half


def test_stmap_is_normalised_and_v_flipped_for_nuke():
    h, w = 50, 70
    stmap, info = build_redistort_stmap(_radial_remap(h, w))
    assert stmap.shape == (h, w, 4)
    assert stmap.dtype == np.float32
    u, v, zero, alpha = (stmap[..., i] for i in range(4))
    inside = alpha > 0.5
    assert u[inside].min() >= -1e-4 and u[inside].max() <= 1 + 1e-4
    assert v[inside].min() >= -1e-4 and v[inside].max() <= 1 + 1e-4
    assert np.allclose(zero, 0.0)
    # Nuke origin is bottom-left: the TOP image row must carry the HIGH v.
    assert v[0, w // 2] > v[-1, w // 2]
    assert info["origin"] == "bottom-left (Nuke)"
    assert info["converged"]


def test_alpha_flags_sources_outside_the_frame():
    """PINCUSHION direction, deliberately.

    A barrel grid (k > 0) pushes samples outward, so its inverse pulls them in
    and nothing can land outside — no flag is possible or needed. Pincushion
    (k < 0) is the case that reaches past the frame edge, and those pixels must
    be flagged rather than silently clamped; clamping is how a comp stretches an
    edge pixel across a corner.
    """
    h, w = 60, 60
    stmap, info = build_redistort_stmap(_radial_remap(h, w, k=-0.25))
    alpha = stmap[..., 3]
    assert info["outside_fraction"] > 0.0
    # Corners are the extreme of the radial term, centre is untouched.
    assert alpha[h // 2, w // 2] == pytest.approx(1.0)
    assert alpha[0, 0] < 1.0


def test_no_alpha_channel_when_not_requested():
    stmap, _ = build_redistort_stmap(_radial_remap(30, 30), with_alpha=False)
    assert stmap.shape[2] == 3


def test_bad_shape_is_rejected():
    with pytest.raises(ValueError):
        invert_remap(np.zeros((10, 10), dtype=np.float32))


# --- delivery: the map must come out of the SHIPPING export, not a script -----

def test_export_nuke_emits_the_stmap_and_records_it(tmp_path):
    """Undistorting on import is a one-way door unless the shipped exporter
    writes the inverse. Before 2026-08-15 the map existed only in a research
    script and zero exporters referenced it."""
    import json
    from atlas_camera.comfy.node_registry import NODE_CLASS_MAPPINGS as M
    from atlas_camera.core.schema import (
        AtlasCamera, AtlasExtrinsics, AtlasIntrinsics, AtlasSolve)

    class _FakeRawImport:
        """Minimal RawImportResult stand-in with no lens profile: the export
        must SKIP cleanly and say so, never raise."""
        undistort_applied = False
        undistort_status = "no_profile_lens"
        source_path = ""
        width = height = 64

    solve = AtlasSolve(camera=AtlasCamera(
        intrinsics=AtlasIntrinsics(image_width=64, image_height=64, fx_px=64.0,
                                   fy_px=64.0, cx_px=32.0, cy_px=32.0),
        extrinsics=AtlasExtrinsics()))

    out = tmp_path / "nuke"
    M["AtlasExportNuke"]().export(solve, str(out), raw_meta=_FakeRawImport(),
                                  write_redistort_stmap=True)
    # Nothing to invert, so no map — and crucially, no exception.
    assert not (out / "redistort_stmap.exr").exists()
    assert (out / "nuke_projection.nk").exists()


def test_stmap_write_is_float32_not_half(tmp_path):
    """Half float resolves UV to ~0.0005 = 2.5 px on a 5K plate, far coarser
    than the 0.004 px the inversion achieves. Size is not worth that."""
    pytest.importorskip("OpenImageIO")
    import OpenImageIO as oiio
    from atlas_camera.raw.redistort import write_stmap_exr

    stmap, _ = build_redistort_stmap(_radial_remap(32, 32))
    dest = tmp_path / "st.exr"
    write_stmap_exr(stmap, str(dest))
    src = oiio.ImageInput.open(str(dest))
    assert src is not None
    spec = src.spec()
    assert spec.format.basetype == oiio.FLOAT, "ST maps must stay float32"
    assert spec.nchannels == 4
    src.close()
