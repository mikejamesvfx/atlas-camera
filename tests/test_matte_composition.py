"""The unseen matte has to say WHY it matted a pixel out.

`primary_camera_validity_mask` ORs six terms and returns one bool array, so a
patch that comes back with a hole in the middle of its own fill tells you
nothing about which test produced it. Measured on the sea-cliff castle
2026-09-04: stripping the matte dropped the fillable residual 9,241 -> 5,051 px,
and the matte's black core matched the residual exactly -- the geometry was
present and the matte hid it. Narrowing that to one term took reading the source
and eliminating the other five by hand, and the answer still needed depth maps
that are not saved anywhere.

So the terms are counted where it matters (inside the hole the patch was
generated for) and reported. `depth_ratio` is the diagnostic that separates the
two remaining candidates: it is `point_depth / sampled` for the depth-shadow
test, so a value clustered just under 1.0 means the two depths disagree about
SCALE, while a bimodal one means the fill invented near content in the core.
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")


def _flat_scene(n=16, depth=10.0):
    """Points on a fronto-parallel plane in front of a camera at the origin."""
    ys, xs = np.mgrid[0:n, 0:n]
    fx = fy = 100.0
    cx = cy = n / 2.0
    X = (xs - cx) / fx * depth
    Y = -(ys - cy) / fy * depth
    Z = -np.full((n, n), depth)
    pts = np.stack([X, Y, Z], axis=-1)
    normals = np.zeros_like(pts)
    normals[..., 2] = 1.0
    view = np.eye(4)
    return pts, normals, view, dict(primary_fx=fx, primary_fy=fy,
                                    primary_cx=cx, primary_cy=cy,
                                    primary_width=n, primary_height=n)


def test_terms_are_returned_and_sum_to_the_mask():
    from atlas_camera.core.depth_geometry import primary_camera_validity_mask

    pts, normals, view, K = _flat_scene()
    ok = np.ones(pts.shape[:2], bool)

    mask, terms = primary_camera_validity_mask(
        pts, ok, normals, ok, primary_view_matrix=view, return_terms=True, **K)

    assert set(terms) >= {"behind", "out_of_frame", "grazing", "shadowed",
                          "invalid_depth", "invalid_normal", "depth_ratio"}
    combined = np.zeros_like(mask)
    for key in ("behind", "out_of_frame", "grazing", "shadowed",
                "invalid_depth", "invalid_normal"):
        combined |= terms[key]
    np.testing.assert_array_equal(combined, mask)


def test_the_shadow_term_fires_only_when_the_point_is_behind_what_is_stored():
    from atlas_camera.core.depth_geometry import primary_camera_validity_mask

    pts, normals, view, K = _flat_scene(depth=10.0)
    ok = np.ones(pts.shape[:2], bool)
    n = pts.shape[0]

    # Left half occluded by something at 4 m, right half sees the plane itself.
    dm = np.full((n, n), 10.0)
    dm[:, : n // 2] = 4.0

    _mask, terms = primary_camera_validity_mask(
        pts, ok, normals, ok, primary_view_matrix=view,
        primary_depth_map=dm, return_terms=True, **K)

    assert terms["shadowed"][:, : n // 2].all(), "hidden half must be shadowed"
    assert not terms["shadowed"][:, n // 2:].any(), "visible half must not be"


def test_depth_ratio_names_a_scale_disagreement():
    """The diagnostic that separates the two live candidates. A patch whose
    depth is uniformly shrunk reads as 'in front of' the primary everywhere,
    and the ratio says so with one number instead of a guess."""
    from atlas_camera.core.depth_geometry import primary_camera_validity_mask

    pts, normals, view, K = _flat_scene(depth=10.0)
    ok = np.ones(pts.shape[:2], bool)
    dm = np.full(pts.shape[:2], 10.0)

    _m, agree = primary_camera_validity_mask(
        pts, ok, normals, ok, primary_view_matrix=view,
        primary_depth_map=dm, return_terms=True, **K)
    # The same geometry pulled 40% nearer -- the shape of a wrong `scale`.
    _m2, shrunk = primary_camera_validity_mask(
        pts * 0.6, ok, normals, ok, primary_view_matrix=view,
        primary_depth_map=dm, return_terms=True, **K)

    assert np.nanmedian(agree["depth_ratio"]) == pytest.approx(1.0, abs=0.02)
    assert np.nanmedian(shrunk["depth_ratio"]) == pytest.approx(0.6, abs=0.02)
    # ...and the shrunk one stops being shadowed, which is the failure seen live
    assert not shrunk["shadowed"].any()


def test_returning_terms_does_not_change_the_mask():
    from atlas_camera.core.depth_geometry import primary_camera_validity_mask

    pts, normals, view, K = _flat_scene()
    ok = np.ones(pts.shape[:2], bool)
    dm = np.full(pts.shape[:2], 8.0)

    plain = primary_camera_validity_mask(
        pts, ok, normals, ok, primary_view_matrix=view,
        primary_depth_map=dm, **K)
    withterms, _t = primary_camera_validity_mask(
        pts, ok, normals, ok, primary_view_matrix=view,
        primary_depth_map=dm, return_terms=True, **K)

    np.testing.assert_array_equal(plain, withterms)


def test_the_patch_reports_what_its_matte_did(monkeypatch):
    """End to end: the numbers have to reach the artist's report, or narrowing
    this needs source-reading and a hand rebuild again."""
    pytest.importorskip("torch")
    pytest.importorskip("PIL")
    import torch

    from atlas_camera.comfy.nodes import AtlasAddPatchView

    from test_add_patch_view import _patch_estimate_depth, _synthetic_primary

    _patch_estimate_depth(monkeypatch)
    solve, _pivot, _eye = _synthetic_primary()

    from types import SimpleNamespace
    ramp = np.linspace(30.0, 5.0, 512)[:, None] * np.ones((1, 512))
    depth = SimpleNamespace(depth=ramp.astype(np.float32), is_metric=True,
                            image_width=512, image_height=512, metadata={})

    out, report = AtlasAddPatchView().add_patch(
        solve, torch.rand(1, 512, 512, 3), patch_azimuth_view="right side view",
        geometry_source="own_depth", relief_grid=48, mask_unseen_only=True,
        primary_depth=depth)

    meta = out.projection_sources[-1].metadata
    assert "matte_paint_fraction" in meta
    for term in ("behind", "out_of_frame", "grazing", "shadowed",
                 "invalid_depth", "invalid_normal"):
        assert f"matte_{term}_px" in meta, term
    assert "matte_depth_ratio_median" in meta
    assert "matte" in report.lower()
