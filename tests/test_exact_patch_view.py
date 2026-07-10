"""Tests for the exact-angle patch channel (the render-conditioned patch
loop's registration contract): AtlasBlockoutViewport's 5th appended
`patch_exact` STRING output (📐's RAW measured orbit floats, before
named-view snapping) and the `exact_view_override` input on
AtlasAddPatchView / AtlasOcclusionMask that reproduces the identical pose.
"""

import json

import pytest

from atlas_camera.comfy.nodes import (
    _ATLAS_BLOCKOUT_CACHE,
    AtlasAddPatchView,
    AtlasBlockoutViewport,
    AtlasOcclusionMask,
    _parse_exact_view,
)
from atlas_camera.core.camera_math import look_at_view_matrix, orbit_camera


def _solve(width=800, height=600, fx=700.0):
    from atlas_camera.core.schema import (
        AtlasExtrinsics, AtlasIntrinsics, AtlasSolve, LatentCamera,
    )
    eye = (1.5, 2.0, 3.0)
    target = (0.5, 0.0, -8.0)
    view, world, rot3 = look_at_view_matrix(eye, target)
    extr = AtlasExtrinsics(
        camera_position=eye, camera_rotation_matrix=rot3,
        camera_world_matrix=world, camera_view_matrix=view,
    )
    intr = AtlasIntrinsics(
        image_width=width, image_height=height, focal_length_mm=35.0,
        sensor_width_mm=36.0, fx_px=fx, fy_px=fx,
        cx_px=width / 2.0, cy_px=height / 2.0,
    )
    return AtlasSolve(camera=LatentCamera(intrinsics=intr, extrinsics=extr))


def _patch_estimate_depth(monkeypatch, size=512):
    np = pytest.importorskip("numpy")
    from dataclasses import dataclass

    @dataclass
    class _FakeDepth:
        depth: object
        is_metric: bool = True
        model_id: str = "fake"

    def fake(image_path, *, model_id=None, device=None, focal_px=None):
        ramp = np.linspace(30.0, 5.0, size)[:, None] * np.ones((1, size))
        return _FakeDepth(depth=ramp.astype(np.float32))

    import atlas_camera.inference.depth_estimator as de
    monkeypatch.setattr(de, "estimate_depth", fake)


# ---------------------------------------------------------------- parsing

def test_parse_exact_view_accepts_key_value_any_order_and_commas():
    assert _parse_exact_view(
        "azimuth_deg=12.5 elevation_deg=-4.0 distance_scale=1.1"
    ) == (12.5, -4.0, 1.1)
    assert _parse_exact_view(
        "distance_scale=0.9, azimuth_deg=-170, elevation_deg=2"
    ) == (-170.0, 2.0, 0.9)


def test_parse_exact_view_rejects_incomplete_or_garbage():
    assert _parse_exact_view("") is None
    assert _parse_exact_view("azimuth_deg=10 elevation_deg=5") is None
    assert _parse_exact_view("<sks> front view eye-level shot medium shot") is None
    assert _parse_exact_view("azimuth_deg=abc elevation_deg=1 distance_scale=1") is None


# ------------------------------------------------- viewport patch_exact

def test_viewport_gains_patch_exact_output_appended():
    assert AtlasBlockoutViewport.RETURN_TYPES[10] == "STRING"
    assert AtlasBlockoutViewport.RETURN_NAMES[10] == "patch_exact"
    # Existing outputs keep their slot indices (saved workflows link by index).
    assert AtlasBlockoutViewport.RETURN_NAMES[:10] == (
        "shaded", "depth", "normal", "mask", "path_frames", "camera_path",
        "patch_azimuth_view", "patch_elevation_view", "patch_distance",
        "patch_prompt")


def test_patch_exact_emits_raw_floats():
    torch = pytest.importorskip("torch")
    from atlas_camera.comfy.nodes import _solve_fingerprint
    _ATLAS_BLOCKOUT_CACHE.clear()
    solve = _solve()
    image = torch.rand(1, 600, 800, 3)
    client_data = json.dumps({"patch_angle": {
        "azimuth_view": "front-right quarter view",
        "elevation_view": "elevated shot",
        "distance_view": "wide shot",
        "prompt": "<sks> front-right quarter view elevated shot wide shot",
        "raw": {"d_azimuth_deg": 38.2, "d_elevation_deg": 24.0,
                "distance_scale": 1.6},
        "fingerprint": _solve_fingerprint(solve, image),
    }})
    out = AtlasBlockoutViewport().render(
        solve, image, resolution=768, client_data=client_data,
        unique_id="test_exact_pass")
    exact = out["result"][10]
    assert _parse_exact_view(exact) == (38.2, 24.0, 1.6)


def test_patch_exact_falls_back_to_named_delta_without_raw():
    torch = pytest.importorskip("torch")
    from atlas_camera.comfy.nodes import _solve_fingerprint
    _ATLAS_BLOCKOUT_CACHE.clear()
    solve = _solve()
    image = torch.rand(1, 600, 800, 3)
    client_data = json.dumps({"patch_angle": {
        "azimuth_view": "front-right quarter view",
        "elevation_view": "elevated shot",
        "distance_view": "wide shot",
        "fingerprint": _solve_fingerprint(solve, image),
    }})
    out = AtlasBlockoutViewport().render(
        solve, image, resolution=768, client_data=client_data,
        unique_id="test_exact_fallback")
    # named-view deltas relative to the assumed front/eye-level source
    assert _parse_exact_view(out["result"][10]) == (45.0, 30.0, 1.8)


def test_patch_exact_default_is_zero_orbit():
    torch = pytest.importorskip("torch")
    _ATLAS_BLOCKOUT_CACHE.clear()
    out = AtlasBlockoutViewport().render(
        _solve(), torch.rand(1, 600, 800, 3), resolution=768, client_data="",
        unique_id="test_exact_default")
    assert _parse_exact_view(out["result"][10]) == (0.0, 0.0, 1.0)


# ---------------------------------------------- AddPatchView exact override

def test_add_patch_exact_override_places_camera_at_raw_delta(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    _patch_estimate_depth(monkeypatch)
    from atlas_camera.core.camera_math import ground_lookat_pivot

    solve = _solve(width=512, height=512, fx=500.0)
    patch_image = torch.rand(1, 512, 512, 3)
    (out,) = AtlasAddPatchView().add_patch(
        solve, patch_image,
        patch_azimuth_view="back view",  # dropdowns must lose to exact
        patch_view_override="<sks> right side view eye-level shot medium shot",
        exact_view_override=(
            "azimuth_deg=12.5 elevation_deg=-4.0 distance_scale=1.1"),
        geometry_source="own_depth",
    )
    assert len(out.projection_sources) == 1
    src = out.projection_sources[0]
    assert (src.azimuth_deg, src.elevation_deg, src.distance_scale) == (
        12.5, -4.0, 1.1)
    assert src.metadata["source"] == "exact_render_patch"
    assert src.metadata["exact_view_override"].startswith("azimuth_deg=12.5")
    expected = orbit_camera(
        solve.camera.extrinsics,
        ground_lookat_pivot(solve.camera.extrinsics),
        d_azimuth_deg=12.5, d_elevation_deg=-4.0, distance_scale=1.1)
    got = src.camera.extrinsics
    for r_exp, r_got in zip(expected.camera_view_matrix,
                            got.camera_view_matrix):
        for a, b in zip(r_exp, r_got):
            assert abs(a - b) < 1e-9


def test_add_patch_exact_override_errors_loudly(monkeypatch):
    torch = pytest.importorskip("torch")
    _patch_estimate_depth(monkeypatch)
    with pytest.raises(ValueError, match="patch_exact"):
        AtlasAddPatchView().add_patch(
            _solve(512, 512, 500.0), torch.rand(1, 512, 512, 3),
            exact_view_override="totally not an exact view string")


def test_occlusion_mask_exact_matches_equivalent_named_view(monkeypatch):
    """The never-drift contract, extended to the exact channel: an exact
    string equal to a named view's own delta must produce the identical
    mask that the named view produces."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("numpy")
    _patch_estimate_depth(monkeypatch)
    solve = _solve(512, 512, 500.0)
    target = torch.rand(1, 512, 512, 3)
    named = AtlasOcclusionMask().generate(
        solve, target, patch_azimuth_view="front-right quarter view",
        patch_elevation_view="eye-level shot", patch_distance="medium shot")
    exact = AtlasOcclusionMask().generate(
        solve, target, patch_azimuth_view="back view",  # must be ignored
        exact_view_override=(
            "azimuth_deg=45.0 elevation_deg=0.0 distance_scale=1.0"))
    assert torch.equal(named[0], exact[0])
    assert torch.equal(named[1], exact[1])
