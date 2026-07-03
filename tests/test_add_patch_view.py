"""Tests for AtlasAddPatchView — the multi-angle patch projection node.

The heavy depth model is monkeypatched with a synthetic ground-plane depth so
the test exercises the node's own logic (patch-camera orbit construction +
ProjectionSource wiring) without the [neural] extra or a model download. Guarded
by importorskip so it skips cleanly where torch/numpy aren't installed.
"""

import math

import pytest

from atlas_camera.comfy.nodes import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    AtlasAddPatchView,
)


def test_node_registered_and_returns_solve():
    assert NODE_CLASS_MAPPINGS["AtlasAddPatchView"] is AtlasAddPatchView
    assert "AtlasAddPatchView" in NODE_DISPLAY_NAME_MAPPINGS
    assert AtlasAddPatchView.RETURN_TYPES == ("ATLAS_SOLVE",)


def test_input_widgets_expose_angle_controls():
    spec = AtlasAddPatchView.INPUT_TYPES()
    assert set(spec["required"]) == {"solve", "patch_image"}
    opt = spec["optional"]
    assert opt["azimuth_deg"][1]["default"] == 35.0
    assert opt["elevation_deg"][1]["default"] == 0.0
    assert opt["distance_scale"][1]["default"] == 1.0


def _synthetic_primary():
    from atlas_camera.core.camera_math import look_at_view_matrix
    from atlas_camera.core.schema import (
        AtlasExtrinsics,
        AtlasIntrinsics,
        AtlasSolve,
        LatentCamera,
    )

    pivot = (0.0, 0.0, 10.0)
    eye = (0.0, 2.0, 0.0)
    view, world, rot3 = look_at_view_matrix(eye, pivot)
    extr = AtlasExtrinsics(
        camera_position=eye,
        camera_rotation_matrix=rot3,
        camera_world_matrix=world,
        camera_view_matrix=view,
    )
    intr = AtlasIntrinsics(
        image_width=512, image_height=512, focal_length_mm=35.0,
        sensor_width_mm=36.0, fx_px=500.0, fy_px=500.0, cx_px=256.0, cy_px=256.0,
    )
    return AtlasSolve(camera=LatentCamera(intrinsics=intr, extrinsics=extr)), pivot, eye


def _patch_estimate_depth(monkeypatch):
    """Replace the depth model with a synthetic downward ground ramp."""
    np = pytest.importorskip("numpy")
    from dataclasses import dataclass

    @dataclass
    class _FakeDepth:
        depth: object
        is_metric: bool = True
        model_id: str = "fake"

    def fake(image_path, *, model_id=None, device=None):
        h = w = 512
        ramp = np.linspace(30.0, 5.0, h)[:, None] * np.ones((1, w))
        return _FakeDepth(depth=ramp.astype(np.float32))

    import atlas_camera.inference.depth_estimator as de
    monkeypatch.setattr(de, "estimate_depth", fake)


def test_add_patch_orbits_camera_and_appends_source(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("PIL")
    _patch_estimate_depth(monkeypatch)

    solve, pivot, eye = _synthetic_primary()
    patch_img = torch.rand(1, 512, 512, 3, dtype=torch.float32)

    (out,) = AtlasAddPatchView().add_patch(
        solve, patch_img, azimuth_deg=35.0, elevation_deg=0.0,
        distance_scale=1.0, name="patch_right", relief_grid=48,
    )

    assert len(out.projection_sources) == 1
    src = out.projection_sources[0]
    assert src.name == "patch_right"
    assert src.azimuth_deg == 35.0
    assert src.image_b64 and src.image_b64.startswith("data:image/jpeg;base64,")
    assert any(p.primitive_type == "mesh" for p in src.proxy_geometry)

    # Patch camera orbited around the pivot: radius preserved, height preserved
    # (pure azimuth), and re-aimed at the pivot (pivot in front → cam-space z<0).
    r_prim = math.dist(eye, pivot)
    r_patch = math.dist(src.camera.extrinsics.camera_position, pivot)
    assert r_patch == pytest.approx(r_prim, abs=1e-3)
    assert src.camera.extrinsics.camera_position[1] == pytest.approx(eye[1], abs=1e-3)


def test_add_patch_does_not_mutate_input_solve(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("PIL")
    _patch_estimate_depth(monkeypatch)

    solve, _pivot, _eye = _synthetic_primary()
    patch_img = torch.rand(1, 256, 384, 3, dtype=torch.float32)

    (out,) = AtlasAddPatchView().add_patch(solve, patch_img, name="patch_left", relief_grid=48)

    assert len(solve.projection_sources) == 0      # input untouched (deep-copied)
    assert len(out.projection_sources) == 1
    # Patch intrinsics follow the patch image resolution, not the primary's.
    pintr = out.projection_sources[0].camera.intrinsics
    assert pintr.image_width == 384
    assert pintr.image_height == 256


def test_add_patch_passes_through_when_primary_has_no_focal(monkeypatch):
    torch = pytest.importorskip("torch")
    from atlas_camera.core.schema import AtlasIntrinsics, AtlasSolve, LatentCamera

    # No fx_px on the primary → cannot back-project a patch; return unchanged.
    intr = AtlasIntrinsics(image_width=512, image_height=512)
    solve = AtlasSolve(camera=LatentCamera(intrinsics=intr))
    patch_img = torch.rand(1, 512, 512, 3, dtype=torch.float32)

    (out,) = AtlasAddPatchView().add_patch(solve, patch_img)
    assert out is solve
    assert len(out.projection_sources) == 0
