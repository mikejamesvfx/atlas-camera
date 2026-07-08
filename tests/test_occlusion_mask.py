"""Tests for AtlasOcclusionMask — the primary-camera-coverage mask node.

Mirrors tests/test_add_patch_view.py's pattern: the heavy depth model is
monkeypatched with a synthetic ground-plane depth ramp so these tests exercise
the node's own logic (patch-camera orbit construction + primary-validity
projection test) without the [neural] extra or a model download.
"""

import pytest

from atlas_camera.comfy.nodes import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    AtlasOcclusionMask,
)


def test_node_registered_and_return_types():
    assert NODE_CLASS_MAPPINGS["AtlasOcclusionMask"] is AtlasOcclusionMask
    assert "AtlasOcclusionMask" in NODE_DISPLAY_NAME_MAPPINGS
    assert AtlasOcclusionMask.RETURN_TYPES == ("MASK", "MASK")


def test_input_widgets_mirror_add_patch_view_named_views():
    spec = AtlasOcclusionMask.INPUT_TYPES()
    assert set(spec["required"]) == {"solve", "target_image"}
    opt = spec["optional"]
    assert "right side view" in opt["patch_azimuth_view"][0]
    assert "front view" in opt["source_azimuth_view"][0]
    assert opt["source_azimuth_view"][1]["default"] == "front view"
    assert opt["angle_threshold"][1]["default"] == 90.0
    assert opt["dilate_px"][1]["default"] == 0


def _synthetic_primary(size=128):
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
        image_width=size, image_height=size, focal_length_mm=35.0,
        sensor_width_mm=36.0, fx_px=size * 0.98, fy_px=size * 0.98,
        cx_px=size / 2.0, cy_px=size / 2.0,
    )
    return AtlasSolve(camera=LatentCamera(intrinsics=intr, extrinsics=extr)), pivot, eye


def _patch_estimate_depth(monkeypatch, size=128):
    """Replace the depth model with a synthetic downward ground ramp."""
    np = pytest.importorskip("numpy")
    from dataclasses import dataclass

    @dataclass
    class _FakeDepth:
        depth: object
        is_metric: bool = True
        model_id: str = "fake"

    def fake(image_path, *, model_id=None, device=None):
        ramp = np.linspace(30.0, 5.0, size)[:, None] * np.ones((1, size))
        return _FakeDepth(depth=ramp.astype(np.float32))

    import atlas_camera.inference.depth_estimator as de
    monkeypatch.setattr(de, "estimate_depth", fake)


def test_zero_orbit_target_is_mostly_covered_by_primary(monkeypatch):
    torch = pytest.importorskip("torch")
    _patch_estimate_depth(monkeypatch)

    solve, _pivot, _eye = _synthetic_primary()
    target_img = torch.rand(1, 128, 128, 3, dtype=torch.float32)

    # Target view == source view (zero orbit): the target camera reproduces
    # the primary's own pose (orbit_camera's zero-delta contract), so nearly
    # every point the target sees, the primary already saw head-on.
    mask, coverage = AtlasOcclusionMask().generate(
        solve, target_img,
        patch_azimuth_view="front view", source_azimuth_view="front view",
    )
    # Only the 1px back-projection border (invalid normals) should be white.
    assert mask.mean().item() < 0.10
    assert coverage.mean().item() > 0.90


def test_facing_away_target_is_mostly_uncovered(monkeypatch):
    torch = pytest.importorskip("torch")
    _patch_estimate_depth(monkeypatch)

    solve, _pivot, _eye = _synthetic_primary()
    target_img = torch.rand(1, 128, 128, 3, dtype=torch.float32)

    # 180 degree orbit: the target camera looks back across the scene from
    # the far side, seeing surfaces largely behind/out-of-frame for the
    # primary — the mask should be majority white (primary can't cover it).
    mask, _coverage = AtlasOcclusionMask().generate(
        solve, target_img,
        patch_azimuth_view="back view", source_azimuth_view="front view",
    )
    assert mask.mean().item() > 0.5


def test_dilate_px_increases_white_area(monkeypatch):
    torch = pytest.importorskip("torch")
    _patch_estimate_depth(monkeypatch)

    solve, _pivot, _eye = _synthetic_primary()
    target_img = torch.rand(1, 128, 128, 3, dtype=torch.float32)

    kwargs = dict(patch_azimuth_view="front-right quarter view", source_azimuth_view="front view")
    mask0, _ = AtlasOcclusionMask().generate(solve, target_img, dilate_px=0, **kwargs)
    mask5, _ = AtlasOcclusionMask().generate(solve, target_img, dilate_px=5, **kwargs)
    assert mask5.sum().item() > mask0.sum().item()


def test_soft_edge_px_creates_gray_pixels(monkeypatch):
    torch = pytest.importorskip("torch")
    _patch_estimate_depth(monkeypatch)

    solve, _pivot, _eye = _synthetic_primary()
    target_img = torch.rand(1, 128, 128, 3, dtype=torch.float32)

    mask, _ = AtlasOcclusionMask().generate(
        solve, target_img,
        patch_azimuth_view="front-right quarter view", source_azimuth_view="front view",
        dilate_px=2, soft_edge_px=3,
    )
    arr = mask.numpy()
    assert ((arr > 0.02) & (arr < 0.98)).mean() > 0.05


def test_angle_threshold_affects_masking(monkeypatch):
    torch = pytest.importorskip("torch")
    _patch_estimate_depth(monkeypatch)

    solve, _pivot, _eye = _synthetic_primary()
    target_img = torch.rand(1, 128, 128, 3, dtype=torch.float32)

    kwargs = dict(patch_azimuth_view="front-right quarter view", source_azimuth_view="front view")
    mask_90, _ = AtlasOcclusionMask().generate(solve, target_img, angle_threshold=90.0, **kwargs)
    mask_30, _ = AtlasOcclusionMask().generate(solve, target_img, angle_threshold=30.0, **kwargs)
    # A tighter facing-angle gate can only mask MORE, never less.
    assert mask_30.sum().item() >= mask_90.sum().item()


def test_depth_shadow_widgets_registered():
    opt = AtlasOcclusionMask.INPUT_TYPES()["optional"]
    assert opt["occlusion_mode"][0] == ["simple", "depth_shadow"]
    assert opt["occlusion_mode"][1]["default"] == "simple"
    assert opt["primary_depth"][0] == "ATLAS_DEPTH_MAP"
    assert opt["depth_bias"][1]["default"] == 0.05


def _primary_depth_result(depth_map, size=128):
    from atlas_camera.inference.depth_estimator import DepthResult

    return DepthResult(
        depth=depth_map, is_metric=True, model_id="fake",
        image_width=size, image_height=size,
        near=float(depth_map.min()), far=float(depth_map.max()),
    )


def test_depth_shadow_mode_flags_points_hidden_behind_near_geometry(monkeypatch):
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    _patch_estimate_depth(monkeypatch)

    solve, _pivot, _eye = _synthetic_primary()
    target_img = torch.rand(1, 128, 128, 3, dtype=torch.float32)
    kwargs = dict(patch_azimuth_view="front view", source_azimuth_view="front view")

    # Primary shadow map: a very near constant surface (0.1m) right in front
    # of the lens — every real scene point is far behind it, so from the
    # primary's view EVERYTHING the target sees is in shadow. The simple mask
    # (zero orbit) is mostly black; depth_shadow must flip it mostly white.
    near_wall = np.full((128, 128), 0.1, dtype=np.float32)
    mask_simple, _ = AtlasOcclusionMask().generate(solve, target_img, **kwargs)
    mask_shadow, _ = AtlasOcclusionMask().generate(
        solve, target_img, occlusion_mode="depth_shadow",
        primary_depth=_primary_depth_result(near_wall), **kwargs)

    assert mask_simple.mean().item() < 0.10
    assert mask_shadow.mean().item() > 0.85
    assert mask_shadow.sum().item() > mask_simple.sum().item()


def test_depth_shadow_falls_back_to_simple_without_primary_depth(monkeypatch):
    torch = pytest.importorskip("torch")
    _patch_estimate_depth(monkeypatch)

    solve, _pivot, _eye = _synthetic_primary()
    target_img = torch.rand(1, 128, 128, 3, dtype=torch.float32)
    kwargs = dict(patch_azimuth_view="front view", source_azimuth_view="front view")

    mask_simple, _ = AtlasOcclusionMask().generate(solve, target_img, **kwargs)
    mask_fallback, _ = AtlasOcclusionMask().generate(
        solve, target_img, occlusion_mode="depth_shadow", primary_depth=None, **kwargs)
    assert torch.equal(mask_simple, mask_fallback)


def test_patch_view_override_matches_explicit_dropdowns(monkeypatch):
    """Wiring 📐 Extract Angle's patch_prompt into patch_view_override must
    produce the identical mask as setting the three dropdowns by hand."""
    torch = pytest.importorskip("torch")
    _patch_estimate_depth(monkeypatch)

    solve, _pivot, _eye = _synthetic_primary()
    target_img = torch.rand(1, 128, 128, 3, dtype=torch.float32)

    explicit, _ = AtlasOcclusionMask().generate(
        solve, target_img,
        patch_azimuth_view="back view", patch_elevation_view="elevated shot",
        patch_distance="wide shot", source_azimuth_view="front view")
    overridden, _ = AtlasOcclusionMask().generate(
        solve, target_img,
        patch_azimuth_view="front view",  # dropdowns say zero-orbit...
        source_azimuth_view="front view",
        patch_view_override="<sks> back view elevated shot wide shot")  # ...override wins
    assert torch.equal(explicit, overridden)


def test_passes_through_full_white_when_primary_has_no_focal():
    torch = pytest.importorskip("torch")
    from atlas_camera.core.schema import AtlasIntrinsics, AtlasSolve, LatentCamera

    intr = AtlasIntrinsics(image_width=128, image_height=128)
    solve = AtlasSolve(camera=LatentCamera(intrinsics=intr))
    target_img = torch.rand(1, 128, 128, 3, dtype=torch.float32)

    mask, coverage = AtlasOcclusionMask().generate(solve, target_img)
    assert mask.mean().item() == pytest.approx(1.0)
    assert coverage.mean().item() == pytest.approx(0.0)
