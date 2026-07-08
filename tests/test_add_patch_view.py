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


def test_input_widgets_expose_named_view_controls():
    spec = AtlasAddPatchView.INPUT_TYPES()
    assert set(spec["required"]) == {"solve", "patch_image"}
    opt = spec["optional"]
    # Named views match the ComfyUI-qwenmultiangle / LoRA options exactly.
    assert "right side view" in opt["patch_azimuth_view"][0]
    assert "front view" in opt["source_azimuth_view"][0]
    assert opt["source_azimuth_view"][1]["default"] == "front view"
    assert "eye-level shot" in opt["patch_elevation_view"][0]
    assert "medium shot" in opt["patch_distance"][0]


def test_absolute_view_maps_to_relative_orbit_delta():
    # The LoRA angle is absolute (subject-relative). Orbit applied = patch - source.
    az = AtlasAddPatchView._AZIMUTH_VIEWS
    assert az["right side view"] - az["front view"] == 90.0
    # front-right quarter from a front source = +45 orbit
    assert az["front-right quarter view"] - az["front view"] == 45.0
    el = AtlasAddPatchView._ELEVATION_VIEWS
    assert el["elevated shot"] - el["eye-level shot"] == 30.0


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

    # Source = front view, patch = right side view (90° absolute) → +90° orbit.
    (out,) = AtlasAddPatchView().add_patch(
        solve, patch_img,
        patch_azimuth_view="right side view", patch_elevation_view="eye-level shot",
        source_azimuth_view="front view", name="patch_right", relief_grid=48,
    )

    assert len(out.projection_sources) == 1
    src = out.projection_sources[0]
    assert src.name == "patch_right"
    assert src.azimuth_deg == 90.0                         # patch − source orbit delta
    assert src.image_b64 and src.image_b64.startswith("data:image/jpeg;base64,")
    assert any(p.primitive_type == "mesh" for p in src.proxy_geometry)

    # Patch camera orbited around the pivot: radius preserved, height preserved
    # (eye-level→eye-level, pure azimuth), re-aimed at the pivot (pivot in front).
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

    (out,) = AtlasAddPatchView().add_patch(
        solve, patch_img, patch_azimuth_view="left side view", name="patch_left", relief_grid=48,
    )

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


def _decode_matte(mask_b64):
    import base64
    import io

    import numpy as np
    from PIL import Image
    return np.asarray(Image.open(io.BytesIO(
        base64.b64decode(mask_b64.split(",", 1)[1]))), dtype=np.float32) / 255.0


def test_unseen_matte_embedded_and_scales_with_orbit(monkeypatch):
    """mask_unseen_only embeds a matte of where the PRIMARY can't project at
    the patch view: near-zero for a same-view patch (primary sees it all),
    majority for a 180-degree patch (primary sees almost none of it)."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("PIL")
    _patch_estimate_depth(monkeypatch)

    solve, _pivot, _eye = _synthetic_primary()
    patch_img = torch.rand(1, 512, 512, 3, dtype=torch.float32)

    (same,) = AtlasAddPatchView().add_patch(
        solve, patch_img, patch_azimuth_view="front view",
        source_azimuth_view="front view", relief_grid=48, unseen_dilate_px=0)
    (back,) = AtlasAddPatchView().add_patch(
        solve, patch_img, patch_azimuth_view="back view",
        source_azimuth_view="front view", relief_grid=48, unseen_dilate_px=0)

    m_same = _decode_matte(same.projection_sources[0].mask_b64)
    m_back = _decode_matte(back.projection_sources[0].mask_b64)
    # Same-view patch: primary covers essentially everything (only the
    # back-projection border rim is invalid).
    assert m_same.mean() < 0.15
    # Far-side patch: substantially more unseen area than the same view.
    # (The absolute fraction depends on scene geometry — frustum-only
    # invalidity on a synthetic ground ramp is modest; the depth-shadow term
    # via primary_depth is what catches true hidden surfaces on real scenes.)
    assert m_back.mean() > 2 * m_same.mean()
    assert m_back.mean() > 0.08


def test_unseen_matte_dilation_and_opt_out(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("PIL")
    _patch_estimate_depth(monkeypatch)
    solve, _pivot, _eye = _synthetic_primary()
    patch_img = torch.rand(1, 512, 512, 3, dtype=torch.float32)

    (d0,) = AtlasAddPatchView().add_patch(
        solve, patch_img, patch_azimuth_view="front-right quarter view",
        relief_grid=48, unseen_dilate_px=0)
    (d16,) = AtlasAddPatchView().add_patch(
        solve, patch_img, patch_azimuth_view="front-right quarter view",
        relief_grid=48, unseen_dilate_px=16)
    assert _decode_matte(d16.projection_sources[0].mask_b64).sum() \
        >= _decode_matte(d0.projection_sources[0].mask_b64).sum()

    (off,) = AtlasAddPatchView().add_patch(
        solve, patch_img, patch_azimuth_view="front-right quarter view",
        relief_grid=48, mask_unseen_only=False)
    assert off.projection_sources[0].mask_b64 is None


def test_scale_registers_against_primary_overlap(monkeypatch):
    """When primary_depth is wired, the patch's metric scale is REGISTERED to
    the primary through the overlap, not independently ground-fit: a same-pose
    patch whose raw depth is exactly 2x the primary's must come out at scale
    0.5 (closed-form: with z_cam=0 at zero orbit, s = m / z_p)."""
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    from dataclasses import dataclass

    from atlas_camera.inference.depth_estimator import DepthResult

    solve, _pivot, _eye = _synthetic_primary()
    h = w = 512
    primary_ramp = (np.linspace(30.0, 5.0, h)[:, None] * np.ones((1, w))).astype(np.float32)

    @dataclass
    class _FakeDepth:
        depth: object
        is_metric: bool = True
        model_id: str = "fake"

    def fake(image_path, *, model_id=None, device=None):
        return _FakeDepth(depth=primary_ramp * 2.0)  # patch depth = 2x primary

    import atlas_camera.inference.depth_estimator as de
    monkeypatch.setattr(de, "estimate_depth", fake)

    # Primary metric map delivered via the shared-depth input. Its own
    # ground fit must come out at 1.0 for the closed-form expectation --
    # monkeypatch estimate_ground_scale to isolate the registration math.
    import atlas_camera.core.relief_mesh as rm
    real_egs = rm.estimate_ground_scale
    monkeypatch.setattr(rm, "estimate_ground_scale",
                        lambda *a, **k: (1.0, {"reason": "test"}))

    primary_dr = DepthResult(depth=primary_ramp, is_metric=True, model_id="fake",
                             image_width=w, image_height=h,
                             near=5.0, far=30.0)
    patch_img = torch.rand(1, h, w, 3, dtype=torch.float32)

    (out,) = AtlasAddPatchView().add_patch(
        solve, patch_img, patch_azimuth_view="front view",
        source_azimuth_view="front view", relief_grid=48,
        primary_depth=primary_dr, unseen_dilate_px=0)

    meta = out.projection_sources[0].metadata
    assert meta["scale_source"] == "primary_registration"
    assert meta["scale"] == pytest.approx(0.5, rel=0.02)


def test_exclude_mask_removes_patch_sky_geometry(monkeypatch):
    """A SAM mask of the PATCH image's sky must keep those pixels out of the
    patch mesh entirely -- hallucinated near-depth sky otherwise triangulates
    into geometry bulging toward the camera (found live)."""
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    _patch_estimate_depth(monkeypatch)

    solve, _pivot, _eye = _synthetic_primary()
    patch_img = torch.rand(1, 512, 512, 3, dtype=torch.float32)
    sky = torch.zeros(1, 512, 512, dtype=torch.float32)
    sky[0, :200, :] = 1.0  # top 200 rows are "sky"

    (plain,) = AtlasAddPatchView().add_patch(
        solve, patch_img, patch_azimuth_view="front-right quarter view",
        relief_grid=48)
    (masked,) = AtlasAddPatchView().add_patch(
        solve, patch_img, patch_azimuth_view="front-right quarter view",
        relief_grid=48, exclude_mask=sky)

    def n_verts(s):
        mesh = next(p for p in s.projection_sources[0].proxy_geometry
                    if p.primitive_type == "mesh")
        return mesh.metadata["n_vertices"]
    assert n_verts(masked) < n_verts(plain)


def _solve_with_geometry():
    """Primary solve carrying one PROXY_ROLE primitive (a stand-in for the
    scene's derived/band geometry)."""
    from atlas_camera.core.proxy_geometry import PROXY_ROLE
    from atlas_camera.core.schema import AtlasProxyPrimitive

    solve, pivot, eye = _synthetic_primary()
    solve.projection_scene.proxy_geometry.append(AtlasProxyPrimitive(
        name="projection_ground", primitive_type="plane",
        dimensions=(20.0, 20.0, 0.0),
        transform_matrix=((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 5.0), (0, 0, 0, 1)),
        metadata={"role": PROXY_ROLE},
    ))
    return solve, pivot, eye


def test_reuse_scene_projects_onto_existing_geometry_without_depth(monkeypatch):
    """Default reuse_scene mode: the patch derives NO geometry — it reuses
    copies of the scene's own primitives (already in the primary's world by
    construction) and never invokes the depth model at all."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("PIL")

    def boom(*a, **k):
        raise AssertionError("estimate_depth must NOT run in reuse_scene mode")
    import atlas_camera.inference.depth_estimator as de
    monkeypatch.setattr(de, "estimate_depth", boom)

    solve, _pivot, _eye = _solve_with_geometry()
    patch_img = torch.rand(1, 256, 256, 3, dtype=torch.float32)
    (out,) = AtlasAddPatchView().add_patch(
        solve, patch_img, patch_azimuth_view="front-right quarter view")

    src = out.projection_sources[0]
    assert src.metadata["scale_source"] == "reuse_scene"
    assert src.metadata["n_reused_primitives"] == 1
    assert len(src.proxy_geometry) == 1
    assert src.proxy_geometry[0].name.endswith("projection_ground")
    assert src.proxy_geometry[0].primitive_type == "plane"
    # The ORIGINAL solve geometry is untouched (deep copies).
    assert solve.projection_scene.proxy_geometry[0].name == "projection_ground"


def test_reuse_scene_falls_back_to_own_depth_without_geometry(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("PIL")
    _patch_estimate_depth(monkeypatch)

    solve, _pivot, _eye = _synthetic_primary()  # no geometry anywhere
    patch_img = torch.rand(1, 512, 512, 3, dtype=torch.float32)
    (out,) = AtlasAddPatchView().add_patch(
        solve, patch_img, patch_azimuth_view="front-right quarter view",
        relief_grid=48)
    src = out.projection_sources[0]
    assert src.metadata["geometry_fallback"] == "no scene geometry to reuse"
    assert src.metadata["scale_source"] in ("ground_fit", "primary_registration")
    assert any(p.primitive_type == "mesh" for p in src.proxy_geometry)


def test_reuse_scene_splat_matte_scales_with_orbit(monkeypatch):
    """The reuse-mode matte comes from forward-splatting the PRIMARY's real
    metric points into the patch view: a same-pose patch is almost fully
    covered (tiny unseen), a far-side patch mostly uncovered."""
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    from atlas_camera.inference.depth_estimator import DepthResult

    def boom(*a, **k):
        raise AssertionError("estimate_depth must NOT run in reuse_scene mode")
    import atlas_camera.inference.depth_estimator as de
    monkeypatch.setattr(de, "estimate_depth", boom)
    import atlas_camera.core.relief_mesh as rm
    monkeypatch.setattr(rm, "estimate_ground_scale",
                        lambda *a, **k: (1.0, {"reason": "test"}))

    solve, _pivot, _eye = _solve_with_geometry()
    h = w = 512
    ramp = (np.linspace(30.0, 5.0, h)[:, None] * np.ones((1, w))).astype(np.float32)
    primary_dr = DepthResult(depth=ramp, is_metric=True, model_id="fake",
                             image_width=w, image_height=h, near=5.0, far=30.0)
    patch_img = torch.rand(1, h, w, 3, dtype=torch.float32)

    (same,) = AtlasAddPatchView().add_patch(
        solve, patch_img, patch_azimuth_view="front view",
        primary_depth=primary_dr, unseen_dilate_px=0)
    (back,) = AtlasAddPatchView().add_patch(
        solve, patch_img, patch_azimuth_view="back view",
        primary_depth=primary_dr, unseen_dilate_px=0)

    m_same = _decode_matte(same.projection_sources[0].mask_b64)
    m_back = _decode_matte(back.projection_sources[0].mask_b64)
    assert m_same.mean() < 0.2
    assert m_back.mean() > 2 * m_same.mean()
