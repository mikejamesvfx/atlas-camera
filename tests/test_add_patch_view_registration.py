"""AtlasAddPatchView `camera_source=register_to_primary` wiring.

The core solver is pinned in test_patch_camera_registration.py; here the node
contract is what matters: widgets APPENDED last, declared_orbit byte-identical
to before, refusals fall back to the declared orbit with a reason, an accepted
registration replaces the patch camera and never touches the primary.
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("PIL")

from atlas_camera.comfy.nodes import AtlasAddPatchView  # noqa: E402
from atlas_camera.core.patch_camera_registration import PatchCameraRegistration  # noqa: E402

from test_add_patch_view import _patch_estimate_depth, _synthetic_primary  # noqa: E402


def _fake_moge(monkeypatch, *, with_points=True):
    """estimate_depth stub returning a MoGe-shaped DepthResult (points + intrinsics)."""
    from dataclasses import dataclass, field

    @dataclass
    class _R:
        depth: object
        points: object = None
        metadata: dict = field(default_factory=dict)
        is_metric: bool = True
        model_id: str = "Ruicheng/moge-2-vitl-normal"

    def fake(image_path, *, model_id=None, device=None, focal_px=None, **kw):
        h = w = 512
        ramp = np.linspace(30.0, 5.0, h)[:, None] * np.ones((1, w))
        pts = None
        if with_points and focal_px is None:
            uu, vv = np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float))
            pts = np.stack([(uu - 256) / 480 * ramp, (vv - 256) / 480 * ramp, ramp], -1)
        return _R(depth=ramp.astype(np.float32), points=pts,
                  metadata={"predicted_focal_px": 480.0, "predicted_fy_px": 480.0,
                            "predicted_cx_px": 256.0, "predicted_cy_px": 256.0,
                            "focal_source": "predicted"})

    import atlas_camera.inference.depth_estimator as de
    monkeypatch.setattr(de, "estimate_depth", fake)


def _fake_primary_depth():
    from types import SimpleNamespace
    ramp = np.linspace(30.0, 5.0, 512)[:, None] * np.ones((1, 512))
    return SimpleNamespace(depth=ramp.astype(np.float32), is_metric=True,
                           image_width=512, image_height=512, metadata={})


def test_widgets_are_appended_after_patch_mask():
    opt = list(AtlasAddPatchView.INPUT_TYPES()["optional"])
    i = opt.index("patch_mask")
    assert opt[i + 1:] == ["camera_source", "primary_image", "registration_min_inliers",
                           "registration_max_residual_m", "registration_max_deviation_deg",
                           "auto_flip_azimuth"]
    spec = AtlasAddPatchView.INPUT_TYPES()["optional"]["camera_source"]
    assert spec[0] == ["declared_orbit", "register_to_primary"]
    assert spec[1]["default"] == "declared_orbit"


def test_declared_orbit_default_records_source_and_nothing_else(monkeypatch):
    _patch_estimate_depth(monkeypatch)
    solve, _, _ = _synthetic_primary()
    img = torch.rand(1, 512, 512, 3)
    out, _report = AtlasAddPatchView().add_patch(solve, img, patch_azimuth_view="right side view",
                                           relief_grid=48)
    meta = out.projection_sources[0].metadata
    assert meta["camera_source"] == "declared_orbit"
    assert not any(k.startswith("registration_") for k in meta)


def test_register_without_primary_depth_falls_back_with_reason(monkeypatch):
    _fake_moge(monkeypatch)
    solve, pivot, eye = _synthetic_primary()
    img = torch.rand(1, 512, 512, 3)
    declared, _report = AtlasAddPatchView().add_patch(
        solve, img, patch_azimuth_view="right side view", relief_grid=48)
    out, _report = AtlasAddPatchView().add_patch(
        solve, img, patch_azimuth_view="right side view", relief_grid=48,
        camera_source="register_to_primary", primary_image=torch.rand(1, 512, 512, 3))
    src = out.projection_sources[0]
    assert src.metadata["camera_source"] == "register_to_primary"
    assert src.metadata["registration_accepted"] is False
    assert "primary_depth" in src.metadata["registration_fallback_reason"]
    # Same camera as the declared path.
    assert src.camera.extrinsics.camera_position == pytest.approx(
        declared.projection_sources[0].camera.extrinsics.camera_position)


def test_accepted_registration_replaces_patch_camera_not_primary(monkeypatch):
    _fake_moge(monkeypatch)
    import atlas_camera.core.patch_camera_registration as pcr
    from atlas_camera.core.camera_math import look_at_view_matrix

    solve, pivot, eye = _synthetic_primary()
    measured_eye = (4.0, 2.5, 3.0)
    view, _, _ = look_at_view_matrix(measured_eye, pivot)
    view = np.asarray(view)

    def fake_register(**kw):
        # The node must hand the core the primary WORLD points and the patch
        # OpenCV pointmap; both non-empty.
        assert np.isfinite(kw["primary_points_world"]).any()
        assert kw["patch_points_cam"].shape[-1] == 3
        assert set(kw["declared_view_matrices"]) == {"noflip", "flip"}
        return PatchCameraRegistration(
            accepted=True, reason="registered", view_matrix=view,
            camera_position=np.array(measured_eye), scale=1.7, n_matches=120,
            n_candidates=110, n_inliers=80, rms_m=0.12, reproj_rms_px=1.4,
            deviation_deg=6.0, deviation_m=1.2, flip_resolved=True)
    monkeypatch.setattr(pcr, "register_patch_camera", fake_register)

    img = torch.rand(1, 512, 512, 3)
    prim_view_before = solve.camera.extrinsics.camera_view_matrix
    out, _report = AtlasAddPatchView().add_patch(
        solve, img, patch_azimuth_view="right side view", relief_grid=48,
        camera_source="register_to_primary", primary_image=torch.rand(1, 512, 512, 3),
        primary_depth=_fake_primary_depth())
    src = out.projection_sources[0]
    m = src.metadata
    assert m["registration_accepted"] is True
    assert m["registration_n_inliers"] == 80 and m["flip_azimuth_resolved"] is True
    assert m["registration_scale"] == pytest.approx(1.7)
    assert m["patch_focal_px_predicted"] == pytest.approx(480.0)
    assert m["evidence_type"] == "generated"                     # firewall: still generated
    assert src.camera.extrinsics.camera_position == pytest.approx(measured_eye, abs=1e-6)
    assert src.camera.intrinsics.fx_px == pytest.approx(480.0)
    assert "REGISTERED" in src.camera.extrinsics.projection_convention
    # Primary untouched, on both the input and the output solve.
    assert solve.camera.extrinsics.camera_view_matrix == prim_view_before
    assert out.camera.extrinsics.camera_view_matrix == prim_view_before


def test_rejected_registration_keeps_declared_orbit_and_reports(monkeypatch):
    _fake_moge(monkeypatch)
    import atlas_camera.core.patch_camera_registration as pcr

    def fake_register(**kw):
        return PatchCameraRegistration(accepted=False, reason="12 inliers < 40",
                                       n_matches=30, n_candidates=25, n_inliers=12,
                                       rms_m=0.9, deviation_deg=40.0)
    monkeypatch.setattr(pcr, "register_patch_camera", fake_register)
    solve, pivot, eye = _synthetic_primary()
    img = torch.rand(1, 512, 512, 3)
    declared, _report = AtlasAddPatchView().add_patch(
        solve, img, patch_azimuth_view="right side view", relief_grid=48)
    out, _report = AtlasAddPatchView().add_patch(
        solve, img, patch_azimuth_view="right side view", relief_grid=48,
        camera_source="register_to_primary", primary_image=torch.rand(1, 512, 512, 3),
        primary_depth=_fake_primary_depth())
    src = out.projection_sources[0]
    assert src.metadata["registration_accepted"] is False
    assert src.metadata["registration_fallback_reason"] == "12 inliers < 40"
    assert src.metadata["registration_n_inliers"] == 12
    assert src.camera.extrinsics.camera_position == pytest.approx(
        declared.projection_sources[0].camera.extrinsics.camera_position)


def test_no_pointmap_from_backend_falls_back(monkeypatch):
    _fake_moge(monkeypatch, with_points=False)
    solve, _, _ = _synthetic_primary()
    img = torch.rand(1, 512, 512, 3)
    out, _report = AtlasAddPatchView().add_patch(
        solve, img, patch_azimuth_view="right side view", relief_grid=48,
        camera_source="register_to_primary", primary_image=torch.rand(1, 512, 512, 3),
        primary_depth=_fake_primary_depth())
    assert "no pointmap" in out.projection_sources[0].metadata["registration_fallback_reason"]


# --- the report is the point: a refusal must be visible AT THE NODE --------

def test_report_names_a_mis_wire_rather_than_hiding_it(monkeypatch):
    """Choosing register_to_primary and forgetting primary_depth is a WIRING
    mistake, not a measurement outcome. Before the report output it degraded
    to the declared orbit with nothing visible unless a health node was wired.
    """
    _fake_moge(monkeypatch)
    solve, _pivot, _eye = _synthetic_primary()
    img = torch.rand(1, 512, 512, 3)
    _out, report = AtlasAddPatchView().add_patch(
        solve, img, patch_azimuth_view="right side view", relief_grid=48,
        camera_source="register_to_primary",
        primary_image=torch.rand(1, 512, 512, 3))
    assert "REGISTRATION REFUSED" in report
    assert "primary_depth" in report
    assert "DECLARED" in report


def test_declared_orbit_report_says_no_measurement_was_attempted(monkeypatch):
    _fake_moge(monkeypatch)
    solve, _pivot, _eye = _synthetic_primary()
    img = torch.rand(1, 512, 512, 3)
    _out, report = AtlasAddPatchView().add_patch(
        solve, img, patch_azimuth_view="right side view", relief_grid=48)
    assert "no measurement was attempted" in report
    assert "REFUSED" not in report
