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
                           "auto_flip_azimuth",
                           # 2026-09-04: the ATLAS_CROP handle, and the two
                           # silhouette-tear thresholds a backmost layer needs
                           # to switch off. Appended last, defaults unchanged.
                           "depth_edge_rel", "max_edge_factor",
                           "sky_heuristic", "crop"]
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


# --- the crop handle: a crop's camera is KNOWN, so MoGe must not guess it ---

def test_crop_handle_beats_moge_prediction_for_the_registered_camera(monkeypatch):
    """MoGe predicts a free focal with a CENTRED principal point. A crop of
    the primary's own photograph has neither an unknown focal nor a centred
    centre, so the handle's intrinsics are the camera the accepted patch is
    stored with — MoGe's number stays only as a cross-check.
    """
    _fake_moge(monkeypatch)
    import atlas_camera.core.patch_camera_registration as pcr
    from atlas_camera.core.camera_math import look_at_view_matrix

    solve, pivot, _eye = _synthetic_primary()
    measured_eye = (4.0, 2.5, 3.0)
    view, _, _ = look_at_view_matrix(measured_eye, pivot)
    view = np.asarray(view)
    seen = {}

    def fake_register(**kw):
        seen["K"] = dict(kw["patch_intrinsics"])
        return PatchCameraRegistration(
            accepted=True, reason="registered", view_matrix=view,
            camera_position=np.array(measured_eye), scale=1.0, n_matches=200,
            n_candidates=180, n_inliers=90, rms_m=0.1, reproj_rms_px=1.1,
            deviation_deg=3.0, deviation_m=0.4, flip_resolved=False)
    monkeypatch.setattr(pcr, "register_patch_camera", fake_register)

    img = torch.rand(1, 192, 256, 3)
    out, _report = AtlasAddPatchView().add_patch(
        solve, img, patch_azimuth_view="right side view", relief_grid=48,
        crop={"empty": False, "x": 128, "y": 64, "width": 256, "height": 192,
              "gen_w": 256, "gen_h": 192},
        camera_source="register_to_primary",
        primary_image=torch.rand(1, 512, 512, 3),
        primary_depth=_fake_primary_depth())

    # The core got the CROP's camera for its reprojection diagnostic, not
    # MoGe's centred 480px guess.
    assert seen["K"]["fx"] == pytest.approx(500.0)
    assert seen["K"]["cx"] == pytest.approx(128.0)
    assert seen["K"]["cy"] == pytest.approx(192.0)
    m = out.projection_sources[0].metadata
    assert m["patch_intrinsics_source"] == "crop_handle"
    assert m["patch_focal_px_predicted"] == pytest.approx(480.0)   # cross-check kept
    got = out.projection_sources[0].camera.intrinsics
    assert got.fx_px == pytest.approx(500.0)
    assert got.cx_px == pytest.approx(128.0)
    assert got.cy_px == pytest.approx(192.0)


def test_without_the_crop_handle_moge_still_supplies_the_camera(monkeypatch):
    """The crop path must not change the full-frame novel-view behaviour: an
    unknown generated camera still takes MoGe's prediction."""
    _fake_moge(monkeypatch)
    import atlas_camera.core.patch_camera_registration as pcr
    from atlas_camera.core.camera_math import look_at_view_matrix

    solve, pivot, _eye = _synthetic_primary()
    view, _, _ = look_at_view_matrix((4.0, 2.5, 3.0), pivot)
    view = np.asarray(view)

    def fake_register(**kw):
        return PatchCameraRegistration(
            accepted=True, reason="registered", view_matrix=view,
            camera_position=np.array((4.0, 2.5, 3.0)), scale=1.0, n_matches=200,
            n_candidates=180, n_inliers=90, rms_m=0.1, reproj_rms_px=1.1,
            deviation_deg=3.0, deviation_m=0.4, flip_resolved=False)
    monkeypatch.setattr(pcr, "register_patch_camera", fake_register)

    out, _report = AtlasAddPatchView().add_patch(
        solve, torch.rand(1, 512, 512, 3), patch_azimuth_view="right side view",
        relief_grid=48, camera_source="register_to_primary",
        primary_image=torch.rand(1, 512, 512, 3),
        primary_depth=_fake_primary_depth())
    m = out.projection_sources[0].metadata
    assert m["patch_intrinsics_source"] == "moge_predicted"
    assert out.projection_sources[0].camera.intrinsics.fx_px == pytest.approx(480.0)
