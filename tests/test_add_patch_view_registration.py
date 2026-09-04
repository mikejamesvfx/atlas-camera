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
                           "sky_heuristic",
                           # 2026-09-04: the scale gate. After the BOOLEAN
                           # it follows and before `crop`, a link input
                           # that occupies no widget slot -- positionally
                           # this is still an append.
                           "scale_max_rel_iqr",
                           # 2026-09-04: the ground cross-check, the gate
                           # that sees what dispersion cannot.
                           "scale_max_ground_disagreement",
                           # 2026-09-05: the sibling check, the first gate
                           # that judges a patch against anything but
                           # itself.
                           "scale_max_sibling_disagreement",
                           "scale_min_siblings", "crop"]
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


# --- the SCALE gate -----------------------------------------------------
# The pose has been gated since register_to_primary landed. The scale never
# was: solve_scale_from_primary takes a median, so it hands back a confident
# number whether its samples agreed exactly or not at all, and the node adopted
# it either way. Castle ROI 5 is the case -- 0.273 against its neighbour's
# 0.645 on the same row at the same distance, ground floating 0.78 m, 5% of its
# own hole painted.

def _run_with_fit(monkeypatch, scale, rel_iqr, **kw):
    _patch_estimate_depth(monkeypatch)
    import atlas_camera.comfy.nodes_geometry as ng
    monkeypatch.setattr(
        ng, "solve_scale_from_primary",
        lambda *a, **k: (scale, {"n_samples": 400, "scale_rel_iqr": rel_iqr,
                                 "scale_rel_mad": 0.0}))
    solve, _p, _e = _synthetic_primary()
    out, report = AtlasAddPatchView().add_patch(
        solve, torch.rand(1, 512, 512, 3), patch_azimuth_view="right side view",
        relief_grid=48, geometry_source="own_depth",
        primary_depth=_fake_primary_depth(), **kw)
    return out.projection_sources[0].metadata, report


def test_the_gate_is_armed_by_default_at_the_measured_threshold():
    """1.0 comes off the sweep in test_patch_scale_occlusion_bias: the spread
    reaches 0.75 on fits that are still exact, and 1.2 on the first wrong one.
    Anything at or below 0.75 refuses good fits; anything above 1.2 lets the
    first bad one through."""
    spec = AtlasAddPatchView.INPUT_TYPES()["optional"]["scale_max_rel_iqr"]
    assert spec[1]["default"] == 1.0


def test_a_broken_fit_is_refused_without_anyone_asking(monkeypatch):
    meta, report = _run_with_fit(monkeypatch, 0.3, rel_iqr=9.0)
    assert meta["scale_source"] == "ground_fit"
    assert meta["scale_refused_rel_iqr"] == 9.0
    assert "REFUSED" in report


def test_zero_disables_the_gate(monkeypatch):
    meta, _r = _run_with_fit(monkeypatch, 0.3, rel_iqr=9.0,
                             scale_max_rel_iqr=0.0)
    assert meta["scale_source"].startswith("primary_registration")
    assert "scale_refused_rel_iqr" not in meta
    assert meta["scale_rel_iqr"] == 9.0


def test_armed_gate_refuses_a_fit_whose_samples_disagree(monkeypatch):
    meta, report = _run_with_fit(monkeypatch, 0.3, rel_iqr=9.0,
                                 scale_max_rel_iqr=0.5)
    assert meta["scale_refused_rel_iqr"] == 9.0
    assert meta["scale_source"] == "ground_fit"
    assert "REFUSED" in report


def test_armed_gate_keeps_a_well_conditioned_fit(monkeypatch):
    meta, _r = _run_with_fit(monkeypatch, 0.3, rel_iqr=0.1,
                             scale_max_rel_iqr=0.5)
    assert meta["scale_source"].startswith("primary_registration")
    assert "scale_refused_rel_iqr" not in meta


def test_the_conditioning_is_reported_whether_or_not_the_gate_is_armed(monkeypatch):
    """A threshold can only be chosen off numbers from a real run, so the
    spread is reported even when nothing acts on it."""
    _meta, report = _run_with_fit(monkeypatch, 0.3, rel_iqr=1.25)
    assert "quartile spread 1.25" in report


# --- the GROUND cross-check ---------------------------------------------
# The dispersion gate reads the fit arguing with itself, and the failure that
# matters does not argue: on the castle the one broken ROI had a TIGHTER spread
# (0.386) than the two good ones (0.518, 0.704). So ask an estimator that
# shares no evidence -- the patch's own ground, landed on Y=0.

def _run_with_both_fits(monkeypatch, scale, ground, ground_info=None, **kw):
    _patch_estimate_depth(monkeypatch)
    import atlas_camera.comfy.nodes_geometry as ng
    monkeypatch.setattr(
        ng, "solve_scale_from_primary",
        lambda *a, **k: (scale, {"n_samples": 400, "scale_rel_iqr": 0.1,
                                 "scale_rel_mad": 0.05}))
    # Imported IN-METHOD, so the module that OWNS it is what to patch.
    import atlas_camera.core.relief_mesh as rm
    monkeypatch.setattr(
        rm, "estimate_ground_scale",
        lambda *a, **k: (ground, ground_info
                         if ground_info is not None else {"inliers": 900,
                                                          "plane_y": 0.0}))
    solve, _p, _e = _synthetic_primary()
    out, report = AtlasAddPatchView().add_patch(
        solve, torch.rand(1, 512, 512, 3), patch_azimuth_view="right side view",
        relief_grid=48, geometry_source="own_depth",
        primary_depth=_fake_primary_depth(), **kw)
    return out.projection_sources[0].metadata, report


def test_the_disagreement_is_measured_and_reported_by_default(monkeypatch):
    meta, report = _run_with_both_fits(monkeypatch, 0.273, 0.645)
    assert meta["scale_ground_fit"] == 0.645
    assert meta["scale_ground_disagreement"] == pytest.approx(0.645 / 0.273,
                                                              rel=1e-3)
    assert "ground cross-check" in report
    # Measured, but not acted on until a threshold is set.
    assert "scale_refused_ground_disagreement" not in meta
    assert meta["scale_source"].startswith("primary_registration")


def test_it_is_symmetric_so_too_large_is_caught_like_too_small(monkeypatch):
    small, _r = _run_with_both_fits(monkeypatch, 0.273, 0.645)
    large, _r2 = _run_with_both_fits(monkeypatch, 0.645, 0.273)
    assert small["scale_ground_disagreement"] == pytest.approx(
        large["scale_ground_disagreement"])


def test_an_armed_threshold_refuses_the_castle_failure(monkeypatch):
    """ROI 5's numbers: registered 0.273 against a ground fit near its
    neighbour's 0.645 -- 2.36x apart, and it painted 5% of its own hole."""
    meta, report = _run_with_both_fits(monkeypatch, 0.273, 0.645,
                                       scale_max_ground_disagreement=1.5)
    assert meta["scale_refused_ground_disagreement"] == pytest.approx(2.363,
                                                                     rel=1e-2)
    assert meta["scale_source"] == "ground_fit"
    assert "REFUSED" in report


def test_an_armed_threshold_keeps_two_estimators_that_agree(monkeypatch):
    meta, _r = _run_with_both_fits(monkeypatch, 0.645, 0.620,
                                   scale_max_ground_disagreement=1.5)
    assert meta["scale_source"].startswith("primary_registration")
    assert "scale_refused_ground_disagreement" not in meta


def test_no_usable_ground_ABSTAINS_rather_than_refusing(monkeypatch):
    """A patch can legitimately be all sky or all facade. Absence of a ground
    fit is not evidence against the registration, and treating it as evidence
    would refuse exactly the patches with the least to check against."""
    meta, report = _run_with_both_fits(
        monkeypatch, 0.273, 1.0,
        ground_info={"reason": "insufficient ground candidates"},
        scale_max_ground_disagreement=1.5)
    assert meta["scale_source"].startswith("primary_registration")
    assert "scale_refused_ground_disagreement" not in meta
    assert meta["scale_ground_fit_reason"] == "insufficient ground candidates"
    assert "ABSTAINED" in report


def test_a_cross_check_that_throws_never_fails_the_patch(monkeypatch):
    # The PRIMARY's own ground fit runs first and is not the cross-check, so
    # only the second call -- the patch's -- is made to throw. Patching both
    # tests nothing about this gate.
    calls = []

    def boom(*a, **k):
        calls.append(1)
        if len(calls) > 1:
            raise RuntimeError("no")
        return 1.0, {"inliers": 900, "plane_y": 0.0}

    _patch_estimate_depth(monkeypatch)
    import atlas_camera.comfy.nodes_geometry as ng
    monkeypatch.setattr(ng, "solve_scale_from_primary",
                        lambda *a, **k: (0.3, {"n_samples": 400,
                                               "scale_rel_iqr": 0.1}))
    import atlas_camera.core.relief_mesh as rm
    monkeypatch.setattr(rm, "estimate_ground_scale", boom)
    solve, _p, _e = _synthetic_primary()
    out, _report = AtlasAddPatchView().add_patch(
        solve, torch.rand(1, 512, 512, 3), patch_azimuth_view="right side view",
        relief_grid=48, geometry_source="own_depth",
        primary_depth=_fake_primary_depth(),
        scale_max_ground_disagreement=1.5)
    meta = out.projection_sources[0].metadata
    assert meta["scale_source"].startswith("primary_registration")
    assert "ground fit failed" in meta["scale_ground_fit_reason"]


# --- the SIBLING check ---------------------------------------------------
# The first gate that judges a patch against something other than itself. Both
# others passed the castle's broken ROI: its samples agreed tightly (spread
# 0.386, tighter than the two ROIs painting 69%) and it disagreed with its
# ground fit by exactly as much as a good patch did (2.15x, both). What it could
# not do was agree with the other patches, which measure the same world.

from atlas_camera.comfy.nodes_geometry import _comparable_sibling_scales  # noqa: E402


def _sib(scale, *, model="m", metric=True, source="primary_registration_visible",
         evidence="generated"):
    from types import SimpleNamespace
    return SimpleNamespace(metadata={
        "scale": scale, "depth_model": model, "depth_is_metric": metric,
        "scale_source": source, "evidence_type": evidence})


def _solve_of(*sources):
    from types import SimpleNamespace
    return SimpleNamespace(projection_sources=list(sources))


def test_siblings_need_the_same_model_and_a_metric_one():
    """A relative model normalises each crop on its own, so two of its scales
    are unrelated numbers however similar the crops were."""
    solve = _solve_of(_sib(0.6), _sib(0.7, model="other"), _sib(0.8, metric=False))
    assert _comparable_sibling_scales(
        solve, depth_model="m", depth_is_metric=True) == [0.6]
    # And this patch's own depth being relative disqualifies the comparison
    # entirely, however many comparable-looking siblings are on the solve.
    assert _comparable_sibling_scales(
        solve, depth_model="m", depth_is_metric=False) == []


def test_a_patch_that_already_fell_back_is_NOT_evidence():
    """Otherwise one failure propagates: a ground-fit fallback, or a scale this
    very check already replaced, would enter the median and drag every later
    patch toward it."""
    solve = _solve_of(_sib(0.6), _sib(0.2, source="ground_fit"),
                      _sib(0.2, source="sibling_median"),
                      _sib(0.9, evidence="photographed"))
    assert _comparable_sibling_scales(
        solve, depth_model="m", depth_is_metric=True) == [0.6]


def _run_with_siblings(monkeypatch, scale, sibling_scales, **kw):
    _patch_estimate_depth(monkeypatch)
    import atlas_camera.comfy.nodes_geometry as ng
    monkeypatch.setattr(
        ng, "solve_scale_from_primary",
        lambda *a, **k: (scale, {"n_samples": 400, "scale_rel_iqr": 0.1}))
    solve, _p, _e = _synthetic_primary()
    model = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"
    for value in sibling_scales:
        solve.projection_sources.append(_sib(value, model=model))
    out, report = AtlasAddPatchView().add_patch(
        solve, torch.rand(1, 512, 512, 3), patch_azimuth_view="right side view",
        relief_grid=48, geometry_source="own_depth", depth_model=model,
        primary_depth=_fake_primary_depth(), **kw)
    return out.projection_sources[-1].metadata, report


def test_the_castle_outlier_is_caught_where_both_other_gates_passed(monkeypatch):
    """ROI 5's real numbers: 0.273 against siblings 0.830/0.611/0.381/0.645,
    median 0.628 — a 2.3x outlier, and the only gate that sees it."""
    meta, report = _run_with_siblings(
        monkeypatch, 0.273, [0.830, 0.611, 0.381, 0.645],
        scale_max_sibling_disagreement=1.8)
    assert meta["scale_sibling_median"] == pytest.approx(0.628)
    assert meta["scale_refused_sibling_disagreement"] == pytest.approx(2.301,
                                                                      rel=1e-2)
    assert meta["scale_replaced_from"] == 0.273
    # It hands back an ANSWER, not just a veto — the siblings measured the same
    # world, so their median IS an estimate of this patch's scale.
    assert meta["scale"] == pytest.approx(0.628)
    assert meta["scale_source"] == "sibling_median"
    assert "REFUSED" in report


def test_the_castle_good_patch_survives_where_the_ground_gate_refused_it(monkeypatch):
    """ROI 2 is the control: 0.611 sits ON the sibling median, and arming the
    GROUND gate hard enough to catch ROI 5 refused this one too."""
    meta, _r = _run_with_siblings(
        monkeypatch, 0.611, [0.830, 0.381, 0.645, 0.273],
        scale_max_sibling_disagreement=1.8)
    assert "scale_refused_sibling_disagreement" not in meta
    assert meta["scale_source"].startswith("primary_registration")


def test_too_few_siblings_ABSTAINS_rather_than_trusting_a_thin_median(monkeypatch):
    meta, report = _run_with_siblings(monkeypatch, 0.273, [0.830, 0.611],
                                      scale_max_sibling_disagreement=1.8,
                                      scale_min_siblings=3)
    assert meta["scale_sibling_n"] == 2
    assert "scale_sibling_median" not in meta
    assert meta["scale_source"].startswith("primary_registration")
    assert "ABSTAINED" in report


def test_it_measures_and_reports_without_a_threshold(monkeypatch):
    meta, report = _run_with_siblings(monkeypatch, 0.273,
                                      [0.830, 0.611, 0.381, 0.645])
    assert meta["scale_sibling_disagreement"] == pytest.approx(2.301, rel=1e-2)
    assert "scale_refused_sibling_disagreement" not in meta
    assert meta["scale_source"].startswith("primary_registration")
    assert "sibling check" in report
