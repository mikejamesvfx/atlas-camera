"""The two Unreal readers: a rendered depth pass and a Sequencer camera.

Both exist so that depth and camera come from ONE render and therefore cannot
disagree about scale or lens. What is pinned here is the part that a wrong
answer would hide rather than announce:

  * `AtlasUnrealCameraPath` REFUSES a stage that is not Y-up metres. Unreal's
    own Z-up centimetres look identical in the file, and reading them as if they
    were Atlas's frame produces a camera that matches on frame one and DRIFTS —
    the failure nothing downstream detects.
  * frame 1 is the identity pose: the source clip's own view, warped by nothing.
  * `AtlasUnrealDepthGeometry` resamples NEAREST, never bilinear: averaging
    across a depth discontinuity invents a surface at a distance nothing in the
    scene occupies.
  * out-of-range depth becomes NaN and rides a mask, rather than a number.

The EXR reader is exercised through a stub `OpenEXR` module. The dependency is
libOpenEXR's business; the node's own logic — resample, scale, clip, mask,
normalised intrinsics, frame repeat — is what these tests are for.
"""
from __future__ import annotations

import json
import sys
import types

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("torch")

from atlas_camera.comfy.nodes import (  # noqa: E402
    NODE_CLASS_MAPPINGS,
    AtlasUnrealCameraPath,
    AtlasUnrealDepthGeometry,
)


# --------------------------------------------------------------- depth reader

class _StubChannel:
    def __init__(self, pixels):
        self.pixels = pixels


class _StubPart:
    def __init__(self, channels):
        self.channels = channels


class _StubFile:
    def __init__(self, channels):
        self.parts = [_StubPart(channels)]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def stub_openexr(monkeypatch):
    """Install a stub `OpenEXR` whose File() yields whatever channels we set."""
    holder = {}

    mod = types.ModuleType("OpenEXR")
    mod.File = lambda path: _StubFile(holder["channels"])
    monkeypatch.setitem(sys.modules, "OpenEXR", mod)

    def _set(**channels):
        holder["channels"] = {k: _StubChannel(np.asarray(v, dtype=np.float32))
                              for k, v in channels.items()}
    return _set


@pytest.fixture
def exr_file(tmp_path):
    p = tmp_path / "shot_0001.exr"
    p.write_bytes(b"not really an exr - the reader is stubbed")
    return p


def test_registered_and_output_names():
    assert NODE_CLASS_MAPPINGS["AtlasUnrealDepthGeometry"] is AtlasUnrealDepthGeometry
    assert NODE_CLASS_MAPPINGS["AtlasUnrealCameraPath"] is AtlasUnrealCameraPath
    assert AtlasUnrealDepthGeometry.RETURN_NAMES == ("moge_geometry", "report")
    assert AtlasUnrealCameraPath.RETURN_NAMES == ("camera_path", "report")


def test_missing_exr_names_the_pass_to_render():
    with pytest.raises(ValueError, match="SceneDepthWorldUnits"):
        AtlasUnrealDepthGeometry().build("no/such/file.exr", 35.0, 23.5, 64, 64)


def test_a_renamed_material_is_reported_with_the_channels_it_did_find(
        stub_openexr, exr_file):
    """MRQ names the layer after the post-process material, so a renamed
    material renames the channel — the report has to say what IS there."""
    stub_openexr(SomethingElse=np.full((16, 16), 5.0))
    with pytest.raises(ValueError, match="SomethingElse"):
        AtlasUnrealDepthGeometry().build(str(exr_file), 35.0, 23.5, 16, 16)


def test_depth_arrives_in_metres_and_rides_a_validity_mask(stub_openexr, exr_file):
    src = np.linspace(2.0, 50.0, 16)[:, None] * np.ones((1, 16))
    src[0, 0] = 0.0                    # nothing rendered here
    src[0, 1] = np.inf
    stub_openexr(FinalImageSceneDepthWorldUnits=src)

    geom, report = AtlasUnrealDepthGeometry().build(
        str(exr_file), 35.0, 23.5, 16, 16)

    d = np.asarray(geom["depth"])
    m = np.asarray(geom["mask"])
    assert d.shape == (1, 16, 16)
    assert not m[0, 0, 0] and not m[0, 0, 1]
    assert np.isnan(d[0, 0, 0]) and np.isnan(d[0, 0, 1])
    assert m.sum() == 16 * 16 - 2
    assert "metric, Unreal world" in report


def test_depth_scale_converts_centimetres(stub_openexr, exr_file):
    stub_openexr(FinalImageSceneDepthWorldUnits=np.full((8, 8), 1200.0))
    geom, _r = AtlasUnrealDepthGeometry().build(
        str(exr_file), 35.0, 23.5, 8, 8, depth_scale=0.01)
    assert float(np.nanmedian(np.asarray(geom["depth"]))) == pytest.approx(12.0)


def test_resample_is_nearest_so_a_depth_cliff_is_not_averaged(stub_openexr, exr_file):
    """Bilinear across a discontinuity invents a surface at a distance nothing
    in the scene occupies, and the warp smears the silhouette across it."""
    src = np.full((16, 16), 100.0)
    src[:, :8] = 2.0                     # a hard cliff down the middle
    stub_openexr(FinalImageSceneDepthWorldUnits=src)

    geom, _r = AtlasUnrealDepthGeometry().build(str(exr_file), 35.0, 23.5, 8, 8)

    d = np.asarray(geom["depth"])[0]
    assert set(np.unique(d[np.isfinite(d)])) == {2.0, 100.0}


def test_far_clip_drops_a_backplate_card_and_says_so(stub_openexr, exr_file):
    """An ImagePlate standing in for sky is real geometry to the renderer and
    comes back at the card's distance, which the warp parallaxes like a wall."""
    src = np.full((16, 16), 10.0)
    src[:4] = 9000.0                    # the sky card
    stub_openexr(FinalImageSceneDepthWorldUnits=src)

    geom, report = AtlasUnrealDepthGeometry().build(
        str(exr_file), 35.0, 23.5, 16, 16, far_clip_m=1000.0)

    m = np.asarray(geom["mask"])[0]
    assert not m[:4].any() and m[4:].all()
    assert "far_clip" in report and "masked out" in report


def test_intrinsics_are_normalised_the_way_the_warp_reads_them_back(
        stub_openexr, exr_file):
    """The warp does `fx = K[0][0,0] * W`, so K carries fx/W, not pixels."""
    stub_openexr(FinalImageSceneDepthWorldUnits=np.full((16, 16), 8.0))
    W, H, focal, sensor = 64, 32, 35.0, 23.5

    geom, report = AtlasUnrealDepthGeometry().build(
        str(exr_file), focal, sensor, W, H)

    K = np.asarray(geom["intrinsics"])[0]
    hfov = np.degrees(2.0 * np.arctan(sensor / (2.0 * focal)))
    fx_px = W / (2.0 * np.tan(np.radians(hfov) / 2.0))
    assert K[0, 0] * W == pytest.approx(fx_px)
    assert K[0, 2] == pytest.approx(0.5) and K[1, 2] == pytest.approx(0.5)
    assert f"{hfov:.2f} deg horizontal" in report


def test_an_aspect_mismatch_is_called_out_not_silently_stretched(
        stub_openexr, exr_file):
    """Depth rendered for a different frame shape does not line up with the
    image, and nothing else in the chain would notice."""
    stub_openexr(FinalImageSceneDepthWorldUnits=np.full((16, 16), 8.0))
    _geom, report = AtlasUnrealDepthGeometry().build(
        str(exr_file), 35.0, 23.5, 64, 16)
    assert "MISMATCH" in report


def test_a_still_plate_is_repeated_for_every_frame_of_the_clip(
        stub_openexr, exr_file):
    stub_openexr(FinalImageSceneDepthWorldUnits=np.full((8, 8), 8.0))
    geom, _r = AtlasUnrealDepthGeometry().build(
        str(exr_file), 35.0, 23.5, 8, 8, frames=5)
    assert np.asarray(geom["depth"]).shape[0] == 5
    assert np.asarray(geom["intrinsics"]).shape[0] == 5


# -------------------------------------------------------------- camera reader

def _write_stage(path, *, up="Y", mpu=1.0, positions=None, focal=35.0,
                 aperture=18.0):
    from pxr import Gf, Sdf, Usd, UsdGeom

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, up)
    UsdGeom.SetStageMetersPerUnit(stage, mpu)
    stage.SetStartTimeCode(0)
    positions = positions if positions is not None else [(0.0, 0.0, float(i))
                                                         for i in range(4)]
    stage.SetEndTimeCode(len(positions) - 1)

    prim = stage.DefinePrim("/Cam", "Xform")
    attr = prim.CreateAttribute("xformOp:transform", Sdf.ValueTypeNames.Matrix4d)
    for i, p in enumerate(positions):
        m = Gf.Matrix4d(1.0)
        m.SetTranslateOnly(Gf.Vec3d(*p))
        attr.Set(m, Usd.TimeCode(i))

    comp = stage.DefinePrim("/Cam/CameraComponent", "Xform")
    fa = comp.CreateAttribute("focalLength", Sdf.ValueTypeNames.Float)
    va = comp.CreateAttribute("verticalAperture", Sdf.ValueTypeNames.Float)
    for i in range(len(positions)):
        fa.Set(focal, Usd.TimeCode(i))
        va.Set(aperture, Usd.TimeCode(i))
    stage.GetRootLayer().Save()
    return path


def test_missing_usd_names_the_exporter():
    with pytest.raises(ValueError, match="LevelSequenceExporterUsd"):
        AtlasUnrealCameraPath().build("no/such/file.usda")


@pytest.mark.parametrize("up,mpu,label", [
    ("Z", 1.0, "z_up"),
    ("Y", 0.01, "centimetres"),
    ("Z", 0.01, "unreals_own_defaults"),
])
def test_a_stage_that_is_not_y_up_metres_is_refused(tmp_path, up, mpu, label):
    """Epic's exporter DEFAULTS are Z-up centimetres, which look identical in
    the file. Converting on a guess produces a camera that matches on frame one
    and drifts — measured 11.3 m out over an 8 m move — so it is refused."""
    pytest.importorskip("pxr")
    p = _write_stage(tmp_path / f"{label}.usda", up=up, mpu=mpu)
    with pytest.raises(ValueError, match="upAxis"):
        AtlasUnrealCameraPath().build(str(p))


def test_frame_one_is_the_identity_pose(tmp_path):
    """The poses are rebased onto the first frame; that is what makes frame 1
    the source clip's own view, warped by nothing."""
    pytest.importorskip("pxr")
    p = _write_stage(tmp_path / "move.usda")

    payload, report = AtlasUnrealCameraPath().build(str(p))
    doc = json.loads(payload)

    assert doc["format"] == "atlas.ltx.crossview_warp.pose"
    first = doc["poses"][0]
    assert first["f"] == 1
    assert first["p"] == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)
    assert first["q"] == pytest.approx([0.0, 0.0, 0.0, 1.0], abs=1e-9)
    assert "frame 1     identity" in report


def test_every_frame_carries_a_pose_and_a_field_of_view(tmp_path):
    pytest.importorskip("pxr")
    p = _write_stage(tmp_path / "move.usda", focal=35.0, aperture=18.0)

    payload, report = AtlasUnrealCameraPath().build(str(p))
    doc = json.loads(payload)

    assert doc["frameCount"] == 4 == len(doc["poses"])
    expect_vfov = np.degrees(2.0 * np.arctan(18.0 / (2.0 * 35.0)))
    for pose in doc["poses"]:
        assert pose["vfov"] == pytest.approx(expect_vfov, abs=1e-3)
        assert len(pose["q"]) == 4
    # The move travelled, and the report measures it.
    assert doc["poses"][-1]["p"] != pytest.approx([0.0, 0.0, 0.0], abs=1e-6)
    assert "travel" in report


def test_frame_count_bounds_what_is_read(tmp_path):
    pytest.importorskip("pxr")
    p = _write_stage(tmp_path / "move.usda")
    doc = json.loads(AtlasUnrealCameraPath().build(str(p), frame_count=2)[0])
    assert doc["frameCount"] == 2


def test_a_lensless_export_with_no_fallback_is_refused(tmp_path):
    """The pose path carries a field of view per frame; it cannot be invented."""
    pytest.importorskip("pxr")
    from pxr import Gf, Sdf, Usd, UsdGeom

    path = tmp_path / "lensless.usda"
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, "Y")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(1)
    prim = stage.DefinePrim("/Cam", "Xform")
    attr = prim.CreateAttribute("xformOp:transform", Sdf.ValueTypeNames.Matrix4d)
    for i in range(2):
        m = Gf.Matrix4d(1.0)
        m.SetTranslateOnly(Gf.Vec3d(0.0, 0.0, float(i)))
        attr.Set(m, Usd.TimeCode(i))
    stage.GetRootLayer().Save()

    with pytest.raises(ValueError, match="cannot be invented"):
        AtlasUnrealCameraPath().build(str(path))

    # ...but the fallback widgets satisfy it, and the report says they were used.
    _payload, report = AtlasUnrealCameraPath().build(
        str(path), focal_mm=35.0, sensor_height_mm=18.0)
    assert "from the fallback widgets, not the file" in report
