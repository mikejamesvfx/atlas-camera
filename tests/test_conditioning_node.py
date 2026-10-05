"""Wrapper contract for AtlasConditioningBundle 🎛 and AtlasConditioningEXR 💾.

core.conditioning's own math is pinned in tests/test_conditioning_bundle.py
against closed forms. This file pins the things only the NODE can get wrong:
socket arity and order, that measured data never leaves through an IMAGE socket,
that the previews declare how they were normalised, and that a missing optional
dependency skips rather than fails.
"""
from __future__ import annotations

import json

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from atlas_camera.comfy.node_registry import NODE_CLASS_MAPPINGS  # noqa: E402
from atlas_camera.core.ghost_pixels import GhostClass  # noqa: E402

BUNDLE = NODE_CLASS_MAPPINGS["AtlasConditioningBundle"]
EXR = NODE_CLASS_MAPPINGS["AtlasWriteConditioningEXR"]
W = H = 96
RES = 128


def _solve(step: bool = True):
    """A solve carrying a real relief mesh with a depth cliff to reveal."""
    from atlas_camera.comfy.nodes_geometry import AtlasDeriveReliefMesh
    from atlas_camera.core.intrinsics import build_intrinsics
    from atlas_camera.core.schema import AtlasCamera, AtlasExtrinsics, AtlasSolve
    from atlas_camera.inference.depth_estimator import DepthResult

    d = np.full((H, W), 12.0, dtype=np.float32)
    if step:
        d[30:70, 30:70] = 3.0
    depth = DepthResult(depth=d, is_metric=True, model_id="t",
                        image_width=W, image_height=H)
    intr = build_intrinsics(image_width=W, image_height=H,
                            focal_length_mm=35.0, sensor_width_mm=36.0)
    cam = AtlasCamera(intrinsics=intr, extrinsics=AtlasExtrinsics(
        camera_position=(0.0, 0.0, 0.0),
        camera_world_matrix=((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0),
                             (0, 0, 0, 1))))
    base = AtlasSolve(camera=cam, image_width=W, image_height=H)
    return AtlasDeriveReliefMesh().derive(base, depth, relief_grid=96)[0]


def _path(move="dolly_pan_left", frames=4, fov_deg=None):
    from atlas_camera.core.camera_path import build_preset_camera_path
    from atlas_camera.core.schema import AtlasExtrinsics
    ex = AtlasExtrinsics(
        camera_position=(0.0, 0.0, 0.0),
        camera_world_matrix=((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0),
                             (0, 0, 0, 1)))
    kw = {"fov_deg": fov_deg} if fov_deg is not None else {}
    return build_preset_camera_path(ex, move, frame_count=frames, **kw)[0]


def _image(seed=1):
    rng = np.random.default_rng(seed)
    return torch.from_numpy(rng.random((1, H, W, 3)).astype(np.float32))


@pytest.fixture
def built():
    return BUNDLE().build(_solve(), _image(), camera_path=_path(),
                          resolution=RES)


def test_socket_arity_matches_return_names(built):
    assert len(built) == len(BUNDLE.RETURN_TYPES) == len(BUNDLE.RETURN_NAMES)


def test_the_bundle_is_not_an_image_socket():
    """Metric depth in metres and flow in pixels both leave [0, 1], so an IMAGE
    socket would quantise them at the first preview or save hop."""
    assert BUNDLE.RETURN_TYPES[0] == "ATLAS_CONDITIONING"
    assert "ATLAS_CONDITIONING" not in BUNDLE.RETURN_TYPES[1:]


def test_the_bundle_carries_every_measured_pass(built):
    seq = built[0]
    assert seq.frames == 4
    for field in ("depth_m", "normal_world", "position_world", "flow_fwd",
                  "flow_bwd", "class_map"):
        assert getattr(seq, field) is not None, field
    assert len(seq.k) == seq.frames


def test_previews_are_in_range_and_declare_their_normalisation(built):
    report = json.loads(built[-1])
    names = BUNDLE.RETURN_NAMES
    for index, name in enumerate(names):
        if not name.endswith("_preview"):
            continue
        tensor = built[index]
        assert tensor.min() >= 0.0 and tensor.max() <= 1.0, name
    norm = report["preview_normalisation"]
    assert "near_m" in norm["depth"] and "far_m" in norm["depth"]
    assert "scale_px" in norm["flow"]
    assert "per SEQUENCE" in norm["depth"]["mapping"]


def test_the_masks_are_disjoint_and_carry_the_class_split(built):
    valid, ghost = built[6], built[7]
    assert valid.shape == ghost.shape
    assert not ((valid > 0.5) & (ghost > 0.5)).any()
    seq = built[0]
    assert (seq.class_map == int(GhostClass.VALID)).any()


def test_the_report_names_the_units_and_the_static_principal_point(built):
    report = json.loads(built[-1])
    assert "metres" in report["units"]["depth"]
    assert "pixels" in report["units"]["flow"]
    assert report["principal_point"] == "static"
    assert report["focal_varies"] is False


def test_a_keyed_zoom_reports_a_varying_focal():
    """A zoom must not render like a dolly. The path's fov channel drives a
    per-frame K, and the report says so rather than leaving a reader to assume
    the focal was fixed."""
    built = BUNDLE().build(_solve(), _image(),
                           camera_path=_path("vertigo", fov_deg=40.0),
                           resolution=RES)
    report = json.loads(built[-1])
    assert report["focal_varies"] is True
    focals = [k[1][1] for k in built[0].k]
    assert len(set(round(f, 6) for f in focals)) > 1


def test_no_geometry_refuses_loudly_and_returns_no_bundle():
    """An empty bundle read as 'nothing to condition on' would send a model off
    inventing a whole frame with no brief."""
    from atlas_camera.core.intrinsics import build_intrinsics
    from atlas_camera.core.schema import AtlasCamera, AtlasExtrinsics, AtlasSolve
    intr = build_intrinsics(image_width=W, image_height=H,
                            focal_length_mm=35.0, sensor_width_mm=36.0)
    bare = AtlasSolve(camera=AtlasCamera(intrinsics=intr,
                                         extrinsics=AtlasExtrinsics()),
                      image_width=W, image_height=H)
    out = BUNDLE().build(bare, _image(), resolution=RES)
    report = json.loads(out[-1])
    assert out[0] is None
    assert "no serialized projection meshes" in report["error"]
    assert "warning" in report


def test_stride_and_last_n_are_recorded_not_silent():
    """A caller reading flow magnitudes needs to know they are per-stride."""
    built = BUNDLE().build(_solve(), _image(), camera_path=_path(frames=8),
                           resolution=RES, stride=2)
    report = json.loads(built[-1])
    assert built[0].frames == 4
    assert report["source_frame_indices"] == [0, 2, 4, 6]
    assert "stride 2" in report["units"]["flow"]


def test_opting_out_of_a_pass_drops_it_from_the_bundle_and_the_report():
    built = BUNDLE().build(_solve(), _image(), camera_path=_path(),
                           resolution=RES, emit_flow=False)
    report = json.loads(built[-1])
    assert built[0].flow_fwd is None
    assert report["emitted"]["flow"] is False


# --------------------------------------------------------------------------
# the EXR writer
# --------------------------------------------------------------------------

def test_the_exr_writer_skips_rather_than_fails_without_oiio(built, tmp_path,
                                                             monkeypatch):
    """The sockets are the primary product and OIIO is optional in this
    package, so a missing dependency must not fail the graph."""
    import atlas_camera.plate.oiio_io as oiio_io
    monkeypatch.setattr(oiio_io, "oiio_available", lambda: False)
    directory, report = EXR().write(built[0], str(tmp_path))
    payload = json.loads(report)
    assert directory == ""
    assert "OpenImageIO" in payload["skipped"]
    assert "SKIPPED, not failed" in payload["note"]


def test_the_exr_writer_reports_no_bundle_instead_of_crashing(tmp_path):
    directory, report = EXR().write(None, str(tmp_path))
    assert directory == ""
    assert "no bundle" in json.loads(report)["error"]


def test_written_passes_record_their_channel_naming(built, tmp_path):
    """A channel called Z that holds a normal is a silent, plausible-looking
    error; the only defence is writing the naming down beside the data."""
    pytest.importorskip("OpenImageIO")
    directory, report = EXR().write(built[0], str(tmp_path / "out"))
    payload = json.loads(report)
    assert payload["bit_depth"] == "float32"
    assert "none" in payload["colour_conversion"]
    assert payload["passes"]["depth"]["channels"] == ["Z"]
    assert payload["passes"]["normal"]["channels"] == ["N.X", "N.Y", "N.Z"]
    assert payload["passes"]["flow_fwd"]["channels"] == ["FWD.X", "FWD.Y"]
    assert payload["camera"]["principal_point"] == "static"
    assert len(payload["camera"]["k_per_frame"]) == built[0].frames
    manifest = list((tmp_path / "out").glob("*_conditioning_manifest.json"))
    assert len(manifest) == 1
