"""Live Record3D stream: parity with the file path, and the traps around it.

The stream and the `.r3d` file carry the SAME payload from the same device, so
the test that matters is not "does the stream produce a solve" but "does it
produce the *same* solve". Everything here is built on a fake stream rather than
hardware, because the conversion math is what can silently go wrong; the USB
handshake either works or fails loudly.
"""
from __future__ import annotations

import json
import zipfile

import pytest

from atlas_camera.importers.record3d import (
    Record3DCapture,
    Record3DStreamError,
    Record3DStreamSource,
    _pose_to_world_matrix,
    build_measured_solve,
)

np = pytest.importorskip("numpy")


# --------------------------------------------------------------------- fakes


class _FakeCoeffs:
    """Mirrors record3d.IntrinsicMatrixCoeffs: tx/ty are the PRINCIPAL POINT."""

    def __init__(self, fx, fy, cx, cy):
        self.fx, self.fy, self.tx, self.ty = fx, fy, cx, cy


class _FakePose:
    """Mirrors record3d.CameraPose: quaternion + translation, named fields."""

    def __init__(self, qx, qy, qz, qw, tx, ty, tz):
        self.qx, self.qy, self.qz, self.qw = qx, qy, qz, qw
        self.tx, self.ty, self.tz = tx, ty, tz


class _FakeStream:
    """Stands in for record3d.Record3DStream with a fixed frame."""

    def __init__(self, rgb, depth, conf, coeffs, pose):
        self._rgb, self._depth, self._conf = rgb, depth, conf
        self._coeffs, self._pose = coeffs, pose
        self.disconnected = False

    def get_rgb_frame(self):
        return self._rgb

    def get_depth_frame(self):
        return self._depth

    def get_confidence_frame(self):
        return self._conf

    def get_intrinsic_mat(self):
        return self._coeffs

    def get_camera_pose(self):
        return self._pose

    def get_device_type(self):
        return 14

    def disconnect(self):
        self.disconnected = True


def _source(rgb=None, depth=None, conf=None, coeffs=None, pose=None):
    rgb = rgb if rgb is not None else np.zeros((48, 64, 3), dtype=np.uint8)
    depth = depth if depth is not None else np.full((12, 16), 2.0, dtype=np.float32)
    conf = conf if conf is not None else np.full((12, 16), 2, dtype=np.uint8)
    coeffs = coeffs or _FakeCoeffs(500.0, 500.0, 32.0, 24.0)
    pose = pose or _FakePose(0.0, 0.0, 0.0, 1.0, 0.0, 1.5, 0.0)

    class _Info:
        udid = "TESTUDID"
        product_id = 1

    return Record3DStreamSource(stream=_FakeStream(rgb, depth, conf, coeffs, pose),
                                device_info=_Info())


# --------------------------------------------------- the quaternion order trap


def test_pose_mapping_is_scalar_last_not_scalar_first():
    """The stream exposes qw first; the file reader wants it LAST.

    Getting this backwards produces a perfectly well-formed rotation matrix that
    is simply wrong, and nothing downstream can detect it. Pinned against a
    quaternion where the two orderings give visibly different results: a 90-degree
    rotation about Y, whose scalar-first misreading is a different rotation
    entirely.
    """
    s = 2 ** -0.5
    pose = _FakePose(qx=0.0, qy=s, qz=0.0, qw=s, tx=0.0, ty=0.0, tz=0.0)
    m = _pose_to_world_matrix(pose)

    # +90 deg about Y maps world +X to -Z. Columns are the rotated basis axes.
    assert m[0][0] == pytest.approx(0.0, abs=1e-9)
    assert m[2][0] == pytest.approx(-1.0, abs=1e-9)
    assert m[0][2] == pytest.approx(1.0, abs=1e-9)

    # The scalar-FIRST misreading (qx=s, qy=0, qz=0, qw=s) is a rotation about X
    # instead, so this assertion is what separates the two.
    assert m[1][1] == pytest.approx(1.0, abs=1e-9), "Y axis must be fixed by a Y rotation"


def test_translation_survives_the_pose_mapping():
    src = _source(pose=_FakePose(0.0, 0.0, 0.0, 1.0, 1.25, 1.5, -3.0))
    frame = src.frame()
    assert frame.camera_world_matrix[0][3] == pytest.approx(1.25)
    assert frame.camera_world_matrix[1][3] == pytest.approx(1.5)
    assert frame.camera_world_matrix[2][3] == pytest.approx(-3.0)


def test_principal_point_comes_from_coeffs_tx_ty_not_a_translation():
    """IntrinsicMatrixCoeffs.tx/ty collide by name with CameraPose.tx/ty."""
    src = _source(coeffs=_FakeCoeffs(500.0, 501.0, 30.0, 20.0),
                  pose=_FakePose(0.0, 0.0, 0.0, 1.0, 9.0, 9.0, 9.0))
    intr = src.frame().intrinsics
    assert intr.fx_px == pytest.approx(500.0)
    assert intr.fy_px == pytest.approx(501.0)
    assert intr.cx_px == pytest.approx(30.0)   # NOT 9.0
    assert intr.cy_px == pytest.approx(20.0)


# ------------------------------------------------------ file <-> stream parity


def _metadata(fx=500.0, w=64, h=48, dw=16, dh=12):
    return {
        "w": w, "h": h, "dw": dw, "dh": dh,
        "K": [fx, 0.0, 0.0, 0.0, fx, 0.0, w / 2.0, h / 2.0, 1.0],
        "fps": 30.0,
        "poses": [[0.0, 0.0, 0.0, 1.0, 0.0, 1.5, 0.0]],
        "deviceType": 14,
    }


def test_stream_and_file_agree_on_the_same_measured_geometry(tmp_path):
    """The whole reason both paths share build_measured_solve.

    Same intrinsics, same pose, same depth size -> the two sources must produce
    an identical camera. If this ever diverges, a scene scanned to file and the
    same scene streamed live would disagree about where the camera was.
    """
    meta = _metadata()
    path = tmp_path / "parity.r3d"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("metadata", json.dumps(meta))
        zf.writestr("rgbd/0.depth",
                    np.full((12, 16), 2.0, dtype=np.float32).tobytes())

    file_solve = Record3DCapture.open(path).solve(0)

    src = _source(coeffs=_FakeCoeffs(500.0, 500.0, 32.0, 24.0),
                  pose=_FakePose(0.0, 0.0, 0.0, 1.0, 0.0, 1.5, 0.0))
    stream_solve = src.solve(src.frame())

    fi, si = file_solve.camera.intrinsics, stream_solve.camera.intrinsics
    assert (fi.fx_px, fi.fy_px, fi.cx_px, fi.cy_px) == \
           pytest.approx((si.fx_px, si.fy_px, si.cx_px, si.cy_px))

    fe, se = file_solve.camera.extrinsics, stream_solve.camera.extrinsics
    assert tuple(fe.camera_position) == pytest.approx(tuple(se.camera_position))
    assert np.allclose(np.asarray(fe.camera_view_matrix, dtype=np.float64),
                       np.asarray(se.camera_view_matrix, dtype=np.float64))
    assert file_solve.source_method == stream_solve.source_method


def test_stream_solve_never_applies_the_negative_z_canonicalization():
    """Measured yaw must pass through — the same invariant as the file path."""
    src = _source()
    solve = src.solve(src.frame())
    assert solve.debug_metadata["canonical_negative_z_applied"] is False
    assert solve.debug_metadata["measured_pose"] is True
    assert solve.debug_metadata["record3d_source"] == "stream"


def test_shared_builder_marks_depth_confidence_only_when_depth_exists():
    with_depth = build_measured_solve(
        intrinsics=_source().frame().intrinsics,
        camera_world_matrix=((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)),
        rgb_size=(64, 48), depth_size=(16, 12), image_path="x")
    without = build_measured_solve(
        intrinsics=_source().frame().intrinsics,
        camera_world_matrix=((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)),
        rgb_size=(64, 48), depth_size=(0, 0), image_path="x")
    assert with_depth.camera.confidence.individual_metrics["depth"] > 0
    assert without.camera.confidence.individual_metrics["depth"] == 0


# ----------------------------------------------------------- stream behaviour


def test_frame_carries_rgb_as_an_array_not_jpeg_bytes():
    """Streamed colour must avoid a needless encode/decode round trip."""
    frame = _source().frame()
    assert frame.rgb_jpeg is None
    assert frame.rgb_array is not None
    assert frame.rgb_array.shape == (48, 64, 3)


def test_empty_frame_is_reported_rather_than_silently_solved():
    src = _source(rgb=np.zeros((0, 0, 3), dtype=np.uint8))
    with pytest.raises(Record3DStreamError, match="empty frame"):
        src.frame()


def test_missing_confidence_plane_is_tolerated():
    src = _source(conf=np.zeros((0, 0), dtype=np.uint8))
    assert src.frame().confidence is None


def test_k_published_at_depth_scale_is_rescaled_to_the_colour_frame():
    """Same guard as the file path: a principal point in the left quarter of a
    frame whose depth buffer is genuinely smaller means K is at depth scale."""
    src = _source(coeffs=_FakeCoeffs(125.0, 125.0, 8.0, 6.0))  # quarter scale
    intr = src.frame().intrinsics
    assert intr.fx_px == pytest.approx(500.0)
    assert intr.cx_px == pytest.approx(32.0)


def test_close_disconnects_the_stream():
    src = _source()
    with src:
        pass
    assert src.stream.disconnected is True


# ------------------------------------------------------------- node contract


def test_node_is_registered_and_never_serves_a_cached_result():
    from atlas_camera.comfy import node_registry as reg

    # Gated into the iOS tier (ATLAS_IOS) for v1, so look it up there.
    cls = reg.IOS_NODE_CLASS_MAPPINGS["AtlasStreamRecord3D"]
    assert cls.RETURN_NAMES == ("image", "solve", "depth", "confidence_mask", "report")
    # A live source must always re-execute: the phone has moved since last run.
    assert cls.IS_CHANGED() != cls.IS_CHANGED(), "IS_CHANGED must not compare equal"


def test_node_matches_the_file_loader_output_contract():
    """Both Record3D nodes are interchangeable downstream, so their slots must
    line up — otherwise swapping one for the other silently rewires a graph."""
    from atlas_camera.comfy import node_registry as reg

    stream = reg.IOS_NODE_CLASS_MAPPINGS["AtlasStreamRecord3D"]
    file_ = reg.IOS_NODE_CLASS_MAPPINGS["AtlasLoadRecord3D"]
    assert stream.RETURN_TYPES == file_.RETURN_TYPES
    assert stream.RETURN_NAMES == file_.RETURN_NAMES


def test_no_device_error_names_the_windows_driver_prerequisite():
    """On Windows a missing Apple driver yields an EMPTY device list, which is
    indistinguishable from an unplugged phone — so the message must say so."""
    from atlas_camera.importers import record3d as mod

    class _NoDevices:
        @staticmethod
        def get_connected_devices():
            return []

    original = mod._require_record3d_stream
    mod._require_record3d_stream = lambda: _NoDevices
    try:
        with pytest.raises(Record3DStreamError) as exc:
            Record3DStreamSource.connect(0)
    finally:
        mod._require_record3d_stream = original

    msg = str(exc.value)
    assert "Apple Mobile Device Support" in msg
    assert "empty" in msg.lower()
