"""Record3D (`.r3d`) importer — measured ARKit geometry into an AtlasSolve.

The load-bearing assertions here are the CONVENTION ones. ARKit's camera basis
happens to match Atlas's exactly (x-right / y-up / z-back, right-handed Y-up
world, metres), so the importer performs no axis flip. That is easy to "fix"
by mistake later, so the pass-through is pinned explicitly, as is the fact that
``solver._face_camera_toward_negative_z`` is NOT applied to a measured pose.
"""

from __future__ import annotations

import io
import json
import math
import zipfile

import numpy as np
import pytest

from atlas_camera.core.schema import AtlasSolve
from atlas_camera.importers.record3d import (
    SOURCE_METHOD,
    Record3DCapture,
    Record3DError,
    quaternion_to_rotation_matrix,
)

RGB_W, RGB_H = 1920, 1440
DEPTH_W, DEPTH_H = 256, 192  # ARKit sceneDepth — the real LiDAR resolution
FX, FY, CX, CY = 1500.0, 1500.0, 960.0, 720.0

IDENTITY_POSE = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]


def _column_major_K(fx=FX, fy=FY, cx=CX, cy=CY):
    """Record3D serializes K column-major: [fx,0,0, 0,fy,0, cx,cy,1]."""
    return [fx, 0.0, 0.0, 0.0, fy, 0.0, cx, cy, 1.0]


def _metadata(poses=None, **overrides):
    meta = {
        "w": RGB_W,
        "h": RGB_H,
        "dw": DEPTH_W,
        "dh": DEPTH_H,
        "K": _column_major_K(),
        "fps": 30.0,
        "poses": poses if poses is not None else [IDENTITY_POSE],
        "initPose": IDENTITY_POSE,
        "deviceType": 14,
    }
    meta.update(overrides)
    return meta


def _write_capture(tmp_path, metadata=None, *, frames=1, depth_value=2.5, name="capture.r3d"):
    """Build a synthetic .r3d: a ZIP of `metadata` + uncompressed rgbd frames.

    Real captures LZFSE-compress the depth/conf planes; the importer falls back
    to a raw read first, so these fixtures exercise every path except the LZFSE
    decompress itself (which is a one-line optional dependency).
    """
    meta = metadata if metadata is not None else _metadata()
    path = tmp_path / name
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("metadata", json.dumps(meta))
        for i in range(frames):
            zf.writestr(f"rgbd/{i}.jpg", b"\xff\xd8\xff\xe0not-a-real-jpeg")
            depth = np.full((DEPTH_H, DEPTH_W), depth_value + i, dtype=np.float32)
            zf.writestr(f"rgbd/{i}.depth", depth.tobytes())
            conf = np.full((DEPTH_H, DEPTH_W), 2, dtype=np.uint8)
            zf.writestr(f"rgbd/{i}.conf", conf.tobytes())
    return path


# --------------------------------------------------------------- quaternions


def test_identity_quaternion_is_identity_rotation():
    r = quaternion_to_rotation_matrix(0.0, 0.0, 0.0, 1.0)
    for i in range(3):
        for j in range(3):
            assert r[i][j] == pytest.approx(1.0 if i == j else 0.0, abs=1e-12)


def test_quaternion_scalar_is_last_not_first():
    """A 90 deg rotation about +Y, read scalar-LAST as Record3D writes it.

    Read scalar-first by mistake and this is a rotation about X — the classic
    way an ARKit import ends up pitched 90 degrees instead of yawed.
    """
    s = math.sin(math.pi / 4)
    c = math.cos(math.pi / 4)
    r = quaternion_to_rotation_matrix(0.0, s, 0.0, c)

    # RotY(90): world +X maps to -Z, world +Z maps to +X. Columns are the
    # rotated basis axes, so column 0 == R @ (1,0,0).
    assert (r[0][0], r[1][0], r[2][0]) == pytest.approx((0.0, 0.0, -1.0), abs=1e-9)
    assert (r[0][1], r[1][1], r[2][1]) == pytest.approx((0.0, 1.0, 0.0), abs=1e-9)
    assert (r[0][2], r[1][2], r[2][2]) == pytest.approx((1.0, 0.0, 0.0), abs=1e-9)


def test_zero_norm_quaternion_is_rejected():
    with pytest.raises(Record3DError, match="Degenerate"):
        quaternion_to_rotation_matrix(0.0, 0.0, 0.0, 0.0)


# ------------------------------------------------------- convention pass-thru


def test_arkit_basis_passes_through_with_no_axis_flip(tmp_path):
    """THE pin: ARKit's camera basis IS Atlas's, so the rotation is untouched.

    An identity ARKit pose means "camera at the world origin looking down -Z,
    +Y up" — which in Atlas is also the identity camera. If someone inserts a
    coordinate conversion here, this fails.
    """
    capture = Record3DCapture.open(_write_capture(tmp_path))
    solve = capture.solve(0)
    ex = solve.camera.extrinsics

    assert ex.camera_position == pytest.approx((0.0, 0.0, 0.0), abs=1e-12)
    for i in range(3):
        for j in range(3):
            expected = 1.0 if i == j else 0.0
            assert ex.camera_rotation_matrix[i][j] == pytest.approx(expected, abs=1e-12)
            assert ex.camera_view_matrix[i][j] == pytest.approx(expected, abs=1e-12)
    assert ex.up_axis == "Y"
    assert ex.coordinate_system == "right_handed"


def test_translation_lands_in_the_last_column_and_view_is_its_inverse(tmp_path):
    """Atlas Matrix4 is row-major, column-vector: translation in the last COLUMN."""
    pose = [0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 3.0]
    capture = Record3DCapture.open(_write_capture(tmp_path, _metadata(poses=[pose])))
    ex = capture.solve(0).camera.extrinsics

    assert ex.camera_position == pytest.approx((1.0, 2.0, 3.0))
    assert (ex.camera_world_matrix[0][3],
            ex.camera_world_matrix[1][3],
            ex.camera_world_matrix[2][3]) == pytest.approx((1.0, 2.0, 3.0))
    # view == inverse(world) for a rigid transform: -R^T @ t with R = I.
    assert (ex.camera_view_matrix[0][3],
            ex.camera_view_matrix[1][3],
            ex.camera_view_matrix[2][3]) == pytest.approx((-1.0, -2.0, -3.0))


def test_measured_pose_skips_the_negative_z_canonicalization(tmp_path):
    """Yaw is MEASURED here, so the single-image yaw convention must not apply.

    A 90 deg yaw must survive as a 90 deg yaw. solver._face_camera_toward_
    negative_z would rotate the world by 180 deg about Y and destroy the
    registration between frames of one capture.
    """
    s, c = math.sin(math.pi / 4), math.cos(math.pi / 4)
    pose = [0.0, s, 0.0, c, 0.0, 0.0, 0.0]
    capture = Record3DCapture.open(_write_capture(tmp_path, _metadata(poses=[pose])))
    solve = capture.solve(0)

    expected = quaternion_to_rotation_matrix(0.0, s, 0.0, c)
    for i in range(3):
        for j in range(3):
            assert solve.camera.extrinsics.camera_rotation_matrix[i][j] == pytest.approx(
                expected[i][j], abs=1e-9
            )
    assert solve.debug_metadata["canonical_negative_z_applied"] is False
    assert solve.debug_metadata["measured_pose"] is True


# -------------------------------------------------------------- intrinsics


def test_column_major_K_is_read_with_the_principal_point_intact(tmp_path):
    """Read row-major by mistake and cx/cy come out 0 (corner principal point)."""
    capture = Record3DCapture.open(_write_capture(tmp_path))
    intr = capture.solve(0).camera.intrinsics

    assert intr.fx_px == pytest.approx(FX)
    assert intr.fy_px == pytest.approx(FY)
    assert intr.cx_px == pytest.approx(CX)
    assert intr.cy_px == pytest.approx(CY)
    assert (intr.image_width, intr.image_height) == (RGB_W, RGB_H)
    assert intr.principal_point_px == pytest.approx((CX, CY))


def test_per_frame_intrinsics_win_over_the_session_K(tmp_path):
    """Switching lens mid-capture changes the focal; per-frame coeffs carry it."""
    meta = _metadata(
        poses=[IDENTITY_POSE, IDENTITY_POSE],
        perFrameIntrinsicCoeffs=[[FX, FY, CX, CY], [800.0, 800.0, CX, CY]],
    )
    capture = Record3DCapture.open(_write_capture(tmp_path, meta, frames=2))

    assert capture.solve(0).camera.intrinsics.fx_px == pytest.approx(FX)
    assert capture.solve(1).camera.intrinsics.fx_px == pytest.approx(800.0)


def test_intrinsics_expressed_at_depth_scale_are_rescaled_to_the_colour_frame(tmp_path):
    """Some exports write K at the 256x192 depth scale; a solve must be at RGB scale."""
    scale = DEPTH_W / RGB_W
    meta = _metadata(K=_column_major_K(FX * scale, FY * scale, CX * scale, CY * scale))
    capture = Record3DCapture.open(_write_capture(tmp_path, meta))
    intr = capture.solve(0).camera.intrinsics

    assert intr.fx_px == pytest.approx(FX, rel=1e-6)
    assert intr.cx_px == pytest.approx(CX, rel=1e-6)


def test_non_positive_focal_is_rejected(tmp_path):
    meta = _metadata(K=_column_major_K(fx=0.0))
    capture = Record3DCapture.open(_write_capture(tmp_path, meta))
    with pytest.raises(Record3DError, match="Non-positive focal"):
        capture.solve(0)


# ------------------------------------------------------------------- frames


def test_depth_frame_decodes_to_metric_float32_at_native_resolution(tmp_path):
    capture = Record3DCapture.open(_write_capture(tmp_path, depth_value=2.5))
    frame = capture.frame(0)

    assert frame.depth.shape == (DEPTH_H, DEPTH_W)
    assert frame.depth.dtype == np.float32
    assert float(frame.depth.mean()) == pytest.approx(2.5)
    assert frame.depth_size == (DEPTH_W, DEPTH_H)
    assert frame.rgb_jpeg is not None


def test_confidence_plane_decodes_to_arkit_levels(tmp_path):
    frame = Record3DCapture.open(_write_capture(tmp_path)).frame(0)
    assert frame.confidence.shape == (DEPTH_H, DEPTH_W)
    assert frame.confidence.dtype == np.uint8
    assert set(np.unique(frame.confidence)) <= {0, 1, 2}


def test_float16_depth_buffers_are_detected_by_length(tmp_path):
    path = tmp_path / "half.r3d"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("metadata", json.dumps(_metadata()))
        zf.writestr("rgbd/0.jpg", b"\xff\xd8\xff")
        zf.writestr("rgbd/0.depth", np.full((DEPTH_H, DEPTH_W), 1.5, np.float16).tobytes())

    frame = Record3DCapture.open(path).frame(0)
    assert frame.depth.dtype == np.float32
    assert float(frame.depth.mean()) == pytest.approx(1.5, abs=1e-3)


def test_wrong_sized_depth_buffer_names_the_mismatch(tmp_path):
    path = tmp_path / "bad.r3d"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("metadata", json.dumps(_metadata()))
        zf.writestr("rgbd/0.depth", b"\x00" * 999)

    with pytest.raises(Record3DError, match="999 bytes"):
        Record3DCapture.open(path).frame(0)


def test_frame_index_out_of_range(tmp_path):
    capture = Record3DCapture.open(_write_capture(tmp_path))
    with pytest.raises(IndexError, match="out of range"):
        capture.frame(7)


def test_extracted_capture_directory_reads_without_the_zip(tmp_path):
    src = _write_capture(tmp_path)
    extracted = tmp_path / "extracted"
    with zipfile.ZipFile(src) as zf:
        zf.extractall(extracted)

    capture = Record3DCapture.open(extracted)
    assert capture.n_frames == 1
    assert capture.frame(0).depth.shape == (DEPTH_H, DEPTH_W)


# -------------------------------------------------------------------- solve


def test_solve_is_stamped_as_a_measured_source_and_serializes(tmp_path):
    """Solve JSON is a contract — _json_ready must handle everything on it."""
    capture = Record3DCapture.open(_write_capture(tmp_path))
    solve = capture.solve(0)

    assert solve.source_method == SOURCE_METHOD
    assert solve.known_intrinsics_used is True
    assert solve.camera.focal_length_inferred is False

    payload = json.loads(solve.to_json())
    assert payload["source_method"] == SOURCE_METHOD
    assert payload["scale_health"]  # scene_health verdict rides along
    assert AtlasSolve.from_dict(payload).source_method == SOURCE_METHOD


def test_measured_confidence_is_high_for_focal_and_extrinsics_zero_for_vps(tmp_path):
    """Nothing here estimated a horizon or a vanishing point — say so honestly."""
    metrics = Record3DCapture.open(_write_capture(tmp_path)).solve(0).camera.confidence.individual_metrics

    assert metrics["focal"] >= 0.9
    assert metrics["extrinsics"] >= 0.9
    assert metrics["scale"] >= 0.8
    assert metrics["horizon"] == 0.0
    assert metrics["vp1"] == metrics["vp2"] == metrics["vp3"] == 0.0


def test_capture_reports_the_depth_resolution_gap_rather_than_hiding_it(tmp_path):
    """256x192 LiDAR under a 1920x1440 plate is the headline caveat, not a detail."""
    capture = Record3DCapture.open(_write_capture(tmp_path))
    joined = " ".join(capture.warnings)

    assert "256x192" in joined
    assert "METRIC" in joined
    assert "not a high-resolution surface" in joined


def test_device_hint_identifies_the_lidar_sensor(tmp_path):
    capture = Record3DCapture.open(_write_capture(tmp_path))
    assert "LiDAR" in capture.device_hint


def test_multi_frame_capture_exposes_every_pose(tmp_path):
    poses = [[0.0, 0.0, 0.0, 1.0, float(i), 0.0, 0.0] for i in range(4)]
    capture = Record3DCapture.open(_write_capture(tmp_path, _metadata(poses=poses), frames=4))

    assert capture.n_frames == 4
    xs = [capture.solve(i).camera.extrinsics.camera_position[0] for i in range(4)]
    assert xs == pytest.approx([0.0, 1.0, 2.0, 3.0])


# ------------------------------------------------------------------- errors


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        Record3DCapture.open(tmp_path / "nope.r3d")


def test_non_zip_is_named_as_such(tmp_path):
    path = tmp_path / "notes.r3d"
    path.write_text("this is not a zip")
    with pytest.raises(Record3DError, match="not a ZIP"):
        Record3DCapture.open(path)


def test_zip_without_metadata(tmp_path):
    path = tmp_path / "empty.r3d"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("rgbd/0.jpg", b"x")
    with pytest.raises(Record3DError, match="No `metadata` entry"):
        Record3DCapture.open(path)


def test_capture_without_poses_is_refused(tmp_path):
    meta = _metadata(poses=[])
    meta.pop("initPose")
    path = tmp_path / "noposes.r3d"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("metadata", json.dumps(meta))
    with pytest.raises(Record3DError, match="no camera poses"):
        Record3DCapture.open(path)


def test_init_pose_only_capture_falls_back_with_a_warning(tmp_path):
    meta = _metadata(poses=[])
    path = tmp_path / "single.r3d"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("metadata", json.dumps(meta))
    capture = Record3DCapture.open(path)

    assert capture.n_frames == 1
    assert any("initPose" in w for w in capture.warnings)


def test_malformed_metadata_json(tmp_path):
    path = tmp_path / "bad.r3d"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("metadata", "{not json")
    with pytest.raises(Record3DError, match="not valid JSON"):
        Record3DCapture.open(path)


def test_capture_closes_its_archive(tmp_path):
    with Record3DCapture.open(_write_capture(tmp_path)) as capture:
        assert capture.n_frames == 1
    assert capture._zip is None
