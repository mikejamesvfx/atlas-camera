"""AtlasFaceScaleReference — the ComfyUI boundary for face-based metric scale.

The core math is verified by construction in test_face_scale.py. What matters
here is the boundary: reading a marked extent off a MASK or a hand-typed bbox,
measuring the metric's OWN axis rather than whichever is longer, honouring the
confirm gate, and refusing loudly instead of emitting a plausible wrong number.
"""

from __future__ import annotations

import json
import math

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from atlas_camera.comfy.nodes_solve import AtlasFaceScaleReference  # noqa: E402
from atlas_camera.core.face_scale import FACE_METRICS  # noqa: E402

FX = FY = 1200.0
CX, CY = 960.0, 540.0
W, H = 1920, 1080


def _solve(camera_height=1.6):
    """Identity-rotation solve at a known height — camera looks down -Z."""
    from atlas_camera.core.intrinsics import build_intrinsics
    from atlas_camera.core.schema import AtlasCamera, AtlasExtrinsics, AtlasSolve

    intr = build_intrinsics(image_width=W, image_height=H, fx_px=FX, fy_px=FY,
                            principal_point_px=(CX, CY))
    identity = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    view = ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, -camera_height),
            (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    world = ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, camera_height),
             (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    return AtlasSolve(
        camera=AtlasCamera(
            intrinsics=intr,
            extrinsics=AtlasExtrinsics(
                camera_position=(0.0, camera_height, 0.0),
                camera_rotation_matrix=identity,
                camera_world_matrix=world,
                camera_view_matrix=view),
        ),
        image_width=W, image_height=H, source_method="test",
    )


def _project(point_world):
    x, y, z = point_world
    return (CX + FX * (x / -z), CY - FY * (y / -z))


def _face_bbox(camera_height=1.6, stature=1.70, distance=3.0,
               metric="head_chin_to_crown"):
    """Pixel bbox of a marked feature on an upright subject at a known height.

    Built in world metres relative to the camera at the origin, then projected —
    same construction as the core tests, so the node must recover camera_height.
    """
    spec = FACE_METRICS[metric]
    anchor_y = -camera_height + spec["anchor_ratio"] * stature
    size = spec["size_m"]
    if spec["axis"] == "vertical":
        bottom = _project((0.0, anchor_y, -distance))
        top = _project((0.0, anchor_y + size, -distance))
        # A head box has width too; give it a plausible one so the node has to
        # pick the vertical axis rather than just taking the larger extent.
        half_w = abs(top[0] - bottom[0]) + 40.0
        return (bottom[0] - half_w, top[1], bottom[0] + half_w, bottom[1])
    left = _project((-size / 2.0, anchor_y, -distance))
    right = _project((size / 2.0, anchor_y, -distance))
    half_h = 30.0
    return (left[0], left[1] - half_h, right[0], right[1] + half_h)


def _mask_from_bbox(bbox):
    x0, y0, x1, y1 = (int(round(v)) for v in bbox)
    m = np.zeros((H, W), dtype=np.float32)
    m[y0:y1 + 1, x0:x1 + 1] = 1.0
    return torch.from_numpy(m).unsqueeze(0)


def _bbox_str(bbox):
    return ",".join(f"{v:.4f}" for v in bbox)


# ------------------------------------------------------------------- measuring


def test_bbox_override_recovers_the_camera_height():
    truth = 1.6
    solve, height, measurement, report = AtlasFaceScaleReference().measure(
        _solve(), bbox_override=_bbox_str(_face_bbox(truth)))

    assert height == pytest.approx(truth, rel=1e-3)
    assert json.loads(measurement)["marked_from"] == "bbox_override"
    assert "camera height" in report


def test_face_mask_recovers_the_camera_height():
    truth = 1.6
    _, height, measurement, _ = AtlasFaceScaleReference().measure(
        _solve(), face_mask=_mask_from_bbox(_face_bbox(truth)))

    # Mask bbox is integer-quantised, so tolerance is looser than the bbox path.
    assert height == pytest.approx(truth, rel=5e-3)
    assert json.loads(measurement)["marked_from"] == "face_mask"


@pytest.mark.parametrize("metric", list(FACE_METRICS))
def test_every_metric_measures_its_own_axis(metric):
    """A horizontal metric must read the box's WIDTH, a vertical one its HEIGHT.

    The fixtures give each box a deliberately non-square shape, so a node that
    just took the larger extent would fail on at least one metric.
    """
    truth = 1.55
    _, height, _, _ = AtlasFaceScaleReference().measure(
        _solve(), metric=metric, bbox_override=_bbox_str(_face_bbox(truth, metric=metric)))
    assert height == pytest.approx(truth, rel=1e-3), metric


def test_wrong_metric_scales_distance_hard_but_camera_height_only_mildly():
    """Where the face-vs-head confusion actually shows up.

    The two constants differ by ~21% (0.185 vs 0.235 m), and that lands almost
    entirely on the measured SUBJECT DISTANCE, which scales proportionally.
    Camera height moves much less, because it is
    ``anchor_height_above_ground - anchor_world_Y`` and only the second term
    rescales — so a face near camera level barely shifts it. Worth pinning: it
    is tempting to describe the mix-up as "scales the whole solve", and that
    overstates it.
    """
    box = _bbox_str(_face_bbox(1.6, metric="head_chin_to_crown"))
    _, as_head, m_head, _ = AtlasFaceScaleReference().measure(
        _solve(), metric="head_chin_to_crown", bbox_override=box)
    _, as_face, m_face, _ = AtlasFaceScaleReference().measure(
        _solve(), metric="face_chin_to_hairline", bbox_override=box)

    d_head = json.loads(m_head)["distance_m"]
    d_face = json.loads(m_face)["distance_m"]
    assert d_face / d_head == pytest.approx(0.185 / 0.235, rel=1e-3)

    # Camera height still moves measurably — just far less than the distance.
    assert 0.01 < abs(as_head - as_face) < 0.05


def test_stature_moves_the_result_and_is_recorded():
    box = _bbox_str(_face_bbox(1.6))
    _, short, m_short, _ = AtlasFaceScaleReference().measure(
        _solve(), bbox_override=box, stature_m=1.55)
    _, tall, _, _ = AtlasFaceScaleReference().measure(
        _solve(), bbox_override=box, stature_m=1.90)

    assert tall > short
    assert json.loads(m_short)["stature_m"] == pytest.approx(1.55)


def test_size_override_replaces_the_population_constant():
    box = _bbox_str(_face_bbox(1.6))
    _, _, measurement, _ = AtlasFaceScaleReference().measure(
        _solve(), bbox_override=box, size_override_m=0.28)
    assert json.loads(measurement)["real_size_m"] == pytest.approx(0.28)


def test_bbox_override_wins_over_a_wired_mask():
    truth = 1.6
    _, height, measurement, _ = AtlasFaceScaleReference().measure(
        _solve(), face_mask=_mask_from_bbox((10.0, 10.0, 40.0, 90.0)),
        bbox_override=_bbox_str(_face_bbox(truth)))
    assert json.loads(measurement)["marked_from"] == "bbox_override"
    assert height == pytest.approx(truth, rel=1e-3)


# ---------------------------------------------------------------- confirm gate


def test_measuring_alone_does_not_rescale_the_solve():
    """An auto-detected reference is never auto-promoted (AtlasApplyScaleReferences doctrine)."""
    solve = _solve(1.6)
    before = solve.camera.extrinsics.camera_position[1]
    out, height, _, report = AtlasFaceScaleReference().measure(
        solve, bbox_override=_bbox_str(_face_bbox(2.4)))

    assert out.camera.extrinsics.camera_position[1] == pytest.approx(before)
    assert height == pytest.approx(2.4, rel=1e-3)   # measured...
    assert "measured only" in report                # ...but not applied
    assert out.debug_metadata["face_scale"]["adopted"] is False
    assert "face_scale" not in out.source_method


def test_confirm_rescales_and_stamps_provenance():
    solve = _solve(1.6)
    out, height, _, report = AtlasFaceScaleReference().measure(
        solve, bbox_override=_bbox_str(_face_bbox(2.4)), confirm=True)

    assert out.camera.extrinsics.camera_position[1] == pytest.approx(2.4, rel=1e-3)
    assert out.debug_metadata["scale_source"] == "face_reference"
    assert out.source_method.endswith("+face_scale")
    assert "RESCALED" in report


def test_rescaled_solve_still_serializes():
    """Solve JSON is a contract — the face measurement must be _json_ready-safe."""
    out, _, _, _ = AtlasFaceScaleReference().measure(
        _solve(), bbox_override=_bbox_str(_face_bbox(1.6)), confirm=True)
    payload = json.loads(out.to_json())
    assert payload["debug_metadata"]["face_scale"]["adopted"] is True
    assert payload["scale_health"]


def test_confirm_does_not_rescale_when_the_measurement_failed():
    solve = _solve(1.6)
    before = solve.camera.extrinsics.camera_position[1]
    out, height, _, report = AtlasFaceScaleReference().measure(
        solve, bbox_override="900,10,1000,60", confirm=True)  # face above the camera

    assert height == 0.0
    assert out.camera.extrinsics.camera_position[1] == pytest.approx(before)
    assert "NO RESULT" in report


# -------------------------------------------------------------------- refusals


def test_nothing_marked_is_refused():
    with pytest.raises(ValueError, match="nothing marked"):
        AtlasFaceScaleReference().measure(_solve())


def test_empty_mask_explains_the_silent_sam3_miss():
    with pytest.raises(ValueError, match="SILENT"):
        AtlasFaceScaleReference().measure(
            _solve(), face_mask=torch.zeros((1, H, W), dtype=torch.float32))


@pytest.mark.parametrize("bad", ["1,2,3", "a,b,c,d", "1,2,3,4,5"])
def test_malformed_bbox_override_is_named(bad):
    with pytest.raises(ValueError, match="bbox_override"):
        AtlasFaceScaleReference().measure(_solve(), bbox_override=bad)


def test_reversed_bbox_corners_are_normalised():
    x0, y0, x1, y1 = _face_bbox(1.6)
    _, height, _, _ = AtlasFaceScaleReference().measure(
        _solve(), bbox_override=_bbox_str((x1, y1, x0, y0)))
    assert height == pytest.approx(1.6, rel=1e-3)


def test_report_flags_the_tier_and_the_stronger_alternative():
    """The node must not oversell itself against a ground reference."""
    _, _, _, report = AtlasFaceScaleReference().measure(
        _solve(), bbox_override=_bbox_str(_face_bbox(1.6)))
    assert "tier-1.5" in report
    assert "AtlasReferenceScaleSolve" in report
    assert "±" in report  # uncertainty band is shown, not hidden


def test_measurement_carries_no_biometric_identifier():
    """Only a distance between two points — no landmarks, descriptors, embeddings."""
    _, _, measurement, _ = AtlasFaceScaleReference().measure(
        _solve(), bbox_override=_bbox_str(_face_bbox(1.6)))
    payload = json.loads(measurement)

    forbidden = {"embedding", "descriptor", "landmarks", "encoding", "identity", "keypoints"}
    assert not (forbidden & set(payload)), payload.keys()
    assert payload["trust_tier"] == "1.5_face_reference"


def test_node_is_registered_under_a_stable_key():
    from atlas_camera.comfy.node_registry import (
        NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS,
    )
    assert NODE_CLASS_MAPPINGS["AtlasFaceScaleReference"] is AtlasFaceScaleReference
    assert NODE_DISPLAY_NAME_MAPPINGS["AtlasFaceScaleReference"] == "Atlas Face Scale Reference 🙂"
