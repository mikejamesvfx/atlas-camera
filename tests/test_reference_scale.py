"""Tests for the reference-object metric scale tier (single-view geometry)."""

import numpy as np
import pytest

from atlas_camera.core.solver import (
    metric_height_from_reference,
    resolve_reference_scale,
    _rotation_from_up_vector,
)


def _project(P_world, R, fx, fy, cx, cy):
    """Project a world point (camera at origin) to a pixel using Atlas conventions."""
    p = R @ np.asarray(P_world, dtype=float)      # world -> cam
    u = cx + fx * p[0] / (-p[2])
    v = cy + fy * p[1] / p[2]
    return (float(u), float(v))


def _scene(height_m, H_obj, R, fx=800.0, fy=800.0, W=1024, H=1024, lateral=0.3):
    """Base/top pixels of a vertical object standing on the ground, camera at Y=0.

    Places the object along the camera's actual forward axis (nudged downward so the
    base lands on the ground plane Y=-height_m and stays in front of the camera),
    which works for any orientation R.
    """
    cx = cy = W / 2.0
    c2w = R.T
    fwd = c2w @ np.array([0.0, 0.0, -1.0])
    right = c2w @ np.array([1.0, 0.0, 0.0])
    d = fwd + np.array([0.0, -0.4, 0.0]) + lateral * right   # ensure a downward ground hit
    d = d / np.linalg.norm(d)
    s = -height_m / d[1]                                     # scale onto ground Y=-height_m
    base = s * d
    top = base + np.array([0.0, H_obj, 0.0])
    return _project(base, R, fx, fy, cx, cy), _project(top, R, fx, fy, cx, cy), (fx, fy, cx, cy)


def test_recovers_camera_height_level_camera():
    R = np.eye(3)
    base_px, top_px, (fx, fy, cx, cy) = _scene(1.6, 1.75, R)
    out = metric_height_from_reference(base_px, top_px, 1.75,
                                       rotation=R, fx=fx, fy=fy, cx=cx, cy=cy)
    assert out["camera_height"] == pytest.approx(1.6, abs=0.02)


def test_recovers_camera_height_pitched_camera():
    # Camera pitched down ~10 degrees (looking toward the ground).
    p = np.radians(-10.0)
    up = (0.0, np.cos(p), -np.sin(p))
    R = _rotation_from_up_vector(up)
    base_px, top_px, (fx, fy, cx, cy) = _scene(2.5, 1.75, R)
    out = metric_height_from_reference(base_px, top_px, 1.75,
                                       rotation=R, fx=fx, fy=fy, cx=cx, cy=cy)
    assert out["camera_height"] == pytest.approx(2.5, abs=0.05)


def test_resolve_uses_registry_reference_id():
    R = np.eye(3)
    base_px, top_px, (fx, fy, cx, cy) = _scene(1.6, 1.75, R)  # person height
    result = resolve_reference_scale(
        [{"reference_id": "person_175cm", "base_px": base_px, "top_px": top_px}],
        rotation=R, fx=fx, fy=fy, cx=cx, cy=cy,
    )
    assert result["camera_height"] == pytest.approx(1.6, abs=0.03)
    assert result["confidence"] > 0.5


def test_resolve_aggregates_multiple_references():
    R = np.eye(3)
    b1, t1, (fx, fy, cx, cy) = _scene(1.6, 1.75, R, lateral=-0.8)
    b2, t2, _ = _scene(1.6, 2.1, R, lateral=1.2)   # a door, off to the side
    result = resolve_reference_scale(
        [
            {"reference_id": "person_175cm", "base_px": b1, "top_px": t1},
            {"reference_id": "door_210cm", "base_px": b2, "top_px": t2},
        ],
        rotation=R, fx=fx, fy=fy, cx=cx, cy=cy,
    )
    assert result["camera_height"] == pytest.approx(1.6, abs=0.05)


def test_bbox_input_form():
    R = np.eye(3)
    base_px, top_px, (fx, fy, cx, cy) = _scene(1.6, 1.75, R)
    # bbox xyxy around the object (base = bottom = larger y).
    bbox = [base_px[0] - 20, top_px[1], base_px[0] + 20, base_px[1]]
    result = resolve_reference_scale(
        [{"reference_id": "person_175cm", "bbox_px": bbox}],
        rotation=R, fx=fx, fy=fy, cx=cx, cy=cy,
    )
    assert result["camera_height"] == pytest.approx(1.6, abs=0.05)


def test_reference_above_horizon_is_rejected():
    R = np.eye(3)
    # An object floating above the horizon (base ray points upward) is invalid.
    out = metric_height_from_reference((512, 300), (512, 100), 1.75,
                                       rotation=R, fx=800, fy=800, cx=512, cy=512)
    assert out["camera_height"] is None
