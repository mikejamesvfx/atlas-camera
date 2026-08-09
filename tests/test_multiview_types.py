"""Contracts for deterministic RAW multi-view registration."""

import random

import numpy as np
import pytest

from atlas_camera.core.multiview_types import (
    MultiViewFrame,
    MultiViewSettings,
    RegistrationOutcome,
    registration_fingerprint,
)
from atlas_camera.raw.pipeline import RawImportResult


def _frames():
    return (
        MultiViewFrame(
            image=np.arange(12, dtype=np.uint8).reshape(2, 2, 3),
            raw_meta={"camera_model": "Atlas A", "orientation": 1},
            label="a",
        ),
        MultiViewFrame(
            image=np.arange(12, 24, dtype=np.uint8).reshape(2, 2, 3),
            raw_meta={"camera_model": "Atlas A", "orientation": 6},
            label="b",
        ),
    )


def _raw_import_result(*, orientation=1):
    image = np.zeros((1, 1, 3), dtype=np.float32)
    return RawImportResult(
        linear_rgb=image,
        display_srgb=image,
        width=1,
        height=1,
        focal_length_mm=35.0,
        sensor_width_mm=36.0,
        sensor_height_mm=24.0,
        sensor_source="camera_db",
        camera_make="Atlas",
        camera_model="A",
        lens_model="35mm",
        undistort_applied=False,
        undistort_status="disabled",
        orientation=orientation,
    )


def test_settings_reject_unknown_values():
    with pytest.raises(ValueError, match="capture_mode"):
        MultiViewSettings(capture_mode="guess")
    with pytest.raises(ValueError, match="match_quality"):
        MultiViewSettings(match_quality="reckless")


def test_fingerprint_changes_with_order_and_seed_but_not_ambient_rng():
    a, b = _frames()
    first = registration_fingerprint([a, b], MultiViewSettings(seed=7))
    random.seed(999)
    np.random.seed(999)
    assert registration_fingerprint([a, b], MultiViewSettings(seed=7)) == first
    assert registration_fingerprint([b, a], MultiViewSettings(seed=7)) != first
    assert registration_fingerprint([a, b], MultiViewSettings(seed=8)) != first


def test_failed_outcome_serializes_without_a_solve():
    out = RegistrationOutcome.failed("insufficient_overlap", "12 matches")
    assert out.solve is None
    assert out.diagnostics.to_dict()["outcome_code"] == "insufficient_overlap"


def test_raw_import_result_preserves_exif_orientation():
    result = _raw_import_result(orientation=6)
    assert result.orientation == 6
