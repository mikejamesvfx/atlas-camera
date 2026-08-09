"""Regression tests for deterministic multi-view SIFT evidence."""

from __future__ import annotations

import hashlib
import random

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

from atlas_camera.core.multiview_features import (
    extract_features,
    match_features,
    render_match_overlay,
)
from atlas_camera.core.multiview_types import QUALITY_PROFILES


def _shifted_checkerboard() -> tuple[object, object]:
    """Make repeatable, locally distinctive texture across the full image."""
    noise = np.random.default_rng(20260809).integers(0, 256, (320, 320), dtype=np.uint8)
    image = np.repeat(noise[..., None], 3, axis=2)
    for y in range(0, 320, 32):
        for x in range(0, 320, 32):
            if (x // 32 + y // 32) % 2:
                image[y:y + 32, x:x + 32] = (210, 210, 210)
            cv2.circle(image, (x + 9, y + 11), 3 + ((x + y) // 32) % 4, (80, 160, 245), -1)
            cv2.line(image, (x + 3, y + 27), (x + 25, y + 5), (255, 80, 40), 1)
    shift = np.float32([[1, 0, 11], [0, 1, 7]])
    return image, cv2.warpAffine(image, shift, (320, 320), borderValue=(0, 0, 0))


def test_feature_and_match_order_is_exactly_repeatable():
    left, right = _shifted_checkerboard()
    profile = QUALITY_PROFILES["balanced"]
    a1 = extract_features(left, profile)
    b1 = extract_features(right, profile)
    m1 = match_features(a1, b1, profile, 0, 1)

    random.seed(41)
    np.random.seed(41)
    cv2.setRNGSeed(41)
    a2 = extract_features(left, profile)
    b2 = extract_features(right, profile)
    m2 = match_features(a2, b2, profile, 0, 1)

    np.testing.assert_array_equal(a1.points_xy, a2.points_xy)
    np.testing.assert_array_equal(a1.stable_indices, a2.stable_indices)
    np.testing.assert_array_equal(m1.indices, m2.indices)
    np.testing.assert_array_equal(m1.distances, m2.distances)


def test_matches_are_mutual_spatially_distributed_and_overlay_is_stable():
    left, right = _shifted_checkerboard()
    profile = QUALITY_PROFILES["balanced"]
    matches = match_features(
        extract_features(left, profile), extract_features(right, profile), profile, 0, 1,
    )

    assert len(matches.indices) >= profile.min_grid_cells
    assert len(set(matches.indices[:, 0])) == len(matches.indices)
    assert len(set(matches.indices[:, 1])) == len(matches.indices)
    assert matches.occupied_grid_cells >= profile.min_grid_cells

    inliers = np.arange(len(matches.indices)) % 2 == 0
    overlay_a = render_match_overlay(left, right, matches, inliers)
    overlay_b = render_match_overlay(left, right, matches, inliers)
    assert overlay_a.shape == (320, 640, 3)
    assert hashlib.sha256(overlay_a.tobytes()).hexdigest() == hashlib.sha256(overlay_b.tobytes()).hexdigest()
