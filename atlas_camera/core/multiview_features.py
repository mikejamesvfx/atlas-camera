"""Deterministic SIFT evidence and correspondence visualisation.

OpenCV and NumPy are deliberately imported only at the vision boundary so the
core package remains importable without the optional vision extra.
"""

from __future__ import annotations

from typing import Any

from atlas_camera.core.multiview_types import FeatureSet, PairMatches, QualityProfile


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("multi-view features need numpy — pip install -e .[vision]") from exc
    return np


def _require_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("multi-view features need opencv-python — pip install -e .[vision]") from exc
    return cv2


def _gray_uint8(image: Any) -> Any:
    """Convert a gray or RGB image to explicitly rounded uint8 grayscale."""
    np = _require_numpy()
    pixels = np.asarray(image)
    if pixels.ndim == 2:
        gray = pixels
    elif pixels.ndim == 3 and pixels.shape[2] >= 3:
        rgb = pixels[..., :3]
        gray = rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114
    else:
        raise ValueError("image must be a grayscale or RGB array")
    gray = np.asarray(gray, dtype=np.float64)
    if np.issubdtype(pixels.dtype, np.floating) and gray.size and float(np.nanmax(gray)) <= 1.0:
        gray *= 255.0
    return np.rint(np.clip(np.nan_to_num(gray, nan=0.0, posinf=255.0, neginf=0.0), 0, 255)).astype(np.uint8)


def extract_features(image: Any, profile: QualityProfile) -> FeatureSet:
    """Extract SIFT features in a process-independent, stable ordering."""
    np = _require_numpy()
    cv2 = _require_cv2()
    gray = _gray_uint8(image)
    keypoints, descriptors = cv2.SIFT_create(nfeatures=profile.max_features).detectAndCompute(
        gray, None,
    )
    keypoints = keypoints or []
    order = sorted(range(len(keypoints)), key=lambda i: (
        round(keypoints[i].pt[1], 6), round(keypoints[i].pt[0], 6),
        -round(keypoints[i].response, 9), keypoints[i].octave,
        round(keypoints[i].size, 6), round(keypoints[i].angle, 6), i,
    ))
    if descriptors is None:
        descriptors = np.empty((0, 128), dtype=np.float32)
    descriptors = np.asarray(descriptors, dtype=np.float32)
    if order:
        descriptors = descriptors[order]
    else:
        descriptors = descriptors.reshape((0, descriptors.shape[1] if descriptors.ndim == 2 else 128))
    points_xy = np.asarray([keypoints[i].pt for i in order], dtype=np.float32).reshape((-1, 2))
    responses = np.asarray([keypoints[i].response for i in order], dtype=np.float32)
    # The original SIFT position remains available for downstream diagnostics,
    # while the arrays above are ordered by the deterministic key.
    stable_indices = np.asarray(order, dtype=np.int64)
    return FeatureSet(points_xy, descriptors, responses, stable_indices,
                      (int(gray.shape[1]), int(gray.shape[0])))


def _ratio_matches(knn_matches: Any, ratio: float) -> dict[int, tuple[int, float]]:
    return {
        int(candidates[0].queryIdx): (int(candidates[0].trainIdx), float(candidates[0].distance))
        for candidates in knn_matches
        if len(candidates) == 2 and candidates[0].distance < ratio * candidates[1].distance
    }


def _occupied_grid_cells(points_xy: Any, image_size: tuple[int, int] | None) -> int:
    np = _require_numpy()
    if len(points_xy) == 0 or image_size is None:
        return 0
    width, height = image_size
    if width <= 0 or height <= 0:
        return 0
    scaled = np.floor(points_xy / np.array((width, height), dtype=np.float32) * 4.0).astype(np.int64)
    scaled = np.clip(scaled, 0, 3)
    return len({(int(cell[0]), int(cell[1])) for cell in scaled})


def match_features(a: FeatureSet, b: FeatureSet, profile: QualityProfile,
                   frame_a: int, frame_b: int) -> PairMatches:
    """Return mutual Lowe-ratio matches sorted by stable feature index."""
    np = _require_numpy()
    cv2 = _require_cv2()
    desc_a = np.asarray(a.descriptors, dtype=np.float32)
    desc_b = np.asarray(b.descriptors, dtype=np.float32)
    if len(desc_a) < 2 or len(desc_b) < 2:
        return PairMatches(frame_a, frame_b, np.empty((0, 2), np.float32),
                           np.empty((0, 2), np.float32), np.empty((0, 2), np.int64),
                           np.empty((0,), np.float32), 0)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward = _ratio_matches(matcher.knnMatch(desc_a, desc_b, k=2), profile.ratio)
    backward = _ratio_matches(matcher.knnMatch(desc_b, desc_a, k=2), profile.ratio)
    pairs = [
        (int(a.stable_indices[query]), int(b.stable_indices[train]), distance, query, train)
        for query, (train, distance) in forward.items()
        if backward.get(train, (None,))[0] == query
    ]
    pairs.sort(key=lambda item: (item[0], item[1], item[2]))
    used_a: set[int] = set()
    used_b: set[int] = set()
    unique_pairs = []
    for pair in pairs:
        if pair[0] not in used_a and pair[1] not in used_b:
            unique_pairs.append(pair)
            used_a.add(pair[0])
            used_b.add(pair[1])
    indices = np.asarray([(pair[0], pair[1]) for pair in unique_pairs], dtype=np.int64).reshape((-1, 2))
    points_a = np.asarray([a.points_xy[pair[3]] for pair in unique_pairs], dtype=np.float32).reshape((-1, 2))
    points_b = np.asarray([b.points_xy[pair[4]] for pair in unique_pairs], dtype=np.float32).reshape((-1, 2))
    distances = np.asarray([pair[2] for pair in unique_pairs], dtype=np.float32)
    return PairMatches(frame_a, frame_b, points_a, points_b, indices, distances,
                       _occupied_grid_cells(points_a, a.image_size))


def _overlay_image(image: Any) -> Any:
    np = _require_numpy()
    pixels = np.asarray(image)
    if pixels.ndim == 2:
        return np.repeat(_gray_uint8(pixels)[..., None], 3, axis=2)
    if pixels.ndim == 3 and pixels.shape[2] >= 3:
        return np.stack([_gray_uint8(pixels[..., channel]) for channel in range(3)], axis=2)
    raise ValueError("image must be a grayscale or RGB array")


def render_match_overlay(image_a: Any, image_b: Any, matches: PairMatches,
                         inlier_mask: Any | None = None) -> Any:
    """Render a deterministic side-by-side correspondence overlay in RGB."""
    np = _require_numpy()
    cv2 = _require_cv2()
    left = _overlay_image(image_a)
    right = _overlay_image(image_b)
    height = max(left.shape[0], right.shape[0])
    canvas = np.zeros((height, left.shape[1] + right.shape[1], 3), dtype=np.uint8)
    canvas[:left.shape[0], :left.shape[1]] = left
    canvas[:right.shape[0], left.shape[1]:] = right
    count = len(matches.indices)
    accepted = np.ones(count, dtype=bool) if inlier_mask is None else np.asarray(inlier_mask, dtype=bool).reshape(-1)
    if len(accepted) != count:
        raise ValueError("inlier_mask must contain one entry per match")
    for index, (point_a, point_b) in enumerate(zip(matches.points_a, matches.points_b)):
        colour = (0, 255, 0) if accepted[index] else (255, 64, 64)
        start = (int(round(float(point_a[0]))), int(round(float(point_a[1]))))
        end = (int(round(float(point_b[0]))) + left.shape[1], int(round(float(point_b[1]))))
        cv2.line(canvas, start, end, colour, 1, lineType=cv2.LINE_8)
        cv2.circle(canvas, start, 3, colour, -1, lineType=cv2.LINE_8)
        cv2.circle(canvas, end, 3, colour, -1, lineType=cv2.LINE_8)
    return canvas
