"""Vanishing-point geometry and image detection utilities."""

from __future__ import annotations

from math import hypot
from typing import Any

from atlas_camera.core.schema import AtlasHorizon, AtlasVanishingPoint, Point2D

Line = tuple[float, float, float]
LineSegment = tuple[Point2D, Point2D]


def _require_vision() -> tuple[Any, Any]:
    try:
        import numpy as np
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "Vanishing point image detection requires numpy and opencv-python. "
            "Install with: pip install -e .[vision]"
        ) from exc
    return np, cv2


def _to_point2d(value: Any) -> Point2D:
    return (float(value[0]), float(value[1]))


def _line_segment_from_array(line: Any) -> LineSegment:
    return ((float(line[0]), float(line[1])), (float(line[2]), float(line[3])))


def normalize_line_segment(line: Any) -> LineSegment:
    """Normalize common artist/API line formats into two image points."""

    if isinstance(line, dict):
        start = line.get("start", line.get("p0", line.get("from")))
        end = line.get("end", line.get("p1", line.get("to")))
        if start is None or end is None:
            raise ValueError("Line dictionaries must contain start/end or p0/p1 points.")
        return _to_point2d(start), _to_point2d(end)

    if len(line) == 4 and not isinstance(line[0], (list, tuple)):
        return (float(line[0]), float(line[1])), (float(line[2]), float(line[3]))

    if len(line) == 2:
        return _to_point2d(line[0]), _to_point2d(line[1])

    raise ValueError(f"Unsupported line segment format: {line!r}")


def flatten_line_segment(line: LineSegment) -> tuple[float, float, float, float]:
    return (line[0][0], line[0][1], line[1][0], line[1][1])


def fit_vanishing_point_from_lines(
    lines: list[Any],
    *,
    direction_label: str | None = None,
) -> AtlasVanishingPoint:
    """Fit a vanishing point to two or more manually supplied line segments."""

    segments = [normalize_line_segment(line) for line in lines]
    if len(segments) < 2:
        raise ValueError("At least two line segments are required for a vanishing point.")

    coefficients = [line_from_points(*segment) for segment in segments]
    sum_aa = sum(a * a for a, _, _ in coefficients)
    sum_ab = sum(a * b for a, b, _ in coefficients)
    sum_bb = sum(b * b for _, b, _ in coefficients)
    sum_ac = sum(a * c for a, _, c in coefficients)
    sum_bc = sum(b * c for _, b, c in coefficients)
    determinant = (sum_aa * sum_bb) - (sum_ab * sum_ab)
    if abs(determinant) < 1e-9:
        raise ValueError("Line group is degenerate; cannot fit a stable vanishing point.")

    x = ((-sum_ac * sum_bb) - (sum_ab * -sum_bc)) / determinant
    y = ((sum_aa * -sum_bc) - (-sum_ac * sum_ab)) / determinant
    return AtlasVanishingPoint(
        position_px=(x, y),
        direction_label=direction_label,
        confidence=min(1.0, len(segments) / 4.0),
        supporting_lines=segments,
    )


def line_from_points(p0: Point2D, p1: Point2D) -> Line:
    x0, y0 = p0
    x1, y1 = p1
    a = y0 - y1
    b = x1 - x0
    c = (x0 * y1) - (x1 * y0)
    scale = hypot(a, b)
    if scale == 0:
        raise ValueError("Cannot create a line from identical points.")
    return (a / scale, b / scale, c / scale)


def intersect_lines(line_a: Line, line_b: Line, *, epsilon: float = 1e-9) -> Point2D | None:
    a0, b0, c0 = line_a
    a1, b1, c1 = line_b
    determinant = (a0 * b1) - (a1 * b0)
    if abs(determinant) < epsilon:
        return None
    x = ((b0 * c1) - (b1 * c0)) / determinant
    y = ((c0 * a1) - (c1 * a0)) / determinant
    return (x, y)


def vanishing_point_from_line_pair(
    line_a: LineSegment,
    line_b: LineSegment,
    *,
    direction_label: str | None = None,
) -> AtlasVanishingPoint | None:
    point = intersect_lines(line_from_points(*line_a), line_from_points(*line_b))
    if point is None:
        return None
    return AtlasVanishingPoint(
        position_px=point,
        direction_label=direction_label,
        confidence=0.5,
        supporting_lines=[line_a, line_b],
    )


def horizon_from_vanishing_points(
    first: AtlasVanishingPoint,
    second: AtlasVanishingPoint,
    *,
    image_width: int | None = None,
) -> AtlasHorizon:
    line = line_from_points(first.position_px, second.position_px)
    endpoints = None
    if image_width is not None:
        a, b, c = line
        if abs(b) > 1e-9:
            y0 = -c / b
            y1 = -(a * image_width + c) / b
            endpoints = ((0.0, y0), (float(image_width), y1))
    return AtlasHorizon(
        line_coefficients=line,
        endpoints_px=endpoints,
        confidence=min(first.confidence, second.confidence),
    )


class VanishingPointDetector:
    """Detect vanishing points from an image using Canny, Hough, and RANSAC."""

    @staticmethod
    def detect_lines(
        image: Any,
        canny_low: int = 50,
        canny_high: int = 150,
        hough_threshold: int = 80,
        min_line_length: int = 50,
        max_line_gap: int = 10,
    ) -> Any:
        np, cv2 = _require_vision()
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        edges = cv2.Canny(gray, canny_low, canny_high, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=hough_threshold,
            minLineLength=min_line_length,
            maxLineGap=max_line_gap,
        )
        if lines is None:
            return np.empty((0, 4), dtype=np.float64)
        return lines.reshape(-1, 4).astype(np.float64)

    @staticmethod
    def filter_lines(
        lines: Any,
        min_angle: float = 5.0,
        max_angle: float = 85.0,
        min_length: float = 30.0,
    ) -> Any:
        np, _ = _require_vision()
        if len(lines) == 0:
            return lines

        dx = lines[:, 2] - lines[:, 0]
        dy = lines[:, 3] - lines[:, 1]
        angles = np.abs(np.degrees(np.arctan2(dy, dx)))
        angles = np.minimum(angles, 180 - angles)
        lengths = np.sqrt(dx**2 + dy**2)
        mask = (angles >= min_angle) & (angles <= max_angle) & (lengths >= min_length)
        return lines[mask]

    @staticmethod
    def line_intersection(line1: Any, line2: Any) -> Any | None:
        np, _ = _require_vision()
        p1 = np.array([line1[0], line1[1], 1.0])
        p2 = np.array([line1[2], line1[3], 1.0])
        p3 = np.array([line2[0], line2[1], 1.0])
        p4 = np.array([line2[2], line2[3], 1.0])

        l1 = np.cross(p1, p2)
        l2 = np.cross(p3, p4)
        intersection = np.cross(l1, l2)
        if abs(intersection[2]) < 1e-10:
            return None
        return intersection[:2] / intersection[2]

    @classmethod
    def classify_lines_by_angle(
        cls,
        lines: Any,
        angle_threshold: float = 30.0,
    ) -> tuple[Any, Any, Any]:
        np, _ = _require_vision()
        if len(lines) == 0:
            empty = np.empty((0, 4), dtype=np.float64)
            return empty, empty, empty

        dx = lines[:, 2] - lines[:, 0]
        dy = lines[:, 3] - lines[:, 1]
        angles = np.degrees(np.arctan2(dy, dx))

        vertical_mask = np.abs(np.abs(angles) - 90) < angle_threshold
        non_vertical = ~vertical_mask
        positive_slope = (dy * dx) > 0
        negative_slope = (dy * dx) < 0

        left_mask = non_vertical & negative_slope
        right_mask = non_vertical & positive_slope
        return lines[left_mask], lines[right_mask], lines[vertical_mask]

    @classmethod
    def ransac_vanishing_point(
        cls,
        lines: Any,
        num_iterations: int = 1000,
        inlier_threshold: float = 5.0,
        image_size: tuple[int, int] | None = None,
        random_seed: int | None = None,
    ) -> tuple[Any, Any] | None:
        np, _ = _require_vision()
        if len(lines) < 2:
            return None

        count = len(lines)
        best_vp = None
        best_inlier_count = 0
        best_inlier_mask = None
        rng = np.random.default_rng(random_seed)

        max_distance = None
        if image_size is not None:
            height, width = image_size
            max_distance = max(height, width) * 10
            center = np.array([width / 2.0, height / 2.0])
        else:
            center = None

        for _ in range(num_iterations):
            indices = rng.choice(count, 2, replace=False)
            point = cls.line_intersection(lines[indices[0]], lines[indices[1]])
            if point is None:
                continue
            if max_distance is not None and np.linalg.norm(point - center) > max_distance:
                continue

            inlier_mask = np.zeros(count, dtype=bool)
            for index in range(count):
                midpoint = np.array(
                    [
                        (lines[index][0] + lines[index][2]) / 2,
                        (lines[index][1] + lines[index][3]) / 2,
                    ]
                )
                line_direction = np.array(
                    [
                        lines[index][2] - lines[index][0],
                        lines[index][3] - lines[index][1],
                    ]
                )
                line_direction = line_direction / (np.linalg.norm(line_direction) + 1e-10)

                vp_direction = point - midpoint
                vp_direction = vp_direction / (np.linalg.norm(vp_direction) + 1e-10)
                cos_angle = np.clip(abs(np.dot(line_direction, vp_direction)), 0, 1)
                angle_deg = np.degrees(np.arccos(cos_angle))
                if angle_deg < inlier_threshold:
                    inlier_mask[index] = True

            inlier_count = int(np.sum(inlier_mask))
            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_vp = point
                best_inlier_mask = inlier_mask

        if best_vp is None or best_inlier_count < 2:
            return None
        return best_vp, best_inlier_mask

    @classmethod
    def detect_vanishing_points(
        cls,
        image: Any,
        canny_low: int = 50,
        canny_high: int = 150,
        hough_threshold: int = 80,
        min_line_length: int = 50,
        max_line_gap: int = 10,
        ransac_iterations: int = 2000,
        ransac_threshold: float = 3.0,
        random_seed: int | None = None,
    ) -> dict[str, Any]:
        h, w = image.shape[:2]
        image_size = (h, w)
        all_lines = cls.detect_lines(
            image,
            canny_low=canny_low,
            canny_high=canny_high,
            hough_threshold=hough_threshold,
            min_line_length=min_line_length,
            max_line_gap=max_line_gap,
        )
        filtered = cls.filter_lines(
            all_lines,
            min_angle=5.0,
            max_angle=85.0,
            min_length=30.0,
        )
        left_lines, right_lines, vertical_lines = cls.classify_lines_by_angle(filtered)

        vp1_result = (
            cls.ransac_vanishing_point(
                left_lines,
                num_iterations=ransac_iterations,
                inlier_threshold=ransac_threshold,
                image_size=image_size,
                random_seed=random_seed,
            )
            if len(left_lines) >= 2
            else None
        )
        vp2_result = (
            cls.ransac_vanishing_point(
                right_lines,
                num_iterations=ransac_iterations,
                inlier_threshold=ransac_threshold,
                image_size=image_size,
                random_seed=None if random_seed is None else random_seed + 1,
            )
            if len(right_lines) >= 2
            else None
        )
        vertical_candidates = cls.filter_lines(
            all_lines,
            min_angle=60.0,
            max_angle=90.0,
            min_length=30.0,
        )
        vp3_result = (
            cls.ransac_vanishing_point(
                vertical_candidates,
                num_iterations=ransac_iterations,
                inlier_threshold=ransac_threshold,
                image_size=image_size,
                random_seed=None if random_seed is None else random_seed + 2,
            )
            if len(vertical_candidates) >= 2
            else None
        )

        return {
            "vp1": vp1_result[0] if vp1_result else None,
            "vp2": vp2_result[0] if vp2_result else None,
            "vp3": vp3_result[0] if vp3_result else None,
            "lines": all_lines,
            "left_lines": left_lines,
            "right_lines": right_lines,
            "vertical_lines": vertical_lines,
            "num_lines_total": len(all_lines),
            "image_size": image_size,
        }

    @staticmethod
    def to_schema_vanishing_points(vp_result: dict[str, Any]) -> list[AtlasVanishingPoint]:
        vanishing_points: list[AtlasVanishingPoint] = []
        groups = (
            ("vp1", "left", "left_lines"),
            ("vp2", "right", "right_lines"),
            ("vp3", "vertical", "vertical_lines"),
        )
        for key, label, line_key in groups:
            point = vp_result.get(key)
            if point is None:
                continue
            lines = [
                _line_segment_from_array(line)
                for line in vp_result.get(line_key, [])
            ]
            confidence = min(1.0, len(lines) / max(1, vp_result.get("num_lines_total", 1)))
            vanishing_points.append(
                AtlasVanishingPoint(
                    position_px=_to_point2d(point),
                    direction_label=label,
                    confidence=confidence,
                    supporting_lines=lines,
                )
            )
        return vanishing_points


def draw_debug_overlay(
    image: Any,
    vp_result: dict[str, Any],
    camera_result: dict[str, Any] | None = None,
) -> Any:
    """Draw detected lines, vanishing points, horizon, and camera metadata."""

    _, cv2 = _require_vision()
    debug = image.copy()
    if len(debug.shape) == 2:
        debug = cv2.cvtColor(debug, cv2.COLOR_GRAY2BGR)

    height, width = debug.shape[:2]
    for line in vp_result.get("lines", []):
        x1, y1, x2, y2 = tuple(int(value) for value in flatten_line_segment(normalize_line_segment(line)))
        cv2.line(debug, (x1, y1), (x2, y2), (100, 100, 100), 1)
    for line in vp_result.get("left_lines", []):
        x1, y1, x2, y2 = tuple(int(value) for value in flatten_line_segment(normalize_line_segment(line)))
        cv2.line(debug, (x1, y1), (x2, y2), (255, 100, 50), 2)
    for line in vp_result.get("right_lines", []):
        x1, y1, x2, y2 = tuple(int(value) for value in flatten_line_segment(normalize_line_segment(line)))
        cv2.line(debug, (x1, y1), (x2, y2), (50, 100, 255), 2)
    for line in vp_result.get("vertical_lines", []):
        x1, y1, x2, y2 = tuple(int(value) for value in flatten_line_segment(normalize_line_segment(line)))
        cv2.line(debug, (x1, y1), (x2, y2), (50, 255, 100), 2)

    colors = [(255, 150, 50), (50, 150, 255), (50, 255, 150)]
    labels = ["VP1 left", "VP2 right", "VP3 vertical"]
    for index, key in enumerate(("vp1", "vp2", "vp3")):
        vp = vp_result.get(key)
        if vp is None:
            continue
        vp_x, vp_y = int(vp[0]), int(vp[1])
        draw_x = max(-width, min(width * 2, vp_x))
        draw_y = max(-height, min(height * 2, vp_y))
        if 0 <= draw_x < width and 0 <= draw_y < height:
            cv2.circle(debug, (draw_x, draw_y), 8, colors[index], 2)
            cv2.circle(debug, (draw_x, draw_y), 3, colors[index], -1)
        cv2.putText(
            debug,
            f"{labels[index]}: ({vp_x}, {vp_y})",
            (8, 22 + index * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            colors[index],
            1,
        )

    if vp_result.get("vp1") is not None and vp_result.get("vp2") is not None:
        vp1 = vp_result["vp1"]
        vp2 = vp_result["vp2"]
        if abs(vp2[0] - vp1[0]) > 1e-6:
            slope = (vp2[1] - vp1[1]) / (vp2[0] - vp1[0])
            y_at_0 = int(vp1[1] + slope * (0 - vp1[0]))
            y_at_w = int(vp1[1] + slope * (width - vp1[0]))
            cv2.line(debug, (0, y_at_0), (width, y_at_w), (0, 255, 255), 2)

    if camera_result is not None:
        info_lines = [
            f"Focal: {camera_result['focal_length_mm']:.1f}mm ({camera_result['focal_length_px']:.0f}px)",
            f"FOV: {camera_result['fov_horizontal_deg']:.1f} x {camera_result['fov_vertical_deg']:.1f}",
            f"Horizon tilt: {camera_result['horizon_angle']:.1f}",
        ]
        for row, text in enumerate(info_lines):
            cv2.putText(
                debug,
                text,
                (8, height - 12 - row * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )
    return debug
