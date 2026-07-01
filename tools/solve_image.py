"""Solve one still image and build an Atlas review package."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atlas_camera.core.solver import solve_still_image
from atlas_camera.exporters.review_package import build_review_package


def _optional_float(value: float | None) -> float | None:
    if value is None:
        return None
    return float(value)


def _build_detection_options(args: argparse.Namespace) -> dict[str, int | float | None]:
    return {
        "canny_low": args.canny_low,
        "canny_high": args.canny_high,
        "hough_threshold": args.hough_threshold,
        "min_line_length": args.min_line_length,
        "max_line_gap": args.max_line_gap,
        "ransac_iterations": args.ransac_iterations,
        "ransac_threshold": args.ransac_threshold,
        "random_seed": args.random_seed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Source image to solve.")
    parser.add_argument(
        "--output-dir",
        default="review_packages",
        help="Directory where the review package folder will be created.",
    )
    parser.add_argument(
        "--package-name",
        default="atlas_review_001",
        help="Review package folder name.",
    )
    parser.add_argument(
        "--focal-length-mm",
        type=float,
        default=None,
        help="Optional known focal length in millimeters.",
    )
    parser.add_argument(
        "--sensor-width-mm",
        type=float,
        default=36.0,
        help="Sensor width in millimeters.",
    )
    parser.add_argument(
        "--camera-height",
        type=float,
        default=1.6,
        help="Assumed camera height in Atlas Y-up world units.",
    )
    parser.add_argument(
        "--principal-point",
        nargs=2,
        type=float,
        metavar=("CX", "CY"),
        help="Optional principal point in pixels.",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Skip OpenCV vanishing-point detection and create a metadata-only package.",
    )
    parser.add_argument("--no-usd", action="store_true", help="Skip optional USD export.")
    parser.add_argument("--canny-low", type=int, default=50)
    parser.add_argument("--canny-high", type=int, default=150)
    parser.add_argument("--hough-threshold", type=int, default=80)
    parser.add_argument("--min-line-length", type=int, default=50)
    parser.add_argument("--max-line-gap", type=int, default=10)
    parser.add_argument("--ransac-iterations", type=int, default=2000)
    parser.add_argument("--ransac-threshold", type=float, default=3.0)
    parser.add_argument("--random-seed", type=int, default=7)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    package_dir = output_dir / args.package_name
    package_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = package_dir / "debug_overlay.png"

    intrinsics_hint = {
        "sensor_width_mm": args.sensor_width_mm,
    }
    focal_length = _optional_float(args.focal_length_mm)
    if focal_length is not None:
        intrinsics_hint["focal_length_mm"] = focal_length
    if args.principal_point is not None:
        intrinsics_hint["principal_point_px"] = tuple(args.principal_point)

    solve = solve_still_image(
        args.image,
        intrinsics_hint=intrinsics_hint,
        detect_vanishing_points=not args.metadata_only,
        debug_overlay_path=None if args.metadata_only else overlay_path,
        camera_height=args.camera_height,
        detection_options=None if args.metadata_only else _build_detection_options(args),
    )

    result = build_review_package(
        solve,
        output_dir,
        package_name=args.package_name,
        source_image_path=args.image,
        debug_overlay_path=None if args.metadata_only else overlay_path,
        include_usd=not args.no_usd,
    )

    print(result.package_dir)
    print(f"solve: {solve.source_method}")
    print(f"vanishing_points: {len(solve.vanishing_points)}")
    print(f"detected_lines: {solve.debug_metadata.get('num_lines_total', 0)}")
    if "camera_estimation" in solve.debug_metadata:
        camera = solve.debug_metadata["camera_estimation"]
        print(f"focal_length_mm: {camera['focal_length_mm']:.3f}")
        print(f"horizon_angle_deg: {camera['horizon_angle']:.3f}")
    for warning in result.warnings:
        print(f"warning: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
