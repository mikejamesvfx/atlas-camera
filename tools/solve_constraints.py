"""Solve one image from artist-guided constraints and build a review package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atlas_camera.core.solver import solve_from_constraints
from atlas_camera.exporters.review_package import build_review_package


def _load_constraints(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Constraints JSON must contain an object at the top level.")
    return data


def _override_hints(args: argparse.Namespace) -> dict:
    hints = {}
    if args.focal_length_mm is not None:
        hints["focal_length_mm"] = args.focal_length_mm
    if args.sensor_width_mm is not None:
        hints["sensor_width_mm"] = args.sensor_width_mm
    if args.principal_point is not None:
        hints["principal_point_px"] = tuple(args.principal_point)
    return hints


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Source image to solve.")
    parser.add_argument(
        "--constraints",
        required=True,
        help="JSON file containing line groups, explicit VPs, and optional hints.",
    )
    parser.add_argument(
        "--output-dir",
        default="review_packages",
        help="Directory where the review package folder will be created.",
    )
    parser.add_argument(
        "--package-name",
        default="atlas_guided_review_001",
        help="Review package folder name.",
    )
    parser.add_argument(
        "--focal-length-mm",
        type=float,
        default=None,
        help="Optional known focal length override in millimeters.",
    )
    parser.add_argument(
        "--sensor-width-mm",
        type=float,
        default=None,
        help="Optional sensor width override in millimeters.",
    )
    parser.add_argument(
        "--principal-point",
        nargs=2,
        type=float,
        metavar=("CX", "CY"),
        help="Optional principal point override in pixels.",
    )
    parser.add_argument("--no-overlay", action="store_true", help="Skip debug overlay generation.")
    parser.add_argument("--no-usd", action="store_true", help="Skip optional USD export.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    package_dir = output_dir / args.package_name
    package_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = None if args.no_overlay else package_dir / "debug_overlay.png"

    constraints = _load_constraints(args.constraints)
    solve = solve_from_constraints(
        args.image,
        constraints,
        intrinsics_hint=_override_hints(args),
        debug_overlay_path=overlay_path,
    )

    result = build_review_package(
        solve,
        output_dir,
        package_name=args.package_name,
        source_image_path=args.image,
        debug_overlay_path=overlay_path,
        include_usd=not args.no_usd,
    )

    summary = solve.debug_metadata.get("constraint_summary", {})
    print(result.package_dir)
    print(f"solve: {solve.source_method}")
    print(f"vanishing_points: {len(solve.vanishing_points)}")
    print(f"guided_lines: {solve.debug_metadata.get('num_lines_total', 0)}")
    print(f"left_lines: {summary.get('left_line_count', 0)}")
    print(f"right_lines: {summary.get('right_line_count', 0)}")
    scale = solve.debug_metadata.get("scale_constraints", {})
    if scale.get("count", 0):
        print(f"scale_references: {scale.get('count')}")
        if scale.get("reference_ids"):
            print(f"reference_ids: {', '.join(scale['reference_ids'])}")
    if "camera_estimation" in solve.debug_metadata:
        camera = solve.debug_metadata["camera_estimation"]
        print(f"focal_length_mm: {camera['focal_length_mm']:.3f}")
        print(f"horizon_angle_deg: {camera['horizon_angle']:.3f}")
    for warning in result.warnings:
        print(f"warning: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
