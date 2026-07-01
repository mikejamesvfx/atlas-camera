"""Build an Atlas review package from an atlas_solve.json file."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atlas_camera.core.io import load_solve_json
from atlas_camera.exporters.review_package import build_review_package


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("solve_json", help="Path to atlas_solve.json.")
    parser.add_argument(
        "--output-dir",
        default="review_packages",
        help="Directory where the package folder will be created.",
    )
    parser.add_argument(
        "--package-name",
        default="atlas_review_001",
        help="Review package folder name.",
    )
    parser.add_argument("--no-usd", action="store_true", help="Skip USD export.")
    args = parser.parse_args()

    solve = load_solve_json(args.solve_json)
    result = build_review_package(
        solve,
        args.output_dir,
        package_name=args.package_name,
        include_usd=not args.no_usd,
    )
    print(result.package_dir)
    for warning in result.warnings:
        print(f"warning: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

