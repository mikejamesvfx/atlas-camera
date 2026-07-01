"""Run Atlas Camera benchmarks against external CV dataset roots."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atlas_camera.datasets import (
    BenchmarkOptions,
    benchmark_eth3d,
    load_dtu_projections,
    write_benchmark_csv,
    write_benchmark_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("eth3d", "dtu"),
        required=True,
        help="Dataset adapter to run.",
    )
    parser.add_argument("--root", required=True, help="External dataset root path.")
    parser.add_argument(
        "--output-json",
        default="validation_output/dataset_benchmark.json",
        help="JSON report destination.",
    )
    parser.add_argument(
        "--output-csv",
        default="validation_output/dataset_benchmark.csv",
        help="CSV report destination.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Maximum images to benchmark.")
    parser.add_argument(
        "--detect-vanishing-points",
        action="store_true",
        help="Run OpenCV vanishing-point detection instead of metadata-only solves.",
    )
    parser.add_argument(
        "--skip-missing-images",
        action="store_true",
        help="Skip entries whose referenced image files are not present.",
    )
    args = parser.parse_args()

    if args.dataset == "dtu":
        projections = load_dtu_projections(args.root)
        print(f"dtu_projection_matrices: {len(projections)}")
        if projections:
            print(f"first_projection: {projections[0].path}")
        print("DTU benchmark scoring is pending image/projection pairing from the local SampleSet layout.")
        return 0

    records = benchmark_eth3d(
        args.root,
        BenchmarkOptions(
            limit=args.limit,
            detect_vanishing_points=args.detect_vanishing_points,
            include_missing_images=not args.skip_missing_images,
        ),
    )
    json_path = write_benchmark_json(records, args.output_json)
    csv_path = write_benchmark_csv(records, args.output_csv)

    ok_count = sum(1 for record in records if record.status == "ok")
    print(f"records: {len(records)}")
    print(f"ok: {ok_count}")
    print(f"errors: {len(records) - ok_count}")
    print(f"json: {json_path}")
    print(f"csv: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
