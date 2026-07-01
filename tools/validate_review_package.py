"""Validate the presence of required Atlas review package files."""

from __future__ import annotations

import argparse
from pathlib import Path

REQUIRED_FILES = [
    "atlas_solve.json",
    "maya_open_scene.py",
    "report.md",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package_dir", help="Review package directory.")
    args = parser.parse_args()

    package_dir = Path(args.package_dir)
    missing = [name for name in REQUIRED_FILES if not (package_dir / name).is_file()]
    if missing:
        for name in missing:
            print(f"missing: {name}")
        return 1
    print(f"valid: {package_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

