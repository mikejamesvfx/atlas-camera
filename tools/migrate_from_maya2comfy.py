"""Report and optionally copy Maya2Comfy migration candidates.

Dry-run is the default. Copying creates a source snapshot under
`migration_artifacts/maya2comfy_sources` for manual review; it does not rewrite
Atlas Camera modules.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import shutil

CANDIDATE_FILES = [
    "camera_estimator.py",
    "maya_usd_camera_loader.py",
    "maya_camera_converter.py",
    "camera_export_formats.py",
    "mesh_depth_renderer.py",
    "utils/validators.py",
    "tests/test_camera_estimator.py",
    "tests/test_usd_camera_loader.py",
    "tests/test_validators.py",
]


def find_candidates(source: Path) -> list[Path]:
    return [source / relative for relative in CANDIDATE_FILES if (source / relative).is_file()]


def write_log(destination: Path, lines: list[str]) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    log_path = destination / "migration_log.md"
    timestamp = datetime.now().isoformat(timespec="seconds")
    log_path.write_text(
        "# Maya2Comfy Migration Log\n\n"
        f"- Timestamp: {timestamp}\n\n"
        "## Candidates\n\n"
        + "\n".join(lines)
        + "\n",
        encoding="utf-8",
    )
    return log_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default="C:/Users/miike/Documents/Maya2Comfy",
        help="Path to the Maya2Comfy prototype.",
    )
    parser.add_argument(
        "--destination",
        default="migration_artifacts/maya2comfy_sources",
        help="Destination for copied source snapshots when --copy is used.",
    )
    parser.add_argument("--copy", action="store_true", help="Copy candidate files.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report candidates without copying. This is the default.",
    )
    args = parser.parse_args()

    source = Path(args.source)
    if not source.is_dir():
        raise SystemExit(f"Source does not exist: {source}")

    candidates = find_candidates(source)
    if not candidates:
        raise SystemExit(f"No migration candidates found under {source}")

    destination = Path(args.destination)
    lines: list[str] = []
    for candidate in candidates:
        relative = candidate.relative_to(source)
        lines.append(f"- `{relative.as_posix()}`")
        print(relative.as_posix())
        if args.copy:
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, target)

    log_path = write_log(destination if args.copy else Path("migration_artifacts"), lines)
    print(f"Migration log: {log_path}")
    if not args.copy:
        print("Dry-run only. Pass --copy to snapshot candidate files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

