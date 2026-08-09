"""Run deterministic two/three-photo RAW registration from a portable manifest.

This is an acceptance tool, not a second solver.  It adapts ordered RAW imports
to :func:`atlas_camera.core.multiview_solver.solve_multiview`, persists the
solver's diagnostics canonically, and writes whatever pair overlays the solver
made available before either success or a structured rejection.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import posixpath
import sys
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atlas_camera.core.multiview_solver import solve_multiview
from atlas_camera.core.multiview_types import MultiViewFrame, MultiViewSettings
from atlas_camera.core.schema import _json_ready
from atlas_camera.raw import import_raw


_MANIFEST_KEYS = {
    "raw_paths", "camera_height_m", "capture_mode", "match_quality", "seed",
}
_PAIR_NAMES = {
    2: ("pair_01.png",),
    3: ("pair_01.png", "pair_02.png", "pair_12.png"),
}
_CANONICAL_OVERLAY_NAMES = ("pair_01.png", "pair_02.png", "pair_12.png")
_METADATA_FIELDS = (
    "camera_make", "camera_model", "lens_model", "focal_length_mm",
    "sensor_width_mm", "sensor_height_mm", "sensor_source", "orientation",
    "capture_datetime", "metadata_source", "undistort_status", "warnings",
)


def canonical_json(value: Any) -> str:
    """Serialize a report deterministically, including a final newline."""
    return json.dumps(
        _canonical_ready(value),
        sort_keys=True,
        indent=2,
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"


def _canonical_ready(value: Any) -> Any:
    value = _json_ready(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, list):
        return [_canonical_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _canonical_ready(item) for key, item in value.items()}
    return value


def _load_manifest(
    manifest: Mapping[str, Any] | str | Path,
) -> tuple[dict[str, Any], Path]:
    if isinstance(manifest, Mapping):
        return dict(manifest), Path.cwd()
    manifest_path = Path(manifest).expanduser().resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("capture manifest must be a JSON object")
    return payload, manifest_path.parent


def _validate_manifest(payload: Mapping[str, Any]) -> None:
    missing = sorted(_MANIFEST_KEYS - set(payload))
    unknown = sorted(set(payload) - _MANIFEST_KEYS)
    if missing:
        raise ValueError(f"capture manifest is missing required keys: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"capture manifest has unknown keys: {', '.join(unknown)}")
    raw_paths = payload["raw_paths"]
    if (
        not isinstance(raw_paths, list)
        or len(raw_paths) not in (2, 3)
        or any(not isinstance(path, str) or not path.strip() for path in raw_paths)
    ):
        raise ValueError("raw_paths must contain exactly 2 or 3 non-empty ordered paths")
    # Let the frozen settings contract validate enum values and seed coercion.
    MultiViewSettings(
        capture_mode=payload["capture_mode"],
        camera_height_m=float(payload["camera_height_m"]),
        match_quality=payload["match_quality"],
        seed=int(payload["seed"]),
    )


def _resolved_paths(raw_paths: list[str], base_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for raw_path in raw_paths:
        path = Path(raw_path).expanduser()
        if _is_authored_absolute(raw_path):
            # Preserve foreign-flavor absolute syntax for the IO probe. Joining
            # it to this host's manifest directory would invent a local path.
            paths.append(path)
        else:
            paths.append((base_dir / path).resolve())
    return paths


def _is_authored_absolute(authored: str) -> bool:
    return (
        PureWindowsPath(authored).is_absolute()
        or PurePosixPath(authored).is_absolute()
    )


def _report_source(authored: str) -> str:
    """Return a stable reference under both Windows and POSIX grammars."""
    windows_path = PureWindowsPath(authored)
    if windows_path.is_absolute():
        return windows_path.name
    posix_path = PurePosixPath(authored)
    if posix_path.is_absolute():
        return posix_path.name
    return posixpath.normpath(authored.replace("\\", "/"))


def _frame_metadata(raw: Any, index: int, source_path: str) -> dict[str, Any]:
    metadata = {
        "frame": index + 1,
        "label": f"photo_{index + 1}",
        "source_path": source_path,
    }
    for field_name in _METADATA_FIELDS:
        value = getattr(raw, field_name, None)
        if field_name == "warnings":
            value = list(value or ())
        metadata[field_name] = _json_ready(value)
    return metadata


def _base_report(payload: Mapping[str, Any], input_count: int) -> dict[str, Any]:
    height = float(payload["camera_height_m"])
    canonical_height = height if math.isfinite(height) else None
    return {
        "schema": "atlas.multiview-registration.v1",
        "input_count": input_count,
        "settings": {
            "camera_height_m": canonical_height,
            "capture_mode": str(payload["capture_mode"]),
            "match_quality": str(payload["match_quality"]),
            "seed": int(payload["seed"]),
        },
        "frames": [],
        "outcome_code": None,
        "summary": "",
        "selected_mode": None,
        "translation_recovered": False,
        "metric_scale_recovered": False,
        "scale": {},
        "warnings": [],
        "diagnostics": {},
        "solve": None,
        "overlays": [],
    }


def _write_report(output_dir: Path, report: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "registration.json").write_text(
        canonical_json(report), encoding="utf-8", newline="\n",
    )


def _clear_canonical_overlays(output_dir: Path) -> None:
    """Remove only runner-owned pair outputs from a prior invocation."""
    for name in _CANONICAL_OVERLAY_NAMES:
        try:
            (output_dir / name).unlink()
        except FileNotFoundError:
            pass


def _privacy_safe_report(
    value: Any,
    resolved_paths: list[Path],
    display_paths: list[str],
) -> Any:
    """Replace every known resolved input spelling in persisted evidence."""
    replacements: list[tuple[str, str]] = []
    for resolved, display in zip(resolved_paths, display_paths):
        spellings = {str(resolved), resolved.as_posix(), os.fspath(resolved)}
        replacements.extend((spelling, display) for spelling in spellings if spelling)
    replacements.sort(key=lambda item: len(item[0]), reverse=True)

    def scrub(item: Any) -> Any:
        if isinstance(item, str):
            for machine_path, display in replacements:
                item = item.replace(machine_path, display)
            return item
        if isinstance(item, list):
            return [scrub(child) for child in item]
        if isinstance(item, tuple):
            return [scrub(child) for child in item]
        if isinstance(item, dict):
            return {str(key): scrub(child) for key, child in item.items()}
        return item

    return scrub(value)


def _write_overlay(path: Path, overlay: Any) -> None:
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Writing registration overlays requires NumPy and Pillow; "
            "install with: pip install -e .[dev,image]"
        ) from exc

    pixels = np.asarray(overlay)
    if pixels.ndim != 3 or pixels.shape[2] not in (3, 4):
        raise ValueError(f"pair overlay must be HWC RGB/RGBA; got shape {pixels.shape}")
    if np.issubdtype(pixels.dtype, np.floating):
        pixels = np.rint(np.clip(pixels, 0.0, 1.0) * 255.0).astype(np.uint8)
    else:
        pixels = np.clip(pixels, 0, 255).astype(np.uint8)
    Image.fromarray(pixels).save(path, format="PNG", optimize=False)


def run_manifest(
    manifest: Mapping[str, Any] | str | Path,
    *,
    output_dir: str | Path | None = None,
    solve_fn: Callable[[Any, MultiViewSettings], Any] = solve_multiview,
    half_size: bool = False,
) -> dict[str, Any]:
    """Run one ordered manifest and always persist a structured solve result.

    Manifest errors remain ``ValueError`` because they are authoring errors.
    RAW decode/metadata errors and deterministic solver rejections become a
    canonical ``registration.json`` suitable for repeat acceptance evidence.
    """
    payload, base_dir = _load_manifest(manifest)
    _validate_manifest(payload)
    paths = _resolved_paths(payload["raw_paths"], base_dir)
    display_paths = [_report_source(path) for path in payload["raw_paths"]]
    destination = Path(output_dir) if output_dir is not None else base_dir / "multiview_acceptance"
    destination = destination.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    _clear_canonical_overlays(destination)
    report = _base_report(payload, len(paths))

    height = float(payload["camera_height_m"])
    frames: list[MultiViewFrame] = []
    current_index = 0
    try:
        for index, (path, display_path) in enumerate(zip(paths, display_paths)):
            current_index = index
            if not path.is_file():
                raise FileNotFoundError("RAW file not found")
            raw = import_raw(str(path), half_size=half_size)
            report["frames"].append(_frame_metadata(raw, index, display_path))
            frames.append(MultiViewFrame(
                image=raw.display_srgb,
                raw_meta=raw,
                plate_ref=None,
                label=f"photo_{index + 1}",
            ))
    except Exception as exc:  # noqa: BLE001 - acceptance evidence must be structured.
        summary = (
            f"RAW import failed for photo_{current_index + 1} "
            f"({display_paths[current_index]}): {type(exc).__name__}"
        )
        report.update({
            "outcome_code": "metadata_mismatch",
            "summary": summary,
            "warnings": [summary],
            "diagnostics": {"outcome_code": "metadata_mismatch", "summary": summary},
        })
        report = _privacy_safe_report(report, paths, display_paths)
        _write_report(destination, report)
        return report

    settings = MultiViewSettings(
        capture_mode=payload["capture_mode"],
        camera_height_m=height,
        match_quality=payload["match_quality"],
        seed=int(payload["seed"]),
    )
    try:
        outcome = solve_fn(frames, settings)
        diagnostics = outcome.diagnostics.to_dict()
        report.update({
            "outcome_code": outcome.diagnostics.outcome_code,
            "summary": outcome.diagnostics.summary,
            "selected_mode": outcome.diagnostics.selected_mode,
            "scale": _json_ready(outcome.diagnostics.scale),
            "warnings": list(outcome.diagnostics.warnings),
            "diagnostics": diagnostics,
            "solve": (
                _json_ready(outcome.solve.to_dict()) if outcome.solve is not None else None
            ),
        })
        selected_mode = outcome.diagnostics.selected_mode or outcome.diagnostics.outcome_code
        report["translation_recovered"] = bool(
            outcome.solve is not None and selected_mode == "translated"
        )
        report["metric_scale_recovered"] = bool(
            report["translation_recovered"]
            and report["scale"].get("source") == "measured_camera_height"
        )
        if selected_mode == "rotation_only":
            warning = "Rotation-only capture cannot recover translation or metric scale."
            if warning not in report["warnings"]:
                report["warnings"].append(warning)

    except Exception as exc:  # noqa: BLE001 - keep unexpected real failures inspectable.
        _clear_canonical_overlays(destination)
        summary = f"deterministic solver failed unexpectedly: {type(exc).__name__}: {exc}"
        report.update({
            "outcome_code": "degenerate_geometry",
            "summary": summary,
            "warnings": [summary],
            "diagnostics": {"outcome_code": "degenerate_geometry", "summary": summary},
            "solve": None,
            "overlays": [],
        })
    else:
        try:
            for name, overlay in zip(_PAIR_NAMES[len(paths)], outcome.overlays):
                if overlay is None:
                    continue
                _write_overlay(destination / name, overlay)
                report["overlays"].append(name)
        except Exception as exc:  # noqa: BLE001 - persist artifact failure safely.
            _clear_canonical_overlays(destination)
            report["overlays"] = []
            report["warnings"].append(
                f"overlay artifact write failed for {name}: {type(exc).__name__}"
            )

    report = _privacy_safe_report(report, paths, display_paths)
    _write_report(destination, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a deterministic two/three-photo RAW capture manifest.",
    )
    parser.add_argument("manifest", type=Path, help="JSON manifest path")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Directory for registration.json and pair overlays")
    parser.add_argument("--half-size", action="store_true",
                        help="Use rawpy half-size decode for faster acceptance iteration")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run_manifest(
        args.manifest,
        output_dir=args.output_dir,
        half_size=args.half_size,
    )
    print(f"{report['outcome_code']}: {report['summary']}")
    print(f"report: {(args.output_dir or args.manifest.parent / 'multiview_acceptance') / 'registration.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
