"""Portable Atlas review package builder."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import shutil

from atlas_camera.core.io import save_solve_json
from atlas_camera.core.schema import AtlasSolve
from atlas_camera.exporters._plate import primary_plate_path
from atlas_camera.exporters.blender_exporter import BlenderExporter
from atlas_camera.exporters.maya_exporter import MayaExporter, write_maya_mel_launcher
from atlas_camera.exporters.nuke_exporter import NukeExporter
from atlas_camera.exporters.usd_exporter import USDExporter

_DCC_EXPORTERS = [
    (BlenderExporter(), "blender_open_scene", "blender_open_scene.py"),
]


@dataclass(slots=True)
class ReviewPackageResult:
    package_dir: Path
    files: dict[str, Path] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _copy_if_present(source: str | Path | None, destination: Path) -> Path | None:
    if not source:
        return None
    source_path = Path(source)
    if not source_path.is_file():
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source_path.resolve() == destination.resolve():
        return destination
    shutil.copy2(source_path, destination)
    return destination


def build_review_package(
    solve: AtlasSolve,
    output_dir: str | Path,
    *,
    package_name: str = "atlas_review_001",
    source_image_path: str | Path | None = None,
    debug_overlay_path: str | Path | None = None,
    include_usd: bool = True,
    relief_mesh_obj_path: str | Path | None = None,
) -> ReviewPackageResult:
    package_dir = Path(output_dir) / package_name
    package_dir.mkdir(parents=True, exist_ok=True)

    result = ReviewPackageResult(package_dir=package_dir)

    source_source = source_image_path or primary_plate_path(solve)
    source_suffix = Path(source_source).suffix if source_source else ".png"
    source_image_name = f"source_image{source_suffix or '.png'}"
    source_copy = _copy_if_present(
        source_source,
        package_dir / source_image_name,
    )
    if source_copy:
        result.files["source_image"] = source_copy
        source_image_name = source_copy.name

    overlay_copy = _copy_if_present(debug_overlay_path, package_dir / "debug_overlay.png")
    if overlay_copy:
        result.files["debug_overlay"] = overlay_copy

    solve_path = save_solve_json(solve, package_dir / "atlas_solve.json")
    result.files["atlas_solve"] = solve_path

    result.files["maya_open_scene"] = MayaExporter().write_scene(
        solve,
        package_dir / "maya_open_scene.py",
        source_image_name=source_image_name,
        relief_mesh_obj_path=relief_mesh_obj_path,
        use_package_source=True,
    )
    result.files["nuke_cards"] = NukeExporter().write_scene(
        solve,
        package_dir / "nuke_cards.py",
        source_image_name=source_image_name,
        use_package_source=True,
    )
    for exporter, key, filename in _DCC_EXPORTERS:
        result.files[key] = exporter.write_scene(solve, package_dir / filename)

    result.files["maya_mel_launcher"] = write_maya_mel_launcher(package_dir, review_name=package_name)

    if include_usd:
        exporter = USDExporter()
        try:
            result.files["camera_usda"] = exporter.export_camera(solve, package_dir / "camera.usda")
            result.files["proxy_scene_usda"] = exporter.export_proxy_scene(
                solve,
                package_dir / "proxy_scene.usda",
            )
            result.files["projection_scene_usda"] = exporter.export_projection_scene(
                solve,
                package_dir / "projection_scene.usda",
                source_image_name=source_image_name,
            )
        except RuntimeError as exc:
            result.warnings.append(str(exc))

    # The reproducibility manifest — the review package is the flagship
    # aggregation point, so it always writes one. Never fatal.
    try:
        from atlas_camera.exporters.manifest import (
            ManifestArtifact,
            write_project_manifest,
        )
        artifacts = [
            ManifestArtifact(name, str(path.name), "AtlasExportReviewPackage")
            for name, path in sorted(result.files.items())
        ]
        manifest_path = write_project_manifest(solve, package_dir,
                                               artifacts=artifacts)
        result.files["manifest"] = manifest_path
    except Exception as exc:  # noqa: BLE001
        result.warnings.append(f"atlas_project.json manifest skipped: {exc}")

    report_path = package_dir / "report.md"
    report_path.write_text(_report_markdown(solve, result), encoding="utf-8")
    result.files["report"] = report_path
    return result


def _report_markdown(solve: AtlasSolve, result: ReviewPackageResult) -> str:
    solve_warnings = [str(warning) for warning in solve.debug_metadata.get("warnings", [])]
    warning_lines = "\n".join(
        f"- {warning}" for warning in [*result.warnings, *solve_warnings]
    ) or "- None"
    file_lines = "\n".join(
        f"- {name}: `{path.name}`"
        for name, path in sorted(result.files.items())
    )
    intrinsics = solve.camera.intrinsics
    camera_estimation = solve.debug_metadata.get("camera_estimation", {})
    horizon_angle = camera_estimation.get("horizon_angle")
    fov_horizontal = camera_estimation.get("fov_horizontal_deg")
    fov_vertical = camera_estimation.get("fov_vertical_deg")
    focal_source = camera_estimation.get("focal_source", "Unavailable")
    focal_inferred = camera_estimation.get("focal_length_inferred", False)
    focal_assumption = camera_estimation.get("focal_assumption") or "None"
    vp_count = len(solve.vanishing_points)
    line_count = solve.debug_metadata.get("num_lines_total", 0)
    scale_constraints = solve.debug_metadata.get("scale_constraints", {})
    scale_count = scale_constraints.get("count", 0)
    scale_status = scale_constraints.get("status", "none_supplied")
    reference_ids = scale_constraints.get("reference_ids", [])
    reference_text = ", ".join(reference_ids) if reference_ids else "None"
    horizon_text = f"{horizon_angle:.2f} deg" if isinstance(horizon_angle, (float, int)) else "Unavailable"
    fov_text = (
        f"{fov_horizontal:.2f} x {fov_vertical:.2f} deg"
        if isinstance(fov_horizontal, (float, int)) and isinstance(fov_vertical, (float, int))
        else "Unavailable"
    )
    from atlas_camera.core.scene_health import scale_health
    sh = scale_health(solve)
    conf_metrics = dict(getattr(solve.camera.confidence, "individual_metrics", {}) or {})
    conf_table = "\n".join(
        f"| {key} | {value:.2f} |" for key, value in sorted(conf_metrics.items())
    ) or "| (none) | — |"
    safe_text = "yes" if sh.safe_to_export else "**NO — verify before delivery**"
    height_text = (f"{sh.camera_height_m:.2f} m"
                   if sh.camera_height_m is not None else "Unavailable")
    conf_text = f"{sh.confidence:.2f}" if sh.confidence is not None else "Unavailable"
    try:
        from atlas_camera.exporters.manifest import manifest_identity_hash
        identity_line = f"\nManifest identity: `{manifest_identity_hash(solve)}`\n"
    except Exception:  # noqa: BLE001
        identity_line = ""
    stamp = solve.debug_metadata.get("scene_health")
    if isinstance(stamp, dict) and stamp.get("level"):
        marks = {"fail": "✖", "warn": "⚠"}
        flag_lines = "\n".join(
            f"- {marks.get(f.get('severity'), '•')} {f.get('message')}"
            for f in (stamp.get("flags") or [])) or "- None"
        ack = "yes" if stamp.get("acknowledged") else "no"
        health_section = f"""## Scene health

- Level: {stamp['level']}
- Acknowledged: {ack} ({stamp.get('evaluated_at', 'unknown time')})

{flag_lines}

"""
    else:
        health_section = ""
    return f"""# Atlas Camera Review Package

## Scale trust

- Status: {sh.status}
- Safe to export: {safe_text}
- Source: {sh.scale_source or "none recorded"}
- Confidence: {conf_text}
- Camera height: {height_text}
- {sh.detail}

## Confidence vector

| metric | value |
|---|---|
{conf_table}

{health_section}## Solve

- Source method: {solve.source_method}
- Confidence: {solve.confidence}
- Image size: {solve.image_width} x {solve.image_height}
- Core coordinates: {solve.camera.extrinsics.coordinate_system}, {solve.camera.extrinsics.up_axis}-up
- Focal length: {intrinsics.focal_length_mm}
- Focal source: {focal_source}
- Focal inferred: {focal_inferred}
- Focal assumption: {focal_assumption}
- Sensor: {intrinsics.sensor_width_mm} x {intrinsics.sensor_height_mm} mm
- Vanishing points: {vp_count}
- Detected lines: {line_count}
- Scale references: {scale_count} ({scale_status})
- Scale reference IDs: {reference_text}
- Horizon angle: {horizon_text}
- Field of view: {fov_text}

## Files

{file_lines}

## Warnings

{warning_lines}

## Limitations

This package is generated from a still-image perspective solve. Results are a
projection-prep starting point, not a replacement for full sequence matchmove.
Scale references are explicit artist guides unless `metric_depth_solved` is true
in `atlas_solve.json`. Metric depth fitting and broader real-image confidence
tuning are planned.
{identity_line}"""
