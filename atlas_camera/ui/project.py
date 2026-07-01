"""Project-folder services for the optional Atlas Camera UI."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import shutil
from typing import Any
from uuid import uuid4

from atlas_camera.core.io import load_solve_json, save_solve_json
from atlas_camera.core.schema import AtlasSolve
from atlas_camera.core.solver import solve_from_constraints, solve_still_image
from atlas_camera.exporters.review_package import ReviewPackageResult, build_review_package
from atlas_camera.inference import create_multimodal_provider, provider_models_response
from atlas_camera.reference_data import list_categories, search_scale_references

PROJECT_META = "atlas_project.json"
CONSTRAINTS_FILE = "constraints.json"
UI_STATE_FILE = "ui_state.json"
SOLVE_FILE = "atlas_solve.json"
OVERLAY_FILE = "debug_overlay.png"
SOURCE_PREFIX = "source_image"

# Keys owned exclusively by the React workbench — never read by the solver.
_UI_ONLY_KEYS: frozenset[str] = frozenset({"viewport3d"})


@dataclass(slots=True)
class AtlasUiProject:
    project_dir: Path
    source_image: Path | None
    constraints_path: Path
    ui_state_path: Path
    solve_path: Path
    overlay_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_dir": str(self.project_dir),
            "source_image": str(self.source_image) if self.source_image else None,
            "constraints_path": str(self.constraints_path),
            "solve_path": str(self.solve_path),
            "overlay_path": str(self.overlay_path),
            "has_solve": self.solve_path.is_file(),
            "has_overlay": self.overlay_path.is_file(),
        }


def _read_json(path: Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    if not path.is_file():
        return dict(default or {})
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _image_size(image_path: Path) -> tuple[int, int]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Atlas UI image loading requires Pillow. Install atlas-camera[ui].") from exc

    with Image.open(image_path) as image:
        return image.size


def _project_from_meta(project_dir: Path) -> AtlasUiProject:
    meta = _read_json(project_dir / PROJECT_META)
    source = meta.get("source_image")
    source_image = Path(source) if source else None
    return AtlasUiProject(
        project_dir=project_dir,
        source_image=source_image,
        constraints_path=project_dir / CONSTRAINTS_FILE,
        ui_state_path=project_dir / UI_STATE_FILE,
        solve_path=project_dir / SOLVE_FILE,
        overlay_path=project_dir / OVERLAY_FILE,
    )


def open_project(project_dir: str | Path) -> AtlasUiProject:
    path = Path(project_dir).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    if not (path / PROJECT_META).is_file():
        _write_json(path / PROJECT_META, {"project_dir": str(path), "source_image": None})
    return _project_from_meta(path)


def create_project(
    project_dir: str | Path,
    *,
    image_path: str | Path | None = None,
    image_filename: str | None = None,
    image_bytes: bytes | None = None,
) -> AtlasUiProject:
    project = open_project(project_dir)
    source_image: Path | None = None

    if image_bytes is not None:
        suffix = Path(image_filename or "source_image.png").suffix or ".png"
        source_image = project.project_dir / f"{SOURCE_PREFIX}{suffix.lower()}"
        source_image.write_bytes(image_bytes)
    elif image_path is not None:
        original = Path(image_path).expanduser().resolve()
        if not original.is_file():
            raise FileNotFoundError(f"Image does not exist: {original}")
        suffix = original.suffix or ".png"
        source_image = project.project_dir / f"{SOURCE_PREFIX}{suffix.lower()}"
        if original != source_image:
            shutil.copy2(original, source_image)

    if source_image is not None:
        width, height = _image_size(source_image)
        constraints = default_constraints(width, height)
        _write_json(project.constraints_path, constraints)
        _write_json(
            project.project_dir / PROJECT_META,
            {
                "project_dir": str(project.project_dir),
                "source_image": str(source_image),
                "image_width": width,
                "image_height": height,
            },
        )

    return _project_from_meta(project.project_dir)


def default_constraints(image_width: int, image_height: int) -> dict[str, Any]:
    return {
        "image_width": int(image_width),
        "image_height": int(image_height),
        "line_groups": {"left": [], "right": [], "vertical": []},
        "scale_constraints": [],
        "intrinsics_hint": {"sensor_width_mm": 36.0},
    }


def load_constraints(project_dir: str | Path) -> dict[str, Any]:
    project = open_project(project_dir)
    solver_fields = _read_json(project.constraints_path)
    ui_fields = _read_json(project.ui_state_path)
    return {**solver_fields, **ui_fields}


def save_constraints(project_dir: str | Path, constraints: dict[str, Any]) -> dict[str, Any]:
    project = open_project(project_dir)
    if project.source_image and ("image_width" not in constraints or "image_height" not in constraints):
        width, height = _image_size(project.source_image)
        constraints = {"image_width": width, "image_height": height, **constraints}
    solver_fields = {k: v for k, v in constraints.items() if k not in _UI_ONLY_KEYS}
    ui_fields = {k: v for k, v in constraints.items() if k in _UI_ONLY_KEYS}
    _write_json(project.constraints_path, solver_fields)
    if ui_fields:
        _write_json(project.ui_state_path, ui_fields)
    return constraints


def solve_project(project_dir: str | Path) -> dict[str, Any]:
    project = open_project(project_dir)
    if project.source_image is None:
        raise ValueError("Project has no source image.")
    constraints = load_constraints(project.project_dir)
    overlay_path = project.overlay_path

    try:
        solve = solve_from_constraints(
            project.source_image,
            constraints,
            debug_overlay_path=overlay_path,
        )
    except ValueError:
        solve = solve_still_image(
            project.source_image,
            image_size=(int(constraints["image_width"]), int(constraints["image_height"])),
            intrinsics_hint=constraints.get("intrinsics_hint"),
            detect_vanishing_points=False,
        )
        solve.debug_metadata["ui_warning"] = (
            "Add left and right guide lines to produce an artist-guided solve."
        )

    save_solve_json(solve, project.solve_path)
    return solve_response(project, solve)


def analyze_project(
    project_dir: str | Path,
    *,
    provider: str = "lmstudio",
    model: str = "",
    base_url: str = "http://127.0.0.1:1234/v1",
    api_key: str | None = None,
    enable_preanalysis: bool = True,
    timeout_seconds: float = 120.0,
) -> dict[str, Any]:
    project = open_project(project_dir)
    if project.source_image is None:
        raise ValueError("Project has no source image.")
    constraints = load_constraints(project.project_dir)
    preanalysis, preanalysis_status, preanalysis_warning = _preanalyze_image(
        project,
        constraints,
        provider=provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
        enabled=enable_preanalysis,
        timeout_seconds=timeout_seconds,
    )
    solve, analysis_mode = _analysis_solve(project, constraints)
    response = solve_response(project, solve)
    analysis = camera_analysis_response(solve, constraints, analysis_mode)
    if preanalysis is not None:
        analysis["notes"] = [
            f"Image reading: {preanalysis['summary']}",
            *analysis["notes"],
        ]
    if preanalysis_warning:
        response["summary"]["warnings"] = [
            *response["summary"].get("warnings", []),
            preanalysis_warning,
        ]
    return {
        **response,
        "analysis": analysis,
        "preanalysis": preanalysis,
        "preanalysis_status": preanalysis_status,
        "preanalysis_warning": preanalysis_warning,
    }


def llm_guidance_project(
    project_dir: str | Path,
    *,
    provider: str = "lmstudio",
    model: str = "",
    base_url: str = "http://127.0.0.1:1234/v1",
    api_key: str | None = None,
    prompt: str | None = None,
    timeout_seconds: float = 120.0,
) -> dict[str, Any]:
    project = open_project(project_dir)
    if project.source_image is None:
        raise ValueError("Project has no source image.")
    constraints = load_constraints(project.project_dir)
    analysis_payload = analyze_project(project.project_dir, enable_preanalysis=False)
    references = search_scale_references(query=None, category=None)
    helper = create_multimodal_provider(
        provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
    )
    observation = helper.analyze_image(
        project.source_image,
        prompt=prompt,
        candidate_reference_ids=[reference.id for reference in references[:20]],
        app_context=_llm_app_context(constraints, analysis_payload),
    )
    guidance_path = project.project_dir / "llm_guidance.json"
    _write_json(guidance_path, observation.to_dict())
    return {
        "project": project.to_dict(),
        "guidance": observation.to_dict(),
        "guidance_path": str(guidance_path),
        "analysis": analysis_payload["analysis"],
        "summary": analysis_payload["summary"],
    }


def load_project_solve(project_dir: str | Path) -> AtlasSolve:
    project = open_project(project_dir)
    if not project.solve_path.is_file():
        raise FileNotFoundError("Project has no solved atlas_solve.json.")
    return load_solve_json(project.solve_path)


def promote_scale_cue(
    project_dir: str | Path,
    reference_id: str,
    bbox_px: list[float],
) -> dict[str, Any]:
    """Convert an LLM-detected bbox into a scale constraint and append it to constraints.json.

    bbox_px is [x1, y1, x2, y2] in image pixels (top-left origin).
    Converts to image_points [[cx, y1], [cx, y2]] — top and bottom of the detected object
    at the horizontal centre of the bbox, matching the height-guide constraint schema.
    Returns {"constraints": ..., "promoted": bool}.
    """
    if len(bbox_px) != 4:
        raise ValueError("bbox_px must have exactly 4 values: [x1, y1, x2, y2]")
    x1, y1, x2, y2 = float(bbox_px[0]), float(bbox_px[1]), float(bbox_px[2]), float(bbox_px[3])
    cx = (x1 + x2) / 2.0
    new_constraint: dict[str, Any] = {
        "reference_id": reference_id,
        "image_points": [[cx, y1], [cx, y2]],
    }

    project = open_project(project_dir)
    constraints = load_constraints(project.project_dir)
    existing = constraints.get("scale_constraints", [])

    if any(sc.get("reference_id") == reference_id for sc in existing):
        return {"constraints": constraints, "promoted": False}

    constraints["scale_constraints"] = [*existing, new_constraint]
    save_constraints(project.project_dir, constraints)
    return {"constraints": constraints, "promoted": True}


def export_review_package(
    project_dir: str | Path,
    *,
    package_name: str = "atlas_review_001",
    include_usd: bool = True,
) -> dict[str, Any]:
    project = open_project(project_dir)
    solve = load_project_solve(project.project_dir)
    result = build_review_package(
        solve,
        project.project_dir / "review_packages",
        package_name=package_name,
        source_image_path=project.source_image,
        debug_overlay_path=project.overlay_path if project.overlay_path.is_file() else None,
        include_usd=include_usd,
    )
    return review_package_response(result)


def reference_response(query: str | None = None, category: str | None = None) -> dict[str, Any]:
    return {
        "categories": list_categories(),
        "references": [
            reference.to_dict()
            for reference in search_scale_references(query=query, category=category)
        ],
    }


def llm_models_response(
    provider: str = "lmstudio",
    *,
    model: str = "",
    base_url: str = "http://127.0.0.1:1234/v1",
    api_key: str | None = None,
) -> dict[str, Any]:
    return provider_models_response(
        provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
    )


def solve_response(project: AtlasUiProject, solve: AtlasSolve) -> dict[str, Any]:
    camera = solve.debug_metadata.get("camera_estimation", {})
    return {
        "project": project.to_dict(),
        "solve": solve.to_dict(),
        "summary": {
            "source_method": solve.source_method,
            "confidence": solve.confidence,
            "vanishing_points": len(solve.vanishing_points),
            "guided_lines": solve.debug_metadata.get("num_lines_total", 0),
            "focal_length_mm": camera.get("focal_length_mm"),
            "horizon_angle_deg": camera.get("horizon_angle"),
            "warnings": _solve_warnings(solve),
        },
    }


def camera_analysis_response(
    solve: AtlasSolve,
    constraints: dict[str, Any],
    analysis_mode: str,
) -> dict[str, Any]:
    camera = solve.camera
    intrinsics = camera.intrinsics
    extrinsics = camera.extrinsics
    fx = intrinsics.fx_px or 0.0
    fy = intrinsics.fy_px or fx
    cx = intrinsics.cx_px if intrinsics.cx_px is not None else intrinsics.image_width / 2.0
    cy = intrinsics.cy_px if intrinsics.cy_px is not None else intrinsics.image_height / 2.0
    intrinsic_matrix = (
        (fx, 0.0, cx),
        (0.0, fy, cy),
        (0.0, 0.0, 1.0),
    )
    view_rows = tuple(row[:4] for row in extrinsics.camera_view_matrix[:3])
    projection_matrix = _matrix3x3_times_3x4(intrinsic_matrix, view_rows)
    rotation = extrinsics.camera_rotation_matrix
    camera_estimation = solve.debug_metadata.get("camera_estimation", {})
    return {
        "mode": analysis_mode,
        "coordinate_system": extrinsics.coordinate_system,
        "up_axis": extrinsics.up_axis,
        "intrinsic_matrix": _matrix_to_lists(intrinsic_matrix),
        "view_matrix": _matrix_to_lists(extrinsics.camera_view_matrix),
        "projection_matrix": _matrix_to_lists(projection_matrix),
        "camera_position": list(extrinsics.camera_position),
        "focal_px": {"fx": fx, "fy": fy},
        "principal_point_px": {"cx": cx, "cy": cy},
        "fov_deg": {
            "horizontal": _fov_degrees(intrinsics.image_width, fx),
            "vertical": _fov_degrees(intrinsics.image_height, fy),
        },
        "rotation_quality": {
            "determinant": _determinant3(rotation),
            "orthogonality_residual": _orthogonality_residual(rotation),
        },
        "vanishing_point_support": {
            "detected": len(solve.vanishing_points),
            "left_lines": len(constraints.get("line_groups", {}).get("left", [])),
            "right_lines": len(constraints.get("line_groups", {}).get("right", [])),
            "vertical_lines": len(constraints.get("line_groups", {}).get("vertical", [])),
            "scale_guides": len(constraints.get("scale_constraints", [])),
            "horizon_angle_deg": camera_estimation.get("horizon_angle"),
            "focal_source": camera_estimation.get("focal_source", "metadata_or_hint"),
        },
        "readiness": _analysis_readiness(solve, constraints),
        "notes": _analysis_notes(solve),
    }


def _llm_app_context(
    constraints: dict[str, Any],
    analysis_payload: dict[str, Any],
) -> dict[str, Any]:
    analysis = analysis_payload.get("analysis", {})
    summary = analysis_payload.get("summary", {})
    return {
        "product": "Atlas Camera",
        "purpose": "Still-image camera inference and projection-prep for DCC handoff.",
        "solver_boundary": (
            "LLM guidance is advisory. Deterministic camera matrices, guide counts, "
            "and benchmark metrics remain the source of truth."
        ),
        "image_size": {
            "width": constraints.get("image_width"),
            "height": constraints.get("image_height"),
        },
        "guide_counts": {
            "left": len(constraints.get("line_groups", {}).get("left", [])),
            "right": len(constraints.get("line_groups", {}).get("right", [])),
            "vertical": len(constraints.get("line_groups", {}).get("vertical", [])),
            "scale": len(constraints.get("scale_constraints", [])),
        },
        "solve_summary": summary,
        "camera_analysis": {
            "mode": analysis.get("mode"),
            "focal_px": analysis.get("focal_px"),
            "principal_point_px": analysis.get("principal_point_px"),
            "fov_deg": analysis.get("fov_deg"),
            "rotation_quality": analysis.get("rotation_quality"),
            "vanishing_point_support": analysis.get("vanishing_point_support"),
            "readiness": analysis.get("readiness"),
        },
        "dataset_evidence_policy": (
            "ETH3D/DTU datasets are used to validate solver behavior and failure cases. "
            "Mention them only as comparable evidence categories, not as direct proof for this image."
        ),
    }


def _preanalysis_app_context(constraints: dict[str, Any]) -> dict[str, Any]:
    return {
        "product": "Atlas Camera",
        "purpose": "Image-first advisory reading before deterministic camera analysis.",
        "authority": (
            "This pre-analysis is advisory only. Do not output final camera matrices, "
            "do not claim the solve is complete, and do not mutate constraints."
        ),
        "image_size": {
            "width": constraints.get("image_width"),
            "height": constraints.get("image_height"),
        },
        "current_artist_guides": {
            "left": len(constraints.get("line_groups", {}).get("left", [])),
            "right": len(constraints.get("line_groups", {}).get("right", [])),
            "vertical": len(constraints.get("line_groups", {}).get("vertical", [])),
            "scale": len(constraints.get("scale_constraints", [])),
        },
        "requested_observations": [
            "scene description and major object groups",
            "likely real-world scale indicators",
            "visible perspective-line families",
            "horizon/depth cues",
            "obvious lens distortion or wide-angle risk",
            "occlusions or ambiguous geometry that could bias recovery",
            "recommended next artist guides",
        ],
    }


def _preanalysis_prompt(candidate_reference_ids: list[str]) -> str:
    return (
        "Read this image before any Atlas camera estimation. Describe the scene, "
        "identify objects that may provide rough scale, obvious perspective families, "
        "horizon and depth cues, lens distortion risks, occlusions, and recommended "
        "artist guide lines. Include scale candidates such as people, cardboard boxes, "
        "corridor or tunnel width/height, wall/floor seams, clothing, equipment, and debris. "
        "Return strict JSON with keys: summary, scene_description, scale_candidates, "
        "scale_cues, perspective_cues, lens_distortion_notes, occlusion_notes, "
        "recommended_guides, technical_guidance, solve_risk_notes, dataset_evidence, warnings. Keep all "
        "claims advisory and uncertainty-aware. Candidate scale reference IDs:\n"
        f"{json.dumps(candidate_reference_ids, indent=2)}"
    )


def _preanalyze_image(
    project: AtlasUiProject,
    constraints: dict[str, Any],
    *,
    provider: str,
    model: str,
    base_url: str,
    api_key: str | None,
    enabled: bool,
    timeout_seconds: float = 120.0,
) -> tuple[dict[str, Any] | None, str, str | None]:
    if not enabled:
        return None, "skipped", None

    references = search_scale_references(query=None, category=None)
    candidate_ids = [reference.id for reference in references[:20]]
    helper = create_multimodal_provider(
        provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
    )
    try:
        observation = helper.analyze_image(
            project.source_image,
            prompt=_preanalysis_prompt(candidate_ids),
            candidate_reference_ids=candidate_ids,
            app_context=_preanalysis_app_context(constraints),
        )
    except Exception as exc:  # noqa: BLE001 - pre-analysis must not block deterministic Analyze.
        return None, "failed", f"Image reading skipped: {exc}"

    payload = observation.to_dict()
    _write_json(project.project_dir / "llm_preanalysis.json", payload)
    return payload, "available", None


def _analysis_solve(project: AtlasUiProject, constraints: dict[str, Any]) -> tuple[AtlasSolve, str]:
    try:
        return (
            solve_from_constraints(
                project.source_image,
                constraints,
                debug_overlay_path=None,
            ),
            "artist_guided_matrix_analysis",
        )
    except ValueError:
        return (
            solve_still_image(
                project.source_image,
                image_size=(int(constraints["image_width"]), int(constraints["image_height"])),
                intrinsics_hint=constraints.get("intrinsics_hint"),
                detect_vanishing_points=False,
            ),
            "metadata_intrinsics_analysis",
        )


def _analysis_readiness(solve: AtlasSolve, constraints: dict[str, Any]) -> list[dict[str, Any]]:
    line_groups = constraints.get("line_groups", {})
    left = len(line_groups.get("left", []))
    right = len(line_groups.get("right", []))
    vertical = len(line_groups.get("vertical", []))
    scale = len(constraints.get("scale_constraints", []))
    return [
        _readiness_item("Left vanishing family", left >= 2, f"{left}/2 guide lines"),
        _readiness_item("Right vanishing family", right >= 2, f"{right}/2 guide lines"),
        _readiness_item("Vertical reference", vertical >= 1, f"{vertical} vertical guide lines"),
        _readiness_item("Scale reference", scale >= 1, f"{scale} scale guides"),
        _readiness_item("Projection matrix", solve.source_method != "automatic_still_image_metadata_only", solve.source_method),
    ]


def _readiness_item(label: str, ok: bool, detail: str) -> dict[str, str]:
    return {
        "label": label,
        "status": "ok" if ok else "needs_input",
        "detail": detail,
    }


def _analysis_notes(solve: AtlasSolve) -> list[str]:
    notes = [
        "K maps camera coordinates into image pixels using Atlas top-left image coordinates.",
        "P = K[R|t] is reported for review; metric translation is not scored from a single still image.",
    ]
    notes.extend(str(note) for note in solve.debug_metadata.get("notes", []))
    if solve.source_method != "artist_guided_constraints":
        notes.append("Add left and right guide families to promote analysis from metadata-only to guided camera geometry.")
    return notes


def _matrix3x3_times_3x4(
    left: tuple[tuple[float, float, float], ...],
    right: tuple[tuple[float, float, float, float], ...],
) -> tuple[tuple[float, float, float, float], ...]:
    return tuple(
        tuple(
            sum(left[row][index] * right[index][col] for index in range(3))
            for col in range(4)
        )
        for row in range(3)
    )


def _matrix_to_lists(matrix: tuple[tuple[float, ...], ...]) -> list[list[float]]:
    return [[float(value) for value in row] for row in matrix]


def _fov_degrees(image_size_px: int, focal_px: float) -> float | None:
    if focal_px <= 0:
        return None
    return 2.0 * math.degrees(math.atan(image_size_px / (2.0 * focal_px)))


def _determinant3(matrix: tuple[tuple[float, float, float], ...]) -> float:
    return (
        matrix[0][0] * ((matrix[1][1] * matrix[2][2]) - (matrix[1][2] * matrix[2][1]))
        - matrix[0][1] * ((matrix[1][0] * matrix[2][2]) - (matrix[1][2] * matrix[2][0]))
        + matrix[0][2] * ((matrix[1][0] * matrix[2][1]) - (matrix[1][1] * matrix[2][0]))
    )


def _orthogonality_residual(matrix: tuple[tuple[float, float, float], ...]) -> float:
    columns = [
        (matrix[0][index], matrix[1][index], matrix[2][index])
        for index in range(3)
    ]
    residuals = []
    for first in range(3):
        for second in range(first + 1, 3):
            residuals.append(abs(sum(columns[first][i] * columns[second][i] for i in range(3))))
    residuals.extend(abs(sum(value * value for value in column) - 1.0) for column in columns)
    return max(residuals) if residuals else 0.0


def review_package_response(result: ReviewPackageResult) -> dict[str, Any]:
    return {
        "package_dir": str(result.package_dir),
        "files": {name: str(path) for name, path in sorted(result.files.items())},
        "warnings": list(result.warnings),
    }


def _solve_warnings(solve: AtlasSolve) -> list[str]:
    warnings = []
    if solve.debug_metadata.get("ui_warning"):
        warnings.append(str(solve.debug_metadata["ui_warning"]))
    warnings.extend(str(warning) for warning in solve.debug_metadata.get("warnings", []))
    if solve.source_method != "artist_guided_constraints":
        warnings.append("Artist-guided solve needs at least two left and two right guide lines.")
    return warnings


def new_project_dir(base_dir: str | Path | None = None) -> Path:
    root = Path(base_dir).expanduser().resolve() if base_dir else Path.cwd() / "atlas_ui_projects"
    return root / f"atlas_project_{uuid4().hex[:8]}"
