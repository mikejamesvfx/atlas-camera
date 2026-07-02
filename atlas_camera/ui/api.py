"""FastAPI app for the optional Atlas Camera local UI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
    from fastapi.responses import FileResponse
except ImportError as exc:  # pragma: no cover - exercised by optional installs.
    raise RuntimeError("Atlas UI requires FastAPI. Install with: pip install -e .[ui]") from exc

from atlas_camera.ui.project import (
    analyze_project,
    create_project,
    export_camera_usd,
    export_review_package,
    llm_guidance_project,
    llm_models_response,
    load_constraints,
    new_project_dir,
    open_project,
    promote_scale_cue,
    reference_response,
    save_constraints,
    solve_project,
)

app = FastAPI(title="Atlas Camera UI", version="0.1.0")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/projects")
async def create_or_open_project(
    project_dir: str | None = Form(default=None),
    base_dir: str | None = Form(default=None),
    image_path: str | None = Form(default=None),
    image: UploadFile | None = File(default=None),
) -> dict[str, Any]:
    target_dir = project_dir or str(new_project_dir(base_dir))
    image_bytes = await image.read() if image is not None else None
    project = create_project(
        target_dir,
        image_path=image_path,
        image_filename=image.filename if image is not None else None,
        image_bytes=image_bytes,
    )
    return {"project": project.to_dict(), "constraints": load_constraints(project.project_dir)}


@app.get("/api/projects")
def get_project(project_dir: str = Query(...)) -> dict[str, Any]:
    project = open_project(project_dir)
    return {"project": project.to_dict(), "constraints": load_constraints(project.project_dir)}


@app.get("/api/references")
def references(
    query: str | None = None,
    category: str | None = None,
) -> dict[str, Any]:
    return reference_response(query=query, category=category)


@app.put("/api/constraints")
def put_constraints(payload: dict[str, Any]) -> dict[str, Any]:
    project_dir = payload.get("project_dir")
    constraints = payload.get("constraints")
    if not project_dir or not isinstance(constraints, dict):
        raise HTTPException(status_code=400, detail="project_dir and constraints are required.")
    return {"constraints": save_constraints(project_dir, constraints)}


@app.post("/api/solve")
def post_solve(payload: dict[str, Any]) -> dict[str, Any]:
    project_dir = payload.get("project_dir")
    if not project_dir:
        raise HTTPException(status_code=400, detail="project_dir is required.")
    try:
        return solve_project(project_dir)
    except Exception as exc:  # noqa: BLE001 - API boundary should report structured failures.
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/analyze")
def post_analyze(payload: dict[str, Any]) -> dict[str, Any]:
    """Run optional LLM image pre-analysis then deterministic matrix analysis.

    Set `enable_preanalysis=false` to skip the LLM step and compute matrix analysis
    only. This is the primary solve-readiness endpoint; it always runs the
    deterministic solver. Use `/api/llm/guidance` for advisory-only LLM queries
    that do not re-run the solver.
    """
    project_dir = payload.get("project_dir")
    if not project_dir:
        raise HTTPException(status_code=400, detail="project_dir is required.")
    try:
        return analyze_project(
            project_dir,
            provider=str(payload.get("provider") or "lmstudio"),
            model=str(payload.get("model") or ""),
            base_url=str(payload.get("base_url") or "http://127.0.0.1:1234/v1"),
            api_key=payload.get("api_key"),
            enable_preanalysis=bool(payload.get("enable_preanalysis", True)),
            timeout_seconds=float(payload.get("timeout_seconds") or 120.0),
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/llm/guidance")
def post_llm_guidance(payload: dict[str, Any]) -> dict[str, Any]:
    """Ask the LLM for advisory guidance about the current project state.

    Unlike `/api/analyze`, this endpoint does NOT run the deterministic solver —
    it sends the source image and an optional artist prompt to the vision model
    and returns advisory text only. Use this for follow-up conversational queries
    after the initial solve.
    """
    project_dir = payload.get("project_dir")
    if not project_dir:
        raise HTTPException(status_code=400, detail="project_dir is required.")
    try:
        return llm_guidance_project(
            project_dir,
            provider=str(payload.get("provider") or "lmstudio"),
            model=str(payload.get("model") or ""),
            base_url=str(payload.get("base_url") or "http://127.0.0.1:1234/v1"),
            api_key=payload.get("api_key"),
            prompt=payload.get("prompt"),
            timeout_seconds=float(payload.get("timeout_seconds") or 120.0),
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/llm/models")
def post_llm_models(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        return llm_models_response(
            provider=str(payload.get("provider") or "lmstudio"),
            model=str(payload.get("model") or ""),
            base_url=str(payload.get("base_url") or "http://127.0.0.1:1234/v1"),
            api_key=payload.get("api_key"),
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/promote_scale_cue")
def post_promote_scale_cue(payload: dict[str, Any]) -> dict[str, Any]:
    project_dir = payload.get("project_dir")
    reference_id = payload.get("reference_id")
    bbox_px = payload.get("bbox_px")
    if not project_dir or not reference_id:
        raise HTTPException(status_code=400, detail="project_dir and reference_id are required.")
    if not isinstance(bbox_px, list) or len(bbox_px) != 4:
        raise HTTPException(status_code=400, detail="bbox_px must be a list of 4 numbers [x1, y1, x2, y2].")
    try:
        return promote_scale_cue(project_dir, reference_id, bbox_px)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/export/camera-usd")
def post_export_camera_usd(payload: dict[str, Any]) -> dict[str, Any]:
    """Export only the solved camera as a USD file (camera.usda in the project directory).

    Lighter alternative to the full review package when you only need the camera
    asset for DCC import. Requires usd-core (`pip install -e .[usd]`).
    """
    project_dir = payload.get("project_dir")
    if not project_dir:
        raise HTTPException(status_code=400, detail="project_dir is required.")
    try:
        return export_camera_usd(project_dir)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/export/review-package")
def post_export(payload: dict[str, Any]) -> dict[str, Any]:
    project_dir = payload.get("project_dir")
    if not project_dir:
        raise HTTPException(status_code=400, detail="project_dir is required.")
    try:
        return export_review_package(
            project_dir,
            package_name=str(payload.get("package_name") or "atlas_review_001"),
            include_usd=bool(payload.get("include_usd", True)),
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/files/{kind}")
def get_file(
    kind: str,
    project_dir: str = Query(...),
    package_name: str = Query(default="atlas_review_001"),
) -> FileResponse:
    project = open_project(project_dir)
    paths = {
        "source": project.source_image,
        "overlay": project.overlay_path,
        "solve": project.solve_path,
        "constraints": project.constraints_path,
        "camera_usd": project.project_dir / "camera.usda",
        "report": project.project_dir / "review_packages" / package_name / "report.md",
    }
    path = paths.get(kind)
    if path is None or not Path(path).is_file():
        raise HTTPException(status_code=404, detail=f"No file available for kind: {kind}")
    return FileResponse(path)
