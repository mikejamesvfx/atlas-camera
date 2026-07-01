import json
import sys

import pytest

from atlas_camera.ui.project import (
    analyze_project,
    create_project,
    export_review_package,
    load_constraints,
    promote_scale_cue,
    reference_response,
    save_constraints,
    solve_project,
    UI_STATE_FILE,
)


def _guided_constraints():
    return {
        "image_width": 160,
        "image_height": 96,
        "line_groups": {
            "left": [
                [[0, 62], [159, 12]],
                [[0, 57], [159, 26]],
            ],
            "right": [
                [[0, 12], [159, 36]],
                [[0, 26], [159, 42]],
            ],
            "vertical": [],
        },
        "scale_constraints": [
            {
                "reference_id": "person_175cm",
                "image_points": [[80, 80], [80, 20]],
            }
        ],
        "intrinsics_hint": {"sensor_width_mm": 36.0},
    }


def test_ui_project_creation_and_constraints_round_trip(synthetic_perspective_image, tmp_path):
    image_path, _ = synthetic_perspective_image
    project = create_project(tmp_path / "lineup", image_path=image_path)

    assert project.project_dir.is_dir()
    assert project.source_image and project.source_image.is_file()
    assert project.constraints_path.is_file()

    constraints = load_constraints(project.project_dir)
    assert constraints["image_width"] == 160
    assert constraints["line_groups"] == {"left": [], "right": [], "vertical": []}

    saved = save_constraints(project.project_dir, _guided_constraints())
    assert saved["scale_constraints"][0]["reference_id"] == "person_175cm"
    assert json.loads(project.constraints_path.read_text(encoding="utf-8")) == saved


def test_ui_reference_search_uses_registry():
    response = reference_response(query="person")

    ids = {item["id"] for item in response["references"]}
    assert "person_175cm" in ids
    assert response["categories"]


def test_ui_solve_and_review_package_export(synthetic_perspective_image, tmp_path):
    image_path, _ = synthetic_perspective_image
    project = create_project(tmp_path / "lineup", image_path=image_path)
    save_constraints(project.project_dir, _guided_constraints())

    solved = solve_project(project.project_dir)
    assert solved["summary"]["source_method"] == "artist_guided_constraints"
    assert solved["summary"]["vanishing_points"] >= 2
    assert project.solve_path.is_file()
    assert project.overlay_path.is_file()

    exported = export_review_package(project.project_dir, include_usd=False)
    assert (project.project_dir / "review_packages" / "atlas_review_001").is_dir()
    assert "atlas_solve" in exported["files"]
    assert "report" in exported["files"]


def test_ui_camera_analysis_reports_matrices(synthetic_perspective_image, tmp_path):
    image_path, _ = synthetic_perspective_image
    project = create_project(tmp_path / "lineup", image_path=image_path)
    save_constraints(project.project_dir, _guided_constraints())

    analyzed = analyze_project(project.project_dir, enable_preanalysis=False)
    analysis = analyzed["analysis"]

    assert analysis["mode"] == "artist_guided_matrix_analysis"
    assert len(analysis["intrinsic_matrix"]) == 3
    assert len(analysis["projection_matrix"]) == 3
    assert len(analysis["projection_matrix"][0]) == 4
    assert analysis["rotation_quality"]["determinant"] > 0.9
    assert analysis["readiness"][0]["status"] == "ok"


def test_ui_analyze_runs_image_preanalysis_before_matrix_analysis(synthetic_perspective_image, tmp_path, monkeypatch):
    image_path, _ = synthetic_perspective_image
    project = create_project(tmp_path / "lineup", image_path=image_path)
    saved_constraints = save_constraints(project.project_dir, _guided_constraints())
    calls = []

    from atlas_camera.inference import MultimodalSceneObservation
    from atlas_camera.ui import project as project_module

    class FakeProvider:
        def __init__(self, *, model, base_url, provider="lmstudio"):
            self.model = model
            self.base_url = base_url
            self.provider = provider

        def analyze_image(self, image_path, *, prompt=None, candidate_reference_ids=None, app_context=None):
            calls.append(
                {
                    "image_path": str(image_path),
                    "prompt": prompt,
                    "candidate_reference_ids": candidate_reference_ids,
                    "app_context": app_context,
                }
            )
            return MultimodalSceneObservation(
                image_path=str(image_path),
                summary="Scene has strong facade edges and a human scale anchor.",
                model=self.model,
                provider=self.provider,
                technical_guidance=["Use facade cornice and floor seams as perspective families."],
                solve_risk_notes=["Mild barrel distortion may bend long horizontal edges."],
                scale_cues=[],
            )

    monkeypatch.setattr(
        project_module,
        "create_multimodal_provider",
        lambda provider, *, model, base_url, api_key=None, timeout_seconds=120.0: FakeProvider(
            model=model,
            base_url=base_url,
            provider=provider,
        ),
    )

    analyzed = project_module.analyze_project(
        project.project_dir,
        provider="lmstudio",
        model="qwen2.5-vl",
        base_url="http://127.0.0.1:1234/v1",
    )

    assert calls
    assert "solve_summary" not in calls[0]["app_context"]
    assert calls[0]["app_context"]["current_artist_guides"] == {
        "left": 2,
        "right": 2,
        "vertical": 0,
        "scale": 1,
    }
    assert analyzed["preanalysis_status"] == "available"
    assert analyzed["preanalysis"]["summary"] == "Scene has strong facade edges and a human scale anchor."
    assert analyzed["analysis"]["notes"][0].startswith("Image reading:")
    assert (project.project_dir / "llm_preanalysis.json").is_file()
    assert load_constraints(project.project_dir) == saved_constraints


def test_ui_analyze_preanalysis_failure_keeps_deterministic_analysis(synthetic_perspective_image, tmp_path, monkeypatch):
    image_path, _ = synthetic_perspective_image
    project = create_project(tmp_path / "lineup", image_path=image_path)
    save_constraints(project.project_dir, _guided_constraints())

    from atlas_camera.ui import project as project_module

    class FailingProvider:
        def __init__(self, *, model, base_url):
            self.model = model
            self.base_url = base_url

        def analyze_image(self, image_path, *, prompt=None, candidate_reference_ids=None, app_context=None):
            raise RuntimeError("Selected model does not support image input")

    monkeypatch.setattr(
        project_module,
        "create_multimodal_provider",
        lambda provider, *, model, base_url, api_key=None, timeout_seconds=120.0: FailingProvider(
            model=model,
            base_url=base_url,
        ),
    )

    analyzed = project_module.analyze_project(project.project_dir)

    assert analyzed["preanalysis"] is None
    assert analyzed["preanalysis_status"] == "failed"
    assert "does not support image input" in analyzed["preanalysis_warning"]
    assert analyzed["analysis"]["projection_matrix"]
    assert any("Image reading skipped" in warning for warning in analyzed["summary"]["warnings"])


def test_ui_llm_guidance_uses_local_ollama_helper(synthetic_perspective_image, tmp_path, monkeypatch):
    image_path, _ = synthetic_perspective_image
    project = create_project(tmp_path / "lineup", image_path=image_path)
    save_constraints(project.project_dir, _guided_constraints())

    class FakeProvider:
        def __init__(self, *, model, base_url, provider="lmstudio"):
            self.model = model
            self.base_url = base_url
            self.provider = provider

        def analyze_image(self, image_path, *, prompt=None, candidate_reference_ids=None, app_context=None):
            return MultimodalSceneObservation(
                image_path=str(image_path),
                summary="Use more vertical structure before export.",
                model=self.model,
                provider=self.provider,
                technical_guidance=["Add a vertical guide near the door frame."],
                dataset_evidence=["Compare against ETH3D indoor scenes for VP stability."],
            )

    from atlas_camera.ui import project as project_module
    from atlas_camera.inference import MultimodalSceneObservation

    monkeypatch.setattr(
        project_module,
        "create_multimodal_provider",
        lambda provider, *, model, base_url, api_key=None, timeout_seconds=120.0: FakeProvider(
            model=model,
            base_url=base_url,
            provider=provider,
        ),
    )

    guided = project_module.llm_guidance_project(
        project.project_dir,
        provider="lmstudio",
        model="qwen2.5-vl",
        base_url="http://127.0.0.1:1234/v1",
    )

    assert guided["guidance"]["summary"] == "Use more vertical structure before export."
    assert guided["guidance"]["technical_guidance"] == ["Add a vertical guide near the door frame."]
    assert (project.project_dir / "llm_guidance.json").is_file()


def test_ui_api_llm_models_reports_provider_diagnostics(monkeypatch):
    fastapi = pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    assert fastapi

    from fastapi.testclient import TestClient

    from atlas_camera.ui import api as api_module

    monkeypatch.setattr(
        api_module,
        "llm_models_response",
        lambda provider, *, model, base_url, api_key=None: {
            "provider": provider,
            "base_url": base_url,
            "model": model,
            "models": [
                {
                    "id": model,
                    "name": model,
                    "vision_capable": True,
                    "capabilities": ["vision"],
                    "raw": None,
                }
            ],
            "vision_capable": True,
            "diagnostic_status": "selected model advertises vision capability",
        },
    )

    client = TestClient(api_module.app)
    response = client.post(
        "/api/llm/models",
        json={
            "provider": "lmstudio",
            "model": "qwen2.5-vl",
            "base_url": "http://127.0.0.1:1234/v1",
        },
    )

    assert response.status_code == 200
    assert response.json()["provider"] == "lmstudio"
    assert response.json()["vision_capable"] is True


def test_ui_main_accepts_custom_port(monkeypatch):
    from atlas_camera.ui import __main__ as main_module

    captured = {}

    class FakeUvicorn:
        @staticmethod
        def run(app, *, host, port, reload):
            captured.update({"app": app, "host": host, "port": port, "reload": reload})

    monkeypatch.setitem(sys.modules, "uvicorn", FakeUvicorn)
    monkeypatch.setattr(sys, "argv", ["atlas-ui", "--host", "0.0.0.0", "--port", "8788", "--reload"])

    main_module.main()

    assert captured == {
        "app": "atlas_camera.ui.api:app",
        "host": "0.0.0.0",
        "port": 8788,
        "reload": True,
    }


def test_ui_api_smoke(synthetic_perspective_image, tmp_path):
    fastapi = pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    assert fastapi

    from fastapi.testclient import TestClient

    from atlas_camera.ui.api import app

    image_path, _ = synthetic_perspective_image
    client = TestClient(app)
    project_dir = tmp_path / "api-lineup"

    created = client.post(
        "/api/projects",
        data={"project_dir": str(project_dir), "image_path": str(image_path)},
    )
    assert created.status_code == 200
    assert created.json()["constraints"]["image_width"] == 160

    constraints = _guided_constraints()
    saved = client.put(
        "/api/constraints",
        json={"project_dir": str(project_dir), "constraints": constraints},
    )
    assert saved.status_code == 200

    solved = client.post("/api/solve", json={"project_dir": str(project_dir)})
    assert solved.status_code == 200
    assert solved.json()["summary"]["source_method"] == "artist_guided_constraints"

    analyzed = client.post(
        "/api/analyze",
        json={"project_dir": str(project_dir), "enable_preanalysis": False},
    )
    assert analyzed.status_code == 200
    assert analyzed.json()["analysis"]["projection_matrix"]

    exported = client.post(
        "/api/export/review-package",
        json={"project_dir": str(project_dir), "include_usd": False},
    )
    assert exported.status_code == 200
    assert "report" in exported.json()["files"]


def test_ui_promote_scale_cue_converts_bbox_to_image_points(synthetic_perspective_image, tmp_path):
    image_path, _ = synthetic_perspective_image
    project = create_project(tmp_path / "lineup", image_path=image_path)

    result = promote_scale_cue(project.project_dir, "person_175cm", [80.0, 20.0, 160.0, 80.0])

    assert result["promoted"] is True
    constraints = load_constraints(project.project_dir)
    sc = constraints["scale_constraints"][0]
    assert sc["reference_id"] == "person_175cm"
    # cx = (80 + 160) / 2 = 120; image_points = [[120, 20], [120, 80]]
    assert sc["image_points"][0] == pytest.approx([120.0, 20.0])
    assert sc["image_points"][1] == pytest.approx([120.0, 80.0])


def test_ui_promote_scale_cue_skips_duplicate_reference_id(synthetic_perspective_image, tmp_path):
    image_path, _ = synthetic_perspective_image
    project = create_project(tmp_path / "lineup", image_path=image_path)

    promote_scale_cue(project.project_dir, "person_175cm", [80.0, 20.0, 160.0, 80.0])
    result = promote_scale_cue(project.project_dir, "person_175cm", [10.0, 5.0, 50.0, 40.0])

    assert result["promoted"] is False
    constraints = load_constraints(project.project_dir)
    assert len(constraints["scale_constraints"]) == 1


def test_ui_promote_scale_cue_same_id_different_bbox_preserves_original_points(synthetic_perspective_image, tmp_path):
    image_path, _ = synthetic_perspective_image
    project = create_project(tmp_path / "dup_bbox", image_path=image_path)

    # First promotion: bbox [80, 20, 160, 80] → cx=120, points [[120,20],[120,80]]
    promote_scale_cue(project.project_dir, "person_175cm", [80.0, 20.0, 160.0, 80.0])

    # Second promotion with same ID but different bbox must be a no-op
    promote_scale_cue(project.project_dir, "person_175cm", [0.0, 0.0, 40.0, 50.0])

    constraints = load_constraints(project.project_dir)
    sc = constraints["scale_constraints"][0]
    assert sc["reference_id"] == "person_175cm"
    # Original image_points must be unchanged despite the second call
    assert sc["image_points"][0] == pytest.approx([120.0, 20.0])
    assert sc["image_points"][1] == pytest.approx([120.0, 80.0])


def test_ui_api_promote_scale_cue_endpoint(synthetic_perspective_image, tmp_path):
    fastapi = pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    assert fastapi

    from fastapi.testclient import TestClient
    from atlas_camera.ui.api import app

    image_path, _ = synthetic_perspective_image
    client = TestClient(app)
    project_dir = tmp_path / "promote-lineup"
    client.post("/api/projects", data={"project_dir": str(project_dir), "image_path": str(image_path)})

    response = client.post(
        "/api/promote_scale_cue",
        json={"project_dir": str(project_dir), "reference_id": "door_210cm", "bbox_px": [100.0, 40.0, 140.0, 90.0]},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["promoted"] is True
    sc = data["constraints"]["scale_constraints"][0]
    assert sc["reference_id"] == "door_210cm"
    # cx = (100 + 140) / 2 = 120
    assert sc["image_points"][0] == pytest.approx([120.0, 40.0])
    assert sc["image_points"][1] == pytest.approx([120.0, 90.0])


def test_save_constraints_splits_viewport3d_to_ui_state(synthetic_perspective_image, tmp_path):
    image_path, _ = synthetic_perspective_image
    project = create_project(tmp_path / "split_test", image_path=image_path)

    constraints_with_ui = {
        **_guided_constraints(),
        "viewport3d": {"schema_version": 1, "display": {"show_projection": True}},
    }
    save_constraints(project.project_dir, constraints_with_ui)

    # constraints.json must not contain viewport3d
    solver_on_disk = json.loads(project.constraints_path.read_text(encoding="utf-8"))
    assert "viewport3d" not in solver_on_disk
    assert "line_groups" in solver_on_disk

    # ui_state.json must contain viewport3d
    ui_state_path = project.project_dir / UI_STATE_FILE
    assert ui_state_path.is_file()
    ui_on_disk = json.loads(ui_state_path.read_text(encoding="utf-8"))
    assert ui_on_disk["viewport3d"]["schema_version"] == 1


def test_load_constraints_merges_solver_and_ui_state(synthetic_perspective_image, tmp_path):
    image_path, _ = synthetic_perspective_image
    project = create_project(tmp_path / "merge_test", image_path=image_path)

    constraints_with_ui = {
        **_guided_constraints(),
        "viewport3d": {"schema_version": 1, "display": {"show_grid": False}},
    }
    save_constraints(project.project_dir, constraints_with_ui)

    loaded = load_constraints(project.project_dir)
    assert loaded["line_groups"]["left"]  # solver field present
    assert loaded["viewport3d"]["display"]["show_grid"] is False  # ui field present


def test_solver_constraints_file_never_contains_ui_keys(synthetic_perspective_image, tmp_path):
    image_path, _ = synthetic_perspective_image
    project = create_project(tmp_path / "clean_test", image_path=image_path)

    # Save with viewport3d multiple times to ensure it never leaks into constraints.json
    for _ in range(3):
        constraints_with_ui = {
            **_guided_constraints(),
            "viewport3d": {"schema_version": 1, "proxy_objects": []},
        }
        save_constraints(project.project_dir, constraints_with_ui)

    on_disk = json.loads(project.constraints_path.read_text(encoding="utf-8"))
    assert "viewport3d" not in on_disk
