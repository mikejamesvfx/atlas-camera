"""atlas_project.json — the versioned reproducibility manifest."""

import json
import time

import pytest

from atlas_camera.core.schema import (
    AtlasExtrinsics,
    AtlasIntrinsics,
    AtlasPlateRef,
    AtlasSolve,
    LatentCamera,
)
from atlas_camera.exporters.manifest import (
    MANIFEST_FILENAME,
    MANIFEST_SCHEMA_VERSION,
    ManifestArtifact,
    build_project_manifest,
    file_md5,
    load_project_manifest,
    manifest_identity_hash,
    solve_content_fingerprint,
    write_project_manifest,
)


def _solve(height=1.6, plate_path=None):
    intr = AtlasIntrinsics(image_width=800, image_height=600, fx_px=700.0,
                           fy_px=700.0, cx_px=400.0, cy_px=300.0,
                           focal_length_mm=35.0, sensor_width_mm=36.0)
    vm = ((1.0, 0, 0, 0), (0, 1.0, 0, -height), (0, 0, 1.0, 0), (0, 0, 0, 1.0))
    extr = AtlasExtrinsics(camera_position=(0.0, height, 0.0),
                           camera_view_matrix=vm)
    s = AtlasSolve(camera=LatentCamera(intrinsics=intr, extrinsics=extr))
    s.debug_metadata["scale_source"] = "manual_override"
    s.debug_metadata["seed"] = 0
    if plate_path:
        s.source_plate = AtlasPlateRef(image_path=str(plate_path),
                                       colorspace="ACEScg", bit_depth="16f",
                                       is_proxy=False)
    return s


def test_build_and_write_round_trip(tmp_path):
    plate = tmp_path / "plate.exr"
    plate.write_bytes(b"fake exr bytes")
    s = _solve(plate_path=plate)
    path = write_project_manifest(
        s, tmp_path, artifacts=[ManifestArtifact("nuke_scene", "a.nk", "Test")])
    data = load_project_manifest(path)
    assert data["schema"] == MANIFEST_SCHEMA_VERSION
    assert data["plate"]["md5"] == file_md5(plate)
    assert data["plate"]["colorspace"] == "ACEScg"
    assert data["scale"]["status"] == "manual"
    assert data["solve"]["confidence_detail"].get("scale") is not None
    assert data["artifacts"] == [{"kind": "nuke_scene", "path": "a.nk",
                                  "exporter": "Test"}]
    assert data["identity_hash"] == manifest_identity_hash(s)


def test_merge_appends_for_same_solve(tmp_path):
    s = _solve()
    write_project_manifest(s, tmp_path,
                           artifacts=[ManifestArtifact("nuke_scene", "a.nk", "T")])
    write_project_manifest(s, tmp_path,
                           artifacts=[ManifestArtifact("maya_scene", "b.ma", "T")])
    # Duplicate append is deduped.
    write_project_manifest(s, tmp_path,
                           artifacts=[ManifestArtifact("maya_scene", "b.ma", "T")])
    data = load_project_manifest(tmp_path / MANIFEST_FILENAME)
    kinds = [a["kind"] for a in data["artifacts"]]
    assert kinds == ["nuke_scene", "maya_scene"]


def test_new_solve_overwrites(tmp_path):
    write_project_manifest(_solve(1.6), tmp_path,
                           artifacts=[ManifestArtifact("nuke_scene", "a.nk", "T")])
    write_project_manifest(_solve(45.0), tmp_path,
                           artifacts=[ManifestArtifact("usd_camera", "c.usda", "T")])
    data = load_project_manifest(tmp_path / MANIFEST_FILENAME)
    assert [a["kind"] for a in data["artifacts"]] == ["usd_camera"]


def test_missing_plate_is_tolerated(tmp_path):
    s = _solve()
    s.image_path = "Z:/nowhere/missing.png"
    data = build_project_manifest(s)
    assert data["plate"]["md5"] is None
    assert data["plate"]["path"]


def test_identity_hash_stable_across_artifacts_and_time(tmp_path):
    s = _solve()
    a = build_project_manifest(s)
    time.sleep(0.01)
    b = build_project_manifest(
        s, artifacts=[ManifestArtifact("nuke_scene", "x.nk", "T")],
        extra={"note": "whatever"})
    assert a["identity_hash"] == b["identity_hash"]
    # But a different solve changes it.
    assert build_project_manifest(_solve(45.0))["identity_hash"] != a["identity_hash"]


def test_fingerprint_includes_plate_content(tmp_path):
    plate = tmp_path / "p.exr"
    plate.write_bytes(b"v1")
    fp1 = solve_content_fingerprint(_solve(plate_path=plate))
    plate.write_bytes(b"v2 different")
    fp2 = solve_content_fingerprint(_solve(plate_path=plate))
    assert fp1 != fp2


def test_loader_rejects_garbage_and_future_schema(tmp_path):
    bad = tmp_path / "x.json"
    bad.write_text(json.dumps({"hello": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="Not an atlas_project"):
        load_project_manifest(bad)
    future = tmp_path / "y.json"
    future.write_text(json.dumps({"schema": 99}), encoding="utf-8")
    with pytest.raises(ValueError, match="newer"):
        load_project_manifest(future)


def test_v1_fixture_loads():
    """Migration seed: this frozen v1 payload must load in every future
    schema version (add the migration in load_project_manifest, then extend
    this test with the migrated expectations)."""
    fixture = {
        "schema": 1, "atlas_version": "0.6.0", "solve_schema_version": "0.2",
        "generated_at": "2026-07-18T12:00:00", "updated_at": "2026-07-18T12:00:00",
        "identity_hash": "abcdef0123456789",
        "plate": {"path": "plate.exr", "md5": None, "colorspace": "ACEScg",
                  "bit_depth": "16f"},
        "solve_fingerprint": "0123456789abcdef",
        "solve": {"source_method": "manual", "seed": 0, "confidence": 0.5,
                  "confidence_detail": {}},
        "models": {"depth_model_id": None, "depth_is_metric": None,
                   "learned_prior": None, "vlm_report_present": False},
        "scale": {"status": "manual", "scale_source": "manual_override",
                  "confidence": 1.0, "camera_height_m": 1.6,
                  "safe_to_export": True, "detail": ""},
        "scene_health": None,
        "settings": {"scale_source": "manual_override"},
        "artifacts": [],
    }
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "atlas_project.json"
        p.write_text(json.dumps(fixture), encoding="utf-8")
        assert load_project_manifest(p)["schema"] == 1


def test_review_package_writes_manifest(tmp_path):
    pytest.importorskip("PIL")
    from atlas_camera.exporters.review_package import build_review_package
    from PIL import Image
    plate = tmp_path / "src.png"
    Image.new("RGB", (32, 24), (90, 90, 90)).save(plate)
    s = _solve(plate_path=plate)
    s.image_path = str(plate)
    result = build_review_package(s, tmp_path, package_name="pkg",
                                  include_usd=False)
    manifest = load_project_manifest(result.files["manifest"])
    assert manifest["scale"]["status"] == "manual"
    assert any(a["kind"] == "atlas_solve" for a in manifest["artifacts"])
    report = result.files["report"].read_text(encoding="utf-8")
    assert "Manifest identity: " in report


def test_export_node_helper_writes_manifest_and_embeds_identity(tmp_path):
    pytest.importorskip("torch")
    from atlas_camera.comfy.nodes import _write_export_manifest
    nk = tmp_path / "scene.nk"
    nk.write_text("Root {\n}\n", encoding="utf-8")
    s = _solve()
    _write_export_manifest(s, tmp_path, [("nuke_scene", str(nk))], "Test")
    data = load_project_manifest(tmp_path / MANIFEST_FILENAME)
    assert data["artifacts"][0]["exporter"] == "Test"
    first_line = nk.read_text(encoding="utf-8").splitlines()[0]
    assert first_line == f"# atlas_project_identity: {data['identity_hash']}"
    # Idempotent: a re-export replaces the comment, never stacks it.
    _write_export_manifest(s, tmp_path, [("nuke_scene", str(nk))], "Test")
    lines = nk.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("# atlas_project_identity: ")
    assert not lines[1].startswith("# atlas_project_identity")
