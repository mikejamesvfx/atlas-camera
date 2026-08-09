"""Photographed/generated projection evidence stays explicit downstream."""

from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import subprocess

import pytest

from atlas_camera.comfy.viewport_payload import _serialize_projection_sources
from atlas_camera.core.camera_math import look_at_view_matrix
from atlas_camera.core.scene_health import evaluate_scene_health
from atlas_camera.core.schema import (
    AtlasExtrinsics,
    AtlasIntrinsics,
    AtlasProjectionScene,
    AtlasProxyPrimitive,
    AtlasSolve,
    LatentCamera,
    ProjectionSource,
)


def _camera(x: float = 0.0) -> LatentCamera:
    eye = (x, 1.6, 0.0)
    view, world, rotation = look_at_view_matrix(eye, (0.0, 0.5, -10.0))
    return LatentCamera(
        intrinsics=AtlasIntrinsics(
            image_width=64, image_height=48, focal_length_mm=35.0,
            sensor_width_mm=36.0, fx_px=60.0, fy_px=60.0,
            cx_px=32.0, cy_px=24.0,
        ),
        extrinsics=AtlasExtrinsics(
            camera_position=eye, camera_rotation_matrix=rotation,
            camera_world_matrix=world, camera_view_matrix=view,
        ),
    )


def _primary_geometry() -> list[AtlasProxyPrimitive]:
    return [AtlasProxyPrimitive(
        name="primary", primitive_type="plane",
        metadata={"n_vertices": 4, "n_faces": 2},
    )]


def _mixed_solve() -> AtlasSolve:
    solve = AtlasSolve(
        camera=_camera(),
        projection_scene=AtlasProjectionScene(proxy_geometry=_primary_geometry()),
    )
    solve.debug_metadata["scale_source"] = "manual_override"
    solve.projection_sources = [
        ProjectionSource(
            camera=_camera(1.0), name="photo", proxy_geometry=[],
            metadata={"evidence_type": "photographed"},
        ),
        ProjectionSource(
            camera=_camera(2.0), name="qwen", proxy_geometry=[],
            metadata={"evidence_type": "generated"},
        ),
    ]
    return solve


def test_viewport_and_health_distinguish_evidence_types():
    solve = _mixed_solve()

    sources = _serialize_projection_sources(solve)
    assert [source["evidence_type"] for source in sources] == [
        "photographed", "generated",
    ]

    health = evaluate_scene_health(solve).to_dict()
    assert health["projection_evidence_counts"] == {
        "photographed": 1, "generated": 1, "unknown": 0,
    }
    assert [layer["evidence_type"] for layer in health["per_layer"]] == [
        "photographed", "generated",
    ]
    mixed = next(flag for flag in health["flags"]
                 if flag["code"] == "mixed_projection_evidence")
    assert "generated cameras did not influence the photographed registration" in mixed["message"]
    zero_layers = [flag["layer"] for flag in health["flags"]
                   if flag["code"] == "zero_vertex_layer"]
    assert zero_layers == ["qwen"]


def test_legacy_projection_evidence_remains_unknown():
    solve = AtlasSolve(camera=_camera())
    solve.debug_metadata["scale_source"] = "manual_override"
    solve.projection_sources = [
        ProjectionSource(camera=_camera(1.0), name="missing", metadata={}),
        ProjectionSource(
            camera=_camera(2.0), name="unrecognized",
            metadata={"evidence_type": "legacy"},
        ),
    ]

    assert [source["evidence_type"] for source in _serialize_projection_sources(solve)] == [
        "unknown", "unknown",
    ]
    health = evaluate_scene_health(solve).to_dict()
    assert health["projection_evidence_counts"] == {
        "photographed": 0, "generated": 0, "unknown": 2,
    }


def test_photographed_source_reuses_primary_geometry_but_generated_does_not():
    from atlas_camera.exporters import _layers

    solve = _mixed_solve()

    photo_geometry, photo_origin = _layers._projection_geometry_for_source(
        solve, solve.projection_sources[0],
    )
    generated_geometry, generated_origin = _layers._projection_geometry_for_source(
        solve, solve.projection_sources[1],
    )

    assert photo_origin == "primary_scene"
    assert photo_geometry == solve.projection_scene.proxy_geometry
    assert generated_origin == "source"
    assert generated_geometry == []


def test_private_source_geometry_always_wins_over_primary_fallback():
    from atlas_camera.exporters import _layers

    solve = _mixed_solve()
    private = AtlasProxyPrimitive(name="private", primitive_type="box")
    solve.projection_sources[0].proxy_geometry = [private]

    geometry, origin = _layers._projection_geometry_for_source(
        solve, solve.projection_sources[0],
    )

    assert geometry == [private]
    assert origin == "source"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_frontend_formats_evidence_labels_and_selects_geometry_fallback():
    """Execute the two pure JS policy helpers, not a Python mirror of them."""
    web = Path(__file__).parents[1] / "atlas_camera" / "comfy" / "web" / "atlas_blockout.js"
    source = web.read_text(encoding="utf-8")
    helper_source = []
    for name in ("projectionEvidenceLabel", "projectionGeometryEntries"):
        match = re.search(rf"function {name}\([^)]*\) \{{[^{{}}]*\}}", source)
        assert match, f"{name} pure helper missing from atlas_blockout.js"
        helper_source.append(match.group(0))
    program = "\n".join(helper_source) + """
console.log(JSON.stringify({
  labels: [projectionEvidenceLabel("photographed"),
           projectionEvidenceLabel("generated"),
           projectionEvidenceLabel("unknown"),
           projectionEvidenceLabel("legacy")],
  photo: projectionGeometryEntries(
    {evidence_type:"photographed", proxy_geometry:[]},
    {proxy_geometry:["primary"]}),
  generated: projectionGeometryEntries(
    {evidence_type:"generated", proxy_geometry:[]},
    {proxy_geometry:["primary"]}),
  own: projectionGeometryEntries(
    {evidence_type:"photographed", proxy_geometry:["private"]},
    {proxy_geometry:["primary"]})
}));
"""
    result = subprocess.run(
        ["node", "--input-type=module", "-e", program],
        check=True, capture_output=True, text=True,
    )
    assert json.loads(result.stdout) == {
        "labels": ["PHOTO", "GENERATED", "SOURCE", "SOURCE"],
        "photo": ["primary"], "generated": [], "own": ["private"],
    }

    build_call = re.search(
        r"function buildPatchSources\([^)]*\) \{.*?"
        r"for \(const e of projectionGeometryEntries\(src, data\)\)",
        source, re.DOTALL,
    )
    assert build_call, "buildPatchSources bypasses projectionGeometryEntries"
    legend_call = re.search(
        r"function refreshLayerLegend\(\) \{.*?"
        r"projectionEvidenceLabel\(s\.evidence_type\)",
        source, re.DOTALL,
    )
    assert legend_call, "projection-source legend bypasses projectionEvidenceLabel"
