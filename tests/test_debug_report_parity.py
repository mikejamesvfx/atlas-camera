"""Frozen-expectation parity for the AtlasDebugReport -> scene_health refactor.

The flag STRINGS below were captured from the PRE-refactor inline
implementation (2026-07-18) on a maximally-flagged synthetic scene. The
refactored node (thin consumer of core.scene_health.evaluate_scene_health)
must reproduce them byte-identically — plus exactly one appended
scale_unverified flag (the deliberate P0 addition, changelogged).
"""

import base64
import io
import json

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("PIL")

from atlas_camera.comfy.nodes import AtlasDebugReport
from atlas_camera.core.scene_health import evaluate_scene_health
from atlas_camera.core.schema import (
    AtlasExtrinsics,
    AtlasIntrinsics,
    AtlasProxyPrimitive,
    AtlasSolve,
    LatentCamera,
    ProjectionSource,
)

FROZEN_LEGACY_FLAGS = [
    "camera height <= 0 — ground-based features (ground depth, "
    "band_geometry=ground) will fail",
    "band_fg: ZERO vertices — this layer contributes no geometry (empty band, "
    "exclude-everything scope, or a failed flat-mode region)",
    "band_bg: matte covers only 0.00% of the frame — layer will paint almost nothing",
    "band GAP between band_fg (far 5.00m) and band_bg (near 9.00m)",
    "scope status_1: scope band_fg: FALLBACK band-only (segment coverage 0.0%)",
    "depth: 5.0% of raw depth is NEGATIVE (DA3 watch-item) — ground-pinning "
    "renormalizes it, but suspect this first if a band's geometry misbehaves "
    "on this shot",
]


def _black_png_b64():
    from PIL import Image
    img = Image.new("L", (16, 16), 0)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _cam(w=640, h=480):
    intr = AtlasIntrinsics(image_width=w, image_height=h, fx_px=500.0, fy_px=500.0,
                           cx_px=w / 2, cy_px=h / 2, focal_length_mm=28.0,
                           sensor_width_mm=36.0)
    vm = ((1.0, 0, 0, 0), (0, 1.0, 0, 0.5), (0, 0, 1.0, 0), (0, 0, 0, 1.0))
    extr = AtlasExtrinsics(camera_position=(0.0, -0.5, 0.0), camera_view_matrix=vm)
    return LatentCamera(intrinsics=intr, extrinsics=extr)


def _prim(name, nverts):
    return AtlasProxyPrimitive(name=name, primitive_type="mesh",
                               metadata={"n_vertices": nverts,
                                         "n_faces": max(0, nverts - 2)})


def _flagged_solve():
    solve = AtlasSolve(camera=_cam())
    solve.debug_metadata["scale_source"] = "assumed_default"
    solve.projection_sources = [
        ProjectionSource(camera=_cam(320, 240), name="band_fg", priority=0,
                         proxy_geometry=[_prim("fg_mesh", 0)],
                         metadata={"projection_mode": "clean_plate",
                                   "band_geometry": "relief",
                                   "near_m": 0.0, "far_m": 5.0}),
        ProjectionSource(camera=_cam(320, 240), name="band_bg", priority=10,
                         proxy_geometry=[_prim("bg_mesh", 1000)],
                         mask_b64=_black_png_b64(),
                         metadata={"projection_mode": "clean_plate",
                                   "band_geometry": "card",
                                   "near_m": 9.0, "far_m": 20.0}),
    ]
    return solve


class _FakeDepth:
    model_id = "fake/depth-model"
    is_metric = True
    near, far = 0.5, 30.0
    image_width, image_height = 64, 48
    metadata = {"negative_fraction": 0.05}
    depth = None


def _run(tmp_path):
    path = tmp_path / "parity_debug.json"
    AtlasDebugReport().report(
        _flagged_solve(), depth=_FakeDepth(), file_path=str(path),
        status_1="scope band_fg: FALLBACK band-only (segment coverage 0.0%)")
    return json.loads(path.read_text(encoding="utf-8"))


def test_flags_are_byte_identical_to_pre_refactor(tmp_path):
    data = _run(tmp_path)
    assert data["flags"][:len(FROZEN_LEGACY_FLAGS)] == FROZEN_LEGACY_FLAGS
    # Exactly one deliberate addition: the scale trust flag (assumed scale).
    extra = data["flags"][len(FROZEN_LEGACY_FLAGS):]
    assert len(extra) == 1 and extra[0].startswith("scale ASSUMED")


def test_per_source_entries_preserved(tmp_path):
    data = _run(tmp_path)
    frozen_fg = {
        "name": "band_fg", "priority": 0, "projection_mode": "clean_plate",
        "band_geometry": "relief", "near_m": 0.0, "far_m": 5.0,
        "n_vertices": 0, "n_faces": 0, "n_filled_cells": None,
        "source_camera_wh": [320, 240], "matte_coverage": None,
        "has_extend_mask": False,
    }
    fg = data["projection_sources"][0]
    for key, value in frozen_fg.items():
        assert fg[key] == value, key
    bg = data["projection_sources"][1]
    assert bg["n_vertices"] == 1000 and bg["matte_coverage"] == 0.0
    assert data["camera"]["camera_height_m"] == -0.5
    assert data["camera"]["scale_source"] == "assumed_default"
    assert data["schema"] == 1


def test_engine_severities_and_level():
    health = evaluate_scene_health(
        _flagged_solve(), _FakeDepth(),
        scope_statuses={"status_1": "FALLBACK"},
        matte_coverage_fn=AtlasDebugReport._matte_coverage)
    codes = {f.code: f.severity for f in health.flags}
    assert codes["camera_below_ground"] == "fail"
    assert codes["zero_vertex_layer"] == "fail"
    assert codes["near_empty_matte"] == "warn"
    assert codes["band_gap"] == "warn"
    assert codes["scale_unverified"] == "warn"
    assert health.level == "fail"
    assert health.scale.status == "assumed"


def test_engine_pass_level_on_clean_scene():
    solve = AtlasSolve(camera=_cam())
    # A level camera above ground + verified scale + no layers = pass.
    solve.camera.extrinsics.camera_view_matrix = (
        (1.0, 0, 0, 0), (0, 1.0, 0, -1.6), (0, 0, 1.0, 0), (0, 0, 0, 1.0))
    solve.debug_metadata["scale_source"] = "manual_override"
    health = evaluate_scene_health(solve)
    assert health.level == "pass"
    assert health.flags == []


def test_engine_degrades_without_matte_fn():
    health = evaluate_scene_health(_flagged_solve(), matte_coverage_fn=None)
    # Coverage unknown -> no near-empty flag, but zero-vertex still fires.
    assert "near_empty_matte" not in {f.code for f in health.flags}
    assert "zero_vertex_layer" in {f.code for f in health.flags}


def _mesh_qa_solve(far_m=None):
    solve = AtlasSolve(camera=_cam())
    solve.camera.extrinsics.camera_view_matrix = (
        (1.0, 0, 0, 0), (0, 1.0, 0, -1.6), (0, 0, 1.0, 0), (0, 0, 0, 1.0))
    solve.debug_metadata["scale_source"] = "manual_override"
    prim = AtlasProxyPrimitive(
        name="m", primitive_type="mesh",
        metadata={"source": "depth_relief_mesh", "n_vertices": 100,
                  "n_faces": 98, "torn_fraction": 0.8,
                  "quad_coherence": True, "stretch_ratio_p95": 20.0,
                  "stretch_fraction_gt12": 0.3})
    solve.projection_sources = [ProjectionSource(
        camera=_cam(320, 240), name="xray", proxy_geometry=[prim],
        metadata={"projection_mode": "clean_plate", "near_m": 0.0,
                  "far_m": far_m})]
    return solve


def test_mesh_qa_flags_fire_on_unbanded_layer():
    """Ported outlier/stretched-edge tier: torn + stretch as health flags."""
    health = evaluate_scene_health(_mesh_qa_solve(far_m=None))
    codes = {f.code for f in health.flags}
    assert "torn_excessive" in codes
    assert "stretch_excessive" in codes
    entry = health.per_layer[0]
    assert entry["torn_fraction"] == 0.8
    assert entry["stretch_ratio_p95"] == 20.0
    assert entry["quad_coherence"] is True


def test_torn_flag_skips_band_clipped_layers():
    """Global torn_fraction always reads high on a deliberately band-clipped
    layer (found live: a correct narrow band flagged at 73.8%) — the torn
    check is scoped to layers without a finite far edge; stretch is not."""
    health = evaluate_scene_health(_mesh_qa_solve(far_m=5.0))
    codes = {f.code for f in health.flags}
    assert "torn_excessive" not in codes
    assert "stretch_excessive" in codes


def test_camera_looks_up_flag():
    """Gravity-flip guard (D810 haze incident): up-looking solves warn."""
    solve = AtlasSolve(camera=_cam())
    solve.debug_metadata["scale_source"] = "manual_override"
    # World matrix whose -Z column (forward) points up ~30 deg.
    import math
    s, c = math.sin(math.radians(30)), math.cos(math.radians(30))
    solve.camera.extrinsics.camera_world_matrix = (
        (1.0, 0.0, 0.0, 0.0), (0.0, c, -s, 10.0),
        (0.0, s, c, 0.0), (0.0, 0.0, 0.0, 1.0))
    solve.camera.extrinsics.camera_view_matrix = (
        (1.0, 0.0, 0.0, 0.0), (0.0, c, s, -10.0 * c),
        (0.0, -s, c, 10.0 * s), (0.0, 0.0, 0.0, 1.0))
    health = evaluate_scene_health(solve)
    codes = {f.code for f in health.flags}
    assert "camera_looks_up" in codes
    assert any("gravity flipped" in f.message for f in health.flags)

    # A level camera does not flag.
    level = AtlasSolve(camera=_cam())
    level.debug_metadata["scale_source"] = "manual_override"
    level.camera.extrinsics.camera_view_matrix = (
        (1.0, 0, 0, 0), (0, 1.0, 0, -1.6), (0, 0, 1.0, 0), (0, 0, 0, 1.0))
    assert "camera_looks_up" not in {f.code for f in evaluate_scene_health(level).flags}
